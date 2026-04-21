#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF → Parquet extractor (CSV input with PDF URLs)
- Reads a CSV where one column contains URLs to PDFs hosted on the internet
- All other CSV columns are written as metadata columns on every page row
- Extracts page text with layout-aware ordering + optional OCR fallback
- Writes one Parquet file per PDF to a local directory or S3 prefix

Args:
  --input      : path to input CSV (local file or s3://bucket/key.csv)
  --out        : local directory OR s3://bucket/prefix/ to write Parquet files
  --url-column : name of the CSV column containing PDF URLs (default: "Report Link")
  --no-ocr     : disable OCR fallback
"""

import argparse
import hashlib
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from urllib.parse import urlparse, parse_qs
import re
import tempfile
import shutil

import requests
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Optional pyarrow fs for s3 write
try:
    import pyarrow.fs as pafs
except Exception:
    pafs = None

import boto3

# ------------------------ Tunables ------------------------

MIN_TEXT_LEN   = int(os.getenv("MIN_TEXT_LEN", "60"))   # OCR trigger threshold
OCR_DPI        = int(os.getenv("OCR_DPI", "250"))
OCR_LANG       = os.getenv("OCR_LANG", "eng")
OCR_TIMEOUT_S  = int(os.getenv("OCR_TIMEOUT_S", "12"))
MAX_OCR_PAGES  = int(os.getenv("MAX_OCR_PAGES", "20"))
ALLOW_OCR      = os.getenv("ALLOW_OCR", "true").lower() != "false"  # --no-ocr overrides

# Two-column detection (tuned)
MIN_GAP_RATIO         = 0.12     # ~12% of page width
EDGE_MARGIN_RATIO     = 0.15     # ignore mids near outer 15% bands
MIN_BLOCKS_PER_COLUMN = 4        # require more evidence per column

# Extra guards
GUTTER_MID_BAND       = (0.35, 0.65)  # require mid in central 35–65% of width
TRIM_TOP_BOTTOM_RATIO = 0.08          # ignore top/bottom 8% of items when finding gutter

Y_TOL = 2.0  # row grouping tolerance (points)

APPLY_ENUMERATOR_CLEAN = True

# ------------------------ Small helpers ------------------------

def sha256_text(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def as_str(s: Optional[str]) -> Optional[str]:
    if s is None: return None
    s2 = str(s).strip()
    return s2 if s2 else None

def is_s3_uri(uri: str) -> bool:
    return uri.strip().lower().startswith("s3://")

def split_s3_uri(uri: str) -> Tuple[str, str]:
    u = uri.strip()[5:]  # drop s3://
    parts = u.split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) == 2 else ""
    return bucket, key

def slugify_filename(name: str) -> str:
    base = os.path.basename(name)
    base = re.sub(r"[^\w\-. ]+", "_", base)
    return base

# ------------------------ OCR ------------------------

def ocr_page_to_text(page: fitz.Page, dpi: int = OCR_DPI, lang: str = OCR_LANG) -> str:
    """Render a PDF page to an image and extract text via Tesseract OCR.

    Used as a fallback when PyMuPDF's native text extraction yields too little
    content (e.g. scanned pages). The page is rasterized at the given DPI,
    then passed to Tesseract. Returns an empty string on timeout.

    Args:
        page: A PyMuPDF page object to OCR.
        dpi: Resolution for rasterization (higher = better accuracy, slower).
        lang: Tesseract language code (e.g. 'eng').

    Returns:
        Extracted text with normalized newlines, or '' on failure.
    """
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    try:
        txt = pytesseract.image_to_string(img, lang=lang, timeout=OCR_TIMEOUT_S) or ""
    except RuntimeError:
        return ""
    return txt.replace("\r\n", "\n").replace("\r", "\n").strip()

# ------------------------ Layout-aware extraction ------------------------

def _collect_items_dict(page: fitz.Page):
    """Extract all text line items from a page using PyMuPDF's dict mode.

    Iterates over text blocks and their lines, collecting each non-empty line
    as a dict with its block index, bounding-box coordinates (x, y, x1), and
    concatenated span text. Non-text blocks (e.g. images) are skipped.

    Args:
        page: A PyMuPDF page object.

    Returns:
        List of dicts, each with keys: 'bidx' (block index), 'y' (top y),
        'x' (left x), 'x1' (right x), 'text' (line text content).
    """
    d = page.get_text("dict") or {}
    items = []
    for b_idx, b in enumerate(d.get("blocks", [])):
        if b.get("type", 0) != 0:
            continue
        for ln in b.get("lines", []):
            spans = ln.get("spans", [])
            if not spans:
                continue
            x0, y0, x1, y1 = ln.get("bbox", (0, 0, 0, 0))
            txt = "".join((s.get("text") or "") for s in spans)
            txt = txt.replace("\r\n", "\n").replace("\r", "\n")
            if not txt.strip():
                continue
            items.append({"bidx": b_idx, "y": y0, "x": x0, "x1": x1, "text": txt.rstrip()})
    return items

def _items_to_columns(items: List[dict], page_width: float):
    """Detect whether a page has a two-column layout and split items accordingly.

    Finds the largest horizontal gap between line centers (the "gutter") after
    trimming header/footer bands. The gutter must pass several sanity checks:
    it must be in the central band of the page, wide enough relative to page
    width, and have enough items on each side.

    Args:
        items: Line items from _collect_items_dict.
        page_width: The width of the PDF page in points.

    Returns:
        Tuple of (left_items, right_items, gutter_x). If no two-column layout
        is detected, returns (items, [], None).
    """
    # Need enough signals to even consider a split
    need = max(4, 2 * MIN_BLOCKS_PER_COLUMN)
    if len(items) < need:
        return items, [], None

    # Trim top/bottom bands to avoid headers/footers polluting gutter search
    ys = sorted(it["y"] for it in items)
    if ys:
        k = int(len(ys) * TRIM_TOP_BOTTOM_RATIO)
        y_lo = ys[k] if k < len(ys) else ys[0]
        y_hi = ys[-k-1] if k > 0 else ys[-1]
        trimmed = [it for it in items if y_lo <= it["y"] <= y_hi]
        if len(trimmed) >= need:
            work = trimmed
        else:
            work = items
    else:
        work = items

    # Use true line centers (x0+x1)/2. Fallback to x if x1 missing.
    centers = []
    for it in work:
        x0 = it.get("x", 0.0)
        x1 = it.get("x1", None)
        c  = 0.5 * (x0 + x1) if isinstance(x1, (int, float)) else float(x0)
        centers.append(c)

    if len(centers) < need:
        return items, [], None

    centers.sort()
    max_gap, mid = 0.0, None
    for i in range(len(centers) - 1):
        gap = centers[i+1] - centers[i]
        if gap > max_gap:
            max_gap = gap
            mid = 0.5 * (centers[i+1] + centers[i])

    if mid is None:
        return items, [], None

    # Stronger interior checks
    if not (EDGE_MARGIN_RATIO * page_width < mid < (1.0 - EDGE_MARGIN_RATIO) * page_width):
        return items, [], None

    lo_band, hi_band = GUTTER_MID_BAND
    if not (lo_band * page_width <= mid <= hi_band * page_width):
        return items, [], None

    if max_gap < (MIN_GAP_RATIO * page_width):
        return items, [], None

    # Split by center (not x+100)
    left, right = [], []
    for it in items:
        x0 = it.get("x", 0.0)
        x1 = it.get("x1", None)
        c  = 0.5 * (x0 + x1) if isinstance(x1, (int, float)) else float(x0)
        (left if c <= mid else right).append(it)

    if len(left) < MIN_BLOCKS_PER_COLUMN or len(right) < MIN_BLOCKS_PER_COLUMN:
        return items, [], None

    return left, right, mid


def _sort_items(items: List[dict]):
    """Sort line items into reading order (top-to-bottom, left-to-right).

    Groups items into rows based on vertical proximity (within Y_TOL points),
    sorts each row left-to-right by x-coordinate, then flattens into a single
    ordered list.

    Args:
        items: Line items from _collect_items_dict or _items_to_columns.

    Returns:
        List of items reordered into natural reading sequence.
    """
    if not items:
        return []
    items = sorted(items, key=lambda it: it["y"])
    rows, cur = [], []
    last_y = None
    for it in items:
        if last_y is None or abs(it["y"] - last_y) <= Y_TOL:
            cur.append(it)
            last_y = it["y"] if last_y is None else (last_y + it["y"]) / 2.0
        else:
            cur.sort(key=lambda z: (round(z["x"], 2), z["bidx"]))
            rows.append(cur)
            cur = [it]
            last_y = it["y"]
    if cur:
        cur.sort(key=lambda z: (round(z["x"], 2), z["bidx"]))
        rows.append(cur)
    return [it for row in rows for it in row]

def page_text_layout(page: fitz.Page) -> str:
    """Extract text from a PDF page with layout-aware reading order.

    Handles both single-column and two-column layouts. For two-column pages,
    text is read left column first (top to bottom), then right column, with
    columns separated by a blank line. For single-column pages, items are
    sorted in standard reading order.

    Args:
        page: A PyMuPDF page object.

    Returns:
        The full page text as a single string in reading order.
    """
    items = _collect_items_dict(page)
    if not items:
        return ""
    left, right, gutter = _items_to_columns(items, page.rect.width)

    def join_items(its: List[dict]) -> str:
        ordered = _sort_items(its)
        return "\n".join(it["text"] for it in ordered if it["text"].strip())

    if gutter is None:
        return join_items(items).strip()
    return "\n\n".join(s for s in (join_items(left), join_items(right)) if s).strip()

# ------------------------ Enumerator cleaner ------------------------

_BARE_ENUM_RE = re.compile(
    r'^([A-Z]\.|\([A-Za-z]\)|\d+\.|\([0-9]{1,3}\)|[ivxlcdm]+\.|\([ivxlcdm]+\))\s*$',
    re.IGNORECASE
)

def remove_orphan_enumerators(text: str) -> str:
    """Remove standalone enumerator lines that are artifacts of column splitting.

    PDF extraction can produce bare enumerator lines (e.g. "A.", "(3)", "iv.")
    that were separated from their content by the layout parser. This function
    drops enumerator-only lines when the next non-empty line is also a bare
    enumerator, indicating a sequence of orphaned labels rather than a real list.

    Args:
        text: Raw extracted page text.

    Returns:
        Cleaned text with orphan enumerator lines removed.
    """
    lines = str(text or "").splitlines()
    out, i, n = [], 0, len(lines)

    def is_bare_enum(s: str) -> bool:
        return bool(_BARE_ENUM_RE.match(s.strip()))

    def next_nonempty(j: int):
        k = j
        while k < n and not lines[k].strip():
            k += 1
        return k if k < n else None

    while i < n:
        raw = lines[i]
        if is_bare_enum(raw):
            k = next_nonempty(i + 1)
            if k is None:
                i += 1
                continue
            nxt = lines[k].strip()
            if is_bare_enum(nxt):
                i += 1
                continue
            out.append(raw)
            i += 1
            continue
        out.append(raw)
        i += 1
    return "\n".join(out)

# ------------------------ Extraction core ------------------------

def extract_pdf_to_records(pdf_path: Path, metadata: Dict, doc_id: Optional[str] = None) -> List[Dict]:
    """Extract text from every page of a PDF and return one record per page.

    For each page, attempts layout-aware text extraction first. Falls back to
    OCR if the extracted text is shorter than MIN_TEXT_LEN and the page has
    very few text blocks (likely a scanned image). Optionally cleans orphan
    enumerators. Each record includes the text, metadata, and a content hash.

    Args:
        pdf_path: Path to the PDF file on disk.
        metadata: Dict of metadata columns (from the CSV row) written into
                  every page record.
        doc_id: Optional stable identifier for this document (e.g. a URL
                parameter). Falls back to a SHA1 hash of the file path if
                not provided.

    Returns:
        List of dicts, one per page, with keys: doc_id, source_name, page,
        text, is_ocr, char_len, sha256, extracted_at, plus all keys in metadata.
    """
    source_name = pdf_path.name
    if doc_id is None:
        doc_id = hashlib.sha1(str(pdf_path).encode("utf-8")).hexdigest()[:20]
    ts = now_iso()
    records: List[Dict] = []
    ocr_used = 0

    with fitz.open(str(pdf_path)) as doc:
        for i in range(doc.page_count):
            page = doc.load_page(i)
            page_num = i + 1

            txt = page_text_layout(page)
            is_ocr = False

            needs_ocr = (
                ALLOW_OCR
                and len(txt.strip()) < MIN_TEXT_LEN
                and ocr_used < MAX_OCR_PAGES
            )
            if needs_ocr:
                items = _collect_items_dict(page)
                if len(items) <= 2:
                    txt_ocr = ocr_page_to_text(page, OCR_DPI, OCR_LANG)
                    if len(txt_ocr.strip()) > len(txt.strip()):
                        txt = txt_ocr
                        is_ocr = True
                        ocr_used += 1

            if APPLY_ENUMERATOR_CLEAN and txt:
                txt = remove_orphan_enumerators(txt)

            rec = {
                "doc_id": doc_id,
                "source_name": source_name,
                "page": page_num,
                "text": txt,
                "is_ocr": is_ocr,
                "char_len": len(txt),
                "sha256": sha256_text(f"{source_name}|{page_num}|{txt}"),
                "extracted_at": ts,
                **metadata,
            }
            records.append(rec)

    return records

# ------------------------ Parquet write ------------------------

def write_parquet(records: List[Dict], out_path: str):
    """Write a list of page records to a Parquet file, locally or on S3.

    Converts records to a Pandas DataFrame, casts partition columns to string
    type, then writes via PyArrow. For S3 paths, uses pyarrow.fs.S3FileSystem
    with credentials from environment variables.

    Args:
        records: List of page record dicts from extract_pdf_to_records.
        out_path: Destination path — either a local file path or an s3:// URI.
    """
    if not records:
        print(f"[warn] No records to write for {out_path}")
        return
    df = pd.DataFrame.from_records(records)
    if "source_name" in df.columns:
        df["source_name"] = df["source_name"].astype("string")
    table = pa.Table.from_pandas(df, preserve_index=False)

    if out_path.startswith("s3://"):
        if pafs is None:
            raise RuntimeError("pyarrow.fs is not available; cannot write to S3")
        _, _, rest = out_path.partition("s3://")
        bucket, _, key = rest.partition("/")
        fs = pafs.S3FileSystem(
            region=os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1")),
            access_key=os.getenv("AWS_ACCESS_KEY_ID"),
            secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )
        with fs.open_output_stream(f"{bucket}/{key}") as sink:
            pq.write_table(table, sink)
        print(f"[ok] wrote {len(df)} rows → {out_path}")
    else:
        out_path = str(Path(out_path))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        pq.write_table(table, out_path)
        print(f"[ok] wrote {len(df)} rows → {out_path}")

# ------------------------ CSV + HTTP helpers ------------------------

def read_csv(input_path: str) -> pd.DataFrame:
    """Read the input CSV from a local path or S3 URI.

    Args:
        input_path: Local file path or s3://bucket/key.csv.

    Returns:
        DataFrame with all columns from the CSV.
    """
    if is_s3_uri(input_path):
        bucket, key = split_s3_uri(input_path)
        s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1")))
        obj = s3.get_object(Bucket=bucket, Key=key)
        return pd.read_csv(obj["Body"])
    return pd.read_csv(input_path)


def download_url_pdf(url: str, local_dir: Path) -> Path:
    """Download a PDF from an HTTP/HTTPS URL to a local directory.

    Args:
        url: HTTP/HTTPS URL pointing to a PDF.
        local_dir: Local directory to save the file into.

    Returns:
        Path to the downloaded local file.
    """
    filename = slugify_filename(url.split("?")[0].rstrip("/").split("/")[-1])
    if not filename.lower().endswith(".pdf"):
        filename += ".pdf"
    local_file = local_dir / filename
    resp = requests.get(url, timeout=60, stream=True)
    resp.raise_for_status()
    with open(local_file, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    return local_file


def build_out_path(pdf_path: Path, out_base: str, row_index: int) -> str:
    """Build the output Parquet file path from the PDF stem and --out base.

    Args:
        pdf_path: Local path to the downloaded PDF.
        out_base: Local directory or s3://bucket/prefix/.
        row_index: CSV row index, prepended to filename to ensure uniqueness
                   when multiple URLs share the same filename.

    Returns:
        Full output path string (local or s3://).
    """
    stem = f"row{row_index}_{pdf_path.stem}_text.parquet"
    if out_base.startswith("s3://"):
        out_bucket, out_key_base = split_s3_uri(out_base)
        out_key_base = out_key_base.strip("/")
        prefix = (out_key_base + "/") if out_key_base else ""
        return f"s3://{out_bucket}/{prefix}{stem}"
    return str(Path(out_base) / stem)



# ------------------------ CLI ------------------------

def main():
    """CLI entry point: read CSV, download PDFs from URLs, extract text, write Parquet.

    Reads an input CSV where one column contains PDF URLs. All other columns
    are written as metadata on every page row. Downloads each PDF to a temp
    directory, processes it, writes a Parquet file, and cleans up on exit.
    """
    ap = argparse.ArgumentParser(description="PDF → Parquet (CSV with PDF URLs)")
    ap.add_argument("--input",      required=True, help="Path to input CSV (local or s3://bucket/key.csv)")
    ap.add_argument("--out",        required=True, help="Output directory or s3://bucket/prefix/")
    ap.add_argument("--url-column", default="Report Link", help="CSV column containing PDF URLs (default: 'Report Link')")
    ap.add_argument("--no-ocr",     action="store_true", help="Disable OCR fallback entirely")
    ap.add_argument("--limit",      type=int, default=0, help="Max number of rows to process (0 = all)")
    args = ap.parse_args()

    global ALLOW_OC
    if args.no_ocr:
        ALLOW_OCR = False

    print(f"[info] reading CSV: {args.input}")
    df = read_csv(args.input)

    if args.url_column not in df.columns:
        print(f"[error] URL column '{args.url_column}' not found in CSV. "
              f"Available columns: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    if args.limit and args.limit > 0:
        df = df.head(args.limit)

    metadata_columns = [c for c in df.columns if c != args.url_column]
    print(f"[info] {len(df)} rows | metadata columns: {metadata_columns}")

    total_pdfs = 0
    total_pages = 0
    t0 = time.time()

    tmp_root = Path(tempfile.mkdtemp(prefix="pdf-extract-"))
    tmp_in = tmp_root / "in"
    tmp_in.mkdir(parents=True, exist_ok=True)

    try:
        for i, row in df.iterrows():
            url = str(row[args.url_column]).strip()
            if not url or url.lower() == "nan":
                print(f"[warn] row {i}: empty URL, skipping")
                continue

            metadata = {col: row[col] for col in metadata_columns}
            metadata[args.url_column] = url

            params = parse_qs(urlparse(url).query)
            doc_id = params.get("apId", [None])[0]
            if doc_id:
                print(f"[info] row {i}: using apId={doc_id} as doc_id")
            else:
                print(f"[warn] row {i}: no apId found in URL, falling back to hash")

            print(f"[info] row {i}: downloading {url}")
            try:
                local_pdf = download_url_pdf(url, tmp_in)
            except Exception as e:
                print(f"[error] row {i}: failed to download {url}: {e}", file=sys.stderr)
                continue

            print(f"[info] row {i}: extracting {local_pdf.name}")
            try:
                records = extract_pdf_to_records(local_pdf, metadata, doc_id=doc_id)
                out_path = build_out_path(local_pdf, args.out, i)
                write_parquet(records, out_path)
                total_pdfs += 1
                total_pages += len(records)
            except Exception as e:
                print(f"[error] row {i}: failed to process {local_pdf.name}: {e}", file=sys.stderr)

        dt = time.time() - t0
        print(f"[done] processed {total_pdfs} PDFs, {total_pages} pages in {dt:.1f}s")

    finally:
        try:
            shutil.rmtree(tmp_root)
        except Exception:
            pass

if __name__ == "__main__":
    main()

