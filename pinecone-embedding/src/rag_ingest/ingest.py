import argparse
import os

import polars as pl

from pinecone_setup import init_pinecone
from s3_loader import load_parquet_from_s3
from embed_dense import embed_dense
from embed_sparse import embed_sparse
from upsert import build_vectors_from_df, upsert


def parse_args():
    parser = argparse.ArgumentParser(description="RAG Ingestion Pipeline")

    parser.add_argument(
        "--index-name",
        required=True,
        help="Name of the Pinecone index to write to",
    )

    parser.add_argument(
        "--bucket",
        required=True,
        help="S3 bucket where parquet files live",
    )

    parser.add_argument(
        "--prefix",
        required=False,
        default=None,
        help="S3 prefix containing parquet shards",
    )

    parser.add_argument(
        "--single-key",
        required=False,
        default=None,
        help="Single parquet file key to load instead of directory",
    )

    parser.add_argument(
        "--metadata-cols",
        nargs="*",       # Allow 0 or more arguments
        required=False,  # Not required
        default=[],      # Default to empty list
        help="Column names to attach as metadata to each vector. If omitted, all columns are used.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize Pinecone index
    pc, index = init_pinecone(
        index_name=args.index_name,
        dimension=1024,
        region="us-east-1",
    )

    # Load parquet(s) from S3 or local
    df = load_parquet_from_s3(
        bucket=args.bucket,
        prefix=args.prefix,
        single_key=args.single_key,
        region="us-east-1",
    )
    print(f"\n[debug] Loaded dataframe: {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"[debug] Columns: {df.columns}")
    if "doc_id" in df.columns:
        unique_docs = df["doc_id"].n_unique()
        print(f"[debug] Unique doc_ids: {unique_docs}")
        print(f"[debug] doc_id values: {df['doc_id'].unique().to_list()}")
    if "source_name" in df.columns:
        print(f"[debug] Source files: {df['source_name'].unique().to_list()}")

    # Drop rows where the text column is null or empty
    df = df.filter(pl.col("text").is_not_null() & (pl.col("text").str.strip_chars().str.len_chars() > 0))
    print(f"[debug] Rows after dropping empty text: {df.shape[0]}")

    # Logic to determine metadata columns
    if not args.metadata_cols:
        meta_cols = df.columns
    else:
        meta_cols = args.metadata_cols
    print(f"[debug] Metadata columns: {meta_cols}")

    # Generate dense embeddings
    print(f"\n[debug] Generating dense embeddings for {df.shape[0]} rows...")
    dense_vecs = embed_dense(
        pc=pc,
        df=df,
        text_col="text",
        embed_model="llama-text-embed-v2",
        batch_size=48,
    )
    print(f"[debug] Dense embeddings generated: {len(dense_vecs)}")

    # Generate sparse embeddings
    print(f"[debug] Generating sparse embeddings for {df.shape[0]} rows...")
    sparse_vecs = embed_sparse(
        pc=pc,
        df=df,
        text_col="text",
        embed_model="pinecone-sparse-english-v0",
        batch_size=96,
    )
    print(f"[debug] Sparse embeddings generated: {len(sparse_vecs)}")

    # Build metadata + vector objects
    vectors, ids = build_vectors_from_df(
        df=df,
        dense_embeddings=dense_vecs,
        sparse_embeddings=sparse_vecs,
        metadata=meta_cols,
        id_template="{doc_id}#page{page}",
    )
    print(f"[debug] Vectors built: {len(vectors)}")
    print(f"[debug] Sample IDs (first 5): {ids[:5]}")

    # Extract metadata list in the same order
    metadata_list = [v["metadata"] for v in vectors]

    # Upsert into Pinecone
    print(f"\n[debug] Upserting {len(ids)} vectors to index '{args.index_name}'...")
    stats = upsert(
        index=index,
        ids=ids,
        dense_vectors=dense_vecs,
        sparse_vectors=sparse_vecs,
        metadata=metadata_list,
        batch_size=100,
    )

    print("\nIngestion Complete!")
    print(stats)


if __name__ == "__main__":
    main()
