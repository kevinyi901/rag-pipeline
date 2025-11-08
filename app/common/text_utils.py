def build_context_string(retrieved_chunks: list, max_chunks: int | None = None) -> str:
    if not retrieved_chunks:
        return "No documents were retrieved."


    items = retrieved_chunks if max_chunks is None else retrieved_chunks[:max_chunks]
    lines = []
    for m in items:
        md = m.get("metadata", {})
        chunk_text = md.get("chunk_text", "")
        state = md.get("state", "N/A")
        county = md.get("county", "N/A")
        section = md.get("section", "N/A")
        tags = []
        for k in ("obligation", "penalty", "permission"):
            if md.get(k) == "Y":
                tags.append(k.capitalize())
        tag_str = f" [{' | '.join(tags)}]" if tags else ""
        lines.append(f"- ({state}/{county}) {section}{tag_str}: {chunk_text}")
    return "\n".join(lines)



def flatten_locations_payload(filters_payload: dict) -> dict:
    normalized_filters = dict(filters_payload)
    nested_locations = normalized_filters.pop("locations", [])
    flat = []
    for loc in nested_locations or []:
        st = loc.get("state")
        for c in loc.get("counties", []) or []:
            flat.append({"state": st, "county": c})
    normalized_filters["flat_locations"] = flat
    return normalized_filters

def build_pinecone_filter(filters: dict) -> dict:
    f = {}
    if state := filters.get("state"):
        f["state"] = {"$eq": state}
    if county := filters.get("county"):
        f["county"] = {"$eq": county}


# support flattened multi-location list
    flats = filters.get("flat_locations", [])
    if flats:
        ors = []
        for loc in flats:
            clause = {}
            if loc.get("state"):
                clause["state"] = {"$eq": loc["state"]}
            if loc.get("county"):
                clause["county"] = {"$eq": loc["county"]}
            if clause:
                ors.append(clause)
        if ors:
            f = {"$or": ors}


    # boolean tags
    for tag in ("obligation", "penalty", "permission"):
        v = filters.get(tag)
        if isinstance(v, str):
            v = v.upper() == "Y"
        if v is True:
            f[tag] = {"$eq": "Y"}
    return f