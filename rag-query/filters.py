"""
Filter processing utilities for RAG pipeline.
"""
from typing import Dict, List, Any


def flatten_locations_payload(filters_payload: dict) -> dict:
    """
    Normalizes a nested filters payload. It takes a 'locations' key
    that has lists of counties and flattens it into a simple
    list of {state, county} pairs.
    
    Args:
        filters_payload: Dictionary with nested location structure
        
    Returns:
        Dictionary with flattened location list
    """
    normalized_filters = filters_payload.copy()
    nested_locations = normalized_filters.pop("locations", [])

    flat_locations = []
    if nested_locations:
        print("\n--- Flattening nested location payload ---")
        for loc_group in nested_locations:
            state = loc_group['state']
            for county in loc_group['county']:
                flat_locations.append({"state": state, "county": county})
                print(f"Added to queue: (state={state}, county={county})")

    # Add flattened list back into the filters
    normalized_filters['locations'] = flat_locations

    return normalized_filters


# need to reconfigure metadata that we would need to implement
def build_pinecone_filter(frontend_filters: dict) -> dict:
    """
    Converts a JSON filter object from the frontend into a
    Pinecone-compatible metadata filter dictionary.
    
    Args:
        frontend_filters: Dictionary containing filter criteria
        
    Returns:
        Pinecone-compatible filter dictionary
    """
    multi_select_fields = [
        'Grant Number',
        'Name',
        'Award Period',
        'Requirement Type',
        'Report Link',
        'Program Area (Award Period) (Award Reports)',
        'Primary Investigator (Award Period) (Award Reports)',
        'Institution (Award Period) (Award Reports)',
        'Project Title (Award Period) (Award Reports)',
        'Award Type (Award Period) (Award Reports)',
        'ORCiD (Award Period) (Award Reports)',
        'City (Award Period) (Award Reports)',
        'State (Award Period) (Award Reports)',
        'Country (Award Period) (Award Reports)',
    ]
    numeric_fields = [
        'Award Amount (Award Period) (Award Reports)',
        'Award Amount (Base) (Award Period) (Award Reports)',
        'Award Budget Total (Award Period) (Award Reports)',
    ]
    date_fields = [
        'Start Date (Award Period) (Award Reports)',
        'End Date (Award Period) (Award Reports)',
        'Received Date',
        'Due Date',
        'Award Start Date Total (Award Period) (Award Reports)',
        'Award End Date Total (Award Period) (Award Reports)',
    ]

    pinecone_filter = {}
    for key, value in frontend_filters.items():
        # --- Handle Multi-Select fields ---
        if key in multi_select_fields:
            if isinstance(value, list) and len(value) > 0:
                pinecone_filter[key] = {"$in": value}

        # --- Handle Numeric range fields ---
        elif key in numeric_fields:
            range_query = {}
            if 'min' in value and value['min'] is not None:
                range_query["$gte"] = value['min']
            if 'max' in value and value['max'] is not None:
                range_query["$lte"] = value['max']
            if range_query:
                pinecone_filter[key] = range_query

        # --- Handle Date range fields ---
        elif key in date_fields:
            range_query = {}
            if 'start' in value and value['start'] is not None:
                range_query["$gte"] = value['start']
            if 'end' in value and value['end'] is not None:
                range_query["$lte"] = value['end']
            if range_query:
                pinecone_filter[key] = range_query

    return pinecone_filter
