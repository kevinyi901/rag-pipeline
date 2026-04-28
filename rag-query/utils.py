"""
Utility functions for RAG pipeline.
"""
import pandas as pd
from typing import List, Dict, Any

from config import Config


def print_chunks(retrieved_chunks: List[dict]) -> None:
    """
    Print retrieved chunks in a formatted table.
    
    Args:
        retrieved_chunks: List of retrieved chunk dictionaries
    """
    if not retrieved_chunks:
        print("No chunks retrieved.")
        return

    results = []
    for c in retrieved_chunks:
        result = {}

        result['score'] = c.get('score', 0)

        metadata = c.get('metadata', {})
        result['Grant Number'] = metadata.get('Grant Number', 'N/A')
        result['Name'] = metadata.get('Name', 'N/A')
        result['Award Period'] = metadata.get('Award Period', 'N/A')
        result['Start Date'] = metadata.get('Start Date (Award Period) (Award Reports)', 'N/A')
        result['End Date'] = metadata.get('End Date (Award Period) (Award Reports)', 'N/A')
        result['Received Date'] = metadata.get('Received Date', 'N/A')
        result['Due Date'] = metadata.get('Due Date', 'N/A')
        result['Requirement Type'] = metadata.get('Requirement Type', 'N/A')
        result['Report Link'] = metadata.get('Report Link', 'N/A')
        result['Program Area'] = metadata.get('Program Area (Award Period) (Award Reports)', 'N/A')
        result['Award Amount (Base)'] = metadata.get('Award Amount (Base) (Award Period) (Award Reports)', 'N/A')
        result['Primary Investigator'] = metadata.get('Primary Investigator (Award Period) (Award Reports)', 'N/A')
        result['Institution'] = metadata.get('Institution (Award Period) (Award Reports)', 'N/A')
        result['Project Title'] = metadata.get('Project Title (Award Period) (Award Reports)', 'N/A')
        result['Award Type'] = metadata.get('Award Type (Award Period) (Award Reports)', 'N/A')
        result['Award Amount'] = metadata.get('Award Amount (Award Period) (Award Reports)', 'N/A')
        result['ORCiD'] = metadata.get('ORCiD (Award Period) (Award Reports)', 'N/A')
        result['City'] = metadata.get('City (Award Period) (Award Reports)', 'N/A')
        result['State'] = metadata.get('State (Award Period) (Award Reports)', 'N/A')
        result['Country'] = metadata.get('Country (Award Period) (Award Reports)', 'N/A')
        result['Award Budget Total'] = metadata.get('Award Budget Total (Award Period) (Award Reports)', 'N/A')
        result['Award Start Date Total'] = metadata.get('Award Start Date Total (Award Period) (Award Reports)', 'N/A')
        result['Award End Date Total'] = metadata.get('Award End Date Total (Award Period) (Award Reports)', 'N/A')
        result['text'] = metadata.get('text', 'N/A')

        results.append(result)

    df = pd.DataFrame(results)

    if 'text' in df.columns and len(df) > 0:
        df['preview'] = df['text'].str.slice(0, 30) + '...'

    output_df = df[[
        'score', 'Grant Number', 'Name', 'Award Period', 'Start Date', 'End Date',
        'Received Date', 'Due Date', 'Requirement Type', 'Report Link', 'Program Area',
        'Award Amount (Base)', 'Primary Investigator', 'Institution', 'Project Title',
        'Award Type', 'Award Amount', 'ORCiD', 'City', 'State', 'Country',
        'Award Budget Total', 'Award Start Date Total', 'Award End Date Total', 'preview'
    ]]

    print(f"Total number of chunks: {len(retrieved_chunks)}")
    print(output_df.to_string())


def print_chunks_reranking(retrieved_chunks: List[dict]) -> None:
    """
    Print retrieved and reranked chunks in a formatted table.
    
    Args:
        retrieved_chunks: List of retrieved chunk dictionaries with rerank scores
    """
    if not retrieved_chunks:
        print("No chunks retrieved.")
        return
        
    results = []
    for c in retrieved_chunks:
        result = {}

        result['score'] = c.get('score', 0)
        result['rerank_score'] = c.get('rerank_score', 0)

        metadata = c.get('metadata', {})
        result['Grant Number'] = metadata.get('Grant Number', 'N/A')
        result['Name'] = metadata.get('Name', 'N/A')
        result['Award Period'] = metadata.get('Award Period', 'N/A')
        result['Start Date'] = metadata.get('Start Date (Award Period) (Award Reports)', 'N/A')
        result['End Date'] = metadata.get('End Date (Award Period) (Award Reports)', 'N/A')
        result['Received Date'] = metadata.get('Received Date', 'N/A')
        result['Due Date'] = metadata.get('Due Date', 'N/A')
        result['Requirement Type'] = metadata.get('Requirement Type', 'N/A')
        result['Report Link'] = metadata.get('Report Link', 'N/A')
        result['Program Area'] = metadata.get('Program Area (Award Period) (Award Reports)', 'N/A')
        result['Award Amount (Base)'] = metadata.get('Award Amount (Base) (Award Period) (Award Reports)', 'N/A')
        result['Primary Investigator'] = metadata.get('Primary Investigator (Award Period) (Award Reports)', 'N/A')
        result['Institution'] = metadata.get('Institution (Award Period) (Award Reports)', 'N/A')
        result['Project Title'] = metadata.get('Project Title (Award Period) (Award Reports)', 'N/A')
        result['Award Type'] = metadata.get('Award Type (Award Period) (Award Reports)', 'N/A')
        result['Award Amount'] = metadata.get('Award Amount (Award Period) (Award Reports)', 'N/A')
        result['ORCiD'] = metadata.get('ORCiD (Award Period) (Award Reports)', 'N/A')
        result['City'] = metadata.get('City (Award Period) (Award Reports)', 'N/A')
        result['State'] = metadata.get('State (Award Period) (Award Reports)', 'N/A')
        result['Country'] = metadata.get('Country (Award Period) (Award Reports)', 'N/A')
        result['Award Budget Total'] = metadata.get('Award Budget Total (Award Period) (Award Reports)', 'N/A')
        result['Award Start Date Total'] = metadata.get('Award Start Date Total (Award Period) (Award Reports)', 'N/A')
        result['Award End Date Total'] = metadata.get('Award End Date Total (Award Period) (Award Reports)', 'N/A')
        result['text'] = metadata.get('text', 'N/A')

        results.append(result)

    df = pd.DataFrame(results)

    if 'text' in df.columns and len(df) > 0:
        df['preview'] = df['text'].str.slice(0, 30) + '...'

    output_df = df[[
        'score', 'rerank_score', 'Grant Number', 'Name', 'Award Period', 'Start Date', 'End Date',
        'Received Date', 'Due Date', 'Requirement Type', 'Report Link', 'Program Area',
        'Award Amount (Base)', 'Primary Investigator', 'Institution', 'Project Title',
        'Award Type', 'Award Amount', 'ORCiD', 'City', 'State', 'Country',
        'Award Budget Total', 'Award Start Date Total', 'Award End Date Total', 'preview'
    ]]

    print(f"Total number of chunks: {len(retrieved_chunks)}")
    print(output_df.to_string())


def generate_csv(csv_filename: str, retrieved_chunks: List[dict]) -> None:
    """
    Generate CSV file from retrieved chunks.
    
    Args:
        csv_filename: Name of the CSV file to generate
        retrieved_chunks: List of retrieved chunk dictionaries
    """
    filename = Config.get_output_path(csv_filename)

    print(f"\n--- Generating CSV File: {filename} ---")
    try:
        processed_data = []
        for chunk in retrieved_chunks:
            # Create a flat dictionary for each row
            row_data = {
                'id': chunk.get('id'),
                'score': chunk.get('score')
            }
            # Add all metadata fields as separate columns
            if 'metadata' in chunk:
                row_data.update(chunk['metadata'])
            processed_data.append(row_data)

        if not processed_data:
            print("No matches found to generate CSV.")
        else:
            # Convert to Pandas DataFrame and save to CSV
            df = pd.DataFrame(processed_data)
            df.to_csv(filename, index=False, encoding='utf-8')
            print(f"Successfully generated CSV with {len(df)} rows at: {filename}")

    except Exception as e:
        print(f"Error generating CSV: {e}")


def generate_csv_reranking(csv_filename: str, reranked_chunks: List[dict]) -> None:
    """
    Generate CSV file from reranked chunks (includes rerank_score).
    
    Args:
        csv_filename: Name of the CSV file to generate
        reranked_chunks: List of reranked chunk dictionaries
    """
    filename = Config.get_output_path(csv_filename)

    print(f"\n--- Generating CSV File: {filename} ---")
    try:
        processed_data = []
        for chunk in reranked_chunks:
            # Create a flat dictionary for each row
            row_data = {
                'id': chunk.get('id'),
                'score': chunk.get('score'),
                'rerank_score': chunk.get('rerank_score'),
            }
            # Add all metadata fields as separate columns
            if 'metadata' in chunk:
                row_data.update(chunk['metadata'])
            processed_data.append(row_data)

        if not processed_data:
            print("No matches found to generate CSV.")
        else:
            # Convert to Pandas DataFrame and save to CSV
            df = pd.DataFrame(processed_data)
            df.to_csv(filename, index=False, encoding='utf-8')
            print(f"Successfully generated CSV with {len(df)} rows at: {filename}")

    except Exception as e:
        print(f"Error generating CSV: {e}")
