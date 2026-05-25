"""
load.py

Ingests PCS-inspired climate scenario content chunks from PCS_storylines.py into a ChromaDB client.
Called at server startup via main.py lifespan with an EphemeralClient.
Run explicitly with `python backend/data/load.py` to rebuild the local chroma_db/.
"""
import os
import sys

# Allow import of PCS_storylines.py from sibling data/ directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.PCS_storylines import STORYLINES

def ingest_to_client(client):
    """
    Drops + recreates the PCS_climate_content collection in the given ChromaDB client.
    Then, batch-ingests all 20 STORYLINE chunks with metadata.
    Returns the populated collection.
    """
    # Drop + recreate existing collections for a clean load every time
    existing_collections = [c.name for c in client.list_collections()]
    if "PCS_climate_content" in existing_collections:
        print("Existing vipr_content collection found — deleting to reload fresh...")
        client.delete_collection("PCS_climate_content")

    collection = client.create_collection(
        name="PCS_climate_content",
        metadata={"description": "PCS-inspired climate scenario content chunks for Soule, France"}
    ) # PCS_climate_content collection

    # Construct parallel lists for batch ingestion
    ids = []
    documents = []
    metadatas = []

    for chunk in STORYLINES:
        ids.append(chunk["chunk_id"])
        documents.append(chunk["content_text"])
        metadata = {
            "storyline": chunk["storyline"],
            "elevation_band": chunk["elevation_band"],
            "season": chunk["season"],
            "variable_type": chunk["variable_type"],
            "abstraction_level": chunk["abstraction_level"],
            "analog_type": chunk["analog_type"],
            "analog_reference": chunk["analog_reference"] if chunk["analog_reference"] is not None else "",
            "farmer_concern": chunk["farmer_concern"],
        } # metadata dictionary for each climate data chunk
        metadatas.append(metadata)
    
    # Ingest 20 climate chunks to ChromaDB with auto-generated vector embedding in 1 batch operation
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    ) # searchable climate database

    # Verify ingestion by checking count - should be 20
    count = collection.count()
    print(f"Ingested: {count} chunks into PCS_climate_content.")
    return collection

if __name__ == "__main__":
    # Local dev: persist to disk at chroma_db/
    import chromadb
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    client = chromadb.PersistentClient(path=os.path.join(project_root, "chroma_db"))
    ingest_to_client(client)
    print("Local climate database ready.")