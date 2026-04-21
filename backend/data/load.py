# Read content from PCS_storylines.py + load it into ChromaDB as vector embeddings
# Script is run 1x to set up climate database, then will persist locally
"""
load.py

Ingests climate data content chunks from PCS_storylines.py into 
a local ChromaDB vector database. 

Script is run once to set up climate knowledge base before starting AI climate agent using:
python backend/data/load.py

Each chunk is stored as a vector embedding with its metadata, enabling
semantic similarity search during retrieval by AI climate agent. 

ChromaDB persists the database locally in a folder called chroma_db/ at the project root.
"""
import chromadb # Import ChromaDB library
import os # Import Python's os module - build file paths on any computer
import sys # Import Python's system module - tells Python where to look for imported files

# Add backend directory to path so can import PCS_storylines
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import STORYLINES database from PCS_storylines.py
from data.PCS_storylines import STORYLINES

def load_knowledge_base():
    """
    Loads all climate content chunks from PCS_storylines.py into ChromaDB.
    
    Creates a persistent ChromaDB client stored locally in chroma_db/.
    Creates a collection called 'PCS_climate_content' if it does not exist.
    Ingests all chunks from STORYLINES from PCS_storylines.py with their metadata.
    
    ChromaDB automatically generates vector embeddings for each chunk
    using its default embedding function (all-MiniLM-L6-v2), enabling
    semantic similarity search during agent retrieval.
    """
    # Get path to project root: Adaptive AI Climate Agent for Pastoral Communities/ so ChromaDB saved database in correct location
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    chroma_path = os.path.join(project_root, "chroma_db")

    # Initialize persistent ChromaDB client
    # Data is saved locally to chroma_db/ & survives between runs
    print(f"Initializing ChromaDB at: {chroma_path}")
    # Create ChromaDB client to save data to disk 
    client = chromadb.PersistentClient(path=chroma_path)

    # Check for existing collections in ChromaDB
    existing_collections = [c.name for c in client.list_collections()]
        # Delete existing collection if it exists so clean runs
    if "PCS_climate_content" in existing_collections:
        print("Existing vipr_content collection found — deleting to reload fresh...")
        client.delete_collection("PCS_climate_content")

    # Create new empty collection: PCS_climate_content
    collection = client.create_collection(
        name="PCS_climate_content",
        metadata={"description": "PCS-inspired climate scenario content chunks for Soule, France"}
    ) # collection

    # Check 20 climate data chunks in collection
    print(f"Loading {len(STORYLINES)} content chunks into ChromaDB...")

    # Prep data for batch ingestion: create 3 empty lists for IDs, actual text, + metadata
    ids = []
    documents = []
    metadatas = []

    # Loop through every climate data chunk in STORYLINES list
    for chunk in STORYLINES:
        # Add climate data chunk id to ids list
        ids.append(chunk["chunk_id"])
        # Add climate data chunk text to list of text content
        documents.append(chunk["content_text"])
        
        # Build metadata dictionary for each climate data chunk
        metadata = {
            "storyline": chunk["storyline"],
            "elevation_band": chunk["elevation_band"],
            "season": chunk["season"],
            "variable_type": chunk["variable_type"],
            "abstraction_level": chunk["abstraction_level"],
            "analog_type": chunk["analog_type"],
            "analog_reference": chunk["analog_reference"] if chunk["analog_reference"] is not None else "",
            "farmer_concern": chunk["farmer_concern"],
        } # metadata
        # Add climate data chunk metadata to metadata list
        metadatas.append(metadata)

    # Ingest all 20 climate chunks to ChromaDB in one batch operation
    # ChromaDB converts it into vector embedding stored w/ ID + metadata
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    ) # collection.add
    # Now have searchable climate database

    # Print success messages
    print(f"Successfully loaded {len(STORYLINES)} chunks into PCS_climate_content collection.")
    print("Climate database is ready.")

    # Verify ingestion by checking count - should be 20
    count = collection.count()
    print(f"Verified: {count} chunks stored in ChromaDB.")

    # Returns collection object so other files can call function
    return collection

# Runs load_knowledge_base() only when file executed directly
if __name__ == "__main__":
    load_knowledge_base()