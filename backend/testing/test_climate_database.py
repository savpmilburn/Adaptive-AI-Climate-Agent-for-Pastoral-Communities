"""
test_climate_database.py

Check to verify ChromaDB database loaded correctly by running:
python testing/test_climate_database.py
"""

import chromadb
import os

# Build path to chroma_db
project_root = os.path.dirname(os.path.abspath(__file__))
chroma_path = os.path.join(project_root, "chroma_db")

# Connect to existing database
client = chromadb.PersistentClient(path=chroma_path)

# Get collection
collection = client.get_collection("PCS_climate_content")

# Test 1 — how many chunks are stored?
print(f"Total chunks stored: {collection.count()}")
print()

# Test 2 — retrieve a specific chunk by ID
result = collection.get(ids=["med_007"])
print("Fetching med_007 (Mediterranean Shift temporal analog):")
print(f"  Text: {result['documents'][0]}")
print(f"  Metadata: {result['metadatas'][0]}")
print()

# Test 3 — semantic search
# Ask a farming question and see what chunks come back
query = "what happens to highland grazing in hot dry summers"
results = collection.query(
    query_texts=[query],
    n_results=3
)

print(f"Semantic search: '{query}'")
print("Top 3 results:")
for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
    print(f"\n  Result {i+1}:")
    print(f"  Storyline: {meta['storyline']}")
    print(f"  Concern: {meta['farmer_concern']}")
    print(f"  Abstraction: {meta['abstraction_level']}")
    print(f"  Text: {doc[:100]}...")