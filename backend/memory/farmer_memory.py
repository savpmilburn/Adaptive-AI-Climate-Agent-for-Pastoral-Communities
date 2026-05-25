"""
farmer_memory.py

Manages cross-session memory for farmer personas using Mem0.

Stores + retrieves key facts about each farmer across separate conversation sessions. 
When a farmer returns for a new conversation, the AI climate agent remembers what was discussed before:
their concerns, skepticism, years they mentioned, + how their beliefs shifted.

This implements the episodic memory component from:
    Park et al. (2023) Generative Agents: Interactive Simulacra of Human Behavior

Mem0 handles:
    - Automatic extraction of key facts from conversation text
    - Storage of memories tied to a farmer user ID
    - Semantic retrieval of relevant memories given a query
    - Memory deduplication and summarization over time

Privacy note:
    Mem0 free tier uses managed cloud storage.
    Acceptable for prototype with synthetic personas.
    Production deployment with real farmers would require local Mem0 deployment for data privacy.
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

from mem0 import MemoryClient

def initialize_memory():
    """
    Initializes + returns the Mem0 cloud MemoryClient using MEM0_API_KEY.
    """
    api_key = os.getenv("MEM0_API_KEY")
    if not api_key:
        raise ValueError("MEM0_API_KEY environment variable is not set.")
    client = MemoryClient(api_key = api_key)
    print("Mem0 cloud client initialized")
    return client
    
## MEMORY OPERATIONS:
def store_memories(memory: MemoryClient, farmer_id: str, conversation_turn: dict) -> list:
    """
    Extracts + stores key facts from 1 conversation exchange in Mem0.

    Args:
        memory: Mem0 MemoryClient instance
        farmer_id: unique farmer identifier for memory namespacing
        conversation_turn: dict with 'farmer' + 'agent' message strings

    Returns:
        list of stored memory result dicts
    """
    messages = [
        {"role": "user", "content": conversation_turn["farmer"]},
        {"role": "assistant", "content": conversation_turn["agent"]}
    ]
    # Mem0 MemoryClient uses user_id explicitly
    result = memory.add(messages, user_id=farmer_id)
    stored_count = len(result.get("results", []))
    print(f"Stored {stored_count} memories for farmer: {farmer_id}")
    return result.get("results", [])

def retrieve_memories(memory: MemoryClient, farmer_id: str, query: str, limit: int = 5) -> list:
    """
    Searches stored memories for a farmer using semantic similarity.
    
    Args:
        memory: Mem0 MemoryClient instance
        farmer_id: unique farmer identifier
        query: text to search against stored memories
        limit: maximum number of memories to return

    Returns:
        list of relevant memory dicts
    """
    # Mem0 MemoryClient uses user_id explicitly
    results = memory.search(query = query, filters = {"user_id": farmer_id}, limit = limit) 
    memories = results.get("results", [])
    print(f"Retrieved {len(memories)} memories for farmer: {farmer_id}")
    return memories

def get_all_memories(memory: MemoryClient, farmer_id: str) -> list:
    """
    Returns all stored memories for a farmer regardless of relevance.
    """
    # MemoryClient requires filters dict for get_all
    results = memory.get_all(filters={"user_id": farmer_id})
    memories = results.get("results", [])
    print(f"Total memories for {farmer_id}: {len(memories)}")
    return memories

# Convert list of Mem0 memory objects into readable string
def format_memories_for_context(memories: list) -> str:
    """
    Formats a list of Mem0 memory dicts into a readable string for system prompt injection.

    Args:
        memories: list of memory dicts from Mem0

    Returns:
        formatted memory string, or empty string if no memories
    """
    if not memories:
        return ""

    lines = ["What I remember about this farmer from previous conversations:"]
    for mem in memories:
        memory_text = mem.get("memory", "")
        if memory_text:
            lines.append(f"  - {memory_text}")

    return "\n".join(lines)

def delete_farmer_memories(memory: MemoryClient, farmer_id: str) -> bool:
    """
    Deletes all stored memories for a specific farmer.
    Used on full persona reset.

    Args:
        memory: Mem0 MemoryClient instance
        farmer_id: unique identifier for this farmer persona

    Returns:
        True if successful
    """
    memory.delete_all(user_id=farmer_id)
    print(f"Deleted all memories for farmer: {farmer_id}")
    return True