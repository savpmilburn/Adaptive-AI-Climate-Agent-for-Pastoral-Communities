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

from mem0 import MemoryClient # Import Mem0 cloud client
# Initiatlize Mem0:
def initialize_memory():
    """
    Initializes Mem0 memory client.
    """
    # Configures Mem0 to use Groq as LLM to read conversations + extract memorable facts
    api_key = os.getenv("MEM0_API_KEY")
    if not api_key:
        raise ValueError("MEM0_API_KEY environment variable is not set.")
    client = MemoryClient(api_key = api_key)
    print("Mem0 cloud client initialized")
    return client
    
# Memory operations:
# Take 1 conversation exchange, format as message list, + passes to Mem0
# Reads conversation, uses Groq LLM to extract key facts, converts to vector embeddings, + stores w/ farmer_id
def store_memories(memory: MemoryClient, farmer_id: str, conversation_turn: dict) -> list:
    messages = [
        {"role": "user", "content": conversation_turn["farmer"]},
        {"role": "assistant", "content": conversation_turn["agent"]}
    ]
    # Mem0 MemoryClient uses user_id explicitly
    result = memory.add(messages, user_id=farmer_id)
    stored_count = len(result.get("results", []))
    print(f"Stored {stored_count} memories for farmer: {farmer_id}")
    return result.get("results", [])

# Searches stored memories for specific farmer using semantic similarity against query string
def retrieve_memories(memory: MemoryClient, farmer_id: str, query: str, limit: int = 5) -> list:
    # Mem0 MemoryClient uses user_id explicitly
    results = memory.search(query = query, filters = {"user_id": farmer_id}, limit = limit) 
    memories = results.get("results", [])
    print(f"Retrieved {len(memories)} memories for farmer: {farmer_id}")
    return memories

# Returns all stored memory for farmer regardless of relevance so AI agent knows complete history
def get_all_memories(memory: MemoryClient, farmer_id: str) -> list:
    # Mem0 v2.0.0 uses filters dict instead of user_id parameter
    results = memory.get_all(filters={"user_id": farmer_id})
    memories = results.get("results", [])
    print(f"Total memories for {farmer_id}: {len(memories)}")
    return memories

# Convert list of Mem0 memory objects into readable string
def format_memories_for_context(memories: list) -> str:
    """
    Formats a list of memory dicts into a readable string
    for injection into the agent's system prompt.

    Args:
        memories: list of memory dicts from Mem0

    Returns:
        formatted string summarizing what the agent remembers
        about this farmer, or empty string if no memories
    """

    if not memories:
        return ""

    lines = ["What I remember about this farmer from previous conversations:"]
    for i, mem in enumerate(memories):
        memory_text = mem.get("memory", "")
        if memory_text:
            lines.append(f"  - {memory_text}")

    return "\n".join(lines)

# Delete all memories for given farmer
def delete_farmer_memories(memory: MemoryClient, farmer_id: str) -> bool:
    """
    Deletes all memories for a specific farmer.

    Used when fully resetting a farmer persona to baseline.

    Args:
        memory: Mem0 Memory instance
        farmer_id: unique identifier for this farmer persona

    Returns:
        True if successful
    """

    memory.delete_all(user_id=farmer_id)
    print(f"Deleted all memories for farmer: {farmer_id}")
    return True