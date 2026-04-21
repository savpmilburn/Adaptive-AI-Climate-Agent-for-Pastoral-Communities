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

from mem0 import Memory # Import Mem0 memory class
# Initiatlize Mem0:
def initialize_memory():
    """
    Initializes Mem0 memory client.
    """
    # Configures Mem0 to use Groq as LLM to read conversations + extract memorable facts
    config = {
        "llm": {
            "provider": "groq",
            "config": {
                "model": "llama-3.3-70b-versatile",
                "api_key": os.getenv("GROQ_API_KEY")
            }
        },
        # Configures Mem0 to use HuggingFace's embedding model instead of OpenAI (requires paid API key)
        "embedder": {
            "provider": "huggingface",
            "config": {
                "model": "multi-qa-MiniLM-L6-cos-v1"
            }
        },
        # Mem0 stores memories in local ChromaDB db at mem0_db/ in project root for local, free memory storage
        "vector_store": {
            "provider": "chroma",
            "config": {
                "collection_name": "farmer_memories",
                "path": os.path.join(
                    os.path.dirname(
                        os.path.dirname(
                            os.path.dirname(os.path.abspath(__file__))
                        )
                    ),
                    "mem0_db"
                )
            }
        }
    }

    # Create Mem0 Memory instance using config above + return it
    try:
        memory = Memory.from_config(config)
        print("Mem0 memory initialized — Groq LLM + HuggingFace embeddings + ChromaDB storage")
        print(f"Memory object type: {type(memory)}")
        return memory
    except Exception as e:
        print(f"Mem0 initialization failed: {e}")
        print(f"Error type: {type(e)}")
        raise
    
# Memory operations:
# Take 1 conversation exchange, format as message list, + passes to Mem0
# Reads conversation, uses Groq LLM to extract key facts, converts to vector embeddings, + stores w/ farmer_id
def store_memories(memory: Memory, farmer_id: str, conversation_turn: dict) -> list:
    messages = [
        {"role": "user", "content": conversation_turn["farmer"]},
        {"role": "assistant", "content": conversation_turn["agent"]}
    ]
    # Mem0 v2.0.0 uses user_id as keyword argument
    result = memory.add(messages, user_id=farmer_id)
    stored_count = len(result.get("results", []))
    print(f"Stored {stored_count} memories for farmer: {farmer_id}")
    return result.get("results", [])

# Searches stored memories for specific farmer using semantic similarity against query string
def retrieve_memories(memory: Memory, farmer_id: str, query: str, limit: int = 5) -> list:
    # Mem0 v2.0.0 uses filters dict for user filtering
    results = memory.search(
        query=query,
        filters={"user_id": farmer_id},
        limit=limit
    )
    memories = results.get("results", [])
    print(f"Retrieved {len(memories)} memories for farmer: {farmer_id}")
    return memories

# Returns all stored memory for farmer regardless of relevance so AI agent knows complete history
def get_all_memories(memory: Memory, farmer_id: str) -> list:
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
def delete_farmer_memories(memory: Memory, farmer_id: str) -> bool:
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