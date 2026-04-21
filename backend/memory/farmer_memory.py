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

from mem0 import Memory
# Initiatlize Mem0:
def initialize_memory():
    """
    Initializes Mem0 memory client.

    Uses Mem0's default configuration which stores memories in local vector database using 
    the same all-MiniLM-L6-v2 embedding model as ChromaDB.
    Memories persist in a local file between server restarts.

    Returns:
        Mem0 Memory instance
    """
    # Initialize with default config
    # This uses local storage — no cloud, no API key needed
    # Memories stored in ~/.mem0/ on your machine
    memory = Memory()
    print("Mem0 memory initialized — local storage")
    return memory

# Memory operations:
def store_memories(memory: Memory, farmer_id: str, conversation_turn: dict) -> list:
    """
    Extracts + stores key facts from 1 conversation turn.

    Mem0 automatically analyzes the conversation text + extracts
    memorable facts including farmer concerns, opinions, years mentioned,
    skepticism signals, + belief-relevant statements.

    Args:
        memory: Mem0 Memory instance
        farmer_id: unique identifier for this farmer persona
        conversation_turn: dict with 'farmer' and 'agent' keys
                          containing the text of one exchange

    Returns:
        list of memory objects that were stored
    """

    # Format conversation as list of message dicts
    # Mem0 expects this format for memory extraction
    messages = [
        {
            "role": "user",
            "content": conversation_turn["farmer"]
        },
        {
            "role": "assistant",
            "content": conversation_turn["agent"]
        }
    ] # messages 

    # Store memories: 
    # Mem0 automatically extracts key facts from conversation + stores them for this farmer
    result = memory.add(
        messages,
        user_id=farmer_id
    ) # result

    stored_count = len(result.get("results", []))
    print(f"Stored {stored_count} memories for farmer: {farmer_id}")

    return result.get("results", [])


def retrieve_memories(memory: Memory, farmer_id: str, query: str, limit: int = 5) -> list:
    """
    Retrieves relevant memories about a farmer given a query.

    Mem0 uses semantic search to find memories most relevant
    to the current conversation context — similar to how the
    hippocampus retrieves relevant episodic memories given
    a current environmental cue.

    Args:
        memory: Mem0 Memory instance
        farmer_id: unique identifier for this farmer persona
        query: current context to search memories against
               usually the farmer's most recent message
        limit: maximum number of memories to retrieve

    Returns:
        list of relevant memory dicts with 'memory' text field
    """

    results = memory.search(
        query=query,
        user_id=farmer_id,
        limit=limit
    )

    memories = results.get("results", [])
    print(f"Retrieved {len(memories)} memories for farmer: {farmer_id}")

    return memories


def get_all_memories(memory: Memory, farmer_id: str) -> list:
    """
    Returns all stored memories for a farmer.

    Used when starting a new session to load complete
    farmer history before the conversation begins.

    Args:
        memory: Mem0 Memory instance
        farmer_id: unique identifier for this farmer persona

    Returns:
        list of all memory dicts for this farmer
    """

    results = memory.get_all(user_id=farmer_id)
    memories = results.get("results", [])
    print(f"Total memories for {farmer_id}: {len(memories)}")
    return memories


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