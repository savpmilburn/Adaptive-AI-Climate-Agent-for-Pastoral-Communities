"""
test_memory.py

Tests cross-session memory using Mem0.
Run two separate simulated sessions & verify AI climate agent
remembers info from the first session in the second.

Run with: python testing/test_memory.py
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.agent.climate_agent import ClimateAgent

print("=== MEMORY TEST ===\n")
print("SESSION 1 — Skeptic farmer mentions 2022\n")

# Session 1
agent = ClimateAgent(persona_key="skeptic")

result1 = agent.chat("I remember 2022 being very dry and difficult for our highland grazing")
print(f"Response: {result1['response'][:150]}...")
print(f"Belief: {result1['belief']}")

result2 = agent.chat("We also had problems with water supply for the animals that summer")
print(f"Response: {result2['response'][:150]}...")

print("\nSession 1 complete. Memories stored in Mem0.")
print("Now simulating server restart by creating new agent instance...\n")

# Session 2 — new agent instance simulates server restart
print("SESSION 2 — New session, same farmer\n")
agent2 = ClimateAgent(persona_key="skeptic")

print("\nStarting new conversation — does agent remember previous session?")
result3 = agent2.chat("Hello, I wanted to continue our discussion about climate")
print(f"Response: {result3['response'][:200]}...")
print(f"\nMemory context injected: {agent2.state.get('memory_context', 'none')[:200]}")

print("\n=== MEMORY TEST COMPLETE ===")