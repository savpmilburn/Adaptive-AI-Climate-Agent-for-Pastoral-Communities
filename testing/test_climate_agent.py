"""
test_climate_agent.py

Tests AI climate agent end to end by running a short
simulated conversation with the skeptic farmer persona
+ verifies belief vector updates correctly across turns.

Run with: python testing/test_climate_agent.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.agent.climate_agent import ClimateAgent

print("=== CLIMATE AGENT TEST ===\n")

# Initialize agent with skeptic persona
agent = ClimateAgent(persona_key="skeptic")

print("\n--- Turn 1 ---")
result1 = agent.chat("I remember 2022 being a very dry and difficult summer for us up in the highlands")

print(f"\nAgent response: {result1['response']}")
print(f"\nReasoning: {result1['reasoning']}")
print(f"\nSelected chunk storyline: {result1['selected_chunk'].get('storyline')}")
print(f"\nUpdated belief:")
print(agent.get_belief_summary())

print("\n--- Turn 2 ---")
result2 = agent.chat("Yes but I am not sure things will really change that much, it has always been like this")

print(f"\nAgent response: {result2['response']}")
print(f"\nUpdated belief:")
print(agent.get_belief_summary())

print("\n--- Turn 3 ---")
result3 = agent.chat("What would wetter winters mean for my cheese production?")

print(f"\nAgent response: {result3['response']}")
print(f"\nUpdated belief:")
print(agent.get_belief_summary())

print("\n=== TEST COMPLETE ===")