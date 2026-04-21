"""
test_belief_model.py

Tests for the farmer belief model using 
python testing/test_belief_model.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.agent.belief_model import (
    get_persona_belief,
    update_belief,
    normalize_belief,
    get_content_priority,
    belief_summary
)

print("=== BELIEF MODEL TESTS ===\n")

# Test 1 — load skeptic persona
print("Test 1: Load skeptic persona")
belief = get_persona_belief("skeptic")
print(belief_summary(belief))
print()

# Test 2 — farmer agrees with Mediterranean Shift content
print("Test 2: Farmer agrees with Mediterranean Shift content")
chunk = {"storyline": "Mediterranean Shift", "abstraction_level": "experiential"}
response = "Yes that makes sense, 2022 was a very dry and hot summer for us"
updated = update_belief(belief, response, chunk)
print("Before:", {k: f"{v:.2f}" for k, v in belief.items()})
print("After: ", {k: f"{v:.2f}" for k, v in updated.items()})
print()

# Test 3 — farmer is skeptical
print("Test 3: Farmer expresses skepticism")
chunk2 = {"storyline": "Tropical Basque", "abstraction_level": "narrative"}
response2 = "I don't think that's right, things have always been the same here"
updated2 = update_belief(belief, response2, chunk2)
print("Before:", {k: f"{v:.2f}" for k, v in belief.items()})
print("After: ", {k: f"{v:.2f}" for k, v in updated2.items()})
print()

# Test 4 — content priority ranking
print("Test 4: Content priority for skeptic farmer")
chunks = [
    {"storyline": "Mediterranean Shift", "abstraction_level": "narrative"},
    {"storyline": "No Change", "abstraction_level": "narrative"},
    {"storyline": "Moist Atlantic", "abstraction_level": "experiential"},
    {"storyline": "Tropical Basque", "abstraction_level": "narrative"},
]
prioritized = get_content_priority(belief, chunks)
print("Priority order for skeptic (No Change = 0.65):")
for i, chunk in enumerate(prioritized):
    print(f"  {i+1}. {chunk['storyline']} ({chunk['abstraction_level']})")
print()

print("=== ALL TESTS COMPLETE ===")