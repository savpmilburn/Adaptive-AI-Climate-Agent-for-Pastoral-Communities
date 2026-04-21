"""
test_fast_api.py

Tests FastAPI backend endpoints using Python's requests library.
Run the server first with: uvicorn backend.main:app --reload
Then in a separate terminal run: python testing/test_fast_api.py
"""

import requests
import json

BASE_URL = "http://localhost:8000"

print("=== FASTAPI BACKEND TESTS ===\n")

# Test 1 — health check
print("Test 1: Health check")
response = requests.get(f"{BASE_URL}/")
print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")
print()

# Test 2 — get personas
print("Test 2: Get all personas")
response = requests.get(f"{BASE_URL}/personas")
data = response.json()
for persona in data["personas"]:
    print(f"  {persona['key']}: {persona['name']} — {persona['response_style']}")
print()

# Test 3 — start a session
print("Test 3: Start session with skeptic persona")
response = requests.post(
    f"{BASE_URL}/session/start",
    json={"persona_key": "skeptic"}
)
session_data = response.json()
session_id = session_data["session_id"]
print(f"Session ID: {session_id[:8]}...")
print(f"Farmer: {session_data['farmer_name']}")
print(f"Initial belief: {session_data['initial_belief']}")
print()

# Test 4 — send a message
print("Test 4: Send farmer message")
response = requests.post(
    f"{BASE_URL}/session/{session_id}/chat",
    json={"message": "I remember 2022 being very dry and difficult for highland grazing"}
)
chat_data = response.json()
print(f"Agent response: {chat_data['response']}")
print(f"Selected storyline: {chat_data['selected_storyline']}")
print(f"Turn number: {chat_data['turn_number']}")
print(f"Updated belief: {chat_data['belief']}")
print()

# Test 5 — get belief state
print("Test 5: Get current belief state")
response = requests.get(f"{BASE_URL}/session/{session_id}/belief")
belief_data = response.json()
print(belief_data["belief_summary"])
print()

# Test 6 — get history
print("Test 6: Get conversation history")
response = requests.get(f"{BASE_URL}/session/{session_id}/history")
history_data = response.json()
print(f"Turn count: {history_data['turn_count']}")
print(f"Messages: {len(history_data['history'])}")
print()

# Test 7 — reset session
print("Test 7: Reset session")
response = requests.delete(f"{BASE_URL}/session/{session_id}")
print(f"Status: {response.status_code}")
print(f"Response: {response.json()['message']}")

print("\n=== ALL TESTS COMPLETE ===")