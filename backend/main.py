# Entry point where FastAPI devines API endpoints
"""
main.py

FastAPI backend server for the Adaptive AI Climate Agent run with:
uvicorn backend.main:app --reload

Exposes HTTP endpoints that React frontend will call.
Acts as bridge between frontend UI & LangGraph AI agent.

Endpoints:
    GET  /                          - Health check
    GET  /personas                  - List available farmer personas
    POST /session/start             - Start new conversation session
    POST /session/{session_id}/chat - Send a farmer message + get response
    GET  /session/{session_id}/belief - Get current belief state
    DELETE /session/{session_id}    - Reset session

Theoretical grounding:
    REST API design follows standard FastAPI patterns.
    Session management allows belief state to persist across
    multiple frontend requests within 1 conversation.
"""

import os
import sys
import uuid # Python library for generating unique IDs for every conversation
from typing import Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException # core FastAPI class + HTTP error handler
from fastapi.middleware.cors import CORSMiddleware # imports Cross-Origin Resource Sharing middleware since browsers default block requests b/w different origins
from pydantic import BaseModel # Pydantic's base class for data models

from backend.agent.climate_agent import ClimateAgent
from backend.agent.belief_model import FARMER_PERSONAS, belief_summary

# FastAPI app:
# Create FastAPI app instance - backend connects to object
app = FastAPI(
    title="Adaptive AI Climate Agent",
    description=(
        "An AI agent that models farmer belief states + adaptively "
        "delivers climate scenario narratives for Soule, France. "
        "Grounded in Bayesian brain theory and the Free Energy Principle."
    ),
    version="1.0.0"
) # app

# CORS middleware:
# Attach CORS middleware to API (running on localhost: 8000) allowing React requests running on localhost:3000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
) # add_middleware

# Storing session:
# PS: in production, would be a db, but using simpler in-memory storage for prototype
# Will reset everytime server restarts, Mem0 will add true persistence 
# Empty dict that stores active AI agent instances in memory so each conversation session
# needs its OWN AI Climate Agent w/ its own belief state + conversation history 
sessions: dict[str, ClimateAgent] = {}

# Request/Response models: 
# Define shape of request body for starting a session: used when frontend calls POST /session/start
class StartSessionRequest(BaseModel):
    """Request body for starting a new session."""
    persona_key: str = "skeptic"  # Default to skeptic persona

# Define request body for chat messages 
class ChatRequest(BaseModel):
    """Request body for sending a farmer message."""
    message: str

class BeliefResponse(BaseModel):
    """Response shape for belief state queries."""
    belief: dict
    belief_summary: str

# Define shape of what is returned by chat endpoint in order to display conversation
class ChatResponse(BaseModel):
    """Response shape for chat interactions."""
    session_id: str
    response: str
    belief: dict
    belief_summary: str
    reasoning: str
    selected_storyline: str
    selected_abstraction: str
    turn_number: int


class SessionResponse(BaseModel):
    """Response shape for session creation."""
    session_id: str
    persona_key: str
    farmer_name: str
    farmer_description: str
    initial_belief: dict
    belief_summary: str


class PersonaInfo(BaseModel):
    """Information about one farmer persona."""
    key: str
    name: str
    description: str
    response_style: str
    initial_belief: dict

# FastAPI Endpoints:
# Simple endpoint that returns status message so frontend can confirm backend is running
@app.get("/")
def health_check():
    """
    Health check endpoint.
    Returns a simple status message confirming the server is running.
    The frontend can call this to verify backend connectivity.
    """
    return {
        "status": "running",
        "service": "VIPR Adaptive Climate Agent",
        "version": "1.0.0"
    }

# Endpoint that returns all 3 farmer personals w/ details 
@app.get("/personas")
def get_personas():
    """
    Returns all available farmer personas.
    The frontend uses this to populate the persona selection UI.
    """
    personas = []
    for key, persona in FARMER_PERSONAS.items():
        personas.append(PersonaInfo(
            key=key,
            name=persona["name"],
            description=persona["description"],
            response_style=persona["response_style"],
            initial_belief=persona["belief"]
        ))
    return {"personas": personas}

# Endpoint that creates a new AI climate agent + returns session ID 
@app.post("/session/start", response_model=SessionResponse)
def start_session(request: StartSessionRequest):
    """
    Starts a new conversation session with a chosen farmer persona.

    Creates a new ClimateAgent instance + stores it in the session store.
    Returns a session_id the frontend uses for all subsequent requests.

    Args:
        request: StartSessionRequest with persona_key

    Returns:
        SessionResponse with session_id and initial state
    """

    # Validate persona key
    if request.persona_key not in FARMER_PERSONAS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown persona: {request.persona_key}. Choose from: {list(FARMER_PERSONAS.keys())}"
        )

    # Generate unique session ID
    session_id = str(uuid.uuid4())

    # Create new agent instance for this session
    # This initializes ChromaDB connection, Groq LLM, and LangGraph
    agent = ClimateAgent(persona_key=request.persona_key)

    # Store in session store
    sessions[session_id] = agent

    persona = FARMER_PERSONAS[request.persona_key]
    initial_belief = agent.state["belief"]

    print(f"New session started: {session_id} | Persona: {request.persona_key}")

    return SessionResponse(
        session_id=session_id,
        persona_key=request.persona_key,
        farmer_name=persona["name"],
        farmer_description=persona["description"],
        initial_belief=initial_belief,
        belief_summary=belief_summary(initial_belief)
    )

# Core endpoint that receives a farmer message, runs AI climate agent, + returns response
@app.post("/session/{session_id}/chat", response_model=ChatResponse)
def chat(session_id: str, request: ChatRequest):
    """
    Sends a farmer message to AI climate agent + returns the response.

    This is the core endpoint. Every message the farmer types in the
    frontend goes through here. The agent runs the full ReAct loop
    and returns the response plus updated belief state.

    Args:
        session_id: UUID string identifying the active session
        request: ChatRequest with the farmer's message text

    Returns:
        ChatResponse with agent response, updated belief, and reasoning
    """

    # Check session exists
    if session_id not in sessions:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id} not found. Start a new session first."
        )

    # Get agent for this session
    agent = sessions[session_id]

    # Validate message is not empty
    if not request.message.strip():
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty."
        )

    # Run the agent
    result = agent.chat(request.message)

    # Count conversation turns
    turn_number = len(agent.state["conversation_history"]) // 2

    print(f"Session {session_id[:8]}... | Turn {turn_number} | Persona: {agent.persona_key}")

    return ChatResponse(
        session_id=session_id,
        response=result["response"],
        belief=result["belief"],
        belief_summary=belief_summary(result["belief"]),
        reasoning=result["reasoning"],
        selected_storyline=result["selected_chunk"].get("storyline", ""),
        selected_abstraction=result["selected_chunk"].get("abstraction_level", ""),
        turn_number=turn_number
    )

# Endpoint that returns current belief state for a session without sending a message
@app.get("/session/{session_id}/belief", response_model=BeliefResponse)
def get_belief(session_id: str):
    """
    Returns the current belief state for a session.

    The frontend can call this to get the latest belief vector
    for display without sending a new message.

    Args:
        session_id: UUID string identifying the active session

    Returns:
        BeliefResponse with current belief dict and formatted summary
    """

    if session_id not in sessions:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id} not found."
        )

    agent = sessions[session_id]
    current_belief = agent.state["belief"]

    return BeliefResponse(
        belief=current_belief,
        belief_summary=belief_summary(current_belief)
    )

# Endpoint that resets a session's belief vector + conversation history
@app.delete("/session/{session_id}")
def reset_session(session_id: str, persona_key: Optional[str] = None):
    """
    Resets a session to its initial state.

    Optionally switches to a different farmer persona.
    Clears conversation history and resets belief vector.

    Args:
        session_id: UUID string identifying the session to reset
        persona_key: optional new persona key to switch to

    Returns:
        confirmation message with new initial state
    """

    if session_id not in sessions:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id} not found."
        )

    agent = sessions[session_id]

    # Validate new persona if provided
    if persona_key and persona_key not in FARMER_PERSONAS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown persona: {persona_key}."
        )

    # Reset agent state
    agent.reset(persona_key=persona_key)

    return {
        "message": "Session reset successfully",
        "session_id": session_id,
        "persona_key": agent.persona_key,
        "initial_belief": agent.state["belief"]
    }

# Endpoint that returns full conversation history for a session
@app.get("/session/{session_id}/history")
def get_history(session_id: str):
    """
    Returns the full conversation history for a session.

    Useful for the frontend to display the chat log and
    for debugging agent behavior during development.

    Args:
        session_id: UUID string identifying the session

    Returns:
        list of message dicts with role and content
    """

    if session_id not in sessions:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id} not found."
        )

    agent = sessions[session_id]

    return {
        "session_id": session_id,
        "persona_key": agent.persona_key,
        "turn_count": len(agent.state["conversation_history"]) // 2,
        "history": agent.state["conversation_history"]
    }