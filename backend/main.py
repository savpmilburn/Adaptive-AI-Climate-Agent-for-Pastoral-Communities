"""
main.py

FastAPI backend server for the Adaptive AI Climate Agent run with:
uvicorn backend.main:app --reload

Exposes HTTP endpoints that React frontend will call.
Acts as bridge between frontend UI & LangGraph AI agent.

Endpoints:
    GET  /                            : Health check
    GET  /personas                    : List available farmer personas
    POST /session/start               : Start new conversation session
    POST /session/{session_id}/chat   : Send a farmer message + get response
    GET  /session/{session_id}/belief : Get current belief state
    DELETE /session/{session_id}      : Reset session

Theoretical grounding:
    REST API design follows standard FastAPI patterns.
    Session management allows belief state to persist across multiple frontend requests within 1 conversation.
"""
import os
import sys
import uuid
from typing import Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.agent.climate_agent import ClimateAgent
from backend.agent.belief_model import FARMER_PERSONAS, belief_summary

from contextlib import asynccontextmanager
import chromadb
from langchain_groq import ChatGroq
from backend.data.load import ingest_to_client
from backend.memory.farmer_memory import initialize_memory

## SINGLETON COMPONENTS (shared across all sessions, initialized once at startup)
_collection = None
_llm = None
_memory = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Creates 1 EphemeralClient that ingests 20 climate content chunks, creates 1 GroqLLM connection, & 1 Mem0 client when server starts.
    Store in _collection, _llm, _memory as module-level globals that every session shares.
    """
    global _collection, _llm, _memory

    # Ephemeral knowledge base, re-ingested on every cold start (~1s for 20 climate chunks)
    chroma_client = chromadb.EphemeralClient()
    _collection = ingest_to_client(chroma_client)

    _llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.7, api_key=os.getenv("GROQ_API_KEY"))

    _memory = initialize_memory()

    print("All components (ChromaDB client, Groq LLM client, Mem0 client) initialized. Server ready.")
    yield

app = FastAPI(
    title="Adaptive AI Climate Agent",
    description=(
        "An AI agent that models farmer belief states + adaptively "
        "delivers climate scenario narratives for Soule, France. "
        "Grounded in Bayesian brain theory + the Free Energy Principle."
    ),
    version="1.0.0", 
    lifespan=lifespan,
) # FastAPI app

_frontend_url = os.getenv("FRONTEND_URL", "")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", *([_frontend_url] if _frontend_url else []), ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
) # CORS: allow requests from frontend origins

# In-memory session store, resets on server restart
sessions: dict[str, ClimateAgent] = {}

# Persists belief vectors across persona switches within a server session
farmer_beliefs: dict[str, dict] = {}

## REQUEST/RESPONSE MODELS:
class StartSessionRequest(BaseModel):
    """Request body for starting a new session used when frontend calls POST /session/start. """
    persona_key: str = "skeptic"  # Default to skeptic persona

class ChatRequest(BaseModel):
    """Request body for sending a farmer message."""
    message: str

class BeliefResponse(BaseModel):
    """Response shape for belief state queries."""
    belief: dict
    belief_summary: str

class ChatResponse(BaseModel):
    """Response shape for chat interactions in displayed conversation."""
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

## FastAPI ENDPOINTS:
@app.get("/")
def health_check():
    """ Returns server status. Used by frontend to verify backend connectivity."""
    return {
        "status": "running",
        "service": "VIPR Adaptive Climate Agent",
        "version": "1.0.0"
    }

@app.get("/personas")
def get_personas():
    """ Returns all available farmer personas for frontend selection UI."""
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

    # Pre-compute farmer_id to find any persisted belief
    farmer_id = f"{request.persona_key}_{FARMER_PERSONAS[request.persona_key]['name'].lower().replace(' ', '_')}"

    # Create new ClimateAgent instance for this session
    # This initializes ChromaDB connection, Groq LLM, and LangGraph
    agent = ClimateAgent(persona_key=request.persona_key, collection = _collection, llm = _llm, memory = _memory, initial_belief = farmer_beliefs.get(farmer_id), )

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

@app.post("/session/{session_id}/chat", response_model=ChatResponse)
def chat(session_id: str, request: ChatRequest):
    """
    Core endpoint that sends a farmer message to AI climate agent + returns the response.
    The agent runs the full ReAct loop + returns the response + updated belief state.

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
    
    agent = sessions[session_id]

    # Validate message is not empty
    if not request.message.strip():
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty."
        )

    # Run the agent
    result = agent.chat(request.message)

    # Persist updated belief for this farmer
    farmer_beliefs[agent.farmer_id] = result["belief"]

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

@app.get("/session/{session_id}/belief", response_model=BeliefResponse)
def get_belief(session_id: str):
    """
    Returns the current belief state for a session without sending a new message.

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

@app.delete("/session/{session_id}")
def reset_session(session_id: str, persona_key: Optional[str] = None):
    """
    Resets a session to its initial state.

    Optionally switches to a different farmer persona.
    Clears conversation history + resets belief vector.

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

    # Clear persisted belief on explicit reset
    farmer_beliefs.pop(agent.farmer_id, None)

    # Reset agent state
    agent.reset(persona_key=persona_key)

    return {
        "message": "Session reset successfully",
        "session_id": session_id,
        "persona_key": agent.persona_key,
        "initial_belief": agent.state["belief"]
    }

@app.get("/session/{session_id}/history")
def get_history(session_id: str):
    """
    Returns the full conversation history for a session.

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