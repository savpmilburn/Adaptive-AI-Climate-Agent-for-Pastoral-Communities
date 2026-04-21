"""
climate_agent.py

Core agent implementation using LangGraph's ReAct loop.

The AI climate agent adaptively selects & delivers climate scenario content 
(from UGA VIPR research) to farmers based on their current belief state & interaction history.

Architecture:
    1. Receive input farmer message
    2. Retrieve relevant content from ChromaDB climate database
    3. Rank content by belief-weighted priority (Free Energy Principle)
    4. Reason about what to say next (ReAct)
    5. Generate adaptive narrative response via Groq LLM
    6. Update farmer belief vector based on response
    7. Store interaction in memory

Theoretical grounding:
    - ReAct reasoning loop (Yao et al., 2023)
    - Free Energy Principle (Friston, 2010)
    - Generative Agents memory (Park et al., 2023)
"""

import os # for file paths
import sys # for file paths
import json # handle structured data
from typing import TypedDict, Annotated # for AgentState + LangGraph
import operator

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Load environment variables like Groq API key
from dotenv import load_dotenv
load_dotenv()

# Third party library imports
import chromadb # ChromaDB: connect to climate database
from langchain_groq import ChatGroq # to talk to Groq LLM
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage # LangChain's structured message types
from langgraph.graph import StateGraph, END # core LangGraph class + constant

# Import belief_model.py
from backend.agent.belief_model import (
    update_belief,
    get_content_priority,
    belief_summary,
    get_persona_belief,
    FARMER_PERSONAS
) # belief_model.py

# Import Mem0 memory
from backend.memory.farmer_memory import (
    initialize_memory,
    store_memories,
    retrieve_memories,
    get_all_memories,
    format_memories_for_context
)

# AI climate agent state
class AgentState(TypedDict):
    """
    The complete state of the AI climate agent at any point in the conversation.
    LangGraph passes this state between nodes in the graph + updates it at each step.

    Every field represents one piece of info the AI climate agent needs to reason + act correctly.
    """
    farmer_message: str # farmer's most recent inputted message
    
    # Full conversation history as a list of message dicts
    # Each dict has 'role' of farmer/agent) +  text 'content'
    conversation_history: Annotated[list, operator.add]
    
    # Current farmer belief vector:dict mapping storyline name to probability float
    belief: dict

    # Retrieved content chunks from ChromaDB climate database: list of chunk dicts with text and metadata
    retrieved_chunks: list

    selected_chunk: dict # single chunk selected for delivery this turn
    reasoning_trace: str # AI climate agent's reasoning trace - why did it choose that chunk?
    agent_response: str # final output response to farmer
    persona_key: str # active farmer persona type

    # Adding memory + farmer_id to AgentState so memory can be accessed in respond_node:
    memory_context: str # formatted String of relevant memories
    farmer_id: str # unique farmer ID for Mem0

# Connect to existing ChromaDB climate database
def initialize_components():
    """
    Initializes ChromaDB client + Groq LLM.
    Called once when the AI climate agent starts up.

    Returns:
        tuple of (chromadb collection, ChatGroq llm)
    """

    # Connect to existing ChromaDB climate database
    project_root = os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
    ) # project_root
    chroma_path = os.path.join(project_root, "chroma_db")

    # Connect to existing persistent database from load.py
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_collection("PCS_climate_content")
    # Check print statement
    print(f"Connected to ChromaDB — {collection.count()} chunks available")

    # Initialize Groq LLM (llama-3.3-70b-versatile) = Groq's best free model
    # temperature=0.7: responses are somewhat creative but not random
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        api_key=os.getenv("GROQ_API_KEY")
    ) # connection to Groq LLM
    # Check print statement
    print("Connected to Groq LLM — llama-3.3-70b-versatile")

    return collection, llm

# LangGraph nodes: each node is 1 step in ReAct loop
# LangGraph calls nodes in sequence + passes state between them.

def retrieve_node(state: AgentState, collection) -> dict:
    """
    AI climate agent acts by querying ChromaDB climate database:
    NODE 1: Retrieve relevant content chunks from ChromaDB climate database.

    Performs semantic search using farmer's message as query +
    returns top 5 most semantically similar climate chunks.
    """

    farmer_message = state["farmer_message"]
    belief = state["belief"]

    # Query ChromaDB for 5 most semantically similar chunks
    results = collection.query(
        query_texts=[farmer_message],
        n_results=5
    ) # results 

    # Package results from ChromaDB climate database into list of chunk dicts
    chunks = []
    # Combine each chunk into 1 self-contained dict w/ text + metadata
    for i, doc in enumerate(results["documents"][0]):
        chunk = {
            "content_text": doc,
            "storyline": results["metadatas"][0][i]["storyline"],
            "abstraction_level": results["metadatas"][0][i]["abstraction_level"],
            "elevation_band": results["metadatas"][0][i]["elevation_band"],
            "season": results["metadatas"][0][i]["season"],
            "farmer_concern": results["metadatas"][0][i]["farmer_concern"],
            "analog_type": results["metadatas"][0][i]["analog_type"],
            "analog_reference": results["metadatas"][0][i]["analog_reference"],
        } # chunk
        chunks.append(chunk)

    print(f"Retrieved {len(chunks)} chunks from ChromaDB")

    return {"retrieved_chunks": chunks}

#
def rank_node(state: AgentState) -> dict:
    """
    Implementing Free Energy Principle: re-ranks 5 retrieved chunks + selects top 
    based on content that most reduces farmer's uncertainty: 
    NODE 2: Rank retrieved chunks by belief-weighted priority.

    Uses get_content_priority from belief_model.py to re-rank
    semantically retrieved chunks based on what would be most
    informative for this farmer given their current belief state.
    """

    belief = state["belief"]
    chunks = state["retrieved_chunks"]

    # Re-rank by belief-weighted informativeness
    prioritized = get_content_priority(belief, chunks)

    # Select top ranked chunk for delivery
    selected = prioritized[0] if prioritized else {}

    print(f"Selected chunk: {selected.get('storyline')} / {selected.get('abstraction_level')}")
    print(belief_summary(belief))

    return {"selected_chunk": selected}

def retrieve_memory_node(state: AgentState, memory) -> dict:
    """
    NODE: Retrieves relevant memories from Mem0 before reasoning.

    Runs semantic search against stored farmer memories using
    the current farmer message as the query.
    Injects formatted memory context into state for use in
    the respond node's system prompt.
    """
    farmer_message = state["farmer_message"]
    farmer_id = state["farmer_id"]

    # Retrieve relevant memories from Mem0 using farmer message as query
    relevant_memories = retrieve_memories(
        memory,
        farmer_id,
        farmer_message,
        limit=3
    )

    # Format memories into readable string for system prompt injection
    memory_context = format_memories_for_context(relevant_memories)

    return {"memory_context": memory_context}


def reason_node(state: AgentState, llm) -> dict:
    """
    Implement Reason step in ReAct loop by asking Groq LLM to explain why 
    the AI climate agent chose that chunk before outputting response:
    NODE 3: Generate reasoning trace explaining why this chunk was selected.

    The reasoning trace is stored + displayed to show the AI climate agent's decision-making process. 
    """

    belief = state["belief"]
    selected_chunk = state["selected_chunk"]
    farmer_message = state["farmer_message"]
    persona_key = state["persona_key"]
    persona = FARMER_PERSONAS[persona_key]

    # Build reasoning prompt: ask the Groq LLM to explain selection decision
    reasoning_prompt = f"""You are an adaptive climate communication agent working with farmers in Soule, France.

Current farmer profile: {persona['description']}

Current farmer belief state:
{belief_summary(belief)}

The farmer just said: "{farmer_message}"

You selected this content to deliver next:
Storyline: {selected_chunk.get('storyline')}
Abstraction level: {selected_chunk.get('abstraction_level')}
Farmer concern: {selected_chunk.get('farmer_concern')}
Content: {selected_chunk.get('content_text')}

In 2-3 sentences, explain why you selected this specific content for this farmer right now.
Consider their belief state, what they just said, and what would most reduce their uncertainty."""

    reasoning_response = llm.invoke([HumanMessage(content=reasoning_prompt)])
    reasoning_trace = reasoning_response.content

    print(f"Reasoning: {reasoning_trace[:100]}...")

    return {"reasoning_trace": reasoning_trace}


def respond_node(state: AgentState, llm) -> dict:
    """
    Groq LLM completes core language generation work:
    NODE 4: Generate actual response delivered to farmer.

    Takes selected chunk + reasoning trace & generates natural, 
    contextually appropriate response in farmer's
    language register (narrative, statistical, experiential) 
    depending on the chunk's abstraction level + the persona's
    response style.
    """

    selected_chunk = state["selected_chunk"]
    farmer_message = state["farmer_message"]
    conversation_history = state["conversation_history"]
    persona_key = state["persona_key"]
    persona = FARMER_PERSONAS[persona_key]
    belief = state["belief"]

    # Build conversation context from history using last 4 exchanges (keeps context window manageable)
    recent_history = conversation_history[-8:] if len(conversation_history) > 8 else conversation_history
    history_text = "\n".join([
        f"{msg['role'].upper()}: {msg['content']}"
        for msg in recent_history
    ]) # history_text

    # Determine framing instruction based on abstraction level
    abstraction_level = selected_chunk.get("abstraction_level", "narrative")
    if abstraction_level == "experiential":
        framing = "Frame this using the temporal analog year as the primary hook. Connect to lived experience."
    elif abstraction_level == "statistical":
        framing = "Present the information with specific numbers and data. Be precise."
    else:
        framing = "Use descriptive narrative language that paints a picture of what the climate would feel like."

    # Retrieve relevant memories about this farmer: give AI agent context from previous sessions
    memory_context = state["memory_context"]

    # System prompt defining AI climate agent's role + behavior
    system_prompt = f"""You are an adaptive climate communication agent helping farmers in Soule (Xiberoa), France understand their climate future.

Your role is to translate complex climate science into meaningful, actionable information for farmers.

Farmer profile: {persona['description']}
Preferred communication style: {persona['response_style']}

Current belief state summary:
{belief_summary(belief)}

{memory_context}

IMPORTANT GUIDELINES:
- Speak directly to the farmer in a warm, respectful tone
- Never use scientific jargon without explaining it
- Connect climate information to farming decisions when possible
- Keep responses concise — 3 to 5 sentences maximum
- {framing}
- Do not mention SSPs, CMIP6, or technical methodology"""

    # Build messages for LLM including conversation history
    messages = [SystemMessage(content=system_prompt)]

    # Add recent conversation history by converting to LangChain message objects
    for msg in recent_history:
        if msg["role"] == "farmer":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))

    # Add current farmer message with the content to deliver
    delivery_prompt = f"""The farmer just said: "{farmer_message}"

Deliver this climate information naturally in the conversation:
{selected_chunk.get('content_text')}

Remember to connect it to what the farmer just said."""

    messages.append(HumanMessage(content=delivery_prompt))

    # Generate response
    response = llm.invoke(messages)
    agent_response = response.content

    # Add this exchange to conversation history
    new_messages = [
        {"role": "farmer", "content": farmer_message},
        {"role": "agent", "content": agent_response}
    ]

    return {
        "agent_response": agent_response,
        "conversation_history": new_messages
    }


def update_belief_node(state: AgentState) -> dict:
    """
    Create the feedback loop of farmer's response to agent's output:
    NODE 5: Update farmer belief vector based on their response.

    Uses update_belief from belief_model.py to shift the probability
    distribution based on agreement/skepticism signals
    + storyline keyword matches in farmer's message.
    """

    farmer_message = state["farmer_message"]
    current_belief = state["belief"]
    selected_chunk = state["selected_chunk"]

    # Update belief based on farmer response + delivered content
    updated_belief = update_belief(
        current_belief,
        farmer_message,
        selected_chunk
    ) # updated_belief

    print("Belief updated:")
    print(belief_summary(updated_belief))

    return {"belief": updated_belief}

# Build LangGraph
def build_agent(collection, llm, memory):
    """
    Assembles the LangGraph state machine connecting all nodes.

    The graph defines the sequence of operations the agent performs
    for each farmer message:
    retrieve -> rank -> reason -> respond -> update_belief

    Returns:
        compiled LangGraph app ready to receive messages
    """
    # Create LangGraph state machine + register each node:
    # Create the state graph with AgentState as state schema
    workflow = StateGraph(AgentState)

    # Add each node to the graph:
    # Each node is a function that takes state + returns updated state
    # We use lambda functions to inject dependencies (collection, llm) without making them part of the state
    workflow.add_node("retrieve", lambda state: retrieve_node(state, collection))
    workflow.add_node("rank", rank_node)
    workflow.add_node("retrieve_memory", lambda state: retrieve_memory_node(state, memory))
    workflow.add_node("reason", lambda state: reason_node(state, llm))
    workflow.add_node("respond", lambda state: respond_node(state, llm))
    workflow.add_node("update_belief", update_belief_node)

    # Define the flow between nodes + compile LangGraph graph using simple linear chain
    workflow.set_entry_point("retrieve") # tells LangGraph were to start
    workflow.add_edge("retrieve", "rank") 
    workflow.add_edge("rank", "retrieve_memory")
    workflow.add_edge("retrieve_memory", "reason")
    workflow.add_edge("reason", "respond")
    workflow.add_edge("respond", "update_belief")
    workflow.add_edge("update_belief", END) # mark finish

    # Compile the graph into a runnable app
    app = workflow.compile()

    print("LangGraph agent compiled with memory successfully")

    return app

# AI climate agent runner:
class ClimateAgent:
    """
    Class that wraps entire LangGraph AI climate agent + manages states across turns:
    For main.py + frontend. 
    """

    def __init__(self, persona_key: str = "skeptic"):
        """
        Initializes AI climate agent with a specific farmer persona.

        Args:
            persona_key: one of 'skeptic', 'worried', 'neutral'
        """

        print(f"\nInitializing Climate Agent with persona: {persona_key}")

        # Initialize ChromaDB climate database + Groq LLM
        self.collection, self.llm = initialize_components()

        # Initialize Mem0 memory
        self.memory = initialize_memory()

        # Build LangGraph app
        self.app = build_agent(self.collection, self.llm, self.memory)

        # Build farmer ID from persona for memory namespacing
        self.persona_key = persona_key
        self.persona = FARMER_PERSONAS[persona_key]
        self.farmer_id = f"{persona_key}_{self.persona['name'].lower().replace(' ', '_')}"

        # Load any existing memories from previous sessions
        existing_memories = get_all_memories(self.memory, self.farmer_id)
        if existing_memories:
            print(f"Loaded {len(existing_memories)} memories from previous sessions")
            for mem in existing_memories:
                print(f"  - {mem.get('memory', '')}")
        else:
            print("No previous memories found — fresh start")
            
        # Initialize AI climate agent state
        self.state = {
            "farmer_message": "",
            "conversation_history": [],
            "belief": get_persona_belief(persona_key),
            "retrieved_chunks": [],
            "selected_chunk": {},
            "reasoning_trace": "",
            "agent_response": "",
            "persona_key": persona_key, 
            "memory_context": "",
            "farmer_id": self.farmer_id
        } # self.state

        print(f"Agent ready. Farmer: {self.persona['name']}")
        print(belief_summary(self.state["belief"]))


    def chat(self, farmer_message: str) -> dict:
        """
        Takes a farmer's message, runs full agent graph, updates state, + returns results:
        Processes 1 farmer message + returns the AI climate agent's response.

        Updates internal state across turns so belief vector +
        conversation history persist throughout the session.

        Args:
            farmer_message: str, what the farmer said

        Returns:
            dict containing:
                - response: the agent's response text
                - belief: updated belief vector
                - reasoning: why the agent chose this content
                - selected_chunk: which content was delivered
        """

        # Update state with new farmer message
        self.state["farmer_message"] = farmer_message

        # Run LangGraph agent (5 node sequence + returned final result)
        result = self.app.invoke(self.state)

        # Update persistent state with results
        self.state["belief"] = result["belief"]
        self.state["conversation_history"] = result["conversation_history"]
        self.state["retrieved_chunks"] = result["retrieved_chunks"]
        self.state["selected_chunk"] = result["selected_chunk"]
        self.state["reasoning_trace"] = result["reasoning_trace"]

        # Store this exchange in Mem0 memory
        store_memories(
            self.memory,
            self.farmer_id,
            {
                "farmer": farmer_message,
                "agent": result["agent_response"]
            }
        )
        
        return {
            "response": result["agent_response"],
            "belief": result["belief"],
            "reasoning": result["reasoning_trace"],
            "selected_chunk": result["selected_chunk"], 
            
        } # return


    def get_belief_summary(self) -> str:
        """Returns readable summary of current belief state."""
        return belief_summary(self.state["belief"])


    def reset(self, persona_key: str = None):
        """
        Resets AI climate agent to a fresh state.
        Optionally switches to a different persona.

        Args:
            persona_key: optional new persona. If None keeps current.
        """

        if persona_key:
            self.persona_key = persona_key
            self.persona = FARMER_PERSONAS[persona_key]

        self.state = {
            "farmer_message": "",
            "conversation_history": [],
            "belief": get_persona_belief(self.persona_key),
            "retrieved_chunks": [],
            "selected_chunk": {},
            "reasoning_trace": "",
            "agent_response": "",
            "persona_key": self.persona_key, 
            "memory_context": "", 
            "farmer_id": self.farmer_id
        }

        print(f"Agent reset. Persona: {self.persona['name']}")