"""
climate_agent.py

Core AI agent implementation using LangGraph's ReAct loop.

The AI climate agent adaptively selects + delivers climate scenario content 
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

import os
import sys
from typing import TypedDict, Annotated
import operator

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END

from backend.agent.belief_model import (
    update_belief,
    get_content_priority,
    belief_summary,
    get_persona_belief,
    FARMER_PERSONAS
) # belief_model.py

from backend.memory.farmer_memory import (
    store_memories,
    retrieve_memories,
    get_all_memories,
    format_memories_for_context
) # Mem0 memory

# AI climate agent state
class AgentState(TypedDict):
    """
    The complete state of the AI climate agent at any point in the conversation.
    LangGraph passes this state between nodes in the graph + updates it at each step.

    Every field represents 1 piece of info the AI climate agent needs to reason + act correctly.
    """
    farmer_message: str # farmer's most recently inputted message
    conversation_history: Annotated[list, operator.add] # list of {role, content} dicts
    belief: dict # current farmer belief vector of storyline + probability 
    retrieved_chunks: list # top 5 ChromaDB content chunks for this turn
    selected_chunk: dict # single chunk selected for delivery this turn
    reasoning_trace: str # AI climate agent's reasoning trace: why it chose that chunk
    agent_response: str # final output response to farmer
    persona_key: str # active farmer persona type
    # Adding memory + farmer_id to AgentState so memory can be accessed in respond_node:
    memory_context: str # formatted Mem0 memories for system prompt
    farmer_id: str # Mem0 namespacing key

## LANGGRAPH NODES: EACH NODE IS 1 STEP IN REACT LOOP
def retrieve_node(state: AgentState, collection) -> dict:
    """
    Queries ChromaDB for the 5 climate content chunks most semantically similar to farmer's message.

    Args:
        state: current AgentState
        collection: ChromaDB collection

    Returns:
        dict with retrieved_chunks list
    """
    farmer_message = state["farmer_message"]

    results = collection.query(
        query_texts=[farmer_message],
        n_results=5
    ) # query ChromaDB for 5 most semantically similar chunks 

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

def rank_node(state: AgentState) -> dict:
    """
    Re-ranks retrieved climate content chunks by belief-weighted informativeness (Free Energy Principle).
    Selects top climate content chunk for delivery this turn.
    
    Args:
        state: current AgentState
    
    Returns:
        dict with selected_chunk
    """
    belief = state["belief"]
    chunks = state["retrieved_chunks"]

    # Re-rank by belief-weighted informativeness
    prioritized = get_content_priority(belief, chunks)
    selected = prioritized[0] if prioritized else {}

    print(f"Selected chunk: {selected.get('storyline')} / {selected.get('abstraction_level')}")
    print(belief_summary(belief))

    return {"selected_chunk": selected}

def retrieve_memory_node(state: AgentState, memory) -> dict:
    """
    Searches Mem0 for memories relevant to current message.
    Injects formatted context into state for the respond node in the ReAct loop.
    
    Args:
        state: current AgentState
        memory: Mem0 MemoryClient
    
    Returns:
        dict with memory_context String
    """
    farmer_message = state["farmer_message"]
    farmer_id = state["farmer_id"]

    relevant_memories = retrieve_memories(
        memory,
        farmer_id,
        farmer_message,
        limit=3
    ) # semantic search against farmer's stored memories

    # Format memories into readable string for system prompt injection
    memory_context = format_memories_for_context(relevant_memories)

    return {"memory_context": memory_context}


def reason_node(state: AgentState, llm) -> dict:
    """
    Asks the Groq LLM to expain in 2-3 sentences why the selected content chunk was chosen given farmer's current belief state + message.

    Args:
        state: current AgentState
        llm: ChatGroq instance
    
    Returns:
        dict with reasoning_trace String
    """
    belief = state["belief"]
    selected_chunk = state["selected_chunk"]
    farmer_message = state["farmer_message"]
    persona_key = state["persona_key"]
    persona = FARMER_PERSONAS[persona_key]

    # Ask Groq LLM to explain the content chunk selection
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
    Generates the farmer-facing response by delivering the selected climate content chunk framed for specific persona.

    Args:
        state: current AgentState
        llm: ChatGroq instance

    Returns:
        dict with agent_response + new conversation_history entries
    """
    selected_chunk = state["selected_chunk"]
    farmer_message = state["farmer_message"]
    conversation_history = state["conversation_history"]
    persona_key = state["persona_key"]
    persona = FARMER_PERSONAS[persona_key]
    belief = state["belief"]

    # Build conversation context from history using last 4 exchanges (keeps context window manageable)
    recent_history = conversation_history[-8:] if len(conversation_history) > 8 else conversation_history

    # Determine framing instruction based on abstraction level
    abstraction_level = selected_chunk.get("abstraction_level", "narrative")
    if abstraction_level == "experiential":
        framing = "Frame this using the temporal analog year as the primary hook. Connect to lived experience."
    elif abstraction_level == "statistical":
        framing = "Present the information with specific numbers and data. Be precise."
    else:
        framing = "Use descriptive narrative language that paints a picture of what the climate would feel like."

    # Memory context from previouse sessions
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

    # Build message list
    messages = [SystemMessage(content=system_prompt)]

    # Add recent conversation history by converting to LangChain message objects
    for msg in recent_history:
        if msg["role"] == "farmer":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))

    # Append delivery prompt
    delivery_prompt = f"""The farmer just said: "{farmer_message}"

Deliver this climate information naturally in the conversation:
{selected_chunk.get('content_text')}

Remember to connect it to what the farmer just said."""

    messages.append(HumanMessage(content=delivery_prompt))

    # Generate response
    response = llm.invoke(messages)
    agent_response = response.content

    new_messages = [
        {"role": "farmer", "content": farmer_message},
        {"role": "agent", "content": agent_response}
    ] # add this exchange to conversation history

    return {
        "agent_response": agent_response,
        "conversation_history": new_messages
    }


def update_belief_node(state: AgentState) -> dict:
    """
    Updates farmer's belief vector based on their response to the delivered content chunk.
    
    Args:
        state: current AgentState

    Returns:
        dict with updated belief
    """
    farmer_message = state["farmer_message"]
    current_belief = state["belief"]
    selected_chunk = state["selected_chunk"]

    updated_belief = update_belief(
        current_belief,
        farmer_message,
        selected_chunk
    ) # update belief based on farmer response + delivered content

    print("Belief updated:")
    print(belief_summary(updated_belief))

    return {"belief": updated_belief}

## BUILD LANGGRAPH:
def build_agent(collection, llm, memory):
    """
    Assembles the LangGraph state graph connecting all nodes in a linear chain.
    The graph defines the sequence of operations the agent performs for each farmer message:
    retrieve -> rank -> reason -> respond -> update_belief
    
    Args:
        collection: ChromaDB collection
        llm: ChatGroq instance
        memory: Mem0 MemoryClient

    Returns:
        compiled LangGraph app
    """
    # Create LangGraph state machine + register each node: create the state graph with AgentState as state schema
    workflow = StateGraph(AgentState)

    # Add each node to the graph:
    # Each node is a function that takes state + returns updated state
    # Register nodes by using lambda functions to inject dependencies without adding to state
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

    # Compile LangGraph graph into a runnable app
    app = workflow.compile()

    print("LangGraph agent compiled with memory successfully")

    return app

class ClimateAgent:
    """
    Wraps the LangGraph agent + manages state across turns within a session.
    Instantiated per session by main.py with shared injected components. 
    """

    def __init__(self, persona_key: str, collection, llm, memory, initial_belief: dict = None):
        """
        Initializes AI climate agent with a specific farmer persona + injected components.

        Args:
            persona_key: one of 'skeptic', 'worried', 'neutral'
            collection: shared ChromaDB collection from main.py lifespan
            llm: shared ChatGroq instance from main.py lifespan
            memory: shared Mem0 MemoryClient from main.py lifespan
            initial_belief: persisted belief vector if returning farmer, else None
        """
        print(f"\nInitializing Climate Agent with persona: {persona_key}")
        # Shared components injected from main.py lifespan
        self.collection = collection
        self.llm = llm
        self.memory = memory

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
            
        self.state = {
            "farmer_message": "",
            "conversation_history": [],
            "belief": initial_belief if initial_belief is not None else get_persona_belief(persona_key),
            "retrieved_chunks": [],
            "selected_chunk": {},
            "reasoning_trace": "",
            "agent_response": "",
            "persona_key": persona_key, 
            "memory_context": "",
            "farmer_id": self.farmer_id
        } # initialize AI climate agent state

        print(f"Agent ready. Farmer: {self.persona['name']}")
        print(belief_summary(self.state["belief"]))


    def chat(self, farmer_message: str) -> dict:
        """
        Runs 1 farmer message through full agent graph + returns results.

        Args:
            farmer_message: farmer's input text

        Returns:
            dict with response, belief, reasoning, + selected_chunk
        """
        # Update state with new farmer message
        self.state["farmer_message"] = farmer_message

        # Run the agent graph
        result = self.app.invoke(self.state)

        # Update persistent state with results
        self.state["belief"] = result["belief"]
        self.state["conversation_history"] = result["conversation_history"]
        self.state["retrieved_chunks"] = result["retrieved_chunks"]
        self.state["selected_chunk"] = result["selected_chunk"]
        self.state["reasoning_trace"] = result["reasoning_trace"]

        store_memories(
            self.memory,
            self.farmer_id,
            {
                "farmer": farmer_message,
                "agent": result["agent_response"]
            }
        ) # store this exchange in Mem0 memory
        
        return {
            "response": result["agent_response"],
            "belief": result["belief"],
            "reasoning": result["reasoning_trace"],
            "selected_chunk": result["selected_chunk"], 
            
        } 


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