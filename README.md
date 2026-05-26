# Adaptive AI Climate Agent for Pastoral Communities
![A flock of sheep begins its descent into the valley at the end of summer, in the Soula Valley, on September 14, 2022 from https://www.theatlantic.com/photo/2022/09/new-generation-shepherds-french-pyrenees/671582/](https://cdn.theatlantic.com/thumbor/XQvLugXKNjHCvUNTdEyaok3jv2g=/1200x800/media/img/photo/2022/09/shepherdess/a06_1243417033/original.jpg)
**Built upon climate science research co-produced with pastoral farming communities in the French Basque Pyrenees, led by Dr. Thomas Mote's WeatherRisk VIPR team @ The University of Georgia.**
## The Problem
Farming communities in the French Basque Pyrenees make critical decisions based on understanding seasonal climate patterns such as when to move livestock and how to manage pastures & optimize production. Climate scientists traditionally produce technical projections for farmers & stakeholders every year; however, these projections are not understandable or useful for the people who need them most. 
## The Solution
This prototype is a full-stack AI agent that models what a specific farmer believes about their future climate and adaptively selects & delivers 
[physical climate storyline](https://doi.org/10.1007/s10584-018-2317-9)-inspired narratives co-produced between UGA's WeatherRisk VIPR team and French pastoralists in order to update that belief. Implementation was guided by [Bayesian brain theory](https://doi.org/10.1016/j.tins.2004.10.007) and the [Free Energy Principle](https://doi.org/10.1038/nrn2787) from computational neuroscience. 

Furthermore, instead of every farmer receiving the same climate information in the same order with the same framing, this AI climate agent tracks distinct belief states & chooses the proper climate narrative for the right farmer at the right moment in order to properly model real belief change. 
## Example
A skeptical farmer, Jean-Pierre, is assigned a 65% probability that the climate will not change. Say that Jean-Pierre messages the AI agent: 
> I remember 2022 being a very dry and difficult summer for grazing. 
The AI agent will recognize 2022 as the correct temporal analog for the Mediterranean Shift climate future (derived from VIPR research), retrieve experiential-register content referencing that year, & deliver it to Jean-Pierre. 

Jean-Pierre's belief for the Mediterranean Shift climate future rises from 10% to 17%, connecting an abstract climate future to an experience that Jean-Pierre actually lived through. The AI agent remembers this exchange & the next session will build on this persistent memory. 

## How It Works
* **RAG knowledge base**: 20 climate content chunks in ChromaDB from VIPR, retrieved by semantic similarity + re-ranked by belief-weighted informativeness
* **Probabilistic belief model**: farmer belief vector updates after every exchange using heuristic Bayesian updating
* **[ReAct](https://doi.org/10.48550/arXiv.2210.03629) agent loop**: LangGraph implements retrieve, rank, reason, respond, update belief for every message
* **Cross-session memory**: Mem0 stores key facts about each farmer across separate conversations
* **Free Energy Principle**: climate content selected to maximally reduce farmer uncertainty given their current belief state

### Tech Stack
**Backend**: Python, FastAPI, LangGraph, LangChain, ChromaDB, Mem0, Groq API <br>
**Frontend**: React, Next.js, Axios

### Local Development
1. Clone the repo
2. Create a `.env` file in the project root (view `.env.example`)
3. Install backend dependencies: `pip install -r requirements.txt`
4. Start the backend: `uvicorn backend.main:app --reload`
5. In a second terminal, navigate to `frontend/` + run: `npm install && npm run dev`
6. Open `http://localhost:3000`
> [!WARNING]
> This prototype uses Groq's free API tier (100,000 tokens/day). If the agent stops responding, the daily limit may have been reached. Please try again later. 