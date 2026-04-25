"use client";

import { useState, useEffect, useRef } from "react";
import Image from "next/image";
import axios from "axios";

const API_BASE = "http://localhost:8000";

const STORYLINE_COLORS = {
  "Mediterranean Shift": "#e34948",
  "Moist Atlantic": "#4a9eff",
  "Tropical Basque": "#4caf7d",
  "No Change": "#8a9187",
};

const PERSONAS = {
  skeptic: {
    name: "Jean-Pierre",
    initial: "JP",
    description:
      "Experienced highland farmer, skeptical of external projections. Anchors on personal memory and past experience.",
  },
  worried: {
    name: "Marie",
    initial: "M",
    description:
      "Younger farmer who has noticed significant changes. Open to data, worried about highland pasture and snow timing.",
  },
  neutral: {
    name: "Arnaud",
    initial: "A",
    description:
      "Mid-career farmer, hasn't formed strong opinions. Responds to concrete comparisons and practical implications.",
  },
};

// Agent avatar — swaps to image when AI_Agent_Icon.png is in /public
function AgentAvatar() {
  const [hasImage, setHasImage] = useState(true);

  if (hasImage) {
    return (
      <div className="agent-avatar">
        <Image
          src="/AI_Agent_Icon.png"
          alt="Climate Agent"
          width={40}
          height={40}
          style={{ borderRadius: "50%", objectFit: "cover", transform: "scale(1.11)", transformOrigin: "center", marginTop: "6px" }}
          onError={() => setHasImage(false)}
        />
      </div>
    );
  }

  // Fallback emoji if image not found
  return <div className="agent-avatar">🌿</div>;
}

export default function Home() {
  const [sessionId, setSessionId] = useState(null);
  const [personaKey, setPersonaKey] = useState("skeptic");
  const [messages, setMessages] = useState([]);
  const [belief, setBelief] = useState({
    "Mediterranean Shift": 0.1,
    "Moist Atlantic": 0.15,
    "Tropical Basque": 0.1,
    "No Change": 0.65,
  });
  const [reasoning, setReasoning] = useState("");
  const [selectedStoryline, setSelectedStoryline] = useState("");
  const [turnCount, setTurnCount] = useState(0);
  const [inputText, setInputText] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isStarting, setIsStarting] = useState(false);
  const chatEndRef = useRef(null);
  // Groq free API key daily token limit estimate
  const [tokensUsed, setTokensUsed] = useState(0);
  const TOKEN_LIMIT = 100000;
  const TOKENS_PER_TURN_ESTIMATE = 3400;

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  async function startSession(key) {
    setIsStarting(true);
    setMessages([]);
    setReasoning("");
    setSelectedStoryline("");
    setTurnCount(0);

    try {
      const response = await axios.post(`${API_BASE}/session/start`, {
        persona_key: key,
      });
      const data = response.data;
      setSessionId(data.session_id);
      setBelief(data.initial_belief);
      setMessages([
        {
          role: "agent",
          content: `Hello, I'm your climate advisor for Soule. I'm here to help you understand what the future climate might look like for your farm. What's on your mind?`,
          storyline: "",
          abstraction: "",
        },
      ]);
    } catch (error) {
      console.error("Failed to start session:", error);
      alert("Could not connect to backend. Make sure the server is running.");
    } finally {
      setIsStarting(false);
    }
  }

  async function sendMessage() {
    if (!inputText.trim() || !sessionId || isLoading) return;
    const userMessage = inputText.trim();
    setInputText("");
    setIsLoading(true);

    setMessages((prev) => [
      ...prev,
      { role: "farmer", content: userMessage },
    ]);

    try {
      const response = await axios.post(
        `${API_BASE}/session/${sessionId}/chat`,
        { message: userMessage }
      );
      const data = response.data;

      setMessages((prev) => [
        ...prev,
        {
          role: "agent",
          content: data.response,
          storyline: data.selected_storyline,
          abstraction: data.selected_abstraction,
        },
      ]);

      setBelief(data.belief);
      setReasoning(data.reasoning);
      setSelectedStoryline(data.selected_storyline);
      setTurnCount(data.turn_number);
      setTokensUsed(prev => prev + TOKENS_PER_TURN_ESTIMATE);
    } catch (error) {
        const isRateLimit = error?.response?.status === 429 || error?.response?.data?.detail?.includes("rate") || error?.message?.includes("429");
        setMessages((prev) => [
          ...prev,
          {
            role: "agent",
            content: isRateLimit
              ? "The free API token limit has been reached for today. Please try again in a few minutes or tomorrow. This is a limitation of the free Groq tier used in this prototype."
              : "Sorry, something went wrong. Please try again.",
            storyline: "",
            abstraction: "",
          },
        ]);
      } finally {
        setIsLoading(false);
      }
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }

  async function handlePersonaChange(e) {
    const newKey = e.target.value;
    setPersonaKey(newKey);
    await startSession(newKey);
  }

  async function handleReset() {
    await startSession(personaKey);
  }

  useEffect(() => {
    startSession("skeptic");
  }, []);

  const currentPersona = PERSONAS[personaKey];

  return (
    <div>
      {/* Header */}
      <div className="header">
        <div>
          <div className="header-title">
            Adaptive AI Climate Agent for Pastoral Communities in Soule, France
          </div>
          <div className="header-subtitle">
            VIPR WeatherRisk: Climate Futures @  The University of Georgia
          </div>
        </div>
        <button className="reset-button" onClick={handleReset}>
          Reset Conversation
        </button>
      </div>

      {/* Explanation Banner */}
      <div style={{
        backgroundColor: "var(--color-navy)",
        borderBottom: `1px solid var(--color-navy-border)`,
        padding: "12px 32px",
        display: "flex",
        alignItems: "center",
        gap: "32px"
      }}>
        <div style={{
          fontSize: "13px",
          color: "#ffffff",
          lineHeight: "1.6",
          fontWeight: 700, 
          fontFamily: "var(--font-ranade)",
          maxWidth: "890px"
        }}>
          This <span style={{ color: "var(--color-accent-yellow)"}}>AI climate agent</span> models a <span style={{ color: "var(--color-accent-red)"}}>farmer&apos;s belief state</span> about their climate future + adaptively selects content from co-produced{" "}
          <span style={{ color: "var(--color-accent-yellow)"}}>VIPR climate scenarios</span>
          {" "}to update that belief, grounded in Bayesian brain theory + the Free Energy Principle. 
          <br/>
          Instead of delivering the same information to every <span style={{ color: "var(--color-accent-red)"}}>farmer</span>, it tracks what <span style={{ color: "var(--color-accent-red)"}}>this specific farmer believes</span> + chooses the framing most likely to reduce their uncertainty regarding <span style={{ color: "var(--color-accent-yellow)"}}>projected climate futures</span>.{" "}
          <br/>
          Left panel: the conversation between the <span style={{ color: "var(--color-accent-yellow)"}}>AI climate agent</span> + <span style={{ color: "var(--color-accent-red)"}}>farmer</span>.{" "}
          <br/>
          Right panel: the <span style={{ color: "var(--color-accent-red)"}}>farmer&apos;s evolving belief vector</span> in real time.
          <br/>
          Use the dropdown to select a <span style={{ color: "var(--color-accent-red)"}}>farmer persona</span> & start a conversation: 
        </div>
      </div>

      {/* Main Layout */}
      <div className="main-layout">

        {/* LEFT PANEL — Light — Chat */}
        <div className="panel panel-light">
          <div className="panel-header-light">
            Farmer Conversation
            {turnCount > 0 && (
              <span className="session-badge">Turn {turnCount}</span>
            )}
          </div>

          <div className="persona-selector">
            <label>Persona</label>
            <select value={personaKey} onChange={handlePersonaChange}>
              <option value="skeptic">Jean-Pierre: The Skeptic</option>
              <option value="worried">Marie: The Worried Adopter</option>
              <option value="neutral">Arnaud: The Neutral</option>
            </select>
          </div>

          <div className="persona-description">
            {currentPersona.description}
          </div>

          <div className="chat-window">
            {isStarting ? (
              <div className="empty-state">Starting session...</div>
            ) : messages.length === 0 ? (
              <div className="empty-state">
                Select a farmer persona to begin
              </div>
            ) : (
              messages.map((msg, i) => (
                <div key={i} className={`message-row ${msg.role}`}>
                  {msg.role === "agent" ? (
                    <AgentAvatar />
                  ) : (
                    <div className="farmer-avatar">
                      {currentPersona.initial}
                    </div>
                  )}
                  <div className={`message ${msg.role}`}>
                    {msg.content}
                    {msg.role === "agent" && msg.storyline && (
                      <div className="message-meta">
                        <span
                          className="meta-dot"
                          style={{
                            backgroundColor:
                              STORYLINE_COLORS[msg.storyline] || "#8a9187",
                          }}
                        />
                        {msg.storyline} · {msg.abstraction}
                      </div>
                    )}
                  </div>
                </div>
              ))
            )}

            {isLoading && (
              <div className="loading-dots">
                <span></span>
                <span></span>
                <span></span>
              </div>
            )}
            <div ref={chatEndRef} />
          </div>

          <div className="chat-input-area">
            <input
              type="text"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={`Type as ${currentPersona.name}...`}
              disabled={isLoading || isStarting || !sessionId}
            />
            <button
              onClick={sendMessage}
              disabled={
                isLoading || isStarting || !sessionId || !inputText.trim()
              }
            >
              {isLoading ? "..." : "Send"}
            </button>
          </div>
        </div>

        {/* RIGHT PANEL — Dark — Belief Dashboard */}
        <div className="panel panel-dark">
          <div className="panel-header-dark">
            Belief Dashboard
          </div>

          <div className="belief-panel">
            <div>
              <div className="belief-section-title">
                Farmer Belief State
              </div>
              {Object.entries(belief).map(([storyline, prob]) => (
                <div key={storyline} className="belief-bar-row">
                  <div className="belief-bar-label">{storyline}</div>
                  <div className="belief-bar-track">
                    <div
                      className="belief-bar-fill"
                      style={{
                        width: `${(prob * 100).toFixed(1)}%`,
                        backgroundColor:
                          STORYLINE_COLORS[storyline] || "#8a9187",
                      }}
                    >
                      <span>{(prob * 100).toFixed(1)}%</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {selectedStoryline && (
              <div className="last-delivered">
                <span
                  style={{
                    width: 8,
                    height: 8,
                    borderRadius: "50%",
                    background:
                      STORYLINE_COLORS[selectedStoryline] || "#8a9187",
                    flexShrink: 0,
                    display: "inline-block",
                  }}
                />
                <span>
                  <strong>Last delivered:</strong> {selectedStoryline}
                </span>
              </div>
            )}

            <div className="reasoning-section">
              <div className="reasoning-title">Agent Reasoning</div>
              {reasoning ? (
                <div className="reasoning-text">{reasoning}</div>
              ) : (
                <div
                  className="reasoning-text"
                  style={{ color: "var(--color-accent-yellow)" }}
                >
                  The agent's reasoning will appear here after the first
                  message.
                </div>
              )}
            </div>

            {turnCount > 0 && (
              <div className="turn-counter">
                {turnCount} conversation{" "}
                {turnCount === 1 ? "turn" : "turns"} completed
              </div>
            )}

            {tokensUsed > 0 && (
                <div style={{
                  fontSize: "12px",
                  fontFamily: "var(--font-archivo)",
                  color: tokensUsed > 80000 
                    ? "var(--color-accent-red)" 
                    : "var(--color-text-light-secondary)",
                  textAlign: "right",
                  marginTop: "4px"
                }}>
                  ~{tokensUsed.toLocaleString()} / {TOKEN_LIMIT.toLocaleString()} tokens used today
                  {tokensUsed > 80000 && " — approaching limit"}
                </div>
              )}
          </div>
        </div>
      </div>
    </div>
  );
}