"""
belief_model.py

Defines the farmer belief model: a probability distribution over 4 possible climate futures for Soule, France. 

The belief model is grounded in Bayesian brain theory which proposes that the brain 
represents knowledge as probability distributions & updates them when new info arrives. 
Here we apply this to model a farmer's evolving understanding of their climate future.

Each farmer maintains a belief vector — 4 probabilities summing to 1.0
representing how strongly they currently believe in each possible future:
    - Mediterranean Shift
    - Moist Atlantic
    - Tropical Basque
    - No Change

The belief vector updates after every agent-farmer exchange based on the
farmer's response, approximating Bayesian updating with a simple heuristic.

Connected to the Free Energy Principle: the agent selects climate content from the database
that maximizes expected informativeness relative to the farmer's current belief state,
choosing what would most reduce the farmer's uncertainty about their climate future.
"""
STORYLINES = [
    "Mediterranean Shift",
    "Moist Atlantic",
    "Tropical Basque",
    "No Change"
] # 4 plausible climate futures where No Change means prior belief that climate will NOT significantly change

DEFAULT_BELIEF = {
    "Mediterranean Shift": 0.25,
    "Moist Atlantic": 0.25,
    "Tropical Basque": 0.25,
    "No Change": 0.25
} # default farmer with no strong prior belief in any direction

FARMER_PERSONAS = {
    "skeptic": {
        "name": "Jean-Pierre",
        "description": (
            "An experienced highland farmer who has worked the same pastures "
            "for 30 years. Skeptical of external climate projections. "
            "Anchors strongly on personal memory and past experience. "
            "Responds better to temporal analogs than statistics."
        ),
        "belief": {
            "Mediterranean Shift": 0.10,
            "Moist Atlantic": 0.15,
            "Tropical Basque": 0.10,
            "No Change": 0.65
        },
        "response_style": "experiential"
    },
    "worried": {
        "name": "Marie",
        "description": (
            "A younger farmer who has noticed significant changes over the "
            "last decade. Already adjusting grazing schedules and worried "
            "about the future. Open to data and projections. "
            "Primary concern is highland pasture availability and snow timing."
        ),
        "belief": {
            "Mediterranean Shift": 0.40,
            "Moist Atlantic": 0.30,
            "Tropical Basque": 0.20,
            "No Change": 0.10
        },
        "response_style": "statistical"
    },
    "neutral": {
        "name": "Arnaud",
        "description": (
            "A mid-career farmer who has noticed some changes but has not "
            "formed strong opinions. Responds well to concrete geographic "
            "comparisons and practical farming implications. "
            "Primary concern is cheese production and winter conditions."
        ),
        "belief": {
            "Mediterranean Shift": 0.25,
            "Moist Atlantic": 0.30,
            "Tropical Basque": 0.25,
            "No Change": 0.20
        },
        "response_style": "narrative"
    }
} # 3 synthetic farmer personas

## AGREEMENT + SKEPTICISIM SIGNALS for BELIEF UPDATING:
AGREEMENT_SIGNALS = [
    "yes", "right", "exactly", "agree", "true", "correct",
    "noticed", "remember", "experienced", "happened", "seen",
    "makes sense", "that's right", "definitely", "absolutely"
] # increase storyline's probability

SKEPTICISM_SIGNALS = [
    "no", "wrong", "disagree", "doubt", "unlikely", "never",
    "don't think", "not sure", "skeptical", "hard to believe",
    "always been", "nothing changed", "normal", "always like this"
] # decrease storyline's probability

STORYLINE_KEYWORDS = {
    "Mediterranean Shift": [
        "dry", "drought", "hot", "heat", "fire", "water",
        "2022", "summer", "drier", "hotter", "fire risk"
    ],
    "Moist Atlantic": [
        "wet", "rain", "flood", "saturated", "soggy", "drainage",
        "2023", "cheese", "hay", "silage", "humid"
    ],
    "Tropical Basque": [
        "storm", "thunder", "pest", "tick", "muggy", "intense",
        "2013", "insects", "armyworm", "heavy rain", "flooding"
    ],
    "No Change": [
        "normal", "always", "same", "nothing", "unchanged",
        "always been", "like always", "no difference"
    ]
} # each match increases storyline's probability weight

## CORE BELIEF FUNCTIONS:
def get_persona_belief(persona_key: str) -> dict:
    """
    Returns starting belief vector for a given farmer persona.

    Args:
        persona_key: 'skeptic', 'worried', 'neutral'

    Returns:
        dict mapping storyline name to starting probability float
    """
    if persona_key not in FARMER_PERSONAS:
        raise ValueError(f"Unknown persona: {persona_key}. Choose from: {list(FARMER_PERSONAS.keys())}")

    # Return a copy so the original persona prior is never modified
    return FARMER_PERSONAS[persona_key]["belief"].copy()


def normalize_belief(belief: dict) -> dict:
    """
    Normalizes a belief vector so all probabilities sum to 1.0.

    This is required after every update to maintain a valid probability distribution for Bayesian belief approximation.

    Args:
        belief: dict mapping storyline name to probability float

    Returns:
        normalized belief dict where values sum to 1.0
    """
    total = sum(belief.values())
    if total == 0:
        return DEFAULT_BELIEF.copy()

    return {storyline: prob / total for storyline, prob in belief.items()}


def update_belief(current_belief: dict, farmer_response: str, delivered_chunk: dict) -> dict:
    """
    Updates farmer belief vector after one agent-farmer exchange.

    This is a heuristic approximation of Bayesian belief updating.
    Full Bayesian updating would require a likelihood model P(response | storyline).
    Instead we use keyword detection to approximate belief shifts in a computationally tractable way.

    The update logic:
    1. Detect if farmer expressed agreement or skepticism
    2. Detect if farmer mentioned keywords associated with any storyline
    3. Apply update multipliers to the relevant storyline probabilities
    4. Normalize the result back to a valid probability distribution

    Args:
        current_belief: dict mapping storyline to current probability
        farmer_response: str, the raw text of what the farmer said
        delivered_chunk: dict, the content chunk the agent just delivered must contain 'storyline' key

    Returns:
        updated and normalized belief dict
    """
    # Work on a copy of belief so we never mutate the original
    belief = current_belief.copy()
    response_lower = farmer_response.lower()

    # Detect agreement or skepticism in the response
    expressed_agreement = any(signal in response_lower for signal in AGREEMENT_SIGNALS)
    expressed_skepticism = any(signal in response_lower for signal in SKEPTICISM_SIGNALS)

    # Get the storyline of the content that was just delivered
    delivered_storyline = delivered_chunk.get("storyline", None)

    # Update Rule 1: agreement means increase delivered storyline by 40%, reduce No Change belief
    if expressed_agreement and delivered_storyline and delivered_storyline in belief:
        belief[delivered_storyline] *= 1.4
        belief["No Change"] *= 0.8

    # Update Rule 2: skepticism means decrease delivered storyline by 20%, increase No Change belief
    if expressed_skepticism and delivered_storyline and delivered_storyline in belief:
        belief[delivered_storyline] *= 0.8
        belief["No Change"] *= 1.2

    # Update Rule 3: storyline-specifici keywords increase storyline's weight by 5% regardless of explicit agreement/skepticism
    for storyline, keywords in STORYLINE_KEYWORDS.items():
        keyword_matches = sum(1 for kw in keywords if kw in response_lower)
        if keyword_matches > 0:
            belief[storyline] *= (1.0 + (0.05 * keyword_matches))

    belief = normalize_belief(belief)
    return belief


def get_content_priority(current_belief: dict, available_chunks: list) -> list:
    """
    Ranks available climate content chunks by expected informativeness
    given the farmer's current belief state.

    Grounded in the Free Energy Principle: agent should select content 
    that would most reduce the farmer's uncertainty, prioritizing chunks 
    from storylines the farmer currently underweights relative to their plausibility.

    Simple implementation: chunks from lower-probability storylines
    are ranked higher because they carry more new information for this farmer. 
    Ex. a farmer who already strongly believes in Med. Shift needs less info about it, they need info about the other scenarios to reduce overall uncertainty.

    Args:
        current_belief: dict mapping storyline to current probability
        available_chunks: list of chunk dicts from ChromaDB retrieval

    Returns:
        list of chunks sorted by priority score, highest first
    """

    def priority_score(chunk):
        """
        Scores a climate content chunk by informativeness with lower belief meaning higher priority. 

        Args:
            chunk: dict containing 'storyline' + 'abstraction_level' keys

        Returns:
            float priority score for a farmer
        """
        storyline = chunk.get("storyline", "")

        # Lower belief = higher priority (more informative)
        storyline_belief = current_belief.get(storyline, 0.25)

        # Invert so lower belief = higher score
        informativeness = 1.0 - storyline_belief

        # Bonus for experiential abstraction level since temporal analogs are more effective
        abstraction_bonus = 0.1 if chunk.get("abstraction_level") == "experiential" else 0.0

        return informativeness + abstraction_bonus

    return sorted(available_chunks, key=priority_score, reverse=True)


def belief_summary(belief: dict) -> str:
    """
    Returns a human readable summary of current belief state.
    Used for agent reasoning traces & frontend display.

    Args:
        belief: dict mapping storyline to probability float

    Returns:
        formatted string showing current belief distribution
    """
    lines = ["Current farmer belief state:"]
    sorted_belief = sorted(belief.items(), key=lambda x: x[1], reverse=True)
    for storyline, prob in sorted_belief:
        # Visual bar using unicode blocks for terminal display
        bar_length = int(prob * 20)
        bar = "█" * bar_length + "░" * (20 - bar_length)
        lines.append(f"  {storyline:<22} {bar} {prob:.1%}")
    return "\n".join(lines)