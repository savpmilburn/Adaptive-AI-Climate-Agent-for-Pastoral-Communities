"""
PCS_storylines.py

Climate data generated using LLM prompt + VIPR Climate Storylines First Draft document 
(3 storyline narratives w/ ecological notes + naturalist narrative summaries) for Soule (Xiberoa), France.

Each entry represents one retrievable content chunk that the AI climate agent
can select + deliver to a farmer during a conversation.

Metadata fields per chunk:
- storyline: which of the 3 climate futures this belongs to (Med. Shift, Moist Atlantic, Trop. Basque)
- elevation_band: which part of Soule this describes
- season: which season this is most relevant to
- variable_type: what climate variable is being described
- abstraction_level: statistical | narrative | experiential
- analog_type: none | temporal | spatial
- analog_reference: specific year or place if analog_type is not none
- farmer_concern: what farming activity this most affects
- content_text: the actual content delivered to the farmer
"""

STORYLINES = [
    # Mediterreanean Shift
    {
        "chunk_id": "med_001",
        "storyline": "Mediterranean Shift",
        "elevation_band": "highland",
        "season": "summer",
        "variable_type": "temperature",
        "abstraction_level": "narrative",
        "analog_type": "none",
        "analog_reference": None,
        "farmer_concern": "grazing",
        "content_text": (
            "Under the Mediterranean Shift, summers in the highland pastures "
            "grow markedly hotter and drier. The freezing level often stays "
            "above the mountain crests, snow melts weeks earlier, and "
            "late-summer dryness reaches higher elevations than today. "
            "The seasonal window for highland grazing tightens significantly."
        )
    },
    {
        "chunk_id": "med_002",
        "storyline": "Mediterranean Shift",
        "elevation_band": "highland",
        "season": "summer",
        "variable_type": "water",
        "abstraction_level": "narrative",
        "analog_type": "none",
        "analog_reference": None,
        "farmer_concern": "water_access",
        "content_text": (
            "Water supply problems at higher elevations are likely most summers "
            "under the Mediterranean Shift. Some farmers may be forced to move "
            "animals to lower pastures earlier than today, or invest in water "
            "retention structures to maintain highland grazing."
        )
    },
    {
        "chunk_id": "med_003",
        "storyline": "Mediterranean Shift",
        "elevation_band": "lowland",
        "season": "summer",
        "variable_type": "temperature",
        "abstraction_level": "narrative",
        "analog_type": "none",
        "analog_reference": None,
        "farmer_concern": "grazing",
        "content_text": (
            "In the valley pastures, late spring turns quickly into summer heat "
            "under the Mediterranean Shift. Stifling warm nights become routine "
            "during hot episodes, and prolonged dry spells with sporadic showers "
            "replace the more regular rainfall of today."
        )
    },
    {
        "chunk_id": "med_004",
        "storyline": "Mediterranean Shift",
        "elevation_band": "all",
        "season": "winter",
        "variable_type": "precipitation",
        "abstraction_level": "narrative",
        "analog_type": "none",
        "analog_reference": None,
        "farmer_concern": "grazing",
        "content_text": (
            "Winters under the Mediterranean Shift turn milder with early rains. "
            "The milder winter temperatures may allow a second season of grazing "
            "in autumn or over winter months, partially offsetting the shortened "
            "summer highland window."
        )
    },
    {
        "chunk_id": "med_005",
        "storyline": "Mediterranean Shift",
        "elevation_band": "lowland",
        "season": "summer",
        "variable_type": "fire",
        "abstraction_level": "narrative",
        "analog_type": "none",
        "analog_reference": None,
        "farmer_concern": "land_management",
        "content_text": (
            "Fire risk increases significantly under the Mediterranean Shift. "
            "Vegetation accumulates in spring then faces hot dry summers, "
            "creating dangerous fire weather conditions. This is the scenario "
            "with the highest fire risk of the three futures."
        )
    },
    {
        "chunk_id": "med_006",
        "storyline": "Mediterranean Shift",
        "elevation_band": "lowland",
        "season": "summer",
        "variable_type": "haymaking",
        "abstraction_level": "narrative",
        "analog_type": "none",
        "analog_reference": None,
        "farmer_concern": "hay_production",
        "content_text": (
            "Haymaking may actually become easier under the Mediterranean Shift "
            "due to reduced rainfall in June, providing more reliable dry windows. "
            "Farmers may also be able to move animals to high pastures earlier "
            "in the season due to favorable spring temperatures."
        )
    },
    {
        "chunk_id": "med_007",
        "storyline": "Mediterranean Shift",
        "elevation_band": "all",
        "season": "all",
        "variable_type": "temperature",
        "abstraction_level": "experiential",
        "analog_type": "temporal",
        "analog_reference": "2022",
        "farmer_concern": "grazing",
        "content_text": (
            "The Mediterranean Shift climate future resembles what farmers in "
            "Soule experienced in 2022 — a year many remember for its heat, "
            "dry summer conditions, and the pressure it placed on highland "
            "grazing and water availability."
        )
    },
    # Moist Atlantic
    {
        "chunk_id": "moist_001",
        "storyline": "Moist Atlantic",
        "elevation_band": "all",
        "season": "winter",
        "variable_type": "precipitation",
        "abstraction_level": "narrative",
        "analog_type": "none",
        "analog_reference": None,
        "farmer_concern": "grazing",
        "content_text": (
            "Under the Moist Atlantic future, the familiar Atlantic rhythm "
            "of Soule persists but shifts warmer and wetter. Winters lose "
            "many frost days at all elevations, rain days become more frequent "
            "across all seasons, and wet fields with saturated soils become "
            "more common from January through April."
        )
    },
    {
        "chunk_id": "moist_002",
        "storyline": "Moist Atlantic",
        "elevation_band": "lowland",
        "season": "spring",
        "variable_type": "precipitation",
        "abstraction_level": "narrative",
        "analog_type": "none",
        "analog_reference": None,
        "farmer_concern": "hay_production",
        "content_text": (
            "Haymaking grows more problematic under the Moist Atlantic scenario "
            "as reliable dry spells become harder to find. Many farming operations "
            "may need to shift toward silage to preserve forage, which could "
            "compromise subsidies tied to traditional cheese production that "
            "depends on hay-fed milk."
        )
    },
    {
        "chunk_id": "moist_003",
        "storyline": "Moist Atlantic",
        "elevation_band": "lowland",
        "season": "winter",
        "variable_type": "soil",
        "abstraction_level": "narrative",
        "analog_type": "none",
        "analog_reference": None,
        "farmer_concern": "pasture_management",
        "content_text": (
            "Saturated soils under the Moist Atlantic future raise the risk "
            "of pasture compaction in heavily grazed fields. Investment in "
            "drainage infrastructure may become necessary to maintain pasture "
            "quality, particularly in the valley and lowland areas."
        )
    },
    {
        "chunk_id": "moist_004",
        "storyline": "Moist Atlantic",
        "elevation_band": "highland",
        "season": "spring",
        "variable_type": "snow",
        "abstraction_level": "narrative",
        "analog_type": "none",
        "analog_reference": None,
        "farmer_concern": "transhumance",
        "content_text": (
            "Under the Moist Atlantic scenario, snow retreats earlier at high "
            "pastures and spring arrives sooner with fewer cold snaps interrupting "
            "leaf-out. The window for moving animals to high pastures may open "
            "a little earlier, though increased snowfall at higher elevations "
            "may require some timing adjustments."
        )
    },
    {
        "chunk_id": "moist_005",
        "storyline": "Moist Atlantic",
        "elevation_band": "all",
        "season": "all",
        "variable_type": "temperature",
        "abstraction_level": "experiential",
        "analog_type": "temporal",
        "analog_reference": "2023",
        "farmer_concern": "grazing",
        "content_text": (
            "The Moist Atlantic climate future resembles conditions farmers "
            "in Soule experienced in 2023 — a year characterized by wetter "
            "than usual conditions and the pasture management challenges "
            "that came with persistent moisture across the seasons."
        )
    },
    {
        "chunk_id": "moist_006",
        "storyline": "Moist Atlantic",
        "elevation_band": "all",
        "season": "all",
        "variable_type": "precipitation",
        "abstraction_level": "narrative",
        "analog_type": "none",
        "analog_reference": None,
        "farmer_concern": "cheese_production",
        "content_text": (
            "Traditional cheese production in Soule could face pressure under "
            "the Moist Atlantic scenario. The shift away from hay toward silage "
            "driven by wetter conditions may conflict with production standards "
            "and subsidies tied to hay-fed milk from highland pastures."
        )
    },
    # Tropical Basque
    {
        "chunk_id": "trop_001",
        "storyline": "Tropical Basque",
        "elevation_band": "lowland",
        "season": "summer",
        "variable_type": "temperature",
        "abstraction_level": "narrative",
        "analog_type": "none",
        "analog_reference": None,
        "farmer_concern": "grazing",
        "content_text": (
            "Under the Tropical Basque scenario, summers turn warmer and muggier "
            "across the valley and lowland pastures. Higher humidity produces "
            "more tropical nights, and convective thunderstorms with intense "
            "downburst winds become more frequent, particularly in the lowlands."
        )
    },
    {
        "chunk_id": "trop_002",
        "storyline": "Tropical Basque",
        "elevation_band": "all",
        "season": "all",
        "variable_type": "precipitation",
        "abstraction_level": "narrative",
        "analog_type": "none",
        "analog_reference": None,
        "farmer_concern": "pasture_management",
        "content_text": (
            "Total rainfall increases in every season under the Tropical Basque "
            "scenario, arriving in larger bursts. Outdoor working conditions "
            "deteriorate for both people and livestock, with animals suffering "
            "more from hoof problems on saturated ground and pastures becoming "
            "increasingly sensitive to compaction — particularly in winter."
        )
    },
    {
        "chunk_id": "trop_003",
        "storyline": "Tropical Basque",
        "elevation_band": "all",
        "season": "all",
        "variable_type": "pests",
        "abstraction_level": "narrative",
        "analog_type": "none",
        "analog_reference": None,
        "farmer_concern": "animal_health",
        "content_text": (
            "Pest and parasite pressures rise significantly under the Tropical "
            "Basque scenario. Tick populations expand and outbreaks of species "
            "like Mythimna unipuncta — the true armyworm — are likely to become "
            "more frequent. This is the scenario with the highest pest pressure "
            "of the three futures."
        )
    },
    {
        "chunk_id": "trop_004",
        "storyline": "Tropical Basque",
        "elevation_band": "lowland",
        "season": "summer",
        "variable_type": "hay_production",
        "abstraction_level": "narrative",
        "analog_type": "none",
        "analog_reference": None,
        "farmer_concern": "hay_production",
        "content_text": (
            "Valley haymaking becomes considerably more difficult under the "
            "Tropical Basque scenario as wetter conditions reduce the number "
            "of suitable drying windows. Farms may need to rely more heavily "
            "on plastic-wrapped silage bales to preserve forage through the "
            "wetter seasons."
        )
    },
    {
        "chunk_id": "trop_005",
        "storyline": "Tropical Basque",
        "elevation_band": "highland",
        "season": "winter",
        "variable_type": "snow",
        "abstraction_level": "narrative",
        "analog_type": "none",
        "analog_reference": None,
        "farmer_concern": "transhumance",
        "content_text": (
            "At higher pastures under the Tropical Basque scenario, snowpack "
            "thins and melts earlier. Winter remains mild and wet but rainfall "
            "distribution tilts toward heavier events, meaning river flow may "
            "become more unpredictable with potential for flooding."
        )
    },
    {
        "chunk_id": "trop_006",
        "storyline": "Tropical Basque",
        "elevation_band": "all",
        "season": "all",
        "variable_type": "temperature",
        "abstraction_level": "experiential",
        "analog_type": "temporal",
        "analog_reference": "2013",
        "farmer_concern": "grazing",
        "content_text": (
            "The Tropical Basque climate future resembles conditions farmers "
            "in Soule experienced in 2013 — a year remembered for its increased "
            "rainfall intensity, challenging outdoor working conditions, and "
            "the moisture-related pressures it placed on pasture management."
        )
    },
    {
        "chunk_id": "trop_007",
        "storyline": "Tropical Basque",
        "elevation_band": "all",
        "season": "summer",
        "variable_type": "heat_stress",
        "abstraction_level": "narrative",
        "analog_type": "none",
        "analog_reference": None,
        "farmer_concern": "animal_health",
        "content_text": (
            "Heat stress becomes a significant concern for livestock under the "
            "Tropical Basque scenario. Higher humidity limits animals' ability "
            "to cool themselves. Managing mountain pastures to include ample "
            "shade trees may help mitigate this, particularly in valley and "
            "mid-elevation areas."
        )
    },
] # STORYLINES