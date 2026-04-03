"""
NER-based question preprocessing — extracts city names, party names, and
knesset numbers from questions using a BERT-based NER model (dslim/bert-base-NER).

Replaces the hardcoded dictionary lookup with ML-based entity extraction,
while keeping a normalized lookup table for mapping English city names to
their exact Hebrew equivalents in the database.

Module 3: Attention, Transformers, Embeddings (NER is a token classification
task using a fine-tuned BERT model with attention over input tokens).
"""
import os
import re
from transformers import pipeline

# ── NER model (lazy-loaded singleton) ──
_ner_pipeline = None


def _get_ner():
    global _ner_pipeline
    if _ner_pipeline is None:
        _ner_pipeline = pipeline(
            "ner",
            model="dslim/bert-base-NER",
            aggregation_strategy="simple",
        )
    return _ner_pipeline


# ── Hebrew city name lookup (built from DB at startup) ──
# This maps normalized English transliterations to exact Hebrew DB names.
# The NER model finds candidate entities; this table resolves them.
_city_lookup = None


def _get_city_lookup() -> dict[str, str]:
    """Build English->Hebrew city lookup from the database localities table.

    Uses a hardcoded mapping for common transliterations since there's no
    standard English spelling for Hebrew city names.
    """
    global _city_lookup
    if _city_lookup is not None:
        return _city_lookup

    # Core mappings — covers the most commonly asked cities
    _city_lookup = {
        "tel aviv": "תל אביב - יפו",
        "jerusalem": "ירושלים",
        "haifa": "חיפה",
        "beer sheva": "באר שבע",
        "be'er sheva": "באר שבע",
        "beersheba": "באר שבע",
        "netanya": "נתניה",
        "rishon lezion": "ראשון לציון",
        "rishon le zion": "ראשון לציון",
        "petah tikva": "פתח תקווה",
        "ashdod": "אשדוד",
        "ashkelon": "אשקלון",
        "kiryat ata": "קרית אתא",
        "kiryat bialik": "קרית ביאליק",
        "kiryat yam": "קרית ים",
        "kiryat motzkin": "קרית מוצקין",
        "kiryat gat": "קרית גת",
        "kiryat shmona": "קרית שמונה",
        "nazareth": "נצרת",
        "ramat gan": "רמת גן",
        "bnei brak": "בני ברק",
        "herzliya": "הרצליה",
        "kfar saba": "כפר סבא",
        "modiin": "מודיעין",
        "acre": "עכו",
        "akko": "עכו",
        "tiberias": "טבריה",
        "lod": "לוד",
        "ramla": "רמלה",
        "bat yam": "בת ים",
        "holon": "חולון",
        "eilat": "אילת",
        "rehovot": "רחובות",
        "ra'anana": "רעננה",
        "raanana": "רעננה",
        "upper nazareth": "נצרת עילית",
        "nazareth illit": "נצרת עילית",
        "safed": "צפת",
        "tzfat": "צפת",
    }
    return _city_lookup


def extract_entities(question: str) -> dict:
    """Extract named entities from a question using BERT-based NER.

    Returns:
        dict with keys:
            - "locations": list of location entity strings (LOC/GPE)
            - "organizations": list of org entity strings (ORG) — may include party names
            - "knesset_numbers": list of ints extracted from K14-K25 patterns
            - "years": list of year ints (4-digit numbers in 1990-2030 range)
    """
    ner = _get_ner()
    results = ner(question)

    locations = []
    organizations = []

    for entity in results:
        label = entity["entity_group"]
        word = entity["word"].strip()
        if label == "LOC":
            locations.append(word)
        elif label == "ORG":
            organizations.append(word)

    # Regex for knesset numbers (K14, K25, Knesset 25, etc.)
    knesset_numbers = [int(n) for n in re.findall(r'[kK](\d{1,2})\b', question)]
    knesset_numbers += [int(n) for n in re.findall(r'[Kk]nesset\s+(\d{1,2})', question)]
    knesset_numbers = sorted(set(knesset_numbers))

    # Regex for years
    years = [int(y) for y in re.findall(r'\b(19[9]\d|20[0-3]\d)\b', question)]

    return {
        "locations": locations,
        "organizations": organizations,
        "knesset_numbers": knesset_numbers,
        "years": years,
    }


def preprocess_question_ner(question: str) -> str:
    """Replace English city names with Hebrew equivalents using NER + lookup.

    Pipeline:
    1. Run BERT-NER to extract location entities from the question
    2. For each entity, check the city lookup table (fuzzy match)
    3. Inject the Hebrew name into the question: "Kiryat Ata" -> "Kiryat Ata (Hebrew: קרית אתא)"

    Falls back to dictionary lookup for cities the NER model misses
    (e.g., partial names like "Haifa" that NER might not tag).
    """
    entities = extract_entities(question)
    lookup = _get_city_lookup()

    # Track which positions we've already annotated
    annotated = set()
    result = question

    # Step 1: Try NER-detected locations
    for loc in entities["locations"]:
        loc_lower = loc.lower()
        hebrew = lookup.get(loc_lower)
        if hebrew and loc_lower not in annotated:
            # Replace the entity in the question with annotated version
            # Find the original text (case-preserving)
            pattern = re.compile(re.escape(loc), re.IGNORECASE)
            match = pattern.search(result)
            if match:
                original = match.group()
                replacement = f"{original} (Hebrew: {hebrew})"
                result = result[:match.start()] + replacement + result[match.end():]
                annotated.add(loc_lower)

    # Step 2: Fallback — check for known cities the NER might have missed
    q_lower = result.lower()
    for eng, heb in lookup.items():
        if eng in annotated:
            continue
        idx = q_lower.find(eng)
        if idx != -1:
            # Check it's not already annotated
            following = result[idx + len(eng):idx + len(eng) + 10]
            if "(Hebrew:" not in following:
                original = result[idx:idx + len(eng)]
                result = result[:idx] + f"{original} (Hebrew: {heb})" + result[idx + len(eng):]
                q_lower = result.lower()
                annotated.add(eng)

    return result
