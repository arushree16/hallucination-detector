"""
evidence_retriever.py  —  Person 2's module (REAL implementation)
==================================================================
Uses Wikipedia search + Sentence-BERT semantic similarity to find
relevant evidence sentences for a given claim.

Also runs Person 2's own keyword-based judge as a first-pass verdict.

Public API (what pipeline.py calls):
    evidence_sentences, p2_verdict = fetch_evidence(claim)

    evidence_sentences : List[str]   – top relevant sentences from Wikipedia
    p2_verdict         : str         – "TRUE" | "FALSE" | "UNCERTAIN"

Install:
    pip install wikipedia-api sentence-transformers nltk requests
"""

import re
import requests
import nltk
import wikipediaapi
from sentence_transformers import SentenceTransformer, util
from typing import List, Tuple

# ── Downloads (safe to call repeatedly) ───────────────────
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

# ── Module-level singletons (load once) ───────────────────
_wiki = wikipediaapi.Wikipedia(
    user_agent="fact-checker/2.0",
    language="en",
    extract_format=wikipediaapi.ExtractFormat.WIKI,
)
_sbert = SentenceTransformer("BAAI/bge-base-en-v1.5")


# ══════════════════════════════════════════════════════════
#  PERSON 2 — INTERNAL HELPERS  (unchanged logic)
# ══════════════════════════════════════════════════════════

def _improve_query(fact: str) -> str:
    """Map common fact phrasings to better Wikipedia search queries."""
    import re
    f = fact.lower()
    if "earth revolves around the sun" in f:
        return "heliocentrism earth orbit sun"
    if "sun revolves around earth" in f:
        return "heliocentrism geocentrism"
    if "xy chromosomes" in f or "xx chromosomes" in f:
        return "human sex chromosomes male female xx xy"
    if "largest country" in f:
        return "largest country by area world"
    # Capital city: search specifically for "capital of COUNTRY"
    cap = re.search(r"capital of (\w+)", f)
    if cap:
        country = cap.group(1)
        return f"capital city of {country}"
    # Speed of light
    if "speed of light" in f:
        return "speed of light physics"
    # Brain myths
    if "brain" in f and ("10 percent" in f or "10%" in f):
        return "ten percent of the brain myth"
    # Human body facts
    if "human body" in f and ("bone" in f or "bones" in f):
        return "human skeleton bones"
    return fact


def _search_wikipedia_titles(query: str) -> List[str]:
    """Return top-3 Wikipedia page titles matching query."""
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list":   "search",
        "srsearch": query,
        "format": "json",
    }
    try:
        resp = requests.get(url, params=params,
                            headers={"User-Agent": "fact-checker/2.0"},
                            timeout=5)
        results = resp.json().get("query", {}).get("search", [])
        return [r["title"] for r in results[:3]]
    except Exception:
        return []


def _clean_sentences(text: str) -> List[str]:
    """Tokenise text into sentences, dropping boilerplate / very short ones."""
    sentences = nltk.sent_tokenize(text)
    clean = []
    for s in sentences:
        s_lower = s.lower()
        if len(s) < 40:
            continue
        if any(x in s_lower for x in ["see also", "references", "external links"]):
            continue
        clean.append(s)
    return clean


def judge_fact_p2(fact: str, evidence: str) -> str:
    """
    Person 2's keyword + pattern-based fact judge.
    Returns "TRUE" | "FALSE" | "UNCERTAIN"
    """
    f_lower = fact.lower()
    e_lower = evidence.lower()

    # Negation detection
    neg_words = ["no", "not", "never", "none", "no evidence",
                 "lack of", "without", "doesn't", "lacks"]
    if any(w in e_lower for w in neg_words):
        return "FALSE"

    # Population numbers — extract and compare
    fact_pop = re.search(r"(\d+\.?\d*)\s*(billion|crore)", f_lower)
    evid_pop = re.search(r"(\d+\.?\d*)\s*(billion|crore)", e_lower)
    if fact_pop and evid_pop:
        f_num = float(fact_pop.group(1))
        e_num = float(evid_pop.group(1))
        return "TRUE" if abs(f_num - e_num) <= 0.1 else "FALSE"

    # Largest country
    if "largest" in f_lower and "country" in f_lower:
        if any(w in e_lower for w in ["largest", "1st", "first", "#1"]):
            return "TRUE"
        if any(w in e_lower for w in ["second", "third", "2nd", "3rd", "smaller"]):
            return "FALSE"
    
    # Capital city detection — strict matching
    if "capital of" in f_lower:
        import re
        cap_match = re.search(r"capital of (\w+) is (\w+)", f_lower)
        if cap_match:
            country_in_fact = cap_match.group(1).lower()
            city_in_fact = cap_match.group(2).lower()
            
            # Must find EXACT match: "capital of {country}" near the city name
            # Check sentence for the pattern: "capital of COUNTRY" AND the city
            if f"capital of {country_in_fact}" in e_lower:
                # Found correct country - now check if city matches
                if city_in_fact in e_lower:
                    return "TRUE"
                # Evidence mentions country but different city as capital
                other_city = re.search(rf"capital of {re.escape(country_in_fact)} is (\w+)", e_lower)
                if other_city and other_city.group(1).lower() != city_in_fact:
                    return "FALSE"
            
            # Check if evidence says this city is capital of a DIFFERENT country
            if city_in_fact in e_lower:
                wrong_country = re.search(rf"capital of (\w+)[^.]*{re.escape(city_in_fact)}|{re.escape(city_in_fact)}[^.]*capital of (\w+)", e_lower)
                if wrong_country:
                    matched = (wrong_country.group(1) or wrong_country.group(2) or "").lower()
                    if matched and matched != country_in_fact:
                        return "FALSE"
                        
    # Geography detection — location claims
    geography_map = {
        "amazon rainforest": "south america",
        "amazon river": "south america",
        "mount everest": "nepal|tibet|himalaya",
        "sahara desert": "africa",
        "nile river": "africa",
    }
    for place, correct_region in geography_map.items():
        if place in f_lower:
            if re.search(correct_region, e_lower):
                return "TRUE"
            # Check for wrong region mentions
            wrong_regions = ["africa", "europe", "asia", "north america", "south america", "australia", "antarctica"]
            for region in wrong_regions:
                if region != correct_region and region in e_lower:
                    return "FALSE"
                        
    # Myth/false claim detection
    myth_indicators = ["myth", "false", "misconception", "debunked", "not true", "incorrect"]
    if any(w in e_lower for w in myth_indicators):
        return "FALSE"

    # Earth revolves / orbits
    if "earth revolves around the sun" in f_lower or "earth orbits the sun" in f_lower:
        orbit_words = ["orbits the sun", "revolves around sun", "revolves around the sun",
                       "heliocentric", "heliocentrism", "orbital motion"]
        return "TRUE" if any(w in e_lower for w in orbit_words) else "FALSE"

    # Sex chromosomes
    if "girls" in f_lower or "female" in f_lower:
        if "xx" in e_lower and "xy" not in e_lower:
            return "FALSE"
        if "xy" in e_lower:
            return "TRUE"

    # Life existence
    if "has life" in f_lower or "life" in f_lower:
        life_words     = ["life exists", "life found", "inhabited", "biosphere"]
        neg_life_words = ["no life", "no evidence", "not found", "unproven"]
        if any(w in e_lower for w in life_words):
            return "TRUE"
        if any(w in e_lower for w in neg_life_words):
            return "FALSE"

    # Kohinoor value
    if "kohinoor" in f_lower and "worth" in f_lower:
        low_value  = ["rupees", "2 rupees", "cheap", "low value"]
        high_value = ["billion", "million", "priceless", "expensive"]
        if any(w in e_lower for w in low_value):
            return "TRUE"
        if any(w in e_lower for w in high_value):
            return "FALSE"

    # Default: genuinely uncertain — do not guess TRUE just because "is" appears
    return "UNCERTAIN"


# ══════════════════════════════════════════════════════════
#  PUBLIC API — called by pipeline.py
# ══════════════════════════════════════════════════════════

def fetch_evidence(claim: str) -> Tuple[List[str], str]:
    """
    Search Wikipedia for evidence sentences relevant to the claim.

    Parameters
    ----------
    claim : A single factual claim string.

    Returns
    -------
    (evidence_sentences, p2_verdict)

    evidence_sentences : top-5 relevant sentences (may be empty)
    p2_verdict         : "TRUE" | "FALSE" | "UNCERTAIN"
                         based on Person 2's keyword judge over
                         the single best-scoring sentence.
    """
    query  = _improve_query(claim)
    titles = _search_wikipedia_titles(query)

    if not titles:
        return [], "UNCERTAIN"

    # Collect all candidate sentences from top Wikipedia pages
    all_candidates: List[Tuple[str, float]] = []

    for title in titles:
        page = _wiki.page(title)
        if not page.exists():
            continue

        sentences = _clean_sentences(page.text[:5000])
        if not sentences:
            continue

        claim_emb = _sbert.encode(claim,     convert_to_tensor=True)
        sent_emb  = _sbert.encode(sentences, convert_to_tensor=True)
        scores    = util.cos_sim(claim_emb, sent_emb)[0]

        for i, score in enumerate(scores):
            if float(score) > 0.60:   # lowered from 0.72 — catches paraphrased facts
                all_candidates.append((sentences[i], float(score)))

    if not all_candidates:
        return [], "UNCERTAIN"

    # Sort by similarity score, keep top 5
    all_candidates.sort(key=lambda x: x[1], reverse=True)
    top_sentences = [s for s, _ in all_candidates[:5]]

    # Person 2's verdict: majority vote over top sentences
    votes = {"TRUE": 0, "FALSE": 0}
    for sent, _ in all_candidates[:5]:
        verdict = judge_fact_p2(claim, sent)
        if verdict in votes:
            votes[verdict] += 1

    total = votes["TRUE"] + votes["FALSE"]
    if total == 0:
        p2_verdict = "UNCERTAIN"
    else:
        p2_verdict = "TRUE" if votes["TRUE"] >= votes["FALSE"] else "FALSE"

    return top_sentences, p2_verdict