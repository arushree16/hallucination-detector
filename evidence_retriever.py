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
    Person 2's verdict - NOW DEPRECATED.
    
    P2 no longer makes keyword-based judgments. The NLI model (P3) 
    is the sole source of truth verification.
    
    Returns "UNCERTAIN" always - P2's role is purely evidence retrieval.
    """
    return "UNCERTAIN"


def check_evidence_for_myth_indicators(evidence_sentences: List[str]) -> bool:
    """
    Check if evidence contains strong myth/false claim indicators.
    
    If evidence mentions "myth", "debunked", "false claim", etc.,
    the claim is likely false regardless of NLI output.
    
    Returns True if evidence strongly suggests the claim is false.
    """
    myth_indicators = [
        "myth", "debunked", "false claim", "misconception",
        "common misconception", "widely debunked", "scientifically false",
        "no scientific evidence", "not true"
    ]
    
    for sent in evidence_sentences:
        sent_lower = sent.lower()
        for indicator in myth_indicators:
            if indicator in sent_lower:
                return True
    return False


# ══════════════════════════════════════════════════════════
#  PUBLIC API — called by pipeline.py
# ══════════════════════════════════════════════════════════

def fetch_evidence(claim: str) -> Tuple[List[str], str]:
    """
    Search Wikipedia for evidence sentences relevant to the claim.
    
    P2 NO LONGER MAKES VERDICTS - it only retrieves evidence.
    The NLI model (DeBERTa) is the sole fact checker.

    Parameters
    ----------
    claim : A single factual claim string.

    Returns
    -------
    (evidence_sentences, "UNCERTAIN")

    evidence_sentences : top relevant sentences from Wikipedia
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
            if float(score) > 0.55:  # Slightly lower to catch more evidence
                all_candidates.append((sentences[i], float(score)))

    if not all_candidates:
        return [], "UNCERTAIN"

    # Sort by similarity score, keep top 7 (more evidence for NLI)
    all_candidates.sort(key=lambda x: x[1], reverse=True)
    top_sentences = [s for s, _ in all_candidates[:7]]

    # P2 verdict is always UNCERTAIN - NLI makes all decisions
    return top_sentences, "UNCERTAIN"