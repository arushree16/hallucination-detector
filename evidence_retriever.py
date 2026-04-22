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
import spacy
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
_nlp = spacy.load("en_core_web_sm")


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


def check_entity_contradiction(claim: str, evidence_sentences: List[str]) -> tuple[bool, str]:
    """
    Check for entity-level contradictions between claim and evidence.
    
    Returns (True, explanation) if entities mismatch (e.g., Germany vs France),
    (False, "") if no contradiction found.
    """
    import re
    from rapidfuzz import fuzz  # For fuzzy string matching
    
    claim_doc = _nlp(claim)
    claim_entities = {ent.text.lower(): ent.label_ for ent in claim_doc.ents}
    
    if not claim_entities:
        return False, ""
    
    evidence_text = " ".join(evidence_sentences)
    evidence_doc = _nlp(evidence_text)
    evidence_entities = {ent.text.lower(): ent.label_ for ent in evidence_doc.ents}
    
    # Check each entity in claim against evidence
    for claim_ent, claim_label in claim_entities.items():
        # Skip small entities
        if len(claim_ent) < 3:
            continue
            
        # Check if similar entity exists in evidence with different value
        for evid_ent, evid_label in evidence_entities.items():
            # Same entity type but different value = contradiction
            if (claim_label == evid_label and 
                claim_label in ["GPE", "PERSON", "ORG", "NORP"] and
                fuzz.ratio(claim_ent, evid_ent) < 70 and  # Not the same entity
                len(evid_ent) > 3):
                
                # Check if they appear in similar context (same sentence)
                for sent in evidence_sentences:
                    sent_lower = sent.lower()
                    if claim_ent in sent_lower or evid_ent in sent_lower:
                        return True, f"Entity mismatch: claim says '{claim_ent}', evidence says '{evid_ent}'"
    
    # Check for country/location contradictions specifically
    claim_lower = claim.lower()
    for place, correct_regions in {
        "amazon": ["south america", "brazil", "peru"],
        "nile": ["africa", "egypt", "sudan"],
        "sahara": ["africa"],
        "everest": ["nepal", "tibet"],
    }.items():
        if place in claim_lower:
            evidence_lower = evidence_text.lower()
            has_correct = any(region in evidence_lower for region in correct_regions)
            
            # Common wrong locations
            wrong_map = {
                "amazon": ["africa"],
                "nile": ["south america", "asia"],
                "sahara": ["asia", "south america"],
                "everest": ["india", "china"],  # China is partial, India is wrong
            }
            
            has_wrong = any(wrong in claim_lower for wrong in wrong_map.get(place, []))
            
            if has_correct and has_wrong:
                return True, f"Geography contradiction: {place} is not in the claimed location"
    
    return False, ""


def check_geography_contradiction(claim: str, evidence_sentences: List[str]) -> bool:
    """
    DEPRECATED: Use check_entity_contradiction instead.
    Kept for backward compatibility.
    """
    result, _ = check_entity_contradiction(claim, evidence_sentences)
    return result


# ══════════════════════════════════════════════════════════
#  PUBLIC API — called by pipeline.py
# ══════════════════════════════════════════════════════════

def check_high_confidence_evidence(claim: str, evidence_sentences: List[str], scores: List[float]) -> bool:
    """
    Check if we have very high-quality evidence that NLI might miss.
    
    Returns True if evidence semantically matches claim very strongly.
    This catches cases where NLI returns "Not Enough Info" despite clear evidence.
    """
    if not evidence_sentences or not scores:
        return False
    
    claim_lower = claim.lower()
    
    # Check for exact or near-exact matches in top evidence
    for i, sent in enumerate(evidence_sentences[:3]):  # Check top 3
        sent_lower = sent.lower()
        score = scores[i] if i < len(scores) else 0
        
        # High similarity score threshold
        if score < 0.65:
            continue
        
        # Check for key entity matches
        # Extract key terms from claim (nouns, proper nouns, numbers)
        claim_doc = _nlp(claim)
        key_terms = []
        for token in claim_doc:
            if token.pos_ in ["PROPN", "NOUN", "NUM"] and len(token.text) > 2:
                key_terms.append(token.lemma_.lower())
        
        # If most key terms appear in evidence with high similarity, it's strong match
        matches = sum(1 for term in key_terms if term in sent_lower)
        if len(key_terms) > 0 and matches / len(key_terms) >= 0.6:
            return True
        
        # Exact date/year matches are very strong signals
        import re
        claim_years = re.findall(r'\b(19|20)\d{2}\b', claim)
        sent_years = re.findall(r'\b(19|20)\d{2}\b', sent)
        if claim_years and sent_years and claim_years[0] == sent_years[0] and score > 0.6:
            return True
    
    return False


def fetch_evidence(claim: str) -> Tuple[List[str], str, List[float]]:
    """
    Search Wikipedia for evidence sentences relevant to the claim.
    
    P2 NO LONGER MAKES VERDICTS - it only retrieves evidence.
    The NLI model (DeBERTa) is the sole fact checker.

    Parameters
    ----------
    claim : A single factual claim string.

    Returns
    -------
    (evidence_sentences, "UNCERTAIN", similarity_scores)

    evidence_sentences : top relevant sentences from Wikipedia
    similarity_scores : SBERT similarity scores for each sentence
    """
    query  = _improve_query(claim)
    titles = _search_wikipedia_titles(query)

    if not titles:
        return [], "UNCERTAIN", []

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
        return [], "UNCERTAIN", []

    # Sort by similarity score, keep top 7 (more evidence for NLI)
    all_candidates.sort(key=lambda x: x[1], reverse=True)
    top_sentences = [s for s, _ in all_candidates[:7]]
    top_scores = [score for _, score in all_candidates[:7]]

    # P2 verdict is always UNCERTAIN - NLI makes all decisions
    return top_sentences, "UNCERTAIN", top_scores