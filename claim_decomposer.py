"""
claim_decomposer.py — Break compound claims into atomic facts
===========================================================

Splits complex claims like:
  "Gift to USA from Germany in 1941"
Into:
  ["Gift to USA", "from Germany", "in 1941"]

Each sub-claim is verified independently for precise fact-checking.
"""

import spacy
from typing import List, Dict, Tuple
from dataclasses import dataclass

_nlp = spacy.load("en_core_web_sm")


@dataclass
class AtomicClaim:
    text: str
    claim_type: str  # "entity", "location", "date", "action", "attribute"
    original_claim: str


def decompose_claim(claim: str) -> List[AtomicClaim]:
    """
    Break a compound claim into atomic sub-claims.
    
    Example:
        "The Statue of Liberty was a gift to USA from France in 1886"
        → [
            AtomicClaim("Statue of Liberty was a gift", "action", ...),
            AtomicClaim("gift to USA", "entity", ...),
            AtomicClaim("from France", "location", ...),
            AtomicClaim("in 1886", "date", ...)
        ]
    """
    doc = _nlp(claim)
    atoms = []
    
    # Extract dates (TEMPORAL entities)
    for ent in doc.ents:
        if ent.label_ in ["DATE", "TIME", "CARDINAL"] and any(char.isdigit() for char in ent.text):
            # Extract year/date phrase
            date_phrase = extract_date_phrase(doc, ent)
            if date_phrase:
                atoms.append(AtomicClaim(
                    text=date_phrase,
                    claim_type="date",
                    original_claim=claim
                ))
    
    # Extract locations (GPE = Geopolitical Entity, LOC = Location)
    for ent in doc.ents:
        if ent.label_ in ["GPE", "LOC"]:
            loc_phrase = extract_location_phrase(doc, ent)
            if loc_phrase and not any(a.text == loc_phrase for a in atoms):
                atoms.append(AtomicClaim(
                    text=loc_phrase,
                    claim_type="location",
                    original_claim=claim
                ))
    
    # Extract main entities and their actions
    main_subject = None
    main_action = None
    
    for token in doc:
        # Find subject (nsubj or nsubjpass)
        if token.dep_ in ["nsubj", "nsubjpass"] and token.head.pos_ == "VERB":
            main_subject = expand_noun_phrase(token)
            main_action = token.head.text
        # Find main verb/action
        if token.pos_ == "VERB" and not main_action:
            main_action = token.text
    
    # If no subject found, use first noun phrase
    if not main_subject:
        for chunk in doc.noun_chunks:
            main_subject = chunk.text
            break
    
    # Add main action claim
    if main_subject and main_action:
        action_claim = f"{main_subject} {main_action}"
        atoms.append(AtomicClaim(
            text=action_claim,
            claim_type="action",
            original_claim=claim
        ))
    
    # Extract secondary entities (objects, prepositional phrases)
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PERSON", "PRODUCT", "WORK_OF_ART"]:
            entity_phrase = extract_entity_phrase(doc, ent)
            if entity_phrase and not any(a.text == entity_phrase for a in atoms):
                atoms.append(AtomicClaim(
                    text=entity_phrase,
                    claim_type="entity",
                    original_claim=claim
                ))
    
    # If decomposition failed or too few atoms, return whole claim
    if len(atoms) < 2:
        return [AtomicClaim(
            text=claim,
            claim_type="full",
            original_claim=claim
        )]
    
    return atoms


def extract_date_phrase(doc, date_ent) -> str:
    """Extract the full date phrase including preposition."""
    start = date_ent.start
    
    # Look for preceding preposition ("in", "on", "during", "since")
    if start > 0:
        prev_token = doc[start - 1]
        if prev_token.text.lower() in ["in", "on", "during", "since", "at", "by"]:
            return f"{prev_token.text} {date_ent.text}"
    
    return date_ent.text


def extract_location_phrase(doc, loc_ent) -> str:
    """Extract location with its preposition."""
    start = loc_ent.start
    
    # Look for preceding preposition ("in", "from", "to", "at", "near")
    if start > 0:
        prev_token = doc[start - 1]
        if prev_token.text.lower() in ["in", "from", "to", "at", "near", "by", "of"]:
            return f"{prev_token.text} {loc_ent.text}"
    
    return f"in {loc_ent.text}"


def extract_entity_phrase(doc, ent) -> str:
    """Extract entity with relevant context."""
    start = ent.start
    
    # Look for preceding preposition
    if start > 0:
        prev_token = doc[start - 1]
        if prev_token.text.lower() in ["to", "from", "by", "of", "for"]:
            return f"{prev_token.text} {ent.text}"
    
    return ent.text


def expand_noun_phrase(token) -> str:
    """Expand a token to its full noun phrase."""
    # Start with the token itself
    words = [token.text]
    
    # Add left modifiers (compound, adjectives)
    for left in token.lefts:
        if left.dep_ in ["compound", "amod", "det", "nummod"]:
            words.insert(0, left.text)
    
    # Add right modifiers
    for right in token.rights:
        if right.dep_ in ["compound", "appos"]:
            words.append(right.text)
    
    return " ".join(words)


def aggregate_atomic_results(original_claim: str, atomic_results: List[Dict]) -> Dict:
    """
    Combine atomic claim results into a single verdict.
    
    Returns:
        {
            "final_verdict": "SUPPORTED" | "CONTRADICTED" | "PARTIALLY_CONTRADICTED" | ...,
            "confidence": float,
            "atomic_breakdown": [...],
            "incorrect_parts": [...],
            "correct_parts": [...]
        }
    """
    if not atomic_results:
        return {
            "final_verdict": "NOT ENOUGH INFO",
            "confidence": 0.5,
            "atomic_breakdown": []
        }
    
    # Count verdicts
    total = len(atomic_results)
    supported = sum(1 for r in atomic_results if r.get("verdict") == "SUPPORTED")
    contradicted = sum(1 for r in atomic_results if r.get("verdict") == "CONTRADICTED")
    refuted = sum(1 for r in atomic_results if r.get("verdict") == "REFUTED")
    
    incorrect_parts = [r for r in atomic_results if r.get("verdict") in ["CONTRADICTED", "REFUTED"]]
    correct_parts = [r for r in atomic_results if r.get("verdict") == "SUPPORTED"]
    
    # Determine aggregate verdict
    if contradicted > 0 or refuted > 0:
        if supported > 0:
            # Mixed results
            incorrect_pct = ((contradicted + refuted) / total) * 100
            return {
                "final_verdict": "PARTIALLY CONTRADICTED",
                "confidence": round(0.6 + (0.25 * (contradicted + refuted) / total), 4),
                "atomic_breakdown": atomic_results,
                "incorrect_parts": incorrect_parts,
                "correct_parts": correct_parts,
                "incorrect_percentage": round(incorrect_pct, 1),
                "note": f"{len(incorrect_parts)} of {total} facts incorrect ({round(incorrect_pct, 0)}%)"
            }
        else:
            # All wrong
            return {
                "final_verdict": "CONTRADICTED",
                "confidence": 0.85,
                "atomic_breakdown": atomic_results,
                "incorrect_parts": incorrect_parts,
                "correct_parts": [],
                "incorrect_percentage": 100,
                "note": "All sub-claims are contradicted by evidence"
            }
    
    if supported == total:
        # All correct
        avg_conf = sum(r.get("confidence", 0.8) for r in atomic_results) / total
        return {
            "final_verdict": "SUPPORTED",
            "confidence": round(min(avg_conf, 0.92), 4),
            "atomic_breakdown": atomic_results,
            "incorrect_parts": [],
            "correct_parts": correct_parts,
            "incorrect_percentage": 0,
            "note": "All sub-claims verified"
        }
    
    # Mixed uncertain
    return {
        "final_verdict": "NOT ENOUGH INFO",
        "confidence": 0.5,
        "atomic_breakdown": atomic_results,
        "incorrect_parts": incorrect_parts,
        "correct_parts": correct_parts,
        "incorrect_percentage": 0,
        "note": "Some claims could not be verified"
    }
