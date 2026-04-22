"""
pipeline.py  —  Full Integration: Person 1 + Person 2 + Person 3
=================================================================

Person 1  →  claim_extractor.py       : Raw text → factual claims  (spaCy)
Person 2  →  evidence_retriever.py    : Claim → Wikipedia evidence  (SBERT)
Person 3  →  hallucination_detector.py: Claim + Evidence → NLI verdict (RoBERTa)

Both P2 and P3 produce a verdict. They are reconciled into a FINAL verdict.

Install:
    pip install spacy transformers torch wikipedia-api sentence-transformers nltk requests
    python -m spacy download en_core_web_sm

Run:
    python pipeline.py
"""

import json
from typing import List, Dict

from claim_extractor      import extract_claims
from evidence_retriever   import (fetch_evidence, judge_fact_p2, 
                                   check_evidence_for_myth_indicators, 
                                   check_entity_contradiction,
                                   check_high_confidence_evidence)
from hallucination_detector import verify_claim
from claim_decomposer     import decompose_claim, aggregate_atomic_results


# ══════════════════════════════════════════════════════════
#  VERDICT RECONCILIATION
# ══════════════════════════════════════════════════════════

def reconcile(claim: str, p2: str, p3_label: str, p3_conf: float, evidence: List[str], scores: List[float]) -> Dict:
    """
    FACT VERIFICATION with entity-level contradiction detection.
    
    Three-tier verdict system:
    - SUPPORTED: Evidence confirms claim
    - CONTRADICTED: Evidence directly contradicts claim (entity/number mismatch)
    - REFUTED: Myth/false claim detected
    - NOT ENOUGH INFO: Insufficient evidence
    """
    # PRIORITY 1: Entity-level contradiction check (NEW)
    # This catches Germany vs France, Amazon in Africa, etc.
    if evidence:
        is_contradiction, explanation = check_entity_contradiction(claim, evidence)
        if is_contradiction:
            return {
                "final": "CONTRADICTED",
                "confidence": 0.85,
                "note": f"❌ {explanation}"
            }
    
    # PRIORITY 2: Myth/false claim detection
    if evidence and check_evidence_for_myth_indicators(evidence):
        return {
            "final": "REFUTED",
            "confidence": 0.85,
            "note": "✗ Evidence indicates this is a myth/false claim"
        }
    
    # PRIORITY 3: High-confidence evidence override for NLI failures
    if p3_label == "Not Enough Info" and evidence and scores:
        if check_high_confidence_evidence(claim, evidence, scores):
            avg_conf = sum(scores[:3]) / min(3, len(scores)) if scores else 0.7
            return {
                "final": "SUPPORTED",
                "confidence": round(avg_conf, 4),
                "note": "✓ Strong semantic evidence match"
            }
    
    # PRIORITY 4: Trust NLI verdict with adjusted confidence
    # Fix: Confidence shouldn't be 100% unless perfect match
    if p3_label == "Supported":
        # Reduce confidence if entities don't align well
        adjusted_conf = min(p3_conf, 0.92)  # Cap at 92% to avoid false 100%
        return {
            "final": "SUPPORTED",
            "confidence": round(adjusted_conf, 4),
            "note": "✓ Evidence supports this claim"
        }
    
    if p3_label == "Refuted":
        return {
            "final": "REFUTED",
            "confidence": round(p3_conf, 4),
            "note": "✗ Evidence refutes this claim"
        }
    
    # Default: Not enough info
    return {
        "final": "NOT ENOUGH INFO",
        "confidence": round(p3_conf, 4) if p3_conf else 0.5,
        "note": "⚠ No conclusive evidence found"
    }


def verify_atomic_claims(claim: str, evidence: List[str], scores: List[float]) -> Dict:
    """
    Decompose claim into atomic sub-claims and verify each independently.
    
    Returns aggregated result with breakdown of correct/incorrect parts.
    """
    from claim_decomposer import decompose_claim, AtomicClaim
    
    # Decompose into atomic claims
    atoms = decompose_claim(claim)
    
    # If only one atom, verify directly
    if len(atoms) == 1:
        p3 = verify_claim(claim, evidence)
        rec = reconcile(claim, "UNCERTAIN", p3["label"], p3["confidence"], evidence, scores)
        return {
            "final_verdict": rec["final"],
            "confidence": rec["confidence"],
            "note": rec["note"],
            "atomic_breakdown": [],
            "used_atomic": False
        }
    
    # Verify each atom independently
    atomic_results = []
    
    for atom in atoms:
        # Get evidence for this specific atom
        atom_evidence, _, atom_scores = fetch_evidence(atom.text)
        
        # Verify atom
        p3 = verify_claim(atom.text, atom_evidence)
        rec = reconcile(atom.text, "UNCERTAIN", p3["label"], p3["confidence"], atom_evidence, atom_scores)
        
        atomic_results.append({
            "atom": atom.text,
            "type": atom.claim_type,
            "verdict": rec["final"],
            "confidence": rec["confidence"],
            "note": rec["note"]
        })
    
    # Aggregate results
    agg = aggregate_atomic_results(claim, atomic_results)
    agg["used_atomic"] = True
    agg["atomic_breakdown"] = atomic_results
    
    return agg


# ══════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════

def run_pipeline(text: str) -> List[Dict]:
    """
    Run the full fact-checking pipeline on a block of raw text.

    Returns a list of result dicts:
    {
        "claim"         : str,
        "evidence"      : List[str],
        "p2_verdict"    : "TRUE" | "FALSE" | "UNCERTAIN",
        "p3_label"      : "Supported" | "Refuted" | "Not Enough Info",
        "p3_confidence" : float,
        "final_verdict" : str,
        "confidence"    : float,
        "note"          : str,
    }
    """
    all_results = []

    # ── STEP 1  ───────────────────────────────────────────
    print("\n" + "═" * 65)
    print("  STEP 1 — Claim Extraction  [Person 1 · spaCy]")
    print("═" * 65)

    claims = extract_claims(text)
    if not claims:
        print("  ⚠  No factual claims detected.")
        return []

    print(f"  ✓ {len(claims)} claim(s) found:")
    for i, c in enumerate(claims, 1):
        print(f"    {i}. {c}")

    # ── STEP 2 + 3  (per claim) ───────────────────────────
    for idx, claim in enumerate(claims, 1):
        print(f"\n{'─' * 65}")
        print(f"  Claim {idx}/{len(claims)}: \"{claim}\"")

        # Step 2 — Person 2
        print(f"\n  STEP 2 — Evidence Retrieval  [Person 2 · Wikipedia + SBERT]")
        evidence, p2_verdict, scores = fetch_evidence(claim)

        if evidence:
            print(f"  ✓ {len(evidence)} evidence sentences retrieved")
            for e in evidence:
                print(f"    • {e[:95]}{'…' if len(e) > 95 else ''}")
        else:
            print(f"  ⚠  No evidence found")

        # Step 3 — Atomic Verification (NEW)
        print(f"\n  STEP 3 — Atomic Verification  [Decomposing claim]")
        atomic_result = verify_atomic_claims(claim, evidence, scores)
        
        # Show atomic breakdown if decomposition was used
        if atomic_result.get("used_atomic"):
            print(f"  Decomposed into {len(atomic_result['atomic_breakdown'])} sub-claims:")
            for atom in atomic_result["atomic_breakdown"]:
                atom_icon = {"SUPPORTED": "✓", "CONTRADICTED": "✗", 
                           "REFUTED": "✗", "NOT ENOUGH INFO": "?"}.get(atom["verdict"], "?")
                print(f"    {atom_icon} [{atom['type']}] {atom['atom'][:60]}... → {atom['verdict']}")
            
            if atomic_result.get("incorrect_percentage", 0) > 0:
                print(f"\n  ⚠  {atomic_result['incorrect_percentage']}% of facts are incorrect")
        
        final_verdict = atomic_result["final_verdict"]
        final_confidence = atomic_result["confidence"]
        final_note = atomic_result["note"]
        
        icon = {"SUPPORTED": "✅", "REFUTED": "❌", "CONTRADICTED": "❌",
                "NOT ENOUGH INFO": "⚠️ ", "PARTIALLY CONTRADICTED": "🔶"}.get(final_verdict, "❓")

        print(f"\n  {icon}  FINAL: {final_verdict}  ({final_confidence:.0%})")
        print(f"      {final_note}")

        all_results.append({
            "claim":          claim,
            "evidence":       evidence,
            "p2_verdict":     p2_verdict,
            "p3_label":       atomic_result.get("atomic_breakdown", [{}])[0].get("verdict", "N/A") if atomic_result.get("atomic_breakdown") else "N/A",
            "p3_confidence":  final_confidence,
            "final_verdict":  final_verdict,
            "confidence":     final_confidence,
            "note":           final_note,
            "atomic_breakdown": atomic_result.get("atomic_breakdown", []),
            "incorrect_parts": atomic_result.get("incorrect_parts", []),
            "correct_parts": atomic_result.get("correct_parts", []),
        })

    # ── SUMMARY  ──────────────────────────────────────────
    icons = {"SUPPORTED": "✅", "REFUTED": "❌",
             "NOT ENOUGH INFO": "⚠️ ", "CONFLICT": "🔶"}
    print("\n\n" + "═" * 65)
    print("  PIPELINE SUMMARY")
    print("═" * 65)
    for r in all_results:
        icon = icons.get(r["final_verdict"], "❓")
        print(f"  {icon}  {r['final_verdict']:<18} "
              f"({r['confidence']:.0%})  NLI: {r['p3_label']}")
        print(f"       → {r['claim'][:70]}")

    return all_results


# ══════════════════════════════════════════════════════════
#  TEST — Person 2's original test claims
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    
    print("\n" + "═" * 65)
    print("  HALLUCINATION DETECTOR — Interactive Mode")
    print("═" * 65)
    print("\nEnter text to fact-check (press Enter twice to submit):")
    print("-" * 65)
    
    lines = []
    while True:
        try:
            line = input()
            if line == "" and lines and lines[-1] == "":
                break
            lines.append(line)
        except EOFError:
            break
    
    user_text = "\n".join(lines).strip()
    
    if not user_text:
        print("\n⚠ No text provided. Using demo text instead.\n")
        user_text = """
        The population of India is 1.4 billion.
        India is the largest country in the world.
        Girls have XY chromosomes.
        The Earth revolves around the Sun.
        Mars has life.
        The Kohinoor diamond is worth 2 rupees.
        """
    
    results = run_pipeline(user_text)

    print("\n\n" + "═" * 65)
    print("  RAW JSON OUTPUT")
    print("═" * 65)
    clean = [{k: v for k, v in r.items() if k != "evidence"} for r in results]
    print(json.dumps(clean, indent=2))