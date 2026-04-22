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
from evidence_retriever   import fetch_evidence, judge_fact_p2, check_evidence_for_myth_indicators
from hallucination_detector import verify_claim


# ══════════════════════════════════════════════════════════
#  VERDICT RECONCILIATION
# ══════════════════════════════════════════════════════════

def reconcile(p2: str, p3_label: str, p3_conf: float, evidence: List[str]) -> Dict:
    """
    SIMPLIFIED: Trust NLI (P3) completely, with myth detection safeguard.
    
    P2 is now purely an evidence retriever - it makes no verdicts.
    The DeBERTa NLI model is the sole fact checker.
    
    SAFEGUARD: If evidence contains "myth", "debunked", etc., override NLI to REFUTED.
    This prevents false positives when NLI gets confused by quoted false statements.
    """
    # Check for myth indicators in evidence - strong signal claim is false
    if evidence and check_evidence_for_myth_indicators(evidence):
        return {
            "final": "REFUTED",
            "confidence": 0.85,
            "note": "✗ Evidence indicates this is a myth/false claim"
        }
    
    # Map NLI labels directly to final verdict
    final_map = {
        "Supported": ("SUPPORTED", "✓ NLI confirms claim is supported by evidence"),
        "Refuted": ("REFUTED", "✗ NLI confirms claim is refuted by evidence"),
        "Not Enough Info": ("NOT ENOUGH INFO", "⚠ No conclusive evidence found")
    }
    
    final_verdict, note = final_map.get(p3_label, ("NOT ENOUGH INFO", "Unknown NLI response"))
    
    return {
        "final": final_verdict,
        "confidence": p3_conf,
        "note": note
    }


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
        evidence, p2_verdict = fetch_evidence(claim)

        if evidence:
            print(f"  ✓ {len(evidence)} evidence sentences retrieved")
            for e in evidence:
                print(f"    • {e[:95]}{'…' if len(e) > 95 else ''}")
        else:
            print(f"  ⚠  No evidence found")

        # Step 3 — Person 3
        print(f"\n  STEP 3 — NLI Verification  [Person 3 · RoBERTa-NLI]")
        p3 = verify_claim(claim, evidence)
        p3_label = p3["label"]
        p3_conf  = p3["confidence"]
        print(f"  P3 label: {p3_label}  |  confidence: {p3_conf:.2%}")

        # Reconcile
        rec  = reconcile(p2_verdict, p3_label, p3_conf, evidence)
        icon = {"SUPPORTED": "✅", "REFUTED": "❌",
                "NOT ENOUGH INFO": "⚠️ ", "CONFLICT": "🔶"}.get(rec["final"], "❓")

        print(f"\n  {icon}  FINAL: {rec['final']}  ({rec['confidence']:.0%})")
        print(f"      {rec['note']}")

        all_results.append({
            "claim":          claim,
            "evidence":       evidence,
            "p2_verdict":     p2_verdict,
            "p3_label":       p3_label,
            "p3_confidence":  p3_conf,
            "final_verdict":  rec["final"],
            "confidence":     rec["confidence"],
            "note":           rec["note"],
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