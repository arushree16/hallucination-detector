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
from evidence_retriever   import fetch_evidence, judge_fact_p2
from hallucination_detector import verify_claim


# ══════════════════════════════════════════════════════════
#  VERDICT RECONCILIATION
# ══════════════════════════════════════════════════════════

def reconcile(p2: str, p3_label: str, p3_conf: float) -> Dict:
    """
    Combine Person 2's keyword verdict with Person 3's NLI verdict.

    ┌──────────────┬───────────────────┬──────────────────────────────────┐
    │ P2           │ P3                │ Final Decision                   │
    ├──────────────┼───────────────────┼──────────────────────────────────┤
    │ TRUE         │ Supported         │ ✅ SUPPORTED   — both agree      │
    │ FALSE        │ Refuted           │ ❌ REFUTED     — both agree      │
    │ TRUE         │ Refuted           │ 🔶 CONFLICT    — disagree        │
    │ FALSE        │ Supported         │ 🔶 CONFLICT    — disagree        │
    │ UNCERTAIN    │ any               │ defer to P3                      │
    │ any          │ Not Enough Info   │ ⚠  NOT ENOUGH INFO               │
    └──────────────┴───────────────────┴──────────────────────────────────┘

    When both agree the confidence is boosted (+0.05, capped at 1.0).
    When they conflict, confidence is penalised (−0.10).
    """
    p3_norm = {"Supported": "TRUE", "Refuted": "FALSE",
               "Not Enough Info": "UNKNOWN"}.get(p3_label, "UNKNOWN")

    if p3_norm == "UNKNOWN":
        # P3 couldn't decide — if P2 is confident, use P2's verdict
        if p2 in ("TRUE", "FALSE"):
            label_map = {"TRUE": "SUPPORTED", "FALSE": "REFUTED"}
            return {"final": label_map[p2],
                    "confidence": round(p3_conf * 0.75, 4),   # discounted — only one signal
                    "note": f"NLI inconclusive — using P2 keyword verdict ({p2})"}
        return {"final": "NOT ENOUGH INFO",
                "confidence": p3_conf,
                "note": "Both P2 and P3 found insufficient evidence"}

    if p2 == "UNCERTAIN":
        return {"final": p3_label.upper(),
                "confidence": p3_conf,
                "note": "P2 uncertain — deferring to NLI (P3)"}

    if p2 == p3_norm:
        return {"final": p3_label.upper(),
                "confidence": min(1.0, p3_conf + 0.05),
                "note": "✓ Both P2 (keyword) and P3 (NLI) agree"}

    # Conflict
    return {"final": "CONFLICT",
            "confidence": max(0.0, p3_conf - 0.10),
            "note": (f"P2 says {p2}, P3 says {p3_label} "
                     f"({p3_conf:.0%}) — manual review advised")}


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
            print(f"  ✓ {len(evidence)} evidence sentence(s) | P2 verdict: {p2_verdict}")
            for e in evidence:
                print(f"    • {e[:95]}{'…' if len(e) > 95 else ''}")
        else:
            print(f"  ⚠  No evidence found | P2 verdict: {p2_verdict}")

        # Step 3 — Person 3
        print(f"\n  STEP 3 — NLI Verification  [Person 3 · RoBERTa-NLI]")
        p3 = verify_claim(claim, evidence)
        p3_label = p3["label"]
        p3_conf  = p3["confidence"]
        print(f"  P3 label: {p3_label}  |  confidence: {p3_conf:.2%}")

        # Reconcile
        rec  = reconcile(p2_verdict, p3_label, p3_conf)
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
              f"({r['confidence']:.0%})  "
              f"P2={r['p2_verdict']:<10} P3={r['p3_label']}")
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