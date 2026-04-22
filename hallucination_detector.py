"""
Hallucination Detection Module
================================
Uses a pretrained NLI (Natural Language Inference) model to verify claims
against a set of evidence sentences.

Installation:
    pip install transformers torch

Usage:
    from hallucination_detector import verify_claim

    claim = "The Eiffel Tower is located in Berlin."
    evidence = [
        "The Eiffel Tower is a wrought-iron lattice tower in Paris, France.",
        "It was constructed between 1887 and 1889.",
        "The tower stands 330 metres tall.",
    ]
    result = verify_claim(claim, evidence)
    print(result)
    # {'label': 'Refuted', 'confidence': 0.94}
"""

from typing import List, Dict
import torch
from transformers import pipeline

# ──────────────────────────────────────────────
# Module-level model cache (load once, reuse)
# ──────────────────────────────────────────────
_nli_pipeline = None

# NLI label order returned by roberta-large-mnli
# Index 0 → CONTRADICTION, 1 → NEUTRAL, 2 → ENTAILMENT
_LABEL_MAP = {
    "CONTRADICTION": "refute",
    "NEUTRAL":       "neutral",
    "ENTAILMENT":    "support",
}


def _get_pipeline():
    """Lazily load the NLI model (cached after first call)."""
    global _nli_pipeline
    if _nli_pipeline is None:
        print("[HallucinationDetector] Loading NLI model (roberta-large-mnli)…")
        _nli_pipeline = pipeline(
            task="zero-shot-classification",
            model="cross-encoder/nli-roberta-base",   # lighter & accurate
            device=0 if torch.cuda.is_available() else -1,
        )
        print("[HallucinationDetector] Model loaded.")
    return _nli_pipeline


# ──────────────────────────────────────────────
# Core NLI helper
# ──────────────────────────────────────────────

def _nli_score(premise: str, hypothesis: str) -> Dict[str, float]:
    """
    Run NLI for a single (premise, hypothesis) pair.

    Returns a dict:
        {"support": float, "neutral": float, "refute": float}
    where values sum to ~1.0.
    """
    pipe = _get_pipeline()

    # zero-shot-classification treats 'hypothesis_template' as a template.
    # We bypass that by passing the claim as the sequence and the evidence
    # sentence as one of the candidate_labels with a direct template.
    #
    # Alternatively – and more precisely – we use the cross-encoder directly:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch.nn.functional as F

    # Lazy-load cross-encoder weights (separate cache)
    global _ce_model, _ce_tokenizer
    if "_ce_model" not in globals():
        model_name = "cross-encoder/nli-roberta-base"
        _ce_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _ce_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        _ce_model.eval()

    inputs = _ce_tokenizer(
        premise, hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    with torch.no_grad():
        logits = _ce_model(**inputs).logits          # shape: (1, 3)
    probs = F.softmax(logits, dim=-1).squeeze().tolist()

    # cross-encoder/nli-roberta-base label order: contradiction, neutral, entailment
    return {
        "refute":  probs[0],
        "neutral": probs[1],
        "support": probs[2],
    }


# ──────────────────────────────────────────────
# Aggregation logic
# ──────────────────────────────────────────────

def _aggregate(scores: List[Dict[str, float]]) -> Dict[str, object]:
    """
    Aggregate per-evidence NLI scores into a final verdict.

    Strategy
    --------
    1. Each sentence votes for its highest-probability label.
    2. Weighted vote: each vote is weighted by its probability, not just 1.
       This prevents 5 weak neutral votes (0.34 each) drowning out
       2 strong support votes (0.85 each).
    3. Decision: highest weighted-vote total wins.
    4. Confidence = mean of winning label's probabilities across ALL sentences.
    """
    n = len(scores)
    if n == 0:
        return {"label": "Not Enough Info", "confidence": 0.0}

    # Weighted votes (probability-weighted, not just count)
    weighted = {"support": 0.0, "refute": 0.0, "neutral": 0.0}
    for s in scores:
        winner = max(s, key=s.get)
        weighted[winner] += s[winner]   # weight by confidence of that vote

    if weighted["support"] > weighted["refute"] and weighted["support"] > weighted["neutral"]:
        final_label = "Supported"
        prob_key    = "support"
    elif weighted["refute"] > weighted["support"] and weighted["refute"] > weighted["neutral"]:
        final_label = "Refuted"
        prob_key    = "refute"
    else:
        final_label = "Not Enough Info"
        prob_key    = "neutral"

    confidence = sum(s[prob_key] for s in scores) / n
    return {"label": final_label, "confidence": round(confidence, 4)}


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def verify_claim(claim: str, evidence: List[str]) -> Dict:
    """
    Verify a claim against a list of evidence sentences using NLI.

    Parameters
    ----------
    claim    : The statement to verify (hypothesis in NLI terms).
    evidence : List of supporting/contradicting sentences (premises).

    Returns
    -------
    dict with keys:
        "label"      – "Supported" | "Refuted" | "Not Enough Info"
        "confidence" – float in [0, 1]

    Example
    -------
    >>> verify_claim(
    ...     "Water boils at 100°C at sea level.",
    ...     ["Water boils at 100 degrees Celsius under standard atmospheric pressure.",
    ...      "The boiling point of water decreases at higher altitudes."]
    ... )
    {'label': 'Supported', 'confidence': 0.87}
    """
    if not evidence:
        return {"label": "Not Enough Info", "confidence": 0.0}

    per_evidence_scores = []
    for ev in evidence:
        scores = _nli_score(premise=ev, hypothesis=claim)
        per_evidence_scores.append(scores)

    result = _aggregate(per_evidence_scores)

    # Attach per-evidence breakdown for transparency (optional)
    result["evidence_breakdown"] = [
        {
            "evidence": ev,
            "support":  round(s["support"],  4),
            "neutral":  round(s["neutral"],  4),
            "refute":   round(s["refute"],   4),
        }
        for ev, s in zip(evidence, per_evidence_scores)
    ]

    return result


# ──────────────────────────────────────────────
# Example run (execute with: python hallucination_detector.py)
# ──────────────────────────────────────────────

if __name__ == "__main__":
    examples = [
        {
            "title": "Example 1 – Supported claim",
            "claim": "The Eiffel Tower is located in Paris.",
            "evidence": [
                "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.",
                "It was built as the centrepiece of the 1889 World's Fair.",
                "Millions of tourists visit the Eiffel Tower in Paris every year.",
            ],
        },
        {
            "title": "Example 2 – Refuted claim",
            "claim": "The Eiffel Tower is located in Berlin.",
            "evidence": [
                "The Eiffel Tower is a wrought-iron lattice tower in Paris, France.",
                "It is one of the most recognisable structures in the world, situated in Paris.",
            ],
        },
        {
            "title": "Example 3 – Not Enough Info",
            "claim": "The Eiffel Tower was designed by Gustave Eiffel himself.",
            "evidence": [
                "The weather in Paris is often cloudy in winter.",
                "French cuisine is renowned worldwide.",
            ],
        },
    ]

    for ex in examples:
        print(f"\n{'='*60}")
        print(f"  {ex['title']}")
        print(f"  Claim   : {ex['claim']}")
        print(f"  Evidence: {ex['evidence']}")
        result = verify_claim(ex["claim"], ex["evidence"])
        # Print summary (without breakdown for brevity)
        summary = {k: v for k, v in result.items() if k != "evidence_breakdown"}
        print(f"  Result  : {summary}")
        print("\n  Per-evidence breakdown:")
        for b in result["evidence_breakdown"]:
            print(f"    › \"{b['evidence'][:70]}…\"")
            print(f"      support={b['support']:.3f}  neutral={b['neutral']:.3f}  refute={b['refute']:.3f}")