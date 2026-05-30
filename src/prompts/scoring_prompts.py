QUALITY_SCORING_SYSTEM = """Score the final technical blog draft.

Return a QualityScore object with 1-10 scores for:
- clarity
- hallucination_risk
- technical_depth
- seo_readiness
- redundancy
- overall

Use low hallucination_risk scores for riskier drafts and higher scores for well-grounded drafts.
Set needs_revision=true if the draft still feels risky, thin, repetitive, or weakly grounded.
"""
