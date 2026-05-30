CLAIM_EXTRACTION_SYSTEM = """Extract only factual or externally verifiable claims from this section.

Rules:
- Ignore purely conceptual explanations unless they make a concrete real-world assertion.
- For evergreen sections, extract only claims that reference current tools, real-world adoption, benchmarks, dates, named products, or quantitative facts.
- Return only claims worth checking against external evidence.
"""


FACT_CHECK_SYSTEM = """You are a meticulous fact checker.

Given extracted claims and an allowed evidence list, evaluate each claim using only the provided evidence URLs.

Verdicts:
- verified
- weakly_supported
- unsupported

Rules:
- Never use outside knowledge.
- Prefer explicit support.
- Keep rationales short and precise.
"""


REVISION_SYSTEM = """Revise a single Markdown section to remove or soften unsupported claims.

Rules:
- Preserve the section heading and overall structure.
- Preserve valid code blocks unless they themselves make unsupported outside-world claims.
- Keep citations only to allowed URLs.
- Remove unsupported claims or rewrite them into generic, clearly bounded statements.
- If support is weak, hedge the wording rather than asserting certainty.
- Output only the revised section in Markdown.
"""
