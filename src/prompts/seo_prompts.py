SEO_SYSTEM = """Produce SEO metadata for a technical blog post.

Return a SeoMetadata object with:
- meta_title
- meta_description
- slug
- keywords
- faq_block

Rules:
- Preserve technical accuracy.
- Avoid keyword stuffing.
- Keep the title concise and strong.
- Build an FAQ block that reflects the actual content, not generic filler.
"""
