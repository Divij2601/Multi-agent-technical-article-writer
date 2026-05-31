"""FastAPI backend for the Blog Writing Agent.

A thin async-job layer over the existing LangGraph engine in `src/`. Because a
single generation can take a long time and free-tier providers throttle hard,
jobs run one-at-a-time in a background worker and progress is streamed to clients.
See CLAUDE.md ("Backend") for the full contract.
"""
