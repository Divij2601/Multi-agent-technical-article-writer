from __future__ import annotations

from typing import Any, Optional

from src.graph.main_graph import run as run_pipeline


def run(topic: str, as_of: Optional[str] = None, audience_mode: str = "engineer", execution_mode: str = "balanced") -> dict[str, Any]:
    return run_pipeline(topic=topic, as_of=as_of, audience_mode=audience_mode, execution_mode=execution_mode)


if __name__ == "__main__":
    result = run("Inside a Transformer: From Attention to Output Tokens")
    print(result.get("output_path", ""))
