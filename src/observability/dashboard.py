from __future__ import annotations

from pathlib import Path
from typing import Any


def write_dashboard(run_dir: Path, payload: dict[str, Any]) -> Path:
    path = run_dir / "dashboard.html"
    html = f"""<!doctype html>
<html>
<head><meta charset="utf-8"><title>Blog Agent Dashboard</title></head>
<body style="font-family:Segoe UI,Arial,sans-serif;padding:24px;max-width:1000px;margin:auto;">
<h1>Blog Agent Run Dashboard</h1>
<p><strong>Run ID:</strong> {payload.get('run_id','')}</p>
<p><strong>Topic:</strong> {payload.get('topic','')}</p>
<p><strong>Output:</strong> {payload.get('output_path','')}</p>
<h2>Quality</h2>
<pre>{payload.get('quality_score', {})}</pre>
<h2>Retries</h2>
<pre>{payload.get('retry_records', [])}</pre>
<h2>Fallbacks</h2>
<pre>{payload.get('fallback_reasons', [])}</pre>
<h2>Metrics</h2>
<pre>{payload.get('metrics_summary', {})}</pre>
</body>
</html>"""
    path.write_text(html, encoding="utf-8")
    return path
