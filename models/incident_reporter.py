"""
models/incident_reporter.py
Plain-English incident reports from SHAP output.

Backends (settings.llm_backend):
  auto     - Ollama if running, else NVIDIA NIM if a key is set, else template
  ollama   - local, free, no key
  nim      - NVIDIA NIM (optional)
  template - built-in, no LLM (always works)
Any backend failure falls through to the next one, ending at the template.

Grounding: the prompt carries real feature values, and any figure in the LLM's
text that does not appear in the alert data is listed in a note on the report.
"""

from __future__ import annotations

import re
import threading
import time

from config.logger import get_logger
from config.settings import settings

log = get_logger(__name__)

_ACTIONS = {
    "DoS": "Rate-limit or block the source at the firewall and check the target's load.",
    "Probe": "Block or monitor the scanning source and review which hosts and ports it touched.",
    "R2L": "Check authentication logs on the target and reset any exposed credentials.",
    "U2R": "Isolate the target host and review it for privilege escalation.",
    "Benign": "No action needed unless the anomaly score stays high for this host.",
}

_IP_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
_NUM_RE = re.compile(r"\d[\d,]*\.?\d*")


def _protocol(p) -> str:
    return {6: "TCP", 17: "UDP", 1: "ICMP"}.get(p, str(p))


def _feature_lines(
    top_features: dict, label: str = "", values: dict | None = None
) -> str:
    values = values or {}
    items = sorted(top_features.items(), key=lambda kv: abs(kv[1]), reverse=True)
    target = f"the {label} classification" if label else "the predicted class"
    lines = []
    for k, v in items:
        val = f"value={values[k]:g}, " if k in values else ""
        lines.append(
            f"  - {k}: {val}{'pushed toward' if v > 0 else 'pushed away from'} "
            f"{target} (SHAP={v:+.3f})"
        )
    return "\n".join(lines) or "  (no explanation available)"


def build_prompt(
    flow: dict,
    label: str,
    confidence: float,
    top_features: dict,
    values: dict | None = None,
) -> str:
    return f"""You are a senior SOC analyst. A network intrusion detection system analysed this flow.
Write a concise incident report using ONLY the data below.
Do not invent details. Do not state any number, unit or time that is not written below.

=== ALERT ===
Classification: {label} (confidence {confidence:.0%})
Source:      {flow.get('src_ip', '?')}:{flow.get('src_port', 0)}
Destination: {flow.get('dst_ip', '?')}:{flow.get('dst_port', 0)}
Protocol:    {_protocol(flow.get('protocol'))}
Duration:    {flow.get('duration', 0):.3f} s
Packets:     {flow.get('fwd_packet_count', 0)} forward, {flow.get('bwd_packet_count', 0)} backward
Bytes:       {flow.get('total_fwd_bytes', 0)} sent, {flow.get('total_bwd_bytes', 0)} received
SYN flags:   {flow.get('syn_flag_count', 0)}
Anomaly detector flagged: {bool(flow.get('is_anomaly', False))}

=== TOP FEATURES (actual value, and SHAP: what drove the classification) ===
{_feature_lines(top_features, label, values)}

If the classification is Benign, say no attack was identified; do not describe it as
suspicious unless the anomaly detector flagged it.
Write 3 short paragraphs: 1) executive summary (one sentence, include severity),
2) technical analysis referencing the features above (2-3 sentences),
3) recommended action (1-2 sentences). Under 150 words total."""


def _allowed_numbers(
    flow: dict, confidence: float, top_features: dict, values: dict | None
) -> list[float]:
    nums = [
        float(x)
        for x in (flow or {}).values()
        if isinstance(x, (int, float)) and not isinstance(x, bool)
    ]
    nums += [float(x) for x in (values or {}).values()]
    nums += [float(x) for x in top_features.values()]
    nums += [confidence * 100, confidence]
    return nums


def ungrounded_numbers(text: str, allowed: list[float]) -> list[str]:
    """Numbers >= 10 in the text that match nothing in the alert data."""
    bad = []
    for raw in _NUM_RE.findall(_IP_RE.sub(" ", text)):
        try:
            n = float(raw.replace(",", "").rstrip("."))
        except ValueError:
            continue
        if abs(n) < 10:
            continue
        if not any(abs(n - a) <= max(0.51, 0.01 * abs(a)) for a in allowed):
            bad.append(raw.rstrip("."))
    return sorted(set(bad))


def template_report(
    flow: dict,
    label: str,
    confidence: float,
    top_features: dict,
    values: dict | None = None,
) -> str:
    values = values or {}
    top = sorted(top_features.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
    feats = ", ".join(
        f"{k}{f' = {values[k]:g}' if k in values else ''} "
        f"({'supports' if v > 0 else 'opposes'} the {label} classification)"
        for k, v in top
    )
    src = f"{flow.get('src_ip', '?')}:{flow.get('src_port', 0)}"
    dst = f"{flow.get('dst_ip', '?')}:{flow.get('dst_port', 0)}"
    return (
        f"Summary: {label} activity from {src} to {dst}, classified with "
        f"{confidence:.0%} confidence.\n\n"
        f"Analysis: the flow lasted {flow.get('duration', 0):.3f}s with "
        f"{flow.get('fwd_packet_count', 0)} forward and {flow.get('bwd_packet_count', 0)} "
        f"backward packets. "
        + (
            f"The features that drove this decision were {feats}."
            if feats
            else "No feature explanation was available."
        )
        + f"\n\nRecommended action: {_ACTIONS.get(label, _ACTIONS['Benign'])}"
        "\n\n(Automatic template report; no LLM was available.)"
    )


def _default_post(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    import httpx

    r = httpx.post(url, json=payload, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _default_get(url: str, timeout: float) -> bool:
    import httpx

    httpx.get(url, timeout=timeout).raise_for_status()
    return True


class IncidentReporter:
    def __init__(
        self,
        backend="auto",
        ollama_url="http://localhost:11434",
        ollama_model="llama3.1:8b",
        nim_key="",
        nim_model="meta/llama-3.1-8b-instruct",
        nim_url="https://integrate.api.nvidia.com/v1",
        timeout=120,
        http_get=None,
        http_post=None,
    ) -> None:
        self.backend = backend
        self._ollama_url = ollama_url.rstrip("/")
        self._ollama_model = ollama_model
        self._nim_key = nim_key
        self._nim_model = nim_model
        self._nim_url = nim_url.rstrip("/")
        self._timeout = timeout
        self._get = http_get or _default_get
        self._post = http_post or _default_post
        self._up: tuple[float, bool] = (0.0, False)
        self._lock = threading.Lock()

    @classmethod
    def from_settings(cls) -> "IncidentReporter":
        s = settings
        return cls(
            s.llm_backend,
            s.ollama_url,
            s.ollama_model,
            s.nvidia_nim_api_key,
            s.nvidia_nim_model,
            s.nvidia_nim_url,
            s.llm_timeout_seconds,
        )

    def _ollama_up(self) -> bool:
        with self._lock:
            ts, up = self._up
            if time.time() - ts < 15:
                return up
        try:
            up = bool(self._get(f"{self._ollama_url}/api/tags", 2))
        except Exception:
            up = False
        with self._lock:
            self._up = (time.time(), up)
        return up

    def _order(self) -> list[str]:
        b = self.backend
        if b == "template":
            return []
        if b == "ollama":
            return ["ollama"]
        if b == "nim":
            return ["nim"] if self._nim_key else []
        order = []
        if self._ollama_up():
            order.append("ollama")
        if self._nim_key:
            order.append("nim")
        return order

    def _call(self, backend: str, prompt: str) -> str:
        body = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 300,
            "temperature": 0.2,
            "stream": False,
        }
        if backend == "ollama":
            url = f"{self._ollama_url}/v1/chat/completions"
            body["model"] = self._ollama_model
            headers = {}
        else:
            url = f"{self._nim_url}/chat/completions"
            body["model"] = self._nim_model
            headers = {"Authorization": f"Bearer {self._nim_key}"}
        data = self._post(url, body, headers, self._timeout)
        return data["choices"][0]["message"]["content"].strip()

    def generate(
        self,
        flow: dict,
        label: str,
        confidence: float,
        top_features: dict,
        values: dict | None = None,
    ) -> dict:
        t0 = time.perf_counter()
        prompt = build_prompt(flow, label, confidence, top_features, values)
        for backend in self._order():
            try:
                text = self._call(backend, prompt)
                if text:
                    bad = ungrounded_numbers(
                        text, _allowed_numbers(flow, confidence, top_features, values)
                    )
                    if bad:
                        log.warning("reporter.ungrounded_numbers", numbers=bad)
                        text += (
                            "\n\n[Note: this report contains figures not found in "
                            "the alert data: "
                            + ", ".join(bad)
                            + ". Verify before use.]"
                        )
                    return {
                        "report": text,
                        "backend": backend,
                        "ungrounded": bad,
                        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                    }
            except Exception as exc:
                log.warning(
                    "reporter.backend_failed",
                    backend=backend,
                    error=str(exc),
                    hint="for Ollama: run `ollama pull " + self._ollama_model + "`",
                )
        return {
            "report": template_report(flow, label, confidence, top_features, values),
            "backend": "template",
            "ungrounded": [],
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }


_reporter: IncidentReporter | None = None


def get_reporter() -> IncidentReporter:
    global _reporter
    if _reporter is None:
        _reporter = IncidentReporter.from_settings()
    return _reporter
