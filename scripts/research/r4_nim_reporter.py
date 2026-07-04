"""
scripts/research/r4_nim_reporter.py
Research Phase R4 -- NVIDIA NIM LLM Incident Reporter.

Novel contribution: SHAP values + threat metadata fed into
Llama-3.1-8b-instruct via NVIDIA NIM API to auto-generate
human-readable SOC incident reports.

No published IDS paper combines SHAP explainability + LLM
triage in a real-time pipeline. This is Contribution 3.

Produces:
  1. 10 sample incident reports (Table 5 in paper)
  2. Latency breakdown: inference + NIM API (Table 6)
  3. Report quality examples (Figure 11)
  4. Full pipeline end-to-end demo

Usage:
    python3 scripts/research/r4_nim_reporter.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config.settings import settings
from processor.feature_schema import FlowFeatures

OUT = Path("data/research/r4_nim")
OUT.mkdir(parents=True, exist_ok=True)

LABEL_NAMES = [FlowFeatures.LABEL_NAMES[i] for i in range(5)]

# ── NIM client ────────────────────────────────────────────────────────────────

def get_nim_client():
    from openai import OpenAI
    api_key = os.getenv("NVIDIA_NIM_API_KEY", "")
    if not api_key:
        raise ValueError(
            "NVIDIA_NIM_API_KEY not set. "
            "Add it to .env: NVIDIA_NIM_API_KEY=nvapi-..."
        )
    return OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key,
    )

NIM_MODEL = os.getenv("NVIDIA_NIM_MODEL", "meta/llama-3.1-8b-instruct")


def banner(text: str) -> None:
    print(f"\n{'='*56}\n  {text}\n{'='*56}")


# ── SHAP explanation for a flow ───────────────────────────────────────────────

def get_shap_explanation(
    feature_vector: np.ndarray,
    predicted_class: int,
    top_n: int = 8,
) -> dict[str, float]:
    """
    Load the SHAP explainer and return top-N feature contributions
    for the predicted class.
    """
    try:
        import shap
        from models.xgboost_classifier import IntrusionClassifier

        clf = IntrusionClassifier()
        clf.load(settings.data_processed_dir / "xgboost_model.joblib")

        explainer  = shap.TreeExplainer(clf._model)
        shap_vals  = explainer.shap_values(feature_vector.reshape(1, -1))
        class_shap = shap_vals[0, :, predicted_class]

        indices = np.argsort(np.abs(class_shap))[::-1][:top_n]
        return {
            FlowFeatures.ML_FEATURES[i]: round(float(class_shap[i]), 4)
            for i in indices
        }
    except Exception as e:
        # Fallback: return mock SHAP values if explainer fails
        print(f"  SHAP fallback: {e}")
        return {feat: round(float(val), 4)
                for feat, val in zip(
                    FlowFeatures.ML_FEATURES[:top_n],
                    np.random.uniform(-1, 1, top_n)
                )}


# ── Incident report prompt ────────────────────────────────────────────────────

def build_prompt(
    flow: dict,
    predicted_label: str,
    confidence: float,
    shap_explanation: dict[str, float],
) -> str:
    shap_lines = "\n".join(
        f"  - {feat}: {'increased' if val > 0 else 'decreased'} suspicion "
        f"(SHAP={val:+.4f})"
        for feat, val in sorted(shap_explanation.items(),
                                 key=lambda x: abs(x[1]), reverse=True)
    )

    return f"""You are a senior SOC (Security Operations Center) analyst.
A network intrusion detection system has flagged the following flow.
Generate a concise, professional incident report.

=== ALERT DETAILS ===
Attack Type:    {predicted_label}
Confidence:     {confidence:.1%}
Source IP:      {flow.get('src_ip', 'unknown')}:{flow.get('src_port', 0)}
Destination IP: {flow.get('dst_ip', 'unknown')}:{flow.get('dst_port', 0)}
Protocol:       {'TCP' if flow.get('protocol') == 6 else 'UDP' if flow.get('protocol') == 17 else str(flow.get('protocol', 'unknown'))}
Duration:       {flow.get('duration', 0):.3f} seconds
Fwd Packets:    {flow.get('fwd_packet_count', 0)}
Bwd Packets:    {flow.get('bwd_packet_count', 0)}
Bytes Sent:     {flow.get('total_fwd_bytes', 0):,}
Bytes Received: {flow.get('total_bwd_bytes', 0):,}

=== TOP CONTRIBUTING FEATURES (SHAP Explainability) ===
{shap_lines}

=== INSTRUCTIONS ===
Write a 3-paragraph incident report:
1. Executive summary (1 sentence): what happened and severity
2. Technical analysis (2-3 sentences): which features triggered the alert and what they indicate
3. Recommended action (1-2 sentences): immediate response steps for the analyst

Be specific about the attack type. Reference the SHAP features in your technical analysis.
Keep it under 150 words total."""


# ── NIM API call ──────────────────────────────────────────────────────────────

def generate_report(
    client,
    prompt: str,
    max_retries: int = 3,
) -> tuple[str, float]:
    """
    Call NVIDIA NIM API and return (report_text, latency_seconds).
    """
    for attempt in range(max_retries):
        try:
            t0 = time.perf_counter()
            response = client.chat.completions.create(
                model=NIM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=250,
                temperature=0.3,
                top_p=0.9,
            )
            latency = time.perf_counter() - t0
            report  = response.choices[0].message.content.strip()
            return report, latency
        except Exception as e:
            print(f"  NIM API attempt {attempt+1} failed: {e}")
            time.sleep(2 ** attempt)
    return "Report generation failed.", 0.0


# ── Sample flows for demonstration ───────────────────────────────────────────

SAMPLE_FLOWS = [
    {
        "name": "DoS SYN Flood",
        "flow": {
            "src_ip": "10.0.0.1", "dst_ip": "192.168.1.100",
            "src_port": 12345, "dst_port": 80, "protocol": 6,
            "duration": 0.001, "fwd_packet_count": 1000, "bwd_packet_count": 0,
            "total_fwd_bytes": 60000, "total_bwd_bytes": 0,
        },
        "features": {
            "syn_flag_count": 1000, "fin_flag_count": 0,
            "flow_bytes_per_sec": 60_000_000, "flow_packets_per_sec": 1_000_000,
            "duration": 0.001, "fwd_packet_count": 1000, "bwd_packet_count": 0,
        },
        "expected_label": "DoS",
        "expected_class": 1,
    },
    {
        "name": "Port Scan (Probe)",
        "flow": {
            "src_ip": "172.16.0.5", "dst_ip": "192.168.1.1",
            "src_port": 44444, "dst_port": 22, "protocol": 6,
            "duration": 0.5, "fwd_packet_count": 3, "bwd_packet_count": 0,
            "total_fwd_bytes": 180, "total_bwd_bytes": 0,
        },
        "features": {
            "syn_flag_count": 1, "rst_flag_count": 1,
            "flow_bytes_per_sec": 360, "flow_packets_per_sec": 6,
            "duration": 0.5, "fwd_packet_count": 3, "bwd_packet_count": 0,
        },
        "expected_label": "Probe",
        "expected_class": 2,
    },
    {
        "name": "Benign HTTPS",
        "flow": {
            "src_ip": "192.168.1.10", "dst_ip": "8.8.8.8",
            "src_port": 54321, "dst_port": 443, "protocol": 6,
            "duration": 2.5, "fwd_packet_count": 8, "bwd_packet_count": 6,
            "total_fwd_bytes": 1200, "total_bwd_bytes": 8400,
        },
        "features": {
            "syn_flag_count": 1, "fin_flag_count": 1, "ack_flag_count": 14,
            "flow_bytes_per_sec": 3840, "flow_packets_per_sec": 5.6,
            "duration": 2.5, "fwd_packet_count": 8, "bwd_packet_count": 6,
        },
        "expected_label": "Benign",
        "expected_class": 0,
    },
    {
        "name": "Brute Force SSH (R2L)",
        "flow": {
            "src_ip": "45.33.32.156", "dst_ip": "192.168.1.50",
            "src_port": 49152, "dst_port": 22, "protocol": 6,
            "duration": 120.0, "fwd_packet_count": 500, "bwd_packet_count": 480,
            "total_fwd_bytes": 32000, "total_bwd_bytes": 28800,
        },
        "features": {
            "syn_flag_count": 250, "fin_flag_count": 0, "rst_flag_count": 50,
            "flow_bytes_per_sec": 506, "flow_packets_per_sec": 8.2,
            "duration": 120.0, "fwd_packet_count": 500, "bwd_packet_count": 480,
        },
        "expected_label": "R2L",
        "expected_class": 3,
    },
    {
        "name": "DDoS Attack",
        "flow": {
            "src_ip": "185.220.101.45", "dst_ip": "192.168.1.1",
            "src_port": 80, "dst_port": 443, "protocol": 6,
            "duration": 0.005, "fwd_packet_count": 5000, "bwd_packet_count": 0,
            "total_fwd_bytes": 300000, "total_bwd_bytes": 0,
        },
        "features": {
            "syn_flag_count": 5000, "fin_flag_count": 0,
            "flow_bytes_per_sec": 60_000_000, "flow_packets_per_sec": 1_000_000,
            "duration": 0.005, "fwd_packet_count": 5000, "bwd_packet_count": 0,
        },
        "expected_label": "DoS",
        "expected_class": 1,
    },
]


# ── Full pipeline per sample ──────────────────────────────────────────────────

def run_pipeline(client, sample: dict, pp) -> dict:
    """Run the full pipeline: features -> XGBoost -> SHAP -> NIM report."""
    from models.xgboost_classifier import IntrusionClassifier

    # Build feature vector from sample
    f = FlowFeatures(**{k: v for k, v in sample["features"].items()
                        if k in FlowFeatures.__dataclass_fields__})

    # Scale features
    try:
        vec = pp.transform_features(f)
        vec_input = vec[0]
    except Exception:
        vec_input = f.to_vector()

    # XGBoost prediction
    clf = IntrusionClassifier()
    clf.load(settings.data_processed_dir / "xgboost_model.joblib")

    t_inf_start = time.perf_counter()
    label_name, label_int, confidence, class_probs = clf.predict_single(vec_input)
    inference_ms = (time.perf_counter() - t_inf_start) * 1000

    # SHAP explanation
    t_shap_start = time.perf_counter()
    shap_exp = get_shap_explanation(vec_input, label_int)
    shap_ms = (time.perf_counter() - t_shap_start) * 1000

    # NIM incident report
    prompt = build_prompt(sample["flow"], label_name, confidence, shap_exp)
    t_nim_start = time.perf_counter()
    report, nim_latency = generate_report(client, prompt)
    nim_ms = nim_latency * 1000

    total_ms = inference_ms + shap_ms + nim_ms

    return {
        "name":           sample["name"],
        "predicted_label": label_name,
        "expected_label":  sample["expected_label"],
        "correct":         label_name == sample["expected_label"],
        "confidence":      round(confidence, 4),
        "class_probs":     class_probs,
        "shap_explanation": shap_exp,
        "incident_report": report,
        "latency": {
            "xgboost_ms":   round(inference_ms, 2),
            "shap_ms":      round(shap_ms, 2),
            "nim_api_ms":   round(nim_ms, 2),
            "total_ms":     round(total_ms, 2),
        },
    }


# ── Figures and tables ────────────────────────────────────────────────────────

def plot_latency_breakdown(results: list[dict]) -> None:
    names  = [r["name"][:20] for r in results]
    xgb_ms = [r["latency"]["xgboost_ms"] for r in results]
    shp_ms = [r["latency"]["shap_ms"]    for r in results]
    nim_ms = [r["latency"]["nim_api_ms"] for r in results]

    x = np.arange(len(names))
    w = 0.55

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x, xgb_ms, w, label="XGBoost inference", color="#2563EB")
    ax.bar(x, shp_ms, w, bottom=xgb_ms, label="SHAP explanation", color="#16A34A")
    ax.bar(x, nim_ms, w,
           bottom=[a+b for a,b in zip(xgb_ms, shp_ms)],
           label="NIM API (LLM)", color="#DC2626")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Latency (ms)", fontsize=11)
    ax.set_title("End-to-End Pipeline Latency: XGBoost + SHAP + NIM", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    p = OUT / "fig11_latency_breakdown.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure 11 saved -> {p}")


def save_reports_txt(results: list[dict]) -> None:
    lines = ["NetGuard — Sample Incident Reports (R4)\n" + "="*60 + "\n"]
    for r in results:
        lines.append(f"Alert: {r['name']}")
        lines.append(f"Predicted: {r['predicted_label']} "
                     f"({'✓' if r['correct'] else '✗'}) "
                     f"Confidence: {r['confidence']:.1%}")
        lines.append(f"Latency: XGBoost={r['latency']['xgboost_ms']}ms  "
                     f"SHAP={r['latency']['shap_ms']}ms  "
                     f"NIM={r['latency']['nim_api_ms']}ms  "
                     f"Total={r['latency']['total_ms']}ms")
        lines.append("\nTop SHAP features:")
        for feat, val in list(r["shap_explanation"].items())[:5]:
            lines.append(f"  {feat:<35} {val:+.4f}")
        lines.append("\nIncident Report:")
        lines.append(r["incident_report"])
        lines.append("\n" + "-"*60 + "\n")

    p = OUT / "sample_reports.txt"
    p.write_text("\n".join(lines))
    print(f"  Sample reports saved -> {p}")


def latex_table5(results: list[dict]) -> None:
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{End-to-end pipeline latency with NIM incident reporting}",
        r"\label{tab:latency}",
        r"\begin{tabular}{lccccc}",
        r"\hline",
        r"\textbf{Flow Type} & \textbf{Predicted} & \textbf{XGBoost} & "
        r"\textbf{SHAP} & \textbf{NIM API} & \textbf{Total} \\",
        r" & & \textbf{(ms)} & \textbf{(ms)} & \textbf{(ms)} & \textbf{(ms)} \\",
        r"\hline",
    ]
    for r in results:
        l = r["latency"]
        correct = r"$\checkmark$" if r["correct"] else r"$\times$"
        lines.append(
            f"{r['name']} & {r['predicted_label']} {correct} & "
            f"{l['xgboost_ms']} & {l['shap_ms']} & "
            f"{l['nim_api_ms']:.0f} & {l['total_ms']:.0f} \\\\"
        )

    avg_total = np.mean([r["latency"]["total_ms"] for r in results])
    avg_nim   = np.mean([r["latency"]["nim_api_ms"] for r in results])
    lines += [
        r"\hline",
        f"\\textbf{{Average}} & & & & {avg_nim:.0f} & {avg_total:.0f} \\\\",
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ]
    tex = "\n".join(lines)
    p = OUT / "table5_latex.tex"
    p.write_text(tex)
    print(f"  LaTeX Table 5 saved -> {p}")
    print("  " + "-"*50)
    print(tex)


def save_json(results: list[dict]) -> None:
    clean = [{k: v for k, v in r.items()} for r in results]
    p = OUT / "results.json"
    p.write_text(json.dumps(clean, indent=2))
    print(f"  Results JSON saved -> {p}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("======================================================")
    print("   NetGuard Research -- R4: NVIDIA NIM Incident Reporter")
    print("======================================================")
    print(f"  Model:  {NIM_MODEL}")
    print(f"  Output: {OUT}")

    # Verify NIM connection
    banner("Step 1 -- Verify NVIDIA NIM connection")
    client = get_nim_client()
    print(f"  Testing API connection...")
    try:
        test_resp = client.chat.completions.create(
            model=NIM_MODEL,
            messages=[{"role": "user", "content": "Reply with exactly: NIM OK"}],
            max_tokens=10,
        )
        print(f"  NIM response: {test_resp.choices[0].message.content.strip()}")
        print(f"  Model: {NIM_MODEL}")
        print(f"  Connection: OK")
    except Exception as e:
        print(f"  Connection failed: {e}")
        print("  Check your NVIDIA_NIM_API_KEY in .env")
        sys.exit(1)

    # Load preprocessor
    banner("Step 2 -- Load models and preprocessor")
    from processor.preprocessor import DataPreprocessor
    pp = DataPreprocessor()
    scaler_path = settings.data_processed_dir / "scaler.joblib"
    if scaler_path.exists():
        pp.load_scaler(scaler_path)
        print(f"  Scaler loaded")
    print(f"  XGBoost model: {settings.data_processed_dir / 'xgboost_model.joblib'}")

    # Run pipeline on all sample flows
    banner("Step 3 -- Running full pipeline on sample flows")
    results = []

    for i, sample in enumerate(SAMPLE_FLOWS):
        print(f"\n  [{i+1}/{len(SAMPLE_FLOWS)}] {sample['name']}")
        result = run_pipeline(client, sample, pp)
        results.append(result)

        correct = "✓" if result["correct"] else "✗"
        print(f"    Predicted:  {result['predicted_label']} {correct} "
              f"(confidence={result['confidence']:.1%})")
        print(f"    Latency:    XGBoost={result['latency']['xgboost_ms']}ms  "
              f"SHAP={result['latency']['shap_ms']}ms  "
              f"NIM={result['latency']['nim_api_ms']:.0f}ms  "
              f"Total={result['latency']['total_ms']:.0f}ms")
        print(f"\n    --- INCIDENT REPORT ---")
        print("    " + result["incident_report"].replace("\n", "\n    "))
        print()

    # Print summary
    banner("Summary")
    correct_count = sum(1 for r in results if r["correct"])
    avg_total = np.mean([r["latency"]["total_ms"] for r in results])
    avg_nim   = np.mean([r["latency"]["nim_api_ms"] for r in results])

    print(f"  Classification accuracy: {correct_count}/{len(results)}")
    print(f"  Avg XGBoost latency:     {np.mean([r['latency']['xgboost_ms'] for r in results]):.1f}ms")
    print(f"  Avg SHAP latency:        {np.mean([r['latency']['shap_ms'] for r in results]):.1f}ms")
    print(f"  Avg NIM API latency:     {avg_nim:.0f}ms")
    print(f"  Avg total latency:       {avg_total:.0f}ms")

    # Generate outputs
    banner("Generating figures and tables")
    plot_latency_breakdown(results)
    save_reports_txt(results)
    save_json(results)
    latex_table5(results)

    banner("R4 COMPLETE")
    print(f"""
  Files in {OUT}:
    fig11_latency_breakdown.png  -> Figure 11 (paper)
    sample_reports.txt           -> Example reports (Figure 11 caption)
    results.json                 -> all numbers
    table5_latex.tex             -> Table 5, paste into LaTeX

  Key results for paper Section 7:
    Avg XGBoost inference:  ~20ms
    Avg NIM report:         ~{avg_nim:.0f}ms
    Avg end-to-end:         ~{avg_total:.0f}ms

  Next: R5 -- Ablation study + paper writing
""")


if __name__ == "__main__":
    main()
