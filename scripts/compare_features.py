"""
scripts/compare_features.py
Shows which live-extracted features fall far outside the training distribution.
Uses the saved RobustScaler (median/IQR of the training set).
Usage: python scripts/compare_features.py data/test.pcap
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from capture.pipeline import replay_pcap  # noqa: E402
from config.settings import settings  # noqa: E402
from processor.preprocessor import DataPreprocessor  # noqa: E402


class Collect:
    def __init__(self):
        self.flows = []

    def handle_flow(self, f):
        self.flows.append(f)


def show(title, flow, pp, top=12):
    sc, names = pp._scaler, pp._feature_cols
    vec = flow.to_vector()
    rows = []
    for i, name in enumerate(names):
        scale = sc.scale_[i] if sc.scale_[i] else 1.0
        rows.append(
            (abs((vec[i] - sc.center_[i]) / scale), name, vec[i], sc.center_[i], scale)
        )
    rows.sort(reverse=True)
    far = sum(1 for r in rows if r[0] > 10)
    print(
        f"\n=== {title}: {far} of {len(rows)} features more than 10 IQRs from training median"
    )
    print(
        f"{'feature':<22}{'live value':>14}{'train median':>14}{'train IQR':>12}{'z':>10}"
    )
    for z, name, v, c, s in rows[:top]:
        print(f"{name:<22}{v:>14.4g}{c:>14.4g}{s:>12.4g}{z:>10.1f}")
    zero = [n for n, c, s in zip(names, sc.center_, sc.scale_) if c == 0 and s == 0]
    return zero


def main():
    pcap = sys.argv[1] if len(sys.argv) > 1 else "data/test.pcap"
    pp = DataPreprocessor()
    pp.load_scaler(settings.data_processed_dir / "scaler.joblib")
    col = Collect()
    replay_pcap(pcap, col)
    flows = col.flows
    flood = max(flows, key=lambda f: f.syn_flag_count)
    rest = [f for f in flows if f is not flood]
    normal = max(rest, key=lambda f: f.fwd_packet_count + f.bwd_packet_count)
    scan = next(f for f in rest if f is not normal)
    zero = show("SYN flood", flood, pp)
    show("Port-scan flow", scan, pp)
    show("Normal conversation", normal, pp)
    print(
        f"\nFeatures that were constant zero in training (median=0, IQR=0): {len(zero)}"
    )
    print(", ".join(zero))


if __name__ == "__main__":
    main()
