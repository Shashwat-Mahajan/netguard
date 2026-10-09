"""
processor/dataset_loader.py
Loads CICIDS2017 and NSL-KDD datasets into pandas DataFrames.

CICIDS2017:
  Download URL: https://www.unb.ca/cic/datasets/ids-2017.html
  Files: Monday-...-WorkingHours.pcap_ISCX.csv (8 CSVs)
  Expected in: data/raw/cicids2017/

NSL-KDD:
  Download URL: https://www.unb.ca/cic/datasets/nsl.html
  Files: KDDTrain+.txt, KDDTest+.txt
  Expected in: data/raw/nslkdd/

The loaders normalise column names and map labels to the canonical
label scheme defined in FlowFeatures.LABEL_MAP.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config.logger import get_logger
from config.settings import settings
from processor.feature_schema import FlowFeatures

logger = get_logger(__name__)


# ── CICIDS2017 ────────────────────────────────────────────────────────────────

# CICIDS column name → our canonical name
CICIDS_RENAME: dict[str, str] = {
    "Flow Duration": "duration",
    "Total Fwd Packets": "fwd_packet_count",
    "Total Backward Packets": "bwd_packet_count",
    "Total Length of Fwd Packets": "total_fwd_bytes",
    "Total Length of Bwd Packets": "total_bwd_bytes",
    "Fwd Packet Length Max": "fwd_packet_len_max",
    "Fwd Packet Length Min": "fwd_packet_len_min",
    "Fwd Packet Length Mean": "fwd_packet_len_mean",
    "Fwd Packet Length Std": "fwd_packet_len_std",
    "Bwd Packet Length Max": "bwd_packet_len_max",
    "Bwd Packet Length Min": "bwd_packet_len_min",
    "Bwd Packet Length Mean": "bwd_packet_len_mean",
    "Bwd Packet Length Std": "bwd_packet_len_std",
    "Flow Bytes/s": "flow_bytes_per_sec",
    "Flow Packets/s": "flow_packets_per_sec",
    "Fwd Packets/s": "fwd_packets_per_sec",
    "Bwd Packets/s": "bwd_packets_per_sec",
    "Flow IAT Mean": "flow_iat_mean",
    "Flow IAT Std": "flow_iat_std",
    "Flow IAT Max": "flow_iat_max",
    "Flow IAT Min": "flow_iat_min",
    "Fwd IAT Mean": "fwd_iat_mean",
    "Fwd IAT Std": "fwd_iat_std",
    "Bwd IAT Mean": "bwd_iat_mean",
    "Bwd IAT Std": "bwd_iat_std",
    "Fwd PSH Flags": "fwd_psh_flags",
    "Bwd PSH Flags": "bwd_psh_flags",
    "Fwd URG Flags": "fwd_urg_flags",
    "Bwd URG Flags": "bwd_urg_flags",
    "FIN Flag Count": "fin_flag_count",
    "SYN Flag Count": "syn_flag_count",
    "RST Flag Count": "rst_flag_count",
    "PSH Flag Count": "psh_flag_count",
    "ACK Flag Count": "ack_flag_count",
    "Fwd Header Length": "fwd_header_len",
    "Bwd Header Length": "bwd_header_len",
    "Init_Win_bytes_forward": "fwd_win_bytes",
    "Init_Win_bytes_backward": "bwd_win_bytes",
    "Protocol": "protocol",
    "Label": "label",
}


def _normalise_col(col: str) -> str:
    """Strip whitespace from column name (CICIDS CSVs have leading spaces)."""
    return col.strip()


class CICIDS2017Loader:
    """
    Load all CICIDS2017 CSV files from a directory into a single DataFrame.

    Each CSV is ~200–600 MB; they are loaded one at a time and concatenated
    to avoid excessive peak RAM. The result is saved as Parquet for fast
    reloads.
    """

    PARQUET_CACHE = "cicids2017_combined.parquet"

    def __init__(self, raw_dir: Path = settings.data_raw_dir / "cicids2017") -> None:
        self._raw_dir = Path(raw_dir)
        self._cache = settings.data_processed_dir / self.PARQUET_CACHE

    def load(self, use_cache: bool = True) -> pd.DataFrame:
        if use_cache and self._cache.exists():
            logger.info("dataset.cicids2017.cache_hit", path=str(self._cache))
            return pd.read_parquet(self._cache)

        csv_files = sorted(self._raw_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files found in {self._raw_dir}. "
                "Run scripts/download_dataset.sh first."
            )

        chunks: list[pd.DataFrame] = []
        for csv_path in csv_files:
            logger.info("dataset.cicids2017.loading", file=csv_path.name)
            df = self._load_single(csv_path)
            chunks.append(df)
            logger.info(
                "dataset.cicids2017.loaded",
                file=csv_path.name,
                rows=len(df),
                label_dist=df["label"].value_counts().to_dict(),
            )

        combined = pd.concat(chunks, ignore_index=True)
        combined = self._postprocess(combined)

        logger.info(
            "dataset.cicids2017.combined",
            total_rows=len(combined),
            label_dist=combined["label_encoded"].value_counts().to_dict(),
        )

        combined.to_parquet(self._cache, index=False)
        logger.info("dataset.cicids2017.cached", path=str(self._cache))
        return combined

    def _load_single(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(
            path,
            encoding="utf-8",
            low_memory=False,
            dtype=str,  # Read everything as string first — CICIDS has mixed types
        )
        # Normalise column names
        df.columns = [_normalise_col(c) for c in df.columns]
        df = df.rename(columns=CICIDS_RENAME)

        # Keep only canonical feature columns
        keep = FlowFeatures.ML_FEATURES + ["label"]
        existing = [c for c in keep if c in df.columns]
        df = df[existing].copy()

        return df

    def _postprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        # Encode labels
        df["label_clean"] = df["label"].str.strip().str.lower()
        df["label_encoded"] = (
            df["label_clean"].map(FlowFeatures.LABEL_MAP).fillna(-1).astype(int)
        )

        # Convert features to numeric, coerce errors → NaN
        for col in FlowFeatures.ML_FEATURES:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Replace inf values with NaN then fill
        df.replace([float("inf"), float("-inf")], float("nan"), inplace=True)

        # Fill NaN with column median
        for col in FlowFeatures.ML_FEATURES:
            if col in df.columns:
                median = df[col].median()
                df[col].fillna(median if not pd.isna(median) else 0.0, inplace=True)

        # Clip extreme values (e.g. flow_bytes_per_sec can be inf on 1-packet flows)
        for col in [
            "flow_bytes_per_sec",
            "flow_packets_per_sec",
            "fwd_packets_per_sec",
            "bwd_packets_per_sec",
        ]:
            if col in df.columns:
                cap = df[col].quantile(0.9999)
                df[col] = df[col].clip(upper=cap)

        # Drop rows with unknown labels for supervised training
        df = df[df["label_encoded"] >= 0].copy()

        return df


# ── NSL-KDD ──────────────────────────────────────────────────────────────────

NSL_KDD_COLUMNS = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label",
    "difficulty",
]

NSL_KDD_LABEL_MAP = {
    "normal": 0,
    "dos": 1,
    "neptune": 1,
    "back": 1,
    "land": 1,
    "pod": 1,
    "smurf": 1,
    "teardrop": 1,
    "mailbomb": 1,
    "udpstorm": 1,
    "processtable": 1,
    "worm": 1,
    "probe": 2,
    "satan": 2,
    "ipsweep": 2,
    "nmap": 2,
    "portsweep": 2,
    "mscan": 2,
    "saint": 2,
    "r2l": 3,
    "guess_passwd": 3,
    "ftp_write": 3,
    "imap": 3,
    "phf": 3,
    "multihop": 3,
    "warezmaster": 3,
    "warezclient": 3,
    "spy": 3,
    "xlock": 3,
    "xsnoop": 3,
    "snmpguess": 3,
    "snmpgetattack": 3,
    "httptunnel": 3,
    "sendmail": 3,
    "named": 3,
    "u2r": 4,
    "buffer_overflow": 4,
    "loadmodule": 4,
    "rootkit": 4,
    "perl": 4,
    "sqlattack": 4,
    "xterm": 4,
    "ps": 4,
}


class NSLKDDLoader:
    """Load NSL-KDD train/test splits."""

    PARQUET_CACHE_TRAIN = "nslkdd_train.parquet"
    PARQUET_CACHE_TEST = "nslkdd_test.parquet"

    def __init__(self, raw_dir: Path = settings.data_raw_dir / "nslkdd") -> None:
        self._raw_dir = Path(raw_dir)

    def load_train(self, use_cache: bool = True) -> pd.DataFrame:
        return self._load(
            self._raw_dir / "KDDTrain+.txt",
            settings.data_processed_dir / self.PARQUET_CACHE_TRAIN,
            use_cache,
            split="train",
        )

    def load_test(self, use_cache: bool = True) -> pd.DataFrame:
        return self._load(
            self._raw_dir / "KDDTest+.txt",
            settings.data_processed_dir / self.PARQUET_CACHE_TEST,
            use_cache,
            split="test",
        )

    def _load(
        self, src: Path, cache: Path, use_cache: bool, split: str
    ) -> pd.DataFrame:
        if use_cache and cache.exists():
            logger.info("dataset.nslkdd.cache_hit", split=split, path=str(cache))
            return pd.read_parquet(cache)

        if not src.exists():
            raise FileNotFoundError(
                f"NSL-KDD file not found: {src}. "
                "Run scripts/download_dataset.sh first."
            )

        df = pd.read_csv(src, header=None, names=NSL_KDD_COLUMNS)
        df = self._encode(df)

        logger.info(
            "dataset.nslkdd.loaded",
            split=split,
            rows=len(df),
            label_dist=df["label_encoded"].value_counts().to_dict(),
        )
        df.to_parquet(cache, index=False)
        return df

    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
        # Rename NSL-KDD columns to canonical schema
        df = df.rename(columns=NSL_KDD_FEATURE_MAP)

        # Add protocol as numeric (NSL-KDD has protocol_type as string)
        proto_map = {"tcp": 6, "udp": 17, "icmp": 1}
        if "protocol_type" in df.columns:
            df["protocol"] = df["protocol_type"].str.lower().map(proto_map).fillna(0).astype(int)

        # One-hot encode remaining categoricals
        cat_cols = [c for c in ["protocol_type", "service", "flag"] if c in df.columns]
        if cat_cols:
            df = pd.get_dummies(df, columns=cat_cols)

        # Encode label
        df["label_encoded"] = (
            df["label"]
            .str.strip()
            .str.lower()
            .map(NSL_KDD_LABEL_MAP)
            .fillna(-1)
            .astype(int)
        )
        df = df[df["label_encoded"] >= 0]

        # Convert all to float
        for col in df.columns:
            if col not in ("label", "label_encoded"):
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        return df


# ── NSL-KDD → canonical feature mapping ──────────────────────
NSL_KDD_FEATURE_MAP = {
    "duration":          "duration",
    "src_bytes":         "total_fwd_bytes",
    "dst_bytes":         "total_bwd_bytes",
    "land":              "fwd_psh_flags",
    "wrong_fragment":    "bwd_psh_flags",
    "urgent":            "fwd_urg_flags",
    "hot":               "bwd_urg_flags",
    "num_failed_logins": "fin_flag_count",
    "logged_in":         "syn_flag_count",
    "num_compromised":   "rst_flag_count",
    "root_shell":        "psh_flag_count",
    "su_attempted":      "ack_flag_count",
    "num_root":          "fwd_header_len",
    "num_file_creations":"bwd_header_len",
    "num_shells":        "fwd_win_bytes",
    "num_access_files":  "bwd_win_bytes",
    "count":             "fwd_packet_count",
    "srv_count":         "bwd_packet_count",
    "serror_rate":       "flow_iat_mean",
    "rerror_rate":       "flow_iat_std",
    "same_srv_rate":     "flow_iat_max",
    "diff_srv_rate":     "flow_iat_min",
    "srv_diff_host_rate":"fwd_iat_mean",
    "dst_host_count":    "fwd_iat_std",
    "dst_host_srv_count":"bwd_iat_mean",
    "dst_host_same_srv_rate": "bwd_iat_std",
    "dst_host_serror_rate":   "flow_bytes_per_sec",
    "dst_host_rerror_rate":   "flow_packets_per_sec",
    "dst_host_srv_serror_rate": "fwd_packets_per_sec",
    "dst_host_srv_rerror_rate": "bwd_packets_per_sec",
    "dst_host_diff_srv_rate":   "fwd_packet_len_mean",
    "dst_host_same_src_port_rate": "bwd_packet_len_mean",
    "dst_host_srv_diff_host_rate": "fwd_packet_len_max",
    "srv_serror_rate":   "fwd_packet_len_min",
    "srv_rerror_rate":   "bwd_packet_len_max",
    "num_outbound_cmds": "bwd_packet_len_min",
    "is_host_login":     "bwd_packet_len_std",
    "is_guest_login":    "fwd_packet_len_std",
}


# ── NSL-KDD → canonical feature mapping ──────────────────────
NSL_KDD_FEATURE_MAP = {
    "duration":          "duration",
    "src_bytes":         "total_fwd_bytes",
    "dst_bytes":         "total_bwd_bytes",
    "land":              "fwd_psh_flags",
    "wrong_fragment":    "bwd_psh_flags",
    "urgent":            "fwd_urg_flags",
    "hot":               "bwd_urg_flags",
    "num_failed_logins": "fin_flag_count",
    "logged_in":         "syn_flag_count",
    "num_compromised":   "rst_flag_count",
    "root_shell":        "psh_flag_count",
    "su_attempted":      "ack_flag_count",
    "num_root":          "fwd_header_len",
    "num_file_creations":"bwd_header_len",
    "num_shells":        "fwd_win_bytes",
    "num_access_files":  "bwd_win_bytes",
    "count":             "fwd_packet_count",
    "srv_count":         "bwd_packet_count",
    "serror_rate":       "flow_iat_mean",
    "rerror_rate":       "flow_iat_std",
    "same_srv_rate":     "flow_iat_max",
    "diff_srv_rate":     "flow_iat_min",
    "srv_diff_host_rate":"fwd_iat_mean",
    "dst_host_count":    "fwd_iat_std",
    "dst_host_srv_count":"bwd_iat_mean",
    "dst_host_same_srv_rate": "bwd_iat_std",
    "dst_host_serror_rate":   "flow_bytes_per_sec",
    "dst_host_rerror_rate":   "flow_packets_per_sec",
    "dst_host_srv_serror_rate": "fwd_packets_per_sec",
    "dst_host_srv_rerror_rate": "bwd_packets_per_sec",
    "dst_host_diff_srv_rate":   "fwd_packet_len_mean",
    "dst_host_same_src_port_rate": "bwd_packet_len_mean",
    "dst_host_srv_diff_host_rate": "fwd_packet_len_max",
    "srv_serror_rate":   "fwd_packet_len_min",
    "srv_rerror_rate":   "bwd_packet_len_max",
    "num_outbound_cmds": "bwd_packet_len_min",
    "is_host_login":     "bwd_packet_len_std",
    "is_guest_login":    "fwd_packet_len_std",
}
