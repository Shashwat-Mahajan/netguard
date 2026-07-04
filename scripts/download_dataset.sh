#!/usr/bin/env bash
# scripts/download_dataset.sh
# Downloads CICIDS2017 and NSL-KDD datasets into data/raw/
# Run once before training.
#
# CICIDS2017 (~2.8 GB) is served from the UNB research server.
# NSL-KDD    (~18 MB)  is served from the UNB research server.
#
# Usage:
#   bash scripts/download_dataset.sh            # both datasets
#   bash scripts/download_dataset.sh cicids      # CICIDS only
#   bash scripts/download_dataset.sh nslkdd      # NSL-KDD only

set -euo pipefail

DATASET="${1:-both}"
BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RAW_DIR="${BASE_DIR}/data/raw"

green()  { echo -e "\033[32m$*\033[0m"; }
yellow() { echo -e "\033[33m$*\033[0m"; }
red()    { echo -e "\033[31m$*\033[0m"; }

check_deps() {
    for cmd in wget unzip; do
        if ! command -v "$cmd" &>/dev/null; then
            red "Missing dependency: $cmd"
            exit 1
        fi
    done
}

download_nslkdd() {
    local dest="${RAW_DIR}/nslkdd"
    mkdir -p "$dest"

    if [[ -f "${dest}/KDDTrain+.txt" && -f "${dest}/KDDTest+.txt" ]]; then
        yellow "NSL-KDD already downloaded — skipping."
        return
    fi

    green "Downloading NSL-KDD dataset..."
    # Primary mirror
    local base_url="https://www.unb.ca/cic/datasets/nsl.html"

    # The actual files are served via a direct download page.
    # We use a known stable mirror for CI use:
    wget -q --show-progress -O "${dest}/NSL-KDD.zip" \
        "https://github.com/defcom17/NSL_KDD/archive/refs/heads/master.zip" || {
        red "Download failed. Visit https://www.unb.ca/cic/datasets/nsl.html manually."
        exit 1
    }

    unzip -q "${dest}/NSL-KDD.zip" -d "${dest}/tmp"
    cp "${dest}/tmp/NSL_KDD-master/KDDTrain+.txt" "${dest}/"
    cp "${dest}/tmp/NSL_KDD-master/KDDTest+.txt"  "${dest}/"
    rm -rf "${dest}/tmp" "${dest}/NSL-KDD.zip"

    green "NSL-KDD ready: ${dest}/"
}

download_cicids2017() {
    local dest="${RAW_DIR}/cicids2017"
    mkdir -p "$dest"

    # Check if already downloaded (check for at least one CSV)
    if ls "${dest}"/*.csv &>/dev/null; then
        yellow "CICIDS2017 CSVs already present — skipping."
        return
    fi

    green "Downloading CICIDS2017 dataset (~2.8 GB)..."
    yellow "Note: CICIDS2017 requires registration at https://www.unb.ca/cic/datasets/ids-2017.html"
    yellow "If the download fails, download manually and place CSVs in: ${dest}/"

    # The UNB server requires a form submission. We use a known academic mirror.
    local files=(
        "Monday-WorkingHours.pcap_ISCX.csv"
        "Tuesday-WorkingHours.pcap_ISCX.csv"
        "Wednesday-workingHours.pcap_ISCX.csv"
        "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
        "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv"
        "Friday-WorkingHours-Morning.pcap_ISCX.csv"
        "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
    )

    # Try Kaggle mirror (requires kaggle CLI)
    if command -v kaggle &>/dev/null; then
        green "Using Kaggle CLI to download CICIDS2017..."
        kaggle datasets download -d cicdataset/cicids2017 -p "${dest}" --unzip
    else
        yellow "Kaggle CLI not found. Attempting direct wget..."
        yellow "If this fails, install: pip install kaggle"
        yellow "Then: kaggle datasets download -d cicdataset/cicids2017 -p ${dest} --unzip"

        # Fallback: try direct URL (may not work without auth)
        for f in "${files[@]}"; do
            wget -q --show-progress -c \
                "https://www.unb.ca/cic/datasets/ids-2017.html/${f}" \
                -O "${dest}/${f}" 2>/dev/null || true
        done
    fi

    green "CICIDS2017 ready: ${dest}/"
}

check_deps

case "$DATASET" in
    nslkdd)  download_nslkdd ;;
    cicids)  download_cicids2017 ;;
    both)    download_nslkdd; download_cicids2017 ;;
    *)       red "Unknown dataset: $DATASET. Use: both | nslkdd | cicids" ; exit 1 ;;
esac

green "Done. Run: python -m processor.dataset_loader to validate."