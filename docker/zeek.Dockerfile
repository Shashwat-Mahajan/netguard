# ════════════════════════════════════════════════════════════
#  NetGuard — Zeek IDS sidecar
#  Runs as a passive network monitor alongside the API.
#
#  Why a sidecar pattern?
#    Zeek needs CAP_NET_RAW + host network to see raw packets.
#    The API runs as non-root. Separating them follows least-privilege.
#
#  Build (from project root):
#    docker build -f docker/zeek.Dockerfile -t netguard-zeek:phase1 .
#
#  Run standalone (for testing):
#    docker run --rm --net=host --cap-add NET_RAW --cap-add NET_ADMIN \
#      -v zeek-logs:/opt/zeek/logs \
#      netguard-zeek:phase1 eth0
# ════════════════════════════════════════════════════════════

FROM zeek/zeek:6.2.0

LABEL maintainer="netguard"
LABEL description="Zeek IDS passive network monitor"

# ── Install Zeek package manager packages ─────────────────────────────────────
# zkg is already installed in the base image.
# CommunityID adds a standardised flow ID that matches other tools (Suricata, etc.)
RUN zkg install zeek/corelight/zeek-community-id --force

# ── Custom Zeek site configuration ────────────────────────────────────────────
# We append to local.zeek so the base image defaults are preserved.
RUN echo '' >> /usr/local/zeek/share/zeek/site/local.zeek && \
    echo '# NetGuard custom config' >> /usr/local/zeek/share/zeek/site/local.zeek && \
    echo '@load corelight/zeek-community-id' >> /usr/local/zeek/share/zeek/site/local.zeek && \
    echo '@load policy/tuning/json-logs' >> /usr/local/zeek/share/zeek/site/local.zeek && \
    echo '@load frameworks/files/hash-all-files' >> /usr/local/zeek/share/zeek/site/local.zeek && \
    echo 'redef Log::default_rotation_interval = 1hr;' >> /usr/local/zeek/share/zeek/site/local.zeek && \
    echo 'redef Log::default_rotation_postprocessor_cmd = "";' >> /usr/local/zeek/share/zeek/site/local.zeek

# ── Log output directory (shared with API container via Docker volume) ─────────
RUN mkdir -p /opt/zeek/logs/current && \
    chmod 777 /opt/zeek/logs/current

VOLUME ["/opt/zeek/logs"]

# ── Health check ──────────────────────────────────────────────────────────────
# Checks whether Zeek is still writing logs (file modified in last 60s)
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD find /opt/zeek/logs/current -name "conn*.log" -newer /tmp/.zeek_start \
    -type f | grep -q . || exit 1

# ── Entrypoint ────────────────────────────────────────────────────────────────
# The interface name is passed from docker-compose via CMD.
# ENTRYPOINT stays fixed; CMD is overridden by docker-compose command: field.
ENTRYPOINT ["zeek", "-i"]

# Default fallback interface (overridden by docker-compose)
CMD ["eth0", "/usr/local/zeek/share/zeek/site/local.zeek"]
