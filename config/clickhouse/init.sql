CREATE DATABASE IF NOT EXISTS netguard;

CREATE TABLE IF NOT EXISTS netguard.flows (
    flow_id              String,
    src_ip               String,
    dst_ip               String,
    src_port             UInt16,
    dst_port             UInt16,
    protocol             UInt8,
    timestamp            DateTime,
    duration             Float64,
    fwd_packet_count     UInt32,
    bwd_packet_count     UInt32,
    total_fwd_bytes      UInt64,
    total_bwd_bytes      UInt64,
    flow_bytes_per_sec   Float64,
    flow_packets_per_sec Float64,
    syn_flag_count       UInt32,
    fin_flag_count       UInt32,
    rst_flag_count       UInt32,
    psh_flag_count       UInt32,
    label                LowCardinality(String),
    label_encoded        Int8,
    confidence           Float32,
    is_anomaly           UInt8,
    anomaly_score        Float32,
    is_threat            UInt8,
    inference_ms         Float32,
    class_probs          String DEFAULT '',
    explanation          String DEFAULT '',
    report               String DEFAULT ''
) ENGINE = MergeTree
PARTITION BY toYYYYMM(timestamp)
ORDER BY (timestamp, src_ip);

CREATE TABLE IF NOT EXISTS netguard.alerts (
    alert_id      String,
    timestamp     DateTime,
    source        LowCardinality(String),
    rule          LowCardinality(String),
    label         LowCardinality(String),
    label_encoded Int8,
    severity      LowCardinality(String),
    src_ip        String,
    dst_ip        String,
    flow_count    UInt32,
    detail        String,
    intel         String DEFAULT '',
    intel_hit     UInt8 DEFAULT 0
) ENGINE = MergeTree
PARTITION BY toYYYYMM(timestamp)
ORDER BY (timestamp, src_ip);