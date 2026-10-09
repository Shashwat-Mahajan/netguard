warning: in the working copy of '.gitignore', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'api/main.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'api/routers/predict.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'docker-compose.phase4.yml', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'models/anomaly_detector.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'models/explainer.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'models/xgboost_classifier.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'processor/cicids_parquet_loader.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'processor/dataset_loader.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'scripts/research/r1_baselines.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'scripts/research/r2_cicids.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'scripts/research/r3_federated.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'scripts/research/r4_nim_reporter.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'scripts/research/r5_ablation.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'scripts/test_phase3.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'streaming/alert_store.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'streaming/clickhouse_writer.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'streaming/consumer.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'streaming/producer.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'tests/test_models.py', LF will be replaced by CRLF the next time Git touches it
 .gitignore                          |  6 [32m++++[m[31m-[m
 api/main.py                         |  2 [32m+[m[31m-[m
 api/routers/predict.py              |  2 [32m+[m[31m-[m
 capture/sniffer.py                  |  4 [32m+[m[31m--[m
 capture/zeek_parser.py              |  5 [32m+[m[31m---[m
 config/logging.yml                  | 13 [32m++++++++++[m
 config/settings.py                  |  2 [32m+[m[31m-[m
 docker-compose.phase4.yml           | 52 [32m+++++++++++++[m[31m------------------------[m
 docker-compose.yml                  |  3 [32m+[m[31m--[m
 models/anomaly_detector.py          |  6 [32m++[m[31m---[m
 models/explainer.py                 |  2 [32m+[m[31m-[m
 models/xgboost_classifier.py        |  2 [32m+[m[31m-[m
 processor/cicids_parquet_loader.py  |  2 [32m+[m[31m-[m
 processor/dataset_loader.py         |  3 [31m---[m
 processor/feature_schema.py         |  2 [32m+[m[31m-[m
 scripts/ingest_data.py              |  4 [32m+[m[31m--[m
 scripts/research/r1_baselines.py    |  8 [32m+++[m[31m---[m
 scripts/research/r2_cicids.py       |  2 [32m+[m[31m-[m
 scripts/research/r3_federated.py    |  7 [32m+++[m[31m--[m
 scripts/research/r4_nim_reporter.py |  8 [32m+++[m[31m---[m
 scripts/research/r5_ablation.py     | 10 [32m++++[m[31m---[m
 scripts/test_phase3.py              |  2 [32m+[m[31m-[m
 streaming/alert_store.py            |  2 [32m+[m[31m-[m
 streaming/clickhouse_writer.py      |  2 [32m+[m[31m-[m
 streaming/consumer.py               |  2 [32m+[m[31m-[m
 streaming/producer.py               |  2 [31m--[m
 tests/test_capture.py               |  1 [31m-[m
 tests/test_clickhouse_writer.py     |  2 [32m+[m[31m-[m
 tests/test_models.py                |  1 [31m-[m
 tests/test_processor.py             |  2 [32m+[m[31m-[m
 30 files changed, 76 insertions(+), 85 deletions(-)
