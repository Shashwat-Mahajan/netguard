# Federated XGBoost experiment (Phase 5)

Real federated learning with Flower 1.42.0: one server and five client
processes on one machine, talking over gRPC (127.0.0.1). Raw flows never
leave a client; only newly trained trees are sent.

This code lives under `scripts/research` and is separate from production.

## Method

- Data: the CICIDS arrays in `data/processed` (39 features, 5 classes),
  already scaled with ONE global RobustScaler.
- Partitions (`partition.py`): IID (random split) and non-IID (per-class
  Dirichlet, alpha = 0.5). Each client keeps 10% of its rows as local validation.
- Clients (`client.py`): XGBoost with the same hyperparameters as the central
  model, 5 local boosting rounds per federated round, inverse-frequency sample
  weights (switchable with `--no-weights`).
- Server (`server.py`): `FedXgbBagging` (trees are concatenated, not averaged),
  10 rounds. After every round the merged model is scored on the central test set.
- Comparison (`compare.py`): central model vs federated models on the same test set.
- Calibration (`calibrate.py`): per-class bias added to the summed tree scores,
  fitted on the VALIDATION set, test set scored once.

## Results (test set, one seed, one split)

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Central (weighted) | 0.9987 | 0.9269 |
| Federated IID, weighted | 0.9744 | 0.5858 |
| Federated non-IID, weighted | 0.9737 | 0.5914 |
| Federated IID, no weights | 0.9941 | 0.5717 |

After post-hoc bias calibration (macro-F1):

| Model | Before | After |
|---|---|---|
| Federated IID, weighted | 0.586 | 0.770 |
| Federated non-IID, weighted | 0.591 | 0.787 |
| Federated IID, no weights | 0.572 | 0.852 |

## Findings

1. Federated recall is high for every class, but rare-class PRECISION collapses
   with weighted clients (Probe 0.17, R2L 0.36, U2R 0.08 in IID).
2. Without weights the false positives disappear, but Probe and U2R recall fall
   to 0.000: the raw ensemble never predicts them.
3. Neither setting is calibrated. A bias per class fixes much of it
   (best: unweighted + calibration, macro-F1 0.852). The trees carry the signal;
   the summed scores were shifted.
4. Non-IID was about as good as IID here. Plausible reason (not tested): bagging
   keeps every client's trees, so a skewed client cannot overwrite the others.
5. Accuracy stops improving after about round 5.
6. U2R stays hard: after calibration, weighted models reach only 0.11 to 0.14
   precision and the unweighted one 0.33 recall.

## Limitations

- One seed, one split, one test set. No confidence intervals.
- Bias values are fitted on pooled validation data, which a real federation
  would not have. Treat calibrated numbers as an upper bound.
- Bias search is a coarse greedy grid (0.5 steps).
- One shared scaler; in real deployments each client would fit its own.
- Flower's `start_server` / `start_client` are deprecated in 1.42; the old API
  still works. A SuperLink/SuperNode version is future work.
- Tree count grows every round with no pruning; U2R recall drifted down
  slightly with more rounds in the IID run.
- The older `data/processed/xgboost_federated.joblib` predates this work and is
  not used in any comparison here.
- Probable cause of the precision collapse (summed weighted ensembles
  over-predict rare classes) is a hypothesis; the ablation supports the
  direction only.

## Reproduce

```cmd
python scripts\research\federated\partition.py --mode iid
python scripts\research\federated\partition.py --mode noniid --alpha 0.5
python scripts\research\federated\run_federated.py --mode iid --rounds 10
python scripts\research\federated\run_federated.py --mode noniid --rounds 10
python scripts\research\federated\run_federated.py --mode iid --no-weights
python scripts\research\federated\compare.py
python scripts\research\federated\calibrate.py
```