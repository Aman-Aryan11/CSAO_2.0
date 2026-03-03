# CSAO 2.0: Cart Super Add-On Recommendation System

End-to-end recommendation pipeline for food-delivery add-on prediction:
- synthetic data generation
- training data preparation
- candidate generation (item-to-item + collaborative filtering)
- candidate fusion
- ranking feature engineering
- baseline ranking (LightGBM, XGBoost)
- neural pairwise ranker with checkpoint resume
- diagnostics and error analysis

## 1) Repository Layout

- `data_pipeline/`
  - `config.py`: scale, paths, environment config
  - `generate_data.py`: raw synthetic data generation
  - `prepare_training_data.py`: interactions, implicit matrix, ranking dataset, splits
  - `utils.py`: parquet helpers, chunking, logging
  - `data/`
    - `raw/`
    - `processed/`
    - `candidates/`
    - `featured/`
- `src/candidate_generation/`
  - `item_similarity.py`
  - `collaborative_filtering.py`
  - `merge_candidates.py`
- `src/features/`
  - `build_ranking_features.py`
- `src/ranking/`
  - `ml_baselines.py`
  - `train_neural_ranker.py`
  - `cycle2_tuning_impact.py`
  - `cycle3_neural_pipeline.py`
  - `debug_neural_ranker.py`
- `src/analysis/`
  - `error_analysis.py`
- `notebooks/`
  - `EDA.ipynb`
- `output/`
  - `baseline_output/`
  - `error_analysis/`
  - `cycle2/`
  - `cycle3/`
- `models/`
  - `neural_ranker/`
  - `neural/`

## 2) Environment Setup

```bash
cd /Users/aman/Desktop/CSAO_2.0
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 3) Data Pipeline Execution Order

### Step A: Generate raw data

Development scale:
```bash
PIPELINE_ENV=development python /Users/aman/Desktop/CSAO_2.0/data_pipeline/generate_data.py
```

Production scale:
```bash
PIPELINE_ENV=production python /Users/aman/Desktop/CSAO_2.0/data_pipeline/generate_data.py
```

### Step B: Build processed interaction data

```bash
python /Users/aman/Desktop/CSAO_2.0/data_pipeline/prepare_training_data.py
```

### Step C: Candidate generation

```bash
python /Users/aman/Desktop/CSAO_2.0/src/candidate_generation/item_similarity.py
python /Users/aman/Desktop/CSAO_2.0/src/candidate_generation/collaborative_filtering.py
python /Users/aman/Desktop/CSAO_2.0/src/candidate_generation/merge_candidates.py
```

### Step D: Ranking feature engineering

```bash
python /Users/aman/Desktop/CSAO_2.0/src/features/build_ranking_features.py
```

Useful controls for large runs:
```bash
RANKING_FEATURES_BATCH_SIZE=750000 RANKING_FEATURES_MAX_BATCHES=0 python /Users/aman/Desktop/CSAO_2.0/src/features/build_ranking_features.py
```

### Step E: Baseline ranking models

```bash
cd /Users/aman/Desktop/CSAO_2.0/src/ranking
ML_BASELINE_SPREAD_SAMPLE=1 ML_BASELINE_DROP_LEAKY_FEATURES=1 ML_BASELINE_LEAK_CORR_THRESHOLD=0.995 python ml_baselines.py
```

### Step F: Error analysis

```bash
cd /Users/aman/Desktop/CSAO_2.0/src/analysis
ERROR_ANALYSIS_MAX_ROWS_VAL=2000000 ERROR_ANALYSIS_MAX_ROWS_TEST=2000000 python error_analysis.py
```

### Step G: Neural ranker (pairwise)

```bash
cd /Users/aman/Desktop/CSAO_2.0/src/ranking
python train_neural_ranker.py
```

Resume behavior:
- Uses `models/neural_ranker/checkpoints/latest_checkpoint.pt` when present.
- Saves per-epoch checkpoint and best model automatically.

## 4) EDA and Insight Notebook

Notebook path:
- `/Users/aman/Desktop/CSAO_2.0/notebooks/EDA.ipynb`

This notebook now includes:
- scale and quality diagnostics
- user/cart/temporal behavior
- target distribution analysis
- detailed insight summary
- featured split label sanity checks

Run with Jupyter:
```bash
jupyter notebook /Users/aman/Desktop/CSAO_2.0/notebooks/EDA.ipynb
```

## 5) Core Output Artifacts

### Raw / Processed
- `data_pipeline/data/raw/users.parquet`
- `data_pipeline/data/raw/restaurants.parquet`
- `data_pipeline/data/raw/items.parquet`
- `data_pipeline/data/raw/sessions.parquet`
- `data_pipeline/data/raw/session_items.parquet`
- `data_pipeline/data/processed/interactions.parquet`
- `data_pipeline/data/processed/user_item_matrix.parquet`
- `data_pipeline/data/processed/ranking_dataset.parquet`

### Candidates
- `data_pipeline/data/candidates/item_similarity.parquet`
- `data_pipeline/data/candidates/cf_candidates.parquet`
- `data_pipeline/data/candidates/candidates_merged.parquet`

### Featured ranking datasets
- `data_pipeline/data/featured/train_ranking_features.parquet`
- `data_pipeline/data/featured/val_ranking_features.parquet`
- `data_pipeline/data/featured/test_ranking_features.parquet`

### Baseline outputs
- `output/baseline_output/evaluation_summary.json`
- `output/baseline_output/evaluation_summary.csv`
- `output/baseline_output/feature_importance_lightgbm.csv`
- `output/baseline_output/feature_importance_xgboost.csv`
- `output/baseline_output/{train,val,test}_predictions.parquet`

### Neural outputs
- `models/neural_ranker/best_model.pt`
- `models/neural_ranker/checkpoints/epoch_*.pt`
- `models/neural_ranker/checkpoints/latest_checkpoint.pt`
- `models/neural_ranker/training_log.json`

### Diagnostics
- `output/error_analysis/summary_report.json`
- `output/error_analysis/segment_metrics.csv`
- `output/error_analysis/feature_importance.csv`
- `reports/debug/*.json`

## 6) Production-Safety Checks (Recommended)

Before every training run:
1. Verify label variance in train/val/test.
2. Verify each split has sessions with positives.
3. Check no leaky target proxies in feature set.
4. Confirm split is time-based and non-overlapping by `session_id`.

Quick check example:
```bash
/Users/aman/Desktop/CSAO_2.0/.venv/bin/python - << 'PY'
import pandas as pd
from pathlib import Path
base=Path('/Users/aman/Desktop/CSAO_2.0/data_pipeline/data/featured')
for n in ['train_ranking_features.parquet','val_ranking_features.parquet','test_ranking_features.parquet']:
    df=pd.read_parquet(base/n,columns=['session_id','label'])
    g=df.groupby('session_id')['label'].sum()
    print(n, 'rows=',len(df), 'pos=',int(df['label'].sum()), 'pos_rate=',float(df['label'].mean()), 'sessions_with_pos=',int((g>0).sum()))
PY
```

## 7) Scalability Notes

- Major scripts are chunk-based and parquet-oriented.
- Candidate merge and feature build are the heaviest stages at production scale.
- Resume checkpoints are implemented in neural training and baseline artifact tracking.
- Use row caps / env flags during debugging to iterate quickly.

## 8) Known Risks and Mitigations

- Risk: label collapse in featured val/test.
  - Mitigation: pre-train assertions + processed fallback + EDA sanity cell.
- Risk: candidate explosion and memory pressure.
  - Mitigation: top-k trimming, chunk writes, row caps for offline experiments.
- Risk: misleading metrics from leakage.
  - Mitigation: leaky-feature drops and correlation threshold checks.

## 9) Suggested Submission Package

Include:
1. `notebooks/EDA.ipynb`
2. model scripts from `src/`
3. evaluation outputs from `output/baseline_output/`, `output/cycle2/`, `output/cycle3/`, `output/error_analysis/`
4. this `README.md`

