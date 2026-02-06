# DeepLog HDFS Anomaly Detection - Reproduction Study

Reproduction of the DeepLog paper's HDFS log anomaly detection experiment.

**Paper**: Min Du, Feifei Li, Guineng Zheng, Vivek Srikumar. "DeepLog: Anomaly Detection and Diagnosis from System Logs through Deep Learning." CCS 2017.

## 📊 Results

### Paper vs Reproduction (g=9)

| Metric | Paper | Ours | Gap |
|--------|-------|------|-----|
| False Positives | 833 | 6,417 | +5,584 |
| False Negatives | 619 | 1,326 | +707 |
| Precision | 95.1% | 59.2% | -35.9pp |
| Recall | 96.4% | 87.5% | -8.9pp |
| **F1-Score** | **96.0%** | **70.7%** | **-25.3pp** |

### Top-g Sensitivity (best result at g=13)

| Metric | g=9 | g=13 (best F1) |
|--------|-----|-----------------|
| FP | 6,417 | 2,252 |
| FN | 1,326 | 2,307 |
| Precision | 59.2% | 78.7% |
| Recall | 87.5% | 78.3% |
| F1 | 70.7% | 78.5% |

### Data Split Verification

| Split | Paper | Ours |
|-------|-------|------|
| Train (Normal) | 4,855 | 4,855 |
| Test (Normal) | 553,366 | 553,368 |
| Test (Anomaly) | 16,838 | 10,647 |

### Gap Analysis

- **FP가 7.7배 높음**: session-level OR aggregation이 sequence-level FP를 증폭시킴
- **FP 세션 특성**: 대부분의 FP 세션은 소수의 시퀀스만 anomaly로 판정됨
- **Rank 분포**: 정상 시퀀스의 대다수가 rank 0~1에 위치하나, 긴 꼬리 분포가 FP를 유발
- **최적 g=13**: g를 높이면 FP 감소/FN 증가 trade-off, F1 최대 78.5%

> 상세 시각화 분석은 `notebooks/analysis.ipynb` 참조.

## 🏗️ Project Structure

```
deeplog-hdfs-reproduction/
├── src/
│   ├── config.py           # Configuration and hyperparameters
│   ├── preprocessing.py    # Data preprocessing pipeline
│   ├── dataset.py          # PyTorch Dataset and DataLoader
│   ├── model.py            # DeepLog LSTM model
│   ├── train.py            # Training pipeline
│   ├── evaluate.py         # Evaluation with top-g prediction
│   └── utils.py            # Utility functions
├── notebooks/
│   └── analysis.ipynb      # Results visualization and analysis
├── data/
│   └── processed/          # Preprocessed sequences (generated)
├── checkpoints/            # Model checkpoints (generated)
├── results/                # Evaluation results (generated)
├── docs/
│   └── architecture.md     # System architecture documentation
├── main.py                 # Main execution pipeline
├── requirements.txt        # Python dependencies
└── README.md
```

## 🚀 Quick Start

### 1. Installation

```bash
python3.11 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Data Setup

Required files (HDFS_v1 from [LogHub](https://github.com/logpai/loghub)):
- `Event_traces.csv` - Log event sequences by block_id
- `anomaly_label.csv` - Ground truth labels

Update `DATA_ROOT` in `src/config.py` to point to your data directory.

### 3. Run Complete Pipeline

```bash
# Run all stages: preprocessing → training → evaluation
python main.py --all
```

### 4. Run Individual Stages

```bash
python main.py --preprocess   # Data preprocessing
python main.py --train        # Model training
python main.py --evaluate     # Model evaluation
```

## 🔧 Configuration

All hyperparameters are centralized in `src/config.py`:

### Model Hyperparameters (Paper Specification)
```python
NUM_CLASSES = 29        # n: Number of distinct log keys
WINDOW_SIZE = 10        # h: History window size
NUM_LAYERS = 2          # L: Number of LSTM layers
HIDDEN_SIZE = 64        # α: LSTM hidden units
TOP_G = 9               # g: Top-g prediction threshold
```

### Training Configuration
```python
BATCH_SIZE = 128
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
EMBEDDING_DIM = 64
```

## 🔬 Reproduction Notes

### Paper Specifications Implemented
- Window size h=10
- LSTM layers L=2, hidden units α=64
- Number of log keys n=29
- Top-g prediction with g=9
- Session-level anomaly aggregation (OR)

### Data Split Fix (Critical)

논문에서 "first 100,000 log entries"로 학습 데이터를 정의합니다. 이 재현에서 발견한 핵심 사항:

- `Event_traces.csv`의 행 순서는 `HDFS.log`에서의 최초 등장 순서와 일치 (100/100 검증)
- 올바른 분할: 누적 **정상 세션 수**가 4,855에 도달할 때까지의 행을 학습 데이터로 사용
- 잘못된 분할: 누적 **이벤트 수**로 100,000을 기준으로 나누면 3,684 세션 (부족)

### Remaining Performance Gap

논문 대비 FP가 크게 높은 원인 후보:
1. Session-level OR aggregation이 sequence-level FP를 증폭
2. 논문에 기술되지 않은 추가 기법이 있을 가능성 (예: threshold tuning, post-processing)
3. 모델 하이퍼파라미터 또는 학습 데이터 커버리지 차이

### Training Details
- Best model epoch: 33 (val loss: 0.1799)
- Training sequences: 78,093
- Test sequences: 5,389,938
- Trainable parameters: 70,301

## 📝 Citation

```bibtex
@inproceedings{du2017deeplog,
  title={DeepLog: Anomaly Detection and Diagnosis from System Logs through Deep Learning},
  author={Du, Min and Li, Feifei and Zheng, Guineng and Srikumar, Vivek},
  booktitle={Proceedings of the 2017 ACM SIGSAC Conference on Computer and Communications Security},
  pages={1285--1298},
  year={2017}
}
```

## 📄 License

This is a reproduction study for research and educational purposes.

---

**Last Updated**: 2026-02-07
