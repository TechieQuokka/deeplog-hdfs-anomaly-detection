# DeepLog HDFS Anomaly Detection - Reproduction Study

Exact reproduction of the DeepLog paper's HDFS log anomaly detection experiment.

**Paper**: Min Du, Feifei Li, Guineng Zheng, Vivek Srikumar. "DeepLog: Anomaly Detection and Diagnosis from System Logs through Deep Learning." CCS 2017.

## 📊 Target Performance (Paper Results)

| Metric | Target Value |
|--------|-------------|
| False Positives (FP) | 833 |
| False Negatives (FN) | 619 |
| Precision | ~95.1% |
| Recall | ~96.4% |
| **F-measure** | **96%** |

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
├── data/
│   └── processed/          # Preprocessed sequences (generated)
├── checkpoints/            # Model checkpoints (generated)
├── results/                # Evaluation results (generated)
├── docs/
│   └── architecture.md     # System architecture documentation
├── main.py                 # Main execution pipeline
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Create virtual environment (recommended)
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Setup

This project expects HDFS data at: `/mnt/e/Big Data/HDFS/HDFS_v1`

The preprocessed data should contain:
- `Event_traces.csv` - Log event sequences by block_id
- `anomaly_label.csv` - Ground truth labels

If your data is in a different location, update `DATA_ROOT` in `src/config.py`.

### 3. Run Complete Pipeline

```bash
# Run all stages: preprocessing → training → evaluation
python main.py --all
```

### 4. Run Individual Stages

```bash
# Stage 1: Data preprocessing
python main.py --preprocess

# Stage 2: Model training
python main.py --train

# Stage 3: Model evaluation
python main.py --evaluate
```

## 📋 Detailed Usage

### Data Preprocessing

The preprocessing pipeline:
1. Loads `Event_traces.csv` and `anomaly_label.csv`
2. Splits data according to paper specification:
   - **Training**: First 100,000 log entries → 4,855 normal sessions
   - **Testing**: Remaining → 553,366 normal + 16,838 abnormal sessions
3. Generates sliding windows of size h=10
4. Saves processed sequences to pickle files

```bash
python main.py --preprocess
```

Output:
- `data/processed/train_sequences.pkl`
- `data/processed/test_sequences.pkl`

### Model Training

Training configuration (matching paper):
- **Window size (h)**: 10
- **LSTM layers (L)**: 2
- **Hidden size (α)**: 64
- **Embedding dim**: 64
- **Batch size**: 128
- **Loss function**: CrossEntropyLoss
- **Optimizer**: Adam

```bash
python main.py --train
```

The trainer includes:
- Early stopping with patience=10
- Learning rate scheduling
- Checkpoint saving
- Validation monitoring

Output:
- `checkpoints/best_model.pth` - Best model checkpoint

### Model Evaluation

Evaluation uses the **top-g prediction strategy** from the paper:
- **g = 9**: Predict top-9 most probable log keys
- If actual key is NOT in top-9 → mark as anomaly
- Aggregate to session level: session is abnormal if ANY key is anomaly

```bash
python main.py --evaluate
```

Output:
- `results/metrics.json` - Detailed evaluation metrics
- Console output with comparison to paper targets

### Testing Individual Components

```bash
# Test data preprocessing
cd src && python preprocessing.py

# Test dataset loading
cd src && python dataset.py

# Test model architecture
cd src && python model.py

# Test utilities
cd src && python utils.py
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
```

### Data Paths
```python
DATA_ROOT = "/mnt/e/Big Data/HDFS/HDFS_v1"
EVENT_TRACES = f"{DATA_ROOT}/preprocessed/Event_traces.csv"
ANOMALY_LABEL = f"{DATA_ROOT}/preprocessed/anomaly_label.csv"
```

## 📊 Expected Results

After running the complete pipeline, you should achieve metrics close to the paper:

```
Evaluation Results
================================================================================
False Positives (FP): ~833
False Negatives (FN): ~619
Precision: ~95.1%
Recall: ~96.4%
F-measure: ~96%
```

## 🐛 Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size in src/config.py
BATCH_SIZE = 64  # or 32
```

### Data Files Not Found
```bash
# Check data path in src/config.py
DATA_ROOT = "/path/to/your/HDFS_v1"
```

### Preprocessed Data Missing
```bash
# Run preprocessing first
python main.py --preprocess
```

### Model Checkpoint Not Found
```bash
# Train model first
python main.py --train
```

## 📖 Additional Documentation

- **System Architecture**: See `docs/architecture.md` for detailed system design
- **Paper Reference**: Min Du et al., CCS 2017
- **Dataset**: HDFS_v1 from LogHub

## 🔬 Reproduction Notes

This implementation follows the DeepLog paper as closely as possible:

**Exact Paper Specifications**:
- ✅ Window size h=10
- ✅ LSTM layers L=2
- ✅ Hidden units α=64
- ✅ Number of log keys n=29
- ✅ Top-g prediction with g=9
- ✅ Training data: first 100k log entries
- ✅ Session-level anomaly aggregation

**Implementation Choices** (not specified in paper):
- Embedding dimension: 64 (common practice)
- Optimizer: Adam (standard for LSTM)
- Learning rate: 0.001 (may need tuning)
- Batch size: 128 (reasonable default)

## 📝 Citation

If you use this reproduction in your research, please cite the original paper:

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

## 🤝 Contributing

This is a reproduction study. For issues or improvements, please open an issue or pull request.

---

**Project Status**: ✅ Implementation Complete

**Last Updated**: 2026-02-06
