# DeepLog HDFS Anomaly Detection - System Architecture

## 1. Overview

This document describes the architecture for reproducing the DeepLog paper's HDFS log anomaly detection experiment (Min Du et al., CCS 2017).

### 1.1 Project Goal
Exact reproduction of DeepLog's HDFS experiment to achieve the reported performance:
- **Target F-measure**: 96%
- **Target Precision**: ~95.1%
- **Target Recall**: ~96.4%

### 1.2 Technology Stack
- **Language**: Python 3.11
- **Deep Learning Framework**: PyTorch
- **Data Processing**: NumPy, Pandas
- **Metrics**: scikit-learn

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DeepLog HDFS System                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────┐  │
│  │   Raw HDFS  │───▶│ Preprocessing│───▶│   Processed   │  │
│  │    Data     │    │   Pipeline   │    │   Sequences   │  │
│  └─────────────┘    └──────────────┘    └───────────────┘  │
│         │                   │                     │          │
│         │                   ▼                     │          │
│         │          ┌──────────────┐               │          │
│         │          │  Session     │               │          │
│         │          │  Splitter    │               │          │
│         │          └──────────────┘               │          │
│         │                   │                     │          │
│         ▼                   ▼                     ▼          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              PyTorch Dataset & DataLoader           │   │
│  └─────────────────────────────────────────────────────┘   │
│                             │                                │
│                             ▼                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │            DeepLog LSTM Model                        │   │
│  │  ┌────────────────────────────────────────────┐    │   │
│  │  │  Embedding Layer (29 log keys)             │    │   │
│  │  └────────────────────────────────────────────┘    │   │
│  │  ┌────────────────────────────────────────────┐    │   │
│  │  │  2-Layer LSTM (hidden_size=64)             │    │   │
│  │  └────────────────────────────────────────────┘    │   │
│  │  ┌────────────────────────────────────────────┐    │   │
│  │  │  Fully Connected Output (29 classes)       │    │   │
│  │  └────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────┘   │
│                             │                                │
│         ┌───────────────────┴───────────────────┐           │
│         ▼                                       ▼           │
│  ┌─────────────┐                        ┌─────────────┐    │
│  │   Training  │                        │  Evaluation │    │
│  │   Pipeline  │                        │   Pipeline  │    │
│  └─────────────┘                        └─────────────┘    │
│         │                                       │           │
│         ▼                                       ▼           │
│  ┌─────────────┐                        ┌─────────────┐    │
│  │ Checkpoints │                        │   Metrics   │    │
│  └─────────────┘                        └─────────────┘    │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. Component Specifications

### 3.1 Data Preprocessing Component

**Purpose**: Transform raw HDFS log data into training/testing sequences

**Input**:
- `Event_traces.csv` - Pre-parsed log event sequences by block_id
- `anomaly_label.csv` - Ground truth labels for sessions

**Output**:
- Training sequences: `train_sequences.pkl`
- Testing sequences: `test_sequences.pkl`

**Processing Steps**:
1. Load event traces grouped by block_id (session identifier)
2. Split sessions into train/test following paper specification:
   - **Train**: First 100,000 log entries → 4,855 normal sessions
   - **Test**: Remaining → 553,366 normal + 16,838 abnormal sessions
3. Generate sliding windows of size h=10
4. Create (input_sequence, label) pairs
   - Input: [k_{t-10}, k_{t-9}, ..., k_{t-1}]
   - Label: k_t

**Key Parameters**:
- `WINDOW_SIZE = 10` (h)
- `TRAIN_RATIO < 0.01` (less than 1% of total logs)

---

### 3.2 Dataset Component

**Purpose**: PyTorch-compatible dataset for efficient batch loading

**Class**: `HDFSDataset`

**Responsibilities**:
- Load preprocessed sequences from disk
- Convert log keys to integer indices (0-28 for 29 log keys)
- Provide `__getitem__` and `__len__` for DataLoader compatibility

**DataLoader Configuration**:
```python
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4
)
```

---

### 3.3 Model Component

**Purpose**: DeepLog LSTM neural network for log sequence prediction

**Class**: `DeepLogLSTM`

**Architecture**:

```
Input: (batch_size, window_size=10)
    │
    ▼
Embedding Layer
    ├─ Input dim: 29 (number of log keys)
    ├─ Output dim: 64 (embedding size)
    └─ Output: (batch_size, 10, 64)
    │
    ▼
2-Layer LSTM
    ├─ Input size: 64
    ├─ Hidden size: 64
    ├─ Num layers: 2
    └─ Output: (batch_size, 10, 64)
    │
    ▼
Take Last Hidden State: [:, -1, :]
    │ (batch_size, 64)
    ▼
Fully Connected Layer
    ├─ Input: 64
    ├─ Output: 29 (number of classes)
    └─ Output: (batch_size, 29)
    │
    ▼
Output: Probability distribution over 29 log keys
```

**Forward Pass**:
```python
def forward(self, x):
    # x: (batch, seq_len=10)
    embedded = self.embedding(x)      # (batch, 10, 64)
    lstm_out, _ = self.lstm(embedded) # (batch, 10, 64)
    last_hidden = lstm_out[:, -1, :]  # (batch, 64)
    output = self.fc(last_hidden)     # (batch, 29)
    return output
```

**Paper Parameters**:
- L = 2 (number of LSTM layers)
- α = 64 (hidden units per LSTM block)
- n = 29 (number of distinct log keys)

---

### 3.4 Training Component

**Purpose**: Train DeepLog model to predict next log key from history

**Training Configuration**:
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam (learning rate to be determined)
- **Epochs**: To be tuned
- **Device**: CUDA if available, else CPU

**Training Loop**:
```
For each epoch:
    For each batch:
        1. Forward pass: predictions = model(sequences)
        2. Compute loss: loss = criterion(predictions, labels)
        3. Backward pass: loss.backward()
        4. Update weights: optimizer.step()

    Save checkpoint if validation improves
```

**Checkpoint Strategy**:
- Save best model based on validation loss
- Store model state_dict, optimizer state, epoch number

---

### 3.5 Evaluation Component

**Purpose**: Evaluate model using top-g prediction strategy and compute metrics

**Top-g Prediction Strategy**:
1. For each test sequence, get model prediction probabilities
2. Select top-g (g=9) most probable log keys
3. If actual log key is in top-g → **Normal**
4. If actual log key is NOT in top-g → **Anomaly**

**Session-Level Detection**:
- A session (block_id trace) is **abnormal** if ANY log key in that session is detected as anomaly
- This matches paper's evaluation methodology

**Metrics Calculation**:
```python
TP = True Positives (correctly detected abnormal sessions)
FP = False Positives (normal sessions flagged as abnormal)
FN = False Negatives (abnormal sessions missed)
TN = True Negatives (correctly identified normal sessions)

Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F-measure = 2 * (Precision * Recall) / (Precision + Recall)
```

**Target Metrics** (Paper Results):
- FP: 833
- FN: 619
- Precision: ~95.1%
- Recall: ~96.4%
- **F-measure: 96%**

---

## 4. Data Flow

### 4.1 Training Phase

```
Raw Data (Event_traces.csv)
    │
    ▼
[Preprocessing]
    ├─ Load sessions by block_id
    ├─ Split train/test (paper specification)
    └─ Generate sliding windows (h=10)
    │
    ▼
Training Sequences (pkl)
    │
    ▼
[HDFSDataset + DataLoader]
    │
    ▼
[DeepLog LSTM Model]
    ├─ Embedding → LSTM → FC
    └─ Predict next log key
    │
    ▼
[Training Loop]
    ├─ CrossEntropyLoss
    ├─ Adam optimizer
    └─ Checkpoint saving
    │
    ▼
Trained Model (best_model.pth)
```

### 4.2 Evaluation Phase

```
Test Sequences (pkl) + Anomaly Labels
    │
    ▼
[HDFSDataset + DataLoader]
    │
    ▼
[Trained DeepLog Model]
    │
    ▼
[Top-9 Prediction]
    ├─ Get probability distribution
    ├─ Select top-9 keys
    └─ Compare with actual key
    │
    ▼
[Key-Level Anomaly Detection]
    │
    ▼
[Session-Level Aggregation]
    ├─ Group detections by block_id
    └─ Mark session if ANY key is anomaly
    │
    ▼
[Metrics Computation]
    ├─ Compare with ground truth labels
    ├─ Calculate TP, FP, FN, TN
    └─ Compute Precision, Recall, F-measure
    │
    ▼
Results (metrics.json)
```

---

## 5. Configuration Management

All system parameters are centralized in `src/config.py`:

### 5.1 Data Paths
- Raw HDFS data location
- Preprocessed data directory
- Output directories for processed data

### 5.2 Model Hyperparameters
- `NUM_CLASSES = 29` - Number of distinct log keys
- `WINDOW_SIZE = 10` - History window size (h)
- `NUM_LAYERS = 2` - LSTM layers (L)
- `HIDDEN_SIZE = 64` - Hidden units (α)
- `TOP_G = 9` - Top-g prediction threshold

### 5.3 Training Parameters
- Batch size
- Learning rate
- Number of epochs
- Device (CPU/GPU)

### 5.4 Paths
- Checkpoint directory
- Results directory
- Processed data directory

---

## 6. Execution Flow

### 6.1 Main Pipeline (`main.py`)

```python
1. Load configuration from config.py
2. Preprocess data if not already done
3. Load datasets and create data loaders
4. Initialize DeepLog model
5. Train model with checkpointing
6. Load best checkpoint
7. Evaluate on test data
8. Save results (FP, FN, Precision, Recall, F-measure)
9. Compare with paper targets
```

### 6.2 Directory Structure After Execution

```
deeplog-hdfs-reproduction/
├── data/
│   └── processed/
│       ├── train_sequences.pkl  ✓ Created
│       └── test_sequences.pkl   ✓ Created
├── checkpoints/
│   └── best_model.pth           ✓ Created
└── results/
    └── metrics.json             ✓ Created
```

---

## 7. Design Principles

### 7.1 Reproducibility
- **Exact Paper Specification**: All hyperparameters match paper
- **Deterministic Processing**: Fixed random seeds for reproducibility
- **Version Control**: Document all library versions

### 7.2 Modularity
- **Separation of Concerns**: Each component has single responsibility
- **Loose Coupling**: Components communicate through well-defined interfaces
- **Reusability**: Components can be tested and modified independently

### 7.3 Maintainability
- **Clear Documentation**: Architecture, API, and implementation details
- **Consistent Naming**: Follow Python conventions
- **Error Handling**: Graceful failures with informative messages

### 7.4 Efficiency
- **PyTorch DataLoader**: Efficient batch processing
- **GPU Support**: Automatic CUDA detection and usage
- **Checkpointing**: Resume training from saved state

---

## 8. Validation Strategy

### 8.1 Component Testing
- Verify data preprocessing outputs correct sequence format
- Test dataset returns proper tensor shapes
- Validate model architecture matches paper specification

### 8.2 Integration Testing
- End-to-end pipeline execution
- Verify data flow through all components
- Check output formats at each stage

### 8.3 Performance Validation
- Compare final metrics with paper targets:
  - **F-measure**: 96% (±1% tolerance)
  - **FP**: ~833 (session-level)
  - **FN**: ~619 (session-level)

---

## 9. Future Enhancements (Out of Scope)

The following are NOT part of the reproduction but could be explored later:
- Parameter value anomaly detection model
- Workflow model for diagnosis
- Online incremental update mechanism
- Real-time streaming detection
- Multi-dataset experiments (OpenStack, Blue Gene/L)

---

## 10. References

1. Min Du, Feifei Li, Guineng Zheng, Vivek Srikumar. "DeepLog: Anomaly Detection and Diagnosis from System Logs through Deep Learning." CCS 2017.

2. HDFS Dataset: https://github.com/logpai/loghub (HDFS_v1)

3. PyTorch Documentation: https://pytorch.org/docs/

---

**Document Version**: 1.0
**Last Updated**: 2026-02-06
**Status**: Design Phase
