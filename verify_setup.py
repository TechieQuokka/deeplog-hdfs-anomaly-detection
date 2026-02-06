#!/usr/bin/env python3
"""
Verification script to check if the DeepLog implementation is ready to run.
"""

import os
import sys

def check_file_exists(filepath, description):
    """Check if a file exists."""
    exists = os.path.exists(filepath)
    status = "✓" if exists else "✗"
    print(f"{status} {description}: {filepath}")
    return exists

def check_directory_exists(dirpath, description):
    """Check if a directory exists."""
    exists = os.path.isdir(dirpath)
    status = "✓" if exists else "✗"
    print(f"{status} {description}: {dirpath}")
    return exists

def main():
    print("\n" + "=" * 80)
    print("DeepLog HDFS Implementation - Setup Verification")
    print("=" * 80)

    all_ok = True

    # Check source files
    print("\n📦 Source Files:")
    source_files = [
        ("src/__init__.py", "Package init"),
        ("src/config.py", "Configuration"),
        ("src/preprocessing.py", "Preprocessing"),
        ("src/dataset.py", "Dataset"),
        ("src/model.py", "Model"),
        ("src/train.py", "Training"),
        ("src/evaluate.py", "Evaluation"),
        ("src/utils.py", "Utilities"),
    ]
    for filepath, desc in source_files:
        if not check_file_exists(filepath, desc):
            all_ok = False

    # Check main files
    print("\n📄 Main Files:")
    main_files = [
        ("main.py", "Main pipeline"),
        ("requirements.txt", "Dependencies"),
        ("README.md", "User guide"),
    ]
    for filepath, desc in main_files:
        if not check_file_exists(filepath, desc):
            all_ok = False

    # Check documentation
    print("\n📚 Documentation:")
    doc_files = [
        ("docs/architecture.md", "Architecture doc"),
        ("docs/IMPLEMENTATION_SUMMARY.md", "Implementation summary"),
    ]
    for filepath, desc in doc_files:
        if not check_file_exists(filepath, desc):
            all_ok = False

    # Check directories
    print("\n📁 Directories:")
    directories = [
        ("src", "Source code"),
        ("data/processed", "Processed data"),
        ("checkpoints", "Model checkpoints"),
        ("results", "Results"),
        ("docs", "Documentation"),
    ]
    for dirpath, desc in directories:
        if not check_directory_exists(dirpath, desc):
            all_ok = False

    # Check data files (optional - may not exist yet)
    print("\n💾 Data Files (Optional - will be created):")
    data_root = "/mnt/e/Big Data/HDFS/HDFS_v1"
    data_files = [
        (f"{data_root}/preprocessed/Event_traces.csv", "Event traces"),
        (f"{data_root}/preprocessed/anomaly_label.csv", "Anomaly labels"),
    ]
    data_ok = True
    for filepath, desc in data_files:
        if not check_file_exists(filepath, desc):
            data_ok = False

    if not data_ok:
        print("\n⚠️  Data files not found. Update DATA_ROOT in src/config.py if needed.")

    # Check Python version
    print("\n🐍 Python Environment:")
    py_version = sys.version_info
    print(f"Python version: {py_version.major}.{py_version.minor}.{py_version.micro}")

    if py_version.major == 3 and py_version.minor == 11:
        print("✓ Python 3.11 detected (recommended)")
    elif py_version.major == 3 and py_version.minor >= 8:
        print("⚠️  Python 3.8+ detected (compatible but 3.11 recommended)")
    else:
        print("✗ Python version not compatible (3.8+ required)")
        all_ok = False

    # Check PyTorch (if installed)
    print("\n🔥 PyTorch:")
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} installed")
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA not available (will use CPU)")
    except ImportError:
        print("✗ PyTorch not installed (run: pip install -r requirements.txt)")
        all_ok = False

    # Final summary
    print("\n" + "=" * 80)
    if all_ok and data_ok:
        print("✅ All checks passed! Ready to run.")
        print("\nNext steps:")
        print("  1. python main.py --all        # Run complete pipeline")
        print("  2. python main.py --preprocess # Run preprocessing only")
    elif all_ok:
        print("✅ Implementation ready!")
        print("⚠️  Data files need to be configured.")
        print("\nNext steps:")
        print("  1. Update DATA_ROOT in src/config.py if needed")
        print("  2. python main.py --preprocess")
    else:
        print("✗ Some checks failed. Please fix the issues above.")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
