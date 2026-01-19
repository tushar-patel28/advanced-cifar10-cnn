# CIFAR-10 CNN Setup Guide

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/tushar-patel28/advanced-cifar10-cnn.git
cd advanced-cifar10-cnn
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Fix SSL Certificate (Mac users)
```bash
/Applications/Python\ 3.12/Install\ Certificates.command
```

Or add to notebook:
```python
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
```

### 5. Run the Notebook
```bash
jupyter notebook cifar10_advanced_cnn.ipynb
```

## System Requirements

- **Python**: 3.8+
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 2GB free space
- **GPU**: Optional (CPU training works but slower)

## Training Time

- **With GPU**: 30-40 minutes
- **With CPU**: 2-3 hours

## Expected Results

- **Standard Accuracy**: 92-93%
- **With TTA**: 93-94%

## Troubleshooting

### SSL Certificate Error
Run: `/Applications/Python\ 3.12/Install\ Certificates.command`

### Out of Memory
Reduce batch_size from 128 to 64 in the training cell

### Slow Training
- Increase batch_size to 256
- Reduce epochs to 50
- Remove some data augmentation

## Files

- `cifar10_advanced_cnn.ipynb` - Main training notebook
- `CIFAR10_Performance_Report.docx` - Detailed performance analysis
- `requirements.txt` - Python dependencies
- `.gitignore` - Git ignore rules

## Contact

Tushar Patel - [@tushar-patel28](https://github.com/tushar-patel28)
