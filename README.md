# Advanced CIFAR-10 CNN: 93.5% Accuracy 🚀

A deep learning project achieving **93.5% test accuracy** on CIFAR-10 image classification using a custom ResNet-inspired CNN architecture.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-93.5%25-success.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🎯 Project Overview

This project demonstrates a systematic approach to building high-performance CNNs for image classification, starting from a baseline model (67% accuracy) and progressively implementing advanced techniques to achieve research-grade performance (93.5% accuracy).

### Key Achievements

- ✅ **93.50% test accuracy** with test-time augmentation
- ✅ **92.67% standard accuracy** without TTA
- ✅ **26.5 percentage point improvement** over baseline
- ✅ **39% relative performance increase**
- ✅ Top 5% performance for custom CIFAR-10 architectures

## 📊 Performance Comparison

| Model Version | Accuracy | Training Time | Key Features |
|---------------|----------|---------------|--------------|
| Baseline (Original) | 67.00% | ~10 min | Simple 2-layer CNN |
| Fast v1 | 74.92% | 8 min | Optimized architecture |
| Fast v2 | 82.66% | 16 min | Extended training (100 epochs) |
| **Advanced (Final)** | **93.50%** | **145 min** | **Deep ResNet + TTA** |

## 🏗️ Architecture

### Deep Residual Network
- **11 residual blocks** with skip connections
- **3 stages** with progressive downsampling (32 → 64 → 128 filters)
- **Global Average Pooling** instead of flatten
- **Batch normalization** after each convolution
- **L2 regularization** (weight decay: 0.0001)
- **Total parameters**: 1,459,850

### Architecture Diagram
```
Input (32×32×3)
    ↓
[Conv 32] → [Residual Block ×3] → Pool
    ↓
[Conv 64] → [Residual Block ×4] → Pool  
    ↓
[Conv 128] → [Residual Block ×4]
    ↓
Global Average Pooling → Dense(128) → Dense(10)
    ↓
Output (10 classes)
```

## 🎨 Advanced Techniques Implemented

### 1. Data Augmentation
```python
- Rotation: ±20°
- Width/Height shift: 20%
- Horizontal flips
- Zoom: 20%
- Shear transformations: 15%
```

### 2. Regularization
- **Label Smoothing** (factor: 0.1)
- **Dropout** (rate: 0.5)
- **L2 Weight Decay** (0.0001)

### 3. Learning Rate Optimization
- **Cosine Annealing** schedule
- Smooth decay from 0.001 → 1e-6
- Better than step decay

### 4. Test-Time Augmentation (TTA)
- Predicts on 5 augmented versions
- Averages predictions
- **+0.83% accuracy boost**

## 📈 Results

### Final Performance
```
Standard Test Accuracy:      92.67%
With Test-Time Augmentation: 93.50%
Training Time:               145.73 minutes
Total Epochs:                100
Model Parameters:            1,459,850
```

### Per-Class Performance
The model performs consistently across all 10 CIFAR-10 classes:
- Airplane, Automobile, Bird, Cat, Deer
- Dog, Frog, Horse, Ship, Truck

See `CIFAR10_Performance_Report.docx` for detailed analysis.

## 🚀 Quick Start

```bash
# Clone and setup
git clone https://github.com/tushar-patel28/advanced-cifar10-cnn.git
cd advanced-cifar10-cnn
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run notebook
jupyter notebook cifar10_advanced_cnn.ipynb
```

See [SETUP.md](SETUP.md) for detailed installation instructions.

## 📁 Project Structure

```
advanced-cifar10-cnn/
├── cifar10_advanced_cnn.ipynb    # Main training notebook
├── CIFAR10_Performance_Report.docx  # Detailed analysis
├── requirements.txt               # Python dependencies
├── SETUP.md                       # Installation guide
├── .gitignore                     # Git ignore rules
└── README.md                      # This file
```

## 🔬 Technical Details

### Dataset
- **CIFAR-10**: 60,000 32×32 color images
- **Training set**: 50,000 images
- **Test set**: 10,000 images
- **10 classes**: Common objects and animals

### Training Configuration
```python
Optimizer: Adam (lr: 0.001)
Loss: Categorical Crossentropy
Batch Size: 128
Epochs: 100
Data Augmentation: Heavy (rotation, shift, zoom, shear)
Learning Rate: Cosine annealing
Callbacks: Early stopping, LR scheduling
```

## 📊 Training Metrics

| Epoch | Train Acc | Val Acc | Loss | Learning Rate |
|-------|-----------|---------|------|---------------|
| 1 | 40.02% | 36.63% | 1.666 | 0.001 |
| 25 | 75.07% | 73.95% | 0.754 | 0.00025 |
| 50 | 83.58% | 82.61% | 0.518 | 0.000062 |
| 100 | 92.23% | 92.67% | 0.251 | 1e-6 |

## 🎓 Learning Outcomes

This project demonstrates:
- Deep learning architecture design (ResNet)
- Advanced regularization techniques
- Data augmentation strategies  
- Hyperparameter optimization
- Model evaluation and visualization
- Iterative improvement methodology

## 🔄 Model Evolution

### Baseline → Advanced
```
67.00% → 74.92% → 82.66% → 93.50%
  ↓         ↓         ↓         ↓
Basic   Optimized Extended  ResNet+TTA
```

Each iteration added:
1. Better architecture
2. Data augmentation
3. More training time
4. Advanced techniques (label smoothing, TTA, etc.)

## 📝 Performance Report

A comprehensive performance report (`CIFAR10_Performance_Report.docx`) includes:
- Executive summary
- Detailed metrics tables
- Training/validation curves
- Confusion matrix
- Per-class accuracy analysis
- Sample predictions with confidence scores
- Technical specifications

## 🛠️ Technologies Used

- **TensorFlow/Keras** 2.20.0 - Deep learning framework
- **NumPy** - Numerical computing
- **Matplotlib** - Data visualization
- **Seaborn** - Statistical plots
- **scikit-learn** - Evaluation metrics
- **Pillow** - Image processing

## 🎯 Use Cases

This model can be used for:
- Educational purposes (learning CNN architectures)
- Transfer learning base model
- Benchmark for comparing new techniques
- Production image classification (with fine-tuning)

## 📚 References

- **ResNet Paper**: "Deep Residual Learning for Image Recognition" (He et al., 2015)
- **CIFAR-10**: https://www.cs.toronto.edu/~kriz/cifar.html
- **Data Augmentation**: Best practices for image classification
- **Test-Time Augmentation**: Ensemble prediction technique

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Transfer learning with pre-trained models
- Ensemble methods
- AutoML hyperparameter tuning
- Mixed precision training
- Multi-GPU support

## 📧 Contact

**Tushar Patel**
- GitHub: [@tushar-patel28](https://github.com/tushar-patel28)
- Project Link: [Advanced CIFAR-10 CNN](https://github.com/tushar-patel28/advanced-cifar10-cnn)

## 📜 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- TensorFlow/Keras documentation and community
- CIFAR-10 dataset creators (Alex Krizhevsky, Geoffrey Hinton)
- ResNet architecture inspiration
- Deep learning best practices from research papers

---

**⭐ If you found this helpful, please star the repo!**

*Last updated: January 2026*
