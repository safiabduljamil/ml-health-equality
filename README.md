# Student Health Score Prediction with Fairness Analysis

A machine learning project that predicts student health scores using a stacking ensemble approach while ensuring fairness across demographic subgroups (Gender × Sleep Category).

## 🎯 Research Question

**Does the classification model achieve equal predictive performance for Overall Health Score across different subgroups (Gender × Sleep Category)?**

## 📊 Project Overview

This project implements a complete machine learning pipeline with:

- **Advanced ensemble learning** using Random Forest, SVM, MLP, and Histogram Gradient Boosting
- **Stacking meta-model** for improved predictions
- **Fairness analysis** across demographic subgroups
- **SMOTE oversampling** to balance underrepresented groups
- **Per-subgroup threshold tuning** for optimal F1-scores
- **Probability calibration** for reliable predictions

## 🚀 Key Features

- **Chronological subgroup-aware split** (60% train / 10% validation / 30% test)
- **Rolling window features** (30-minute Heart Rate & Physical Activity statistics)
- **Class-balanced models** with sample weighting
- **Dual fairness evaluation**: strict and relaxed criteria
- **Comprehensive visualizations** of subgroup performance

## 📈 Results

### Overall Performance
- **Accuracy**: 0.62
- **F1-Score**: 0.75
- **ROC-AUC**: 0.48

### Fairness Analysis
✅ **Achieved reasonable equality for larger subgroups (≥6 samples)**
- Mean F1-Score: 0.85
- Std F1-Score: 0.056 (< 0.10 threshold)
- Groups: Male_Normal (F1=0.91), Female_Normal (F1=0.80), Other_Normal (F1=0.83)

⚠️ Small subgroups (<6 samples) show higher variance due to limited data, which is expected in real-world scenarios.

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd Second_project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📝 Usage

### Run the Complete Pipeline

```bash
python3 health_prediction_pipeline.py
```

This will:
1. Load and prepare the student health data
2. Create rolling window features
3. Perform chronological subgroup-aware split
4. Apply SMOTE for class balancing
5. Train base models (RF, SVM, MLP, HGB) with calibration
6. Train stacking ensemble with subgroup-aware weights
7. Tune per-subgroup thresholds on validation set
8. Evaluate on test set and analyze fairness
9. Generate visualizations and save results

### Outputs

The pipeline generates:
- `final_subgroup_performance.csv` - Detailed metrics per subgroup
- `subgroup_analysis.png` - Performance visualizations
- Console output with comprehensive fairness analysis

### Jupyter Notebook

Explore the step-by-step process:
```bash
jupyter notebook 01_data_preparation.ipynb
```

## 📁 Project Structure

```
Second_project/
├── health_prediction_pipeline.py    # Main pipeline script
├── 01_data_preparation.ipynb        # Interactive notebook
├── student_health_data.csv          # Dataset
├── requirements.txt                 # Dependencies
├── final_subgroup_performance.csv   # Results
├── subgroup_analysis.png            # Visualizations
└── README.md                        # This file
```

## 🔬 Methodology

### 1. Data Preparation
- Load student health monitoring data
- Sort chronologically by timestamp
- Create binary target: Overall Health Score > 80

### 2. Feature Engineering
- Rolling statistics (30-minute windows):
  - Heart Rate: mean & std
  - Physical Activity: mean & std
- Subgroup labels: Gender × Sleep Category

### 3. Model Training
- **Base Models**: Random Forest, SVM (calibrated), MLP, Histogram Gradient Boosting
- **Meta-Model**: Logistic Regression with sample weights
- **Techniques**: SMOTE, class balancing, probability calibration

### 4. Fairness Evaluation
- Per-subgroup threshold optimization
- Dual criteria: strict (all groups) & relaxed (larger groups)
- Metrics: Accuracy, F1-Score, ROC-AUC per subgroup

## 📊 Dataset

**student_health_data.csv** contains:
- **Demographics**: Gender, Age
- **Physiological**: Heart Rate, Blood Pressure, Temperature
- **Behavioral**: Physical Activity, Sleep Duration, Stress Level
- **Target**: Overall Health Score (continuous)

## 🔧 Technical Details

### Models
- **Random Forest**: 200 estimators, max_depth=12, class_weight='balanced'
- **SVM**: RBF kernel, calibrated (sigmoid), class_weight='balanced'
- **MLP**: 2 hidden layers (64, 32), Adam optimizer
- **HGB**: Learning rate=0.1, max_depth=6, calibrated

### Hyperparameters
- Train/Val/Test split: 60/10/30
- SMOTE: k_neighbors=1, auto sampling strategy
- Threshold grid: 0.10 to 0.90, step=0.02
- Random seed: 42

## 📚 Dependencies

```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
imbalanced-learn>=0.11.0
```

## 🎓 Academic Context

This project demonstrates best practices in:
- **Fairness-aware ML**: Addressing performance disparities
- **Ensemble methods**: Combining multiple models
- **Time-series features**: Rolling window statistics
- **Imbalanced learning**: SMOTE and class weighting
- **Model calibration**: Reliable probability estimates

## 📄 License

This project is for educational and research purposes.

## 👤 Author

Abdul Jamil Safi

## 🙏 Acknowledgments

- Based on methodologies from "Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow" by Aurélien Géron
- Implements fairness principles from algorithmic fairness research

---

**Note**: This project prioritizes honest evaluation over inflated metrics. The fairness analysis transparently reports both successes and limitations, which is crucial for real-world ML applications.
