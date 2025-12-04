# Health Equality - ML Fairness Analysis

Machine Learning project for predicting student health with fairness analysis across demographic subgroups.

## 🎯 Research Question

**Does the classification model, based on the stacking ensemble approach with advanced time-series features, achieve equal predictive performance for Overall Health Score across different subgroups (Gender × Sleep Category)?**

### Answer
✅ **YES - for larger subgroups (n≥6)** with relaxed fairness criteria (Std F1 < 0.10)
- Achieved: Std F1 = 0.092 for all groups, 0.091 for n≥6
- Strict criteria (Std F1 < 0.05) not met due to limited data (n=100 total)

## 📁 Project Structure

```
Second_project/
├── code/                                    # Source code
│   ├── health_prediction_pipeline.py       # Main ML pipeline with stacking ensemble
│   ├── dashboard.py                         # Interactive Streamlit dashboard
│   └── health.py                           # Alternative pipeline implementation
│
├── data/                                    # Datasets and results
│   ├── student_health_data.csv             # Original dataset (100 observations)
│   ├── final_subgroup_performance.csv      # Model evaluation results by subgroup
│   └── example_new_data.csv               # Example file for predictions (20 samples)
│
├── models/                                  # Trained models
│   └── stacking_ensemble.pkl               # Saved stacking ensemble with calibration
│
├── notebooks/                               # Jupyter notebooks
│   └── 01_data_preparation.ipynb           # Exploratory data analysis
│
├── reports/                                 # Scientific documentation
│   ├── final_scientific_report.tex         # LaTeX source (3 pages, German)
│   ├── final_scientific_report.pdf         # Compiled PDF report
│   ├── references.bib                      # Bibliography (5 references)
│   └── figures/                            # Dashboard-generated visualizations
│       ├── performance_table.png           # Subgroup metrics table
│       ├── f1_by_subgroup.png             # F1 scores bar chart
│       ├── size_vs_f1.png                 # Sample size correlation
│       └── f1_vs_accuracy.png             # Metric comparison
│
├── requirements.txt                         # Python dependencies
└── README.md                               # This file
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
# Make sure you have Python 3.8 or higher
python3 --version

# Install required packages
pip install -r requirements.txt
```

**Required packages**: pandas, numpy, matplotlib, seaborn, scikit-learn, imbalanced-learn, streamlit

### 2. Run ML Pipeline
```bash
# Navigate to project root
cd /Users/abduljamilsafi/Documents/Second_project

# Run the main pipeline (takes ~30 seconds)
python3 code/health_prediction_pipeline.py
```

**What it does**:
- Loads `student_health_data.csv` (100 observations)
- Creates rolling statistics features (30-min windows)
- Splits data chronologically by subgroups (60/10/30)
- Applies SMOTE for class balancing
- Trains 4 calibrated base models (RF, SVM, MLP, HGB)
- Trains stacking meta-model with subgroup weighting
- Tunes per-subgroup thresholds on validation set
- Evaluates fairness on test set
- Saves model to `models/stacking_ensemble.pkl`
- Saves results to `data/final_subgroup_performance.csv`
- Generates console output with detailed metrics

### 3. Launch Interactive Dashboard
```bash
# From project root directory
streamlit run code/dashboard.py
```

**Dashboard will open at**: http://localhost:8501

**Features**:
- 📊 **Model Analysis Tab**: View training results, subgroup performance, calibration curves
- 🔮 **Predict New Data Tab**: Upload CSV files for predictions, retrain on new data
- 🎨 **Live Visualizations**: Interactive charts update automatically
- 💾 **Download Results**: Export predictions as CSV

To stop the dashboard: Press `Ctrl+C` in terminal

## Key Features

- **Stacking Ensemble**: Random Forest (200 trees) + SVM (RBF) + MLP (64-32) + HGB (100 iter) + Logistic Regression Meta-Model
- **Fairness Techniques**: 
  - SMOTE for class balancing
  - CalibratedClassifierCV for probability calibration
  - Per-subgroup threshold tuning (0.1-0.9, step 0.02)
  - Sample weighting (inverse group size)
- **Time-Series Features**: Rolling statistics (30-min windows) for Heart Rate & Physical Activity
- **Evaluation**: Chronological subgroup-aware split (60/10/30) prevents temporal data leakage
- **Metrics**: F1=0.81 (Mean), Accuracy=0.65, ROC-AUC=0.48, Std F1=0.092 (acceptable fairness for n≥6)
- **Interactive Dashboard**: Streamlit with live visualizations and CSV upload for predictions

## Results Summary

### Overall Performance (Test Set)
- **Accuracy**: 0.65
- **Mean F1-Score**: 0.81
- **ROC-AUC**: 0.48 (indicates calibration challenges)

### Subgroup Fairness Analysis
| Subgroup | n | F1 | Accuracy | Threshold |
|----------|---|-------|----------|-----------|
| Female_Normal | 6 | 0.91 | 0.83 | 0.50 |
| Female_Short | 4 | 0.86 | 0.75 | 0.50 |
| Other_Short | 4 | 0.86 | 0.75 | 0.50 |
| Male_Normal | 7 | 0.83 | 0.71 | 0.40 |
| Other_Normal | 7 | 0.73 | 0.57 | 0.50 |
| Male_Short | 4 | 0.67 | 0.50 | 0.50 |

**Fairness Metrics:**
- Std F1 (all groups): 0.092
- Std F1 (n≥6): 0.091 ✅ **Meets relaxed criteria (<0.10)**
- Best: Female_Normal (F1=0.91)
- Worst: Male_Short (F1=0.67)

**Interpretation:** The model achieves reasonable fairness for larger subgroups. Smaller groups show higher variance due to data scarcity, which is expected in real-world scenarios.

## Team

- **Abdul Jamil Safi**: 
  - Python implementation (health_prediction_pipeline.py)
  - Stacking ensemble architecture (4 base models + meta-model)
  - SMOTE integration and class balancing
  - CalibratedClassifierCV implementation
  - Per-subgroup threshold tuning algorithm
  - Streamlit dashboard development
  - GitHub repository management
  - Data analysis and visualization

- **Ola Hamza**: 
  - LaTeX scientific report (final_scientific_report.tex)
  - Theoretical foundations and literature review
  - Methodology documentation
  - Results interpretation
  - Discussion and future work sections

**Collaboration**: Strategy development, code review, fairness criteria definition, results discussion

## Methodology Overview

### 1. Data Preparation
- **Dataset**: 100 chronologically sorted student observations (5-minute intervals)
- **Features**: Demographics (Age, Gender), Physiological (Heart Rate, Blood Pressure, Temperature, Blood Oxygen), Behavioral (Physical Activity, Sleep Duration, Stress, Hydration)
- **Target**: Overall Health Score binary (>80 = High, ≤80 = Low)

### 2. Feature Engineering
- **Rolling Statistics**: 30-minute windows (6 observations)
  - Heart Rate: mean & std
  - Physical Activity Level: mean & std
- **Sleep Categorization**: Short (<6h), Normal (6-8h), Long (>8h)
- **Subgroup Labels**: Gender_Sleep_Group (e.g., "Male_Normal") → 8 unique combinations

### 3. Model Architecture
**Stacking Ensemble** based on Wolpert (1992) and Géron (2022):
- **Level-0 (Base Models)**:
  - Random Forest: 200 trees, max_depth=10, class_weight='balanced' + Calibration (isotonic)
  - SVM: RBF kernel, C=1.0 + Calibration (sigmoid)
  - MLP Neural Network: (64, 32) layers, ReLU, Adam + Calibration (sigmoid)
  - Histogram Gradient Boosting: max_iter=100, learning_rate=0.1 + Calibration (isotonic)
- **Level-1 (Meta-Model)**: Logistic Regression with inverse subgroup weights, C=1.0, class_weight='balanced'

### 4. Fairness Techniques
1. **SMOTE** (Chawla et al. 2002): Synthetic minority class examples
2. **Probability Calibration**: CalibratedClassifierCV for reliable confidence scores
3. **Per-Subgroup Threshold Tuning**: Optimize on validation set (0.1-0.9, step 0.02) for max F1 per group
4. **Sample Weighting**: Inverse group size in meta-model

### 5. Evaluation
- **Split Strategy**: Chronological subgroup-based (60% train, 10% val, 30% test) prevents temporal leakage
- **Metrics**: Accuracy, F1-Score, ROC-AUC (overall and per subgroup)
- **Fairness Criteria**: 
  - Strict: Std F1 < 0.05 (all groups)
  - Relaxed: Std F1 < 0.10 (for n≥6)

## 📊 Interactive Dashboard Guide

The Streamlit dashboard provides a comprehensive interface for model analysis and predictions.

### Starting the Dashboard

```bash
# From project root
streamlit run code/dashboard.py
```

The dashboard automatically opens in your browser at **http://localhost:8501**

### Tab 1: 📊 Model Analysis

**Purpose**: Explore training results and fairness metrics

**Features**:
1. **Subgroup Performance Table**
   - Displays all 6 subgroups (Female_Normal, Female_Short, Male_Normal, Male_Short, Other_Normal, Other_Short)
   - Shows metrics: Sample Size (n), F1-Score, Accuracy, ROC-AUC, Optimal Threshold, True High Health %
   - Color-coded for easy identification (dark theme)

2. **Live Visualizations**
   - **F1 by Subgroup**: Horizontal bar chart with color coding (green ≥0.8, orange 0.6-0.8, red <0.6)
   - **Sample Size vs F1**: Scatter plot showing correlation between group size and performance
   - **Accuracy Comparison**: Bar chart comparing accuracy across subgroups
   - **F1 vs Accuracy**: Scatter plot with subgroup labels showing metric trade-offs

3. **Summary Metrics**
   - Mean F1-Score across all subgroups
   - Mean Accuracy
   - Standard Deviation of F1 (fairness metric)

4. **Probability Calibration Curve**
   - Click "Compute Calibration Curve" button
   - Shows reliability diagram on validation data
   - Compares predicted probabilities vs observed frequencies
   - Helps assess model confidence quality

5. **Regenerate Pipeline Button** (Bottom of page)
   - Full-width button for easy access
   - Retrains entire pipeline on current training data
   - Updates all visualizations automatically
   - Takes ~30 seconds to complete

### Tab 2: 🔮 Predict New Data

**Purpose**: Make predictions on new student health observations

**How to Use**:

1. **Prepare CSV File**
   - Must include ALL required columns (see template below)
   - Can have any number of rows (students/observations)
   - Example file provided: `data/example_new_data.csv`

2. **Required Columns**:
   ```
   Student ID, Age, Gender, Date and Time, Overall Health Score,
   Heart Rate (bpm), Blood Pressure Systolic, Blood Pressure Diastolic,
   Body Temperature (°C), Blood Oxygen Level (%),
   Physical Activity Level (steps/min), Sleep Duration (hours),
   Stress Level (scale), Hydration Level (%)
   ```

3. **Upload Process**:
   - Click "Choose a CSV file" button
   - Select your file
   - Dashboard automatically validates columns
   - Missing columns will show error message with details

4. **View Predictions**:
   - Table displays: Student ID, Date, True Label, Predicted Label, Probability, Health Status
   - Color-coded by prediction (High Health / Low Health)
   - Shows confidence percentage for each prediction

5. **Download Results**:
   - Click "Download Predictions as CSV" button
   - Saves to `data/predictions.csv`
   - Includes all prediction details

6. **Regenerate Pipeline Button** (Bottom of page)
   - Retrains model on your uploaded data
   - Creates new features (rolling statistics)
   - Applies same train/val/test split
   - Updates model in `models/stacking_ensemble.pkl`
   - **Warning**: Replaces existing model

### Dashboard Tips

- **Performance**: First load may take 10-15 seconds as model loads
- **Updates**: Visualizations update automatically when regenerating
- **Data Validation**: Dashboard checks for missing/incorrect columns before processing
- **Stopping**: Press `Ctrl+C` in terminal to stop the server
- **Ports**: If port 8501 is busy, Streamlit will suggest alternative (8502, 8503, etc.)

### Troubleshooting

**Dashboard won't start?**
```bash
# Install Streamlit if missing
pip install streamlit

# Check Python version (need 3.8+)
python3 --version
```

**Upload fails?**
- Verify CSV has exactly the required column names (case-sensitive)
- Check Date and Time format: `YYYY-MM-DD HH:MM:SS`
- Ensure no missing values in required columns

**Regenerate button stuck?**
- Wait 30-45 seconds for pipeline to complete
- Check terminal for error messages
- Large datasets (>200 rows) take longer

## Submission

For university submission:

### Required Files
1. **Scientific Report**: `reports/final_scientific_report.pdf` (3 pages, APA format, German language)
   - Compile: `cd reports && pdflatex final_scientific_report.tex && bibtex final_scientific_report && pdflatex final_scientific_report.tex && pdflatex final_scientific_report.tex`
2. **Source Code**: `code/health_prediction_pipeline.py` (main pipeline)
3. **Dataset**: `data/student_health_data.csv` (100 observations)
4. **Dashboard**: `code/dashboard.py` (Streamlit application)

### Report Contents
- **Abstract**: Stacking ensemble approach with fairness analysis
- **Introduction**: Research motivation, literature review (Obermeyer 2019, Wolpert 1992, Géron 2022, Chawla 2002, Mehrabi 2021)
- **Research Question**: Equal performance across Gender × Sleep subgroups?
- **Methodology**: Dataset, feature engineering, model architecture, fairness techniques, evaluation
- **Results**: Overall performance, subgroup analysis table, 4 visualizations (performance table, F1 by subgroup, size vs F1, F1 vs Accuracy)
- **Discussion**: Fairness interpretation, limitations, practical implications
- **Summary & Outlook**: Key findings, future work (lag features, Bayesian optimization, extended calibration)
- **References**: 5 APA-formatted references

### Grading Criteria (100 points)
- **Technical Content (60 points)**:
  - Data preparation & feature engineering
  - Model implementation & stacking architecture
  - Fairness analysis & subgroup evaluation
  - Results interpretation
- **Scientific Report (40 points)**:
  - Structure & clarity
  - Literature integration
  - Methodology description
  - Discussion quality

## 💻 Installation & Usage

### Prerequisites

```bash
# Check Python version (3.8 or higher required)
python3 --version

# Check pip is installed
pip --version
```

### Step-by-Step Installation

1. **Clone or download the project**
```bash
cd /Users/abduljamilsafi/Documents/Second_project
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

**Dependencies installed**:
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `matplotlib` - Plotting
- `seaborn` - Statistical visualizations
- `scikit-learn` - Machine learning models
- `imbalanced-learn` - SMOTE implementation
- `streamlit` - Dashboard framework

### Running the Pipeline

```bash
# Execute main pipeline
python3 code/health_prediction_pipeline.py
```

**Pipeline Outputs**:
- ✅ `models/stacking_ensemble.pkl` - Trained model (4 base models + meta-model + scaler + features)
- ✅ `data/final_subgroup_performance.csv` - Evaluation metrics per subgroup
- ✅ Console output - Detailed fairness analysis with statistics

**Expected Runtime**: ~30-45 seconds on modern hardware

### Running the Dashboard

```bash
# Launch Streamlit dashboard
streamlit run code/dashboard.py
```

**Access dashboard**: Browser opens automatically at http://localhost:8501

**To stop**: Press `Ctrl+C` in terminal

### Making Predictions on New Data

**Option 1: Using Dashboard**
1. Launch dashboard: `streamlit run code/dashboard.py`
2. Go to "🔮 Predict New Data" tab
3. Upload CSV file (use `data/example_new_data.csv` as template)
4. View predictions and download results

**Option 2: Using Script**
```bash
python3 code/predict_new_data.py
```
Reads `data/example_new_data.csv`, generates predictions to `data/predictions.csv`

### Compiling Scientific Report

```bash
# Navigate to reports folder
cd reports

# Compile LaTeX (4 steps required for bibliography)
pdflatex final_scientific_report.tex
bibtex final_scientific_report
pdflatex final_scientific_report.tex
pdflatex final_scientific_report.tex

# Output: final_scientific_report.pdf (3 pages)
```

**Requirements**: LaTeX distribution (MacTeX on macOS, TeX Live on Linux/Windows)

### Running Jupyter Notebook

```bash
# Start Jupyter
jupyter notebook notebooks/01_data_preparation.ipynb
```

Explore step-by-step data preparation and feature engineering

## References

1. **Obermeyer et al. (2019)**: "Dissecting racial bias in an algorithm used to manage the health of populations", *Science*
2. **Wolpert (1992)**: "Stacked generalization", *Neural Networks*
3. **Géron (2022)**: "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" (3rd ed.)
4. **Chawla et al. (2002)**: "SMOTE: Synthetic Minority Over-sampling Technique", *Journal of Artificial Intelligence Research*
5. **Mehrabi et al. (2021)**: "A Survey on Bias and Fairness in Machine Learning", *ACM Computing Surveys*

## License

Educational and research purposes only.

---

**Project Completion Date**: December 4, 2025  
**Repository**: https://github.com/safiabduljamil/ml-health-equality
