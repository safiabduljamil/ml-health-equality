import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE
import warnings
import os
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Configuration - Updated paths for new folder structure
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Check if training on uploaded data (from environment variable)
DATA_FILE = os.environ.get('TRAINING_DATA_FILE', os.path.join(BASE_DIR, 'data', 'student_health_data.csv'))
OUTPUT_CSV = os.path.join(BASE_DIR, 'data', 'final_subgroup_performance.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'visualizations')
TRAIN_RATIO = 0.6
VAL_RATIO = 0.1
TEST_RATIO = 0.3
MIN_SUBGROUP_SAMPLES = 3

# Create
# output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_and_prepare_data(filepath):
    """Load dataset and perform initial preparation."""
    print("=" * 80)
    print("LOADING AND PREPARING DATA")
    print("=" * 80)
    
    df = pd.read_csv(filepath)
    print(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Convert datetime
    df['Date and Time'] = pd.to_datetime(df['Date and Time'])
    
    # Remove Student ID
    df = df.drop(columns=['Student ID'])
    
    # Create binary target
    df['target'] = (df['Overall Health Score'] > 80).astype(int)
    
    # Sort chronologically
    df = df.sort_values('Date and Time').reset_index(drop=True)
    print(f"✓ Data sorted chronologically")
    
    return df


def create_features(df):
    """Create rolling statistics and subgroup features."""
    print("\n" + "=" * 80)
    print("FEATURE ENGINEERING")
    print("=" * 80)
    
    # Rolling statistics (30-minute window = 6 rows)
    window_size = 6
    df['Heart_Rate_rolling_mean_30min'] = df['Heart Rate (bpm)'].rolling(window=window_size, min_periods=1).mean()
    df['Heart_Rate_rolling_std_30min'] = df['Heart Rate (bpm)'].rolling(window=window_size, min_periods=1).std()
    df['Physical_Activity_rolling_mean_30min'] = df['Physical Activity Level (METs)'].rolling(window=window_size, min_periods=1).mean()
    df['Physical_Activity_rolling_std_30min'] = df['Physical Activity Level (METs)'].rolling(window=window_size, min_periods=1).std()
    
    # Forward fill missing values
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].ffill()
    
    # Create Sleep_Category
    sleep_bins = [0, 6, 8, np.inf]
    sleep_labels = ['Short', 'Normal', 'Long']
    df['Sleep_Category'] = pd.cut(df['Sleep Duration (hours)'], bins=sleep_bins, labels=sleep_labels, right=False)
    
    # Create Gender_Sleep_Group
    df['Gender_Sleep_Group'] = df['Gender'].astype(str) + '_' + df['Sleep_Category'].astype(str)
    
    print(f"✓ Rolling features created (30-min window)")
    print(f"✓ Subgroup feature created: Gender_Sleep_Group")
    print(f"  Unique subgroups: {df['Gender_Sleep_Group'].nunique()}")
    
    return df


def subgroup_aware_split(df, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
    """Perform chronological split within each subgroup to ensure test coverage."""
    print("\n" + "=" * 80)
    print("CHRONOLOGICAL SUBGROUP-AWARE SPLIT")
    print("=" * 80)
    
    train_parts = []
    val_parts = []
    test_parts = []
    
    for group, gdf in df.sort_values('Date and Time').groupby('Gender_Sleep_Group', sort=False):
        n = len(gdf)
        if n == 0:
            continue
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_parts.append(gdf.iloc[:train_end])
        val_parts.append(gdf.iloc[train_end:val_end])
        test_parts.append(gdf.iloc[val_end:])
    
    train_df = pd.concat(train_parts).sort_values('Date and Time').reset_index(drop=True)
    val_df = pd.concat(val_parts).sort_values('Date and Time').reset_index(drop=True)
    test_df = pd.concat(test_parts).sort_values('Date and Time').reset_index(drop=True)
    
    print(f"✓ Split completed:")
    print(f"  - Training:   {len(train_df):>4} samples ({train_ratio*100:.0f}%)")
    print(f"  - Validation: {len(val_df):>4} samples ({val_ratio*100:.0f}%)")
    print(f"  - Test:       {len(test_df):>4} samples ({test_ratio*100:.0f}%)")
    print(f"\n✓ Test subgroup coverage:")
    print(test_df['Gender_Sleep_Group'].value_counts().to_string())
    
    return train_df, val_df, test_df


def prepare_features(train_df, val_df, test_df):
    """Prepare feature matrices and target vectors with SMOTE for subgroup balancing."""
    print("\n" + "=" * 80)
    print("FEATURE PREPARATION")
    print("=" * 80)
    
    # Define columns to drop
    cols_to_drop = ['Date and Time', 'Overall Health Score', 'target', 
                    'Gender_Sleep_Group', 'Sleep_Category']
    
    # Separate features and target
    X_train = train_df.drop(columns=cols_to_drop)
    X_val = val_df.drop(columns=cols_to_drop)
    X_test = test_df.drop(columns=cols_to_drop)
    
    y_train = train_df['target']
    y_val = val_df['target']
    y_test = test_df['target']
    
    # Store subgroup labels for later analysis
    test_groups = test_df['Gender_Sleep_Group']
    train_groups_original = train_df['Gender_Sleep_Group'].copy()
    
    # One-hot encode categorical variables
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_cols:
        X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
        X_val = pd.get_dummies(X_val, columns=categorical_cols, drop_first=True)
        X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)
        
        # Ensure all sets have same columns
        X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    
    # Impute and scale
    imputer = SimpleImputer(strategy='median')
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_val = pd.DataFrame(imputer.transform(X_val), columns=X_val.columns, index=X_val.index)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns, index=X_test.index)
    
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns, index=X_val.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    
    # Apply SMOTE to balance subgroups in training data
    # Create stratified labels combining target and subgroup
    train_df_temp = train_df.copy()
    train_df_temp.index = X_train_scaled.index
    stratify_labels = train_df_temp['Gender_Sleep_Group'].astype(str) + '_' + y_train.astype(str)
    
    # Count samples per stratum
    stratum_counts = stratify_labels.value_counts()
    min_samples_needed = 6  # SMOTE needs at least 6 samples
    
    # Only apply SMOTE if we have strata with very few samples
    small_strata = stratum_counts[stratum_counts < min_samples_needed]
    if len(small_strata) > 0:
        print(f"\n⚠️  Small strata detected (< {min_samples_needed} samples):")
        for stratum, count in small_strata.items():
            print(f"    {stratum}: {count} samples")
        
        # Apply moderate SMOTE: balance to 80% of largest group (not full balance)
        target_ratio = 0.8
        try:
            smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=1, sampling_strategy='auto')
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
            
            # Convert back to DataFrame
            X_train_scaled = pd.DataFrame(X_train_resampled, columns=X_train_scaled.columns)
            y_train = pd.Series(y_train_resampled)
            
            # For sample weights, replicate group labels proportionally
            n_synthetic = len(y_train) - len(train_groups_original)
            synthetic_groups = train_groups_original.sample(n=n_synthetic, replace=True, random_state=RANDOM_STATE)
            train_groups_extended = pd.concat([train_groups_original, pd.Series(synthetic_groups.values)])
            
            print(f"✓ SMOTE applied: {len(train_groups_original)} → {len(X_train_scaled)} samples")
        except Exception as e:
            print(f"⚠️  SMOTE failed ({e}), proceeding without resampling")
            train_groups_extended = train_groups_original
    else:
        train_groups_extended = train_groups_original
    
    print(f"✓ Features prepared and scaled")
    print(f"  - Feature count: {X_train_scaled.shape[1]}")
    print(f"  - Train shape: {X_train_scaled.shape}")
    print(f"  - Val shape:   {X_val_scaled.shape}")
    print(f"  - Test shape:  {X_test_scaled.shape}")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, test_groups, train_groups_extended, scaler, X_train_scaled.columns.tolist()


def train_base_models(X_train, y_train, X_val, y_val):
    """Train base models (RF, SVM, MLP, HGB) with probability calibration where helpful."""
    print("\n" + "=" * 80)
    print("TRAINING BASE MODELS (LEVEL-0)")
    print("=" * 80)
    
    # Random Forest
    print("\n[1/4] Training Random Forest...")
    base_rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        class_weight='balanced'
    )
    # Calibrate RF probabilities for better reliability
    rf_model = CalibratedClassifierCV(base_rf, cv=3, method='isotonic')
    rf_model.fit(X_train, y_train)
    rf_val_pred = rf_model.predict(X_val)
    rf_val_proba = rf_model.predict_proba(X_val)[:, 1]
    print(f"✓ RF - Acc: {accuracy_score(y_val, rf_val_pred):.4f} | F1: {f1_score(y_val, rf_val_pred):.4f}")
    
    # SVM
    print("\n[2/4] Training SVM...")
    base_svm = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        probability=True,
        random_state=RANDOM_STATE,
        class_weight='balanced'
    )
    # Calibrate SVM probabilities on validation via cross-validation
    svm_model = CalibratedClassifierCV(base_svm, cv=3, method='sigmoid')
    svm_model.fit(X_train, y_train)
    svm_val_pred = svm_model.predict(X_val)
    svm_val_proba = svm_model.predict_proba(X_val)[:, 1]
    print(f"✓ SVM - Acc: {accuracy_score(y_val, svm_val_pred):.4f} | F1: {f1_score(y_val, svm_val_pred):.4f}")
    
    # MLP
    print("\n[3/4] Training MLP Neural Network...")
    base_mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=RANDOM_STATE
    )
    # Calibrate MLP probabilities
    mlp_model = CalibratedClassifierCV(base_mlp, cv=3, method='sigmoid')
    mlp_model.fit(X_train, y_train)
    mlp_val_pred = mlp_model.predict(X_val)
    mlp_val_proba = mlp_model.predict_proba(X_val)[:, 1]
    print(f"✓ MLP - Acc: {accuracy_score(y_val, mlp_val_pred):.4f} | F1: {f1_score(y_val, mlp_val_pred):.4f}")
    
    # HistGradientBoosting (robust to imbalance and non-linearities)
    print("\n[4/4] Training HistGradientBoosting...")
    hgb_model = HistGradientBoostingClassifier(
        learning_rate=0.1,
        max_depth=6,
        random_state=RANDOM_STATE
    )
    hgb_model.fit(X_train, y_train)
    # Calibrate HGB probabilities (supports predict_proba)
    hgb_calibrated = CalibratedClassifierCV(hgb_model, cv=3, method='isotonic')
    hgb_calibrated.fit(X_train, y_train)
    hgb_val_pred = hgb_calibrated.predict(X_val)
    hgb_val_proba = hgb_calibrated.predict_proba(X_val)[:, 1]
    print(f"✓ HGB - Acc: {accuracy_score(y_val, hgb_val_pred):.4f} | F1: {f1_score(y_val, hgb_val_pred):.4f}")

    return rf_model, svm_model, mlp_model, hgb_calibrated, (rf_val_proba, svm_val_proba, mlp_val_proba, hgb_val_proba)


def train_stacking_ensemble(rf_model, svm_model, mlp_model, hgb_model, X_train, y_train, X_val, y_val, val_probas, train_groups):
    """Train meta-model for stacking ensemble with subgroup-aware sample weights."""
    print("\n" + "=" * 80)
    print("TRAINING META-MODEL (STACKING ENSEMBLE)")
    print("=" * 80)
    
    # Create meta-features
    # Build meta-train with 4 base models (RF from health.py, SVM, MLP, HGB)
    rf_tr = rf_model.predict_proba(X_train)[:, 1]
    
    svm_tr = svm_model.predict_proba(X_train)[:, 1]
    mlp_tr = mlp_model.predict_proba(X_train)[:, 1]
    try:
        hgb_tr = hgb_model.predict_proba(X_train)[:, 1]
    except Exception:
        hgb_tr = hgb_model.decision_function(X_train)
        hgb_tr = (hgb_tr - hgb_tr.min()) / (hgb_tr.ptp() + 1e-9)
    meta_train = np.column_stack([rf_tr, svm_tr, mlp_tr, hgb_tr])
    
    meta_val = np.column_stack(val_probas)
    
    # Compute sample weights inversely proportional to subgroup size
    group_counts = train_groups.value_counts()
    sample_weights = train_groups.map(lambda g: 1.0 / group_counts[g]).values
    sample_weights = sample_weights / sample_weights.sum() * len(sample_weights)
    
    # Train meta-model with sample weights
    meta_model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, class_weight='balanced')
    meta_model.fit(meta_train, y_train, sample_weight=sample_weights)
    
    stacking_val_pred = meta_model.predict(meta_val)
    stacking_val_proba = meta_model.predict_proba(meta_val)[:, 1]
    
    print(f"✓ Stacking Ensemble trained")
    print(f"  Validation F1-Score: {f1_score(y_val, stacking_val_pred):.4f}")
    
    # Tune threshold on validation to maximize F1
    thresholds = np.arange(0.10, 0.91, 0.02)
    best_thr, best_f1 = 0.5, -1
    for thr in thresholds:
        preds = (stacking_val_proba >= thr).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    print(f"  Selected threshold for meta-model: {best_thr:.3f} (F1={best_f1:.4f})")
    meta_model.best_threshold_ = best_thr
    return meta_model


def tune_subgroup_thresholds(meta_model, rf_model, svm_model, mlp_model, hgb_model, X_val, y_val, val_df):
    """Tune per-subgroup thresholds on validation to maximize F1."""
    # Build meta-features for validation
    rf_val_proba = rf_model.predict_proba(X_val)[:, 1]
    
    svm_val_proba = svm_model.predict_proba(X_val)[:, 1]
    mlp_val_proba = mlp_model.predict_proba(X_val)[:, 1]
    try:
        hgb_val_proba = hgb_model.predict_proba(X_val)[:, 1]
    except Exception:
        hgb_val_proba = hgb_model.decision_function(X_val)
        hgb_val_proba = (hgb_val_proba - hgb_val_proba.min()) / (hgb_val_proba.ptp() + 1e-9)
    meta_val = np.column_stack([rf_val_proba, svm_val_proba, mlp_val_proba, hgb_val_proba])
    val_probs = meta_model.predict_proba(meta_val)[:, 1]

    thresholds = np.arange(0.10, 0.91, 0.02)
    subgroup_thresholds = {}
    groups = val_df['Gender_Sleep_Group']
    for group in groups.unique():
        idx = (groups == group)
        if idx.sum() < MIN_SUBGROUP_SAMPLES:
            continue
        best_thr, best_f1 = getattr(meta_model, 'best_threshold_', 0.5), -1
        for thr in thresholds:
            preds = (val_probs[idx] >= thr).astype(int)
            f1 = f1_score(y_val[idx], preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_thr = f1, thr
        subgroup_thresholds[group] = best_thr
    return subgroup_thresholds


def evaluate_on_test(rf_model, svm_model, mlp_model, hgb_model, meta_model, X_test, y_test):
    """Generate predictions on test set."""
    print("\n" + "=" * 80)
    print("TEST SET EVALUATION")
    print("=" * 80)
    
    # Base model predictions
    rf_test_proba = rf_model.predict_proba(X_test)[:, 1]
    
    svm_test_proba = svm_model.predict_proba(X_test)[:, 1]
    mlp_test_proba = mlp_model.predict_proba(X_test)[:, 1]
    try:
        hgb_test_proba = hgb_model.predict_proba(X_test)[:, 1]
    except Exception:
        hgb_test_proba = hgb_model.decision_function(X_test)
        hgb_test_proba = (hgb_test_proba - hgb_test_proba.min()) / (hgb_test_proba.ptp() + 1e-9)
    
    # Meta-features and final prediction
    meta_test = np.column_stack([rf_test_proba, svm_test_proba, mlp_test_proba, hgb_test_proba])
    stacking_test_proba = meta_model.predict_proba(meta_test)[:, 1]
    # Apply tuned threshold
    thr = getattr(meta_model, 'best_threshold_', 0.5)
    stacking_test_pred = (stacking_test_proba >= thr).astype(int)
    
    # Overall metrics
    acc = accuracy_score(y_test, stacking_test_pred)
    f1 = f1_score(y_test, stacking_test_pred, zero_division=0)
    
    print(f"\n✓ OVERALL TEST PERFORMANCE:")
    print(f"  - Accuracy:  {acc:.4f}")
    print(f"  - F1-Score:  {f1:.4f}")
    
    if len(np.unique(y_test)) > 1:
        try:
            auc = roc_auc_score(y_test, stacking_test_proba)
            print(f"  - ROC-AUC:   {auc:.4f}")
        except:
            print(f"  - ROC-AUC:   N/A")
    
    print(f"\n✓ Classification Report:")
    print(classification_report(y_test, stacking_test_pred,
                                labels=[0, 1],
                                target_names=['Low Health', 'High Health'],
                                zero_division=0))
    
    return stacking_test_pred, stacking_test_proba


def analyze_subgroups(y_test, stacking_test_pred, stacking_test_proba, test_groups, min_samples=3):
    """Perform subgroup fairness analysis."""
    print("\n" + "=" * 80)
    print("🔬 SUBGROUP FAIRNESS ANALYSIS")
    print("=" * 80)
    
    subgroup_results = []
    
    for group in test_groups.unique():
        mask = (test_groups == group)
        y_true_group = y_test[mask]
        y_pred_group = stacking_test_pred[mask]
        y_proba_group = stacking_test_proba[mask]
        
        if len(y_true_group) < min_samples:
            continue
        
        acc = accuracy_score(y_true_group, y_pred_group)
        f1 = f1_score(y_true_group, y_pred_group, zero_division=0)
        auc = roc_auc_score(y_true_group, y_proba_group) if len(np.unique(y_true_group)) > 1 else np.nan
        
        subgroup_results.append({
            'Subgroup': group,
            'Sample_Size': len(y_true_group),
            'Accuracy': acc,
            'F1_Score': f1,
            'ROC_AUC': auc,
            'True_High_Health_%': (y_true_group.mean() * 100)
        })
    
    subgroup_df = pd.DataFrame(subgroup_results)
    
    if subgroup_df.empty:
        print("\n⚠️ No subgroups with sufficient samples for analysis.")
        return None
    
    subgroup_df = subgroup_df.sort_values('F1_Score', ascending=False)
    
    print(f"\n📊 SUBGROUP PERFORMANCE TABLE:")
    print("=" * 80)
    print(subgroup_df.to_string(index=False))
    print("=" * 80)
    
    # Statistical summary
    valid_f1 = subgroup_df['F1_Score'].dropna()
    if not valid_f1.empty and len(valid_f1) > 1:
        print(f"\n📈 STATISTICAL SUMMARY:")
        print(f"  - Mean F1-Score:  {valid_f1.mean():.4f}")
        print(f"  - Std F1-Score:   {valid_f1.std():.4f}")
        print(f"  - Best subgroup:  {subgroup_df.iloc[0]['Subgroup']} (F1={subgroup_df.iloc[0]['F1_Score']:.4f})")
        print(f"  - Worst subgroup: {subgroup_df.iloc[-1]['Subgroup']} (F1={subgroup_df.iloc[-1]['F1_Score']:.4f})")
        
        # Answer research question
        print(f"\n{'='*80}")
        print("✅ ANSWER TO RESEARCH QUESTION:")
        print(f"{'='*80}")
        if valid_f1.std() < 0.05:
            print("✓ YES - Model achieves EQUAL performance across subgroups.")
            print(f"  (F1 std = {valid_f1.std():.4f} < 0.05)")
        else:
            print("✗ NO - Model shows UNEQUAL performance across subgroups.")
            print(f"  (F1 std = {valid_f1.std():.4f} ≥ 0.05)")
            print("\n⚠️  Some subgroups have significantly different performance.")
        print(f"{'='*80}")
    
    return subgroup_df


def analyze_subgroups_with_thresholds(y_test, stacking_test_proba, test_groups, subgroup_thresholds, global_thr, min_samples=3):
    """Analyze subgroups using per-group thresholds when available."""
    print("\n" + "=" * 80)
    print("🔬 SUBGROUP FAIRNESS ANALYSIS (PER-GROUP THRESHOLDS)")
    print("=" * 80)

    subgroup_results = []
    for group in test_groups.unique():
        mask = (test_groups == group)
        y_true_group = y_test[mask]
        proba_group = stacking_test_proba[mask]
        if len(y_true_group) < min_samples:
            continue
        thr = subgroup_thresholds.get(group, global_thr)
        y_pred_group = (proba_group >= thr).astype(int)
        acc = accuracy_score(y_true_group, y_pred_group)
        f1 = f1_score(y_true_group, y_pred_group, zero_division=0)
        auc = roc_auc_score(y_true_group, proba_group) if len(np.unique(y_true_group)) > 1 else np.nan
        subgroup_results.append({
            'Subgroup': group,
            'Sample_Size': len(y_true_group),
            'Threshold': thr,
            'Accuracy': acc,
            'F1_Score': f1,
            'ROC_AUC': auc,
            'True_High_Health_%': (y_true_group.mean() * 100)
        })

    subgroup_df = pd.DataFrame(subgroup_results)
    if subgroup_df.empty:
        print("\n⚠️ No subgroups with sufficient samples for analysis.")
        return None
    subgroup_df = subgroup_df.sort_values('F1_Score', ascending=False)

    print(f"\n📊 SUBGROUP PERFORMANCE TABLE:")
    print("=" * 80)
    print(subgroup_df.to_string(index=False))
    print("=" * 80)

    valid_f1 = subgroup_df['F1_Score'].dropna()
    
    # Separate evaluation: with and without very small groups
    large_subgroups = subgroup_df[subgroup_df['Sample_Size'] >= 6]
    if not large_subgroups.empty:
        large_f1 = large_subgroups['F1_Score'].dropna()
        print(f"\n📈 STATISTICAL SUMMARY (ALL SUBGROUPS):")
        print(f"  - Mean F1-Score:  {valid_f1.mean():.4f}")
        print(f"  - Std F1-Score:   {valid_f1.std():.4f}")
        print(f"  - Best subgroup:  {subgroup_df.iloc[0]['Subgroup']} (F1={subgroup_df.iloc[0]['F1_Score']:.4f})")
        print(f"  - Worst subgroup: {subgroup_df.iloc[-1]['Subgroup']} (F1={subgroup_df.iloc[-1]['F1_Score']:.4f})")
        
        if len(large_f1) > 1:
            print(f"\n📈 ROBUST EVALUATION (≥6 samples per group):")
            print(f"  - Mean F1-Score:  {large_f1.mean():.4f}")
            print(f"  - Std F1-Score:   {large_f1.std():.4f}")
            print(f"  - Groups included: {len(large_f1)}/{len(valid_f1)}")
    else:
        print(f"\n📈 STATISTICAL SUMMARY:")
        print(f"  - Mean F1-Score:  {valid_f1.mean():.4f}")
        print(f"  - Std F1-Score:   {valid_f1.std():.4f}")
        print(f"  - Best subgroup:  {subgroup_df.iloc[0]['Subgroup']} (F1={subgroup_df.iloc[0]['F1_Score']:.4f})")
        print(f"  - Worst subgroup: {subgroup_df.iloc[-1]['Subgroup']} (F1={subgroup_df.iloc[-1]['F1_Score']:.4f})")
        large_f1 = valid_f1
    
    print(f"\n{'='*80}")
    print("✅ ANSWER TO RESEARCH QUESTION:")
    print(f"{'='*80}")
    
    # Strict criterion (all groups)
    if valid_f1.std() < 0.05:
        print("✓ YES - Model achieves EQUAL performance across ALL subgroups.")
        print(f"  (F1 std = {valid_f1.std():.4f} < 0.05)")
    else:
        print("✗ NO (Strict) - Model shows UNEQUAL performance across all subgroups.")
        print(f"  (F1 std = {valid_f1.std():.4f} ≥ 0.05)")
    
    # Relaxed criterion (larger groups only)
    if len(large_f1) > 1 and large_f1.std() < 0.10:
        print("\n✓ YES (Relaxed) - Model achieves REASONABLE EQUALITY for larger subgroups.")
        print(f"  (F1 std = {large_f1.std():.4f} < 0.10 for groups with ≥6 samples)")
        print("  ⚠️  Very small subgroups (<6 samples) show higher variance due to limited data.")
    elif len(large_f1) > 1:
        print(f"\n✗ NO (Relaxed) - Even larger subgroups show significant variance.")
        print(f"  (F1 std = {large_f1.std():.4f} ≥ 0.10)")
    
    print(f"{'='*80}")
    return subgroup_df


def visualize_subgroups(subgroup_df, output_file='subgroup_analysis.png'):
    """Create and save subgroup performance visualizations."""
    if subgroup_df is None or subgroup_df.empty:
        print("\n⚠️ No subgroup data to visualize.")
        return
    
    print(f"\n📊 Creating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Plot 1: F1-Score by subgroup
    axes[0, 0].barh(subgroup_df['Subgroup'], subgroup_df['F1_Score'], color='#3498db')
    axes[0, 0].set_xlabel('F1-Score')
    axes[0, 0].set_title('Model Performance (F1-Score) by Subgroup', fontweight='bold')
    axes[0, 0].axvline(x=subgroup_df['F1_Score'].mean(), color='red', linestyle='--', label='Mean')
    axes[0, 0].legend()
    
    # Plot 2: Accuracy by subgroup
    axes[0, 1].barh(subgroup_df['Subgroup'], subgroup_df['Accuracy'], color='#2ecc71')
    axes[0, 1].set_xlabel('Accuracy')
    axes[0, 1].set_title('Model Performance (Accuracy) by Subgroup', fontweight='bold')
    axes[0, 1].axvline(x=subgroup_df['Accuracy'].mean(), color='red', linestyle='--', label='Mean')
    axes[0, 1].legend()
    
    # Plot 3: Sample sizes
    axes[1, 0].bar(subgroup_df['Subgroup'], subgroup_df['Sample_Size'], color='#e74c3c', alpha=0.7)
    axes[1, 0].set_ylabel('Sample Size')
    axes[1, 0].set_title('Test Sample Size by Subgroup', fontweight='bold')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Plot 4: Performance vs Sample Size
    scatter = axes[1, 1].scatter(subgroup_df['Sample_Size'], subgroup_df['F1_Score'], 
                                 s=100, c=subgroup_df['F1_Score'], cmap='viridis', alpha=0.7)
    axes[1, 1].set_xlabel('Sample Size')
    axes[1, 1].set_ylabel('F1-Score')
    axes[1, 1].set_title('Performance vs Sample Size', fontweight='bold')
    plt.colorbar(scatter, ax=axes[1, 1], label='F1-Score')
    
    for idx, row in subgroup_df.iterrows():
        try:
            axes[1, 1].annotate(row['Subgroup'], (row['Sample_Size'], row['F1_Score']), 
                               fontsize=8, ha='right')
        except:
            pass
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, os.path.basename(output_file))
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Visualizations saved to: {output_path}")
    plt.close()


def save_results(subgroup_df, output_prefix='final'):
    """Save analysis results to CSV files."""
    print(f"\n💾 Saving results...")
    
    if subgroup_df is not None and not subgroup_df.empty:
        output_path = os.path.join(BASE_DIR, 'data', f'{output_prefix}_subgroup_performance.csv')
        subgroup_df.to_csv(output_path, index=False)
        print(f"✓ Subgroup results saved to: {output_path}")
    
    print(f"✓ All results saved successfully!")


def save_model(meta_model, rf_model, svm_model, mlp_model, hgb_model, scaler, feature_columns, threshold=0.5):
    """Save trained model and preprocessing objects for future predictions."""
    import pickle
    
    print(f"\n💾 Saving trained model...")
    
    model_dir = os.path.join(BASE_DIR, 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    model_info = {
        'model': meta_model,
        'base_models': {
            'rf': rf_model,
            'svm': svm_model,
            'mlp': mlp_model,
            'hgb': hgb_model
        },
        'scaler': scaler,
        'feature_columns': feature_columns,
        'threshold': threshold
    }
    
    model_path = os.path.join(model_dir, 'stacking_ensemble.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model_info, f)
    
    print(f"✓ Model saved to: {model_path}")
    print(f"   Features: {len(feature_columns)}")
    print(f"   Threshold: {threshold:.3f}")


def main():
    """Main pipeline execution."""
    print("\n" + "=" * 80)
    print("HEALTH PREDICTION PIPELINE - STACKING ENSEMBLE")
    print("=" * 80)
    
    # Step 1: Load and prepare data
    df = load_and_prepare_data(DATA_FILE)
    
    # Step 2: Feature engineering
    df = create_features(df)
    
    # Step 3: Subgroup-aware split
    train_df, val_df, test_df = subgroup_aware_split(df, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
    
    # Step 4: Prepare features
    X_train, X_val, X_test, y_train, y_val, y_test, test_groups, train_groups, scaler, feature_columns = prepare_features(train_df, val_df, test_df)
    
    # Step 5: Train base models
    rf_model, svm_model, mlp_model, hgb_model, val_probas = train_base_models(X_train, y_train, X_val, y_val)
    
    # Step 6: Train stacking ensemble
    meta_model = train_stacking_ensemble(rf_model, svm_model, mlp_model, hgb_model, X_train, y_train, X_val, y_val, val_probas, train_groups)
    subgroup_thresholds = tune_subgroup_thresholds(meta_model, rf_model, svm_model, mlp_model, hgb_model, X_val, y_val, val_df)
    
    # Step 7: Evaluate on test set
    stacking_test_pred, stacking_test_proba = evaluate_on_test(rf_model, svm_model, mlp_model, hgb_model, meta_model, X_test, y_test)
    
    # Step 8: Subgroup analysis (with per-group thresholds)
    global_thr = getattr(meta_model, 'best_threshold_', 0.5)
    subgroup_df = analyze_subgroups_with_thresholds(y_test, stacking_test_proba, test_groups, subgroup_thresholds, global_thr, MIN_SUBGROUP_SAMPLES)
    
    # Step 9: Visualize and save
    visualize_subgroups(subgroup_df, 'subgroup_analysis.png')
    save_results(subgroup_df, 'final')
    
    # Step 10: Save trained model for future predictions
    save_model(meta_model, rf_model, svm_model, mlp_model, hgb_model, scaler, feature_columns, global_thr)
    
    print("\n" + "=" * 80)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
