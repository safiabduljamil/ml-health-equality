import os
import subprocess
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
from sklearn.calibration import calibration_curve

# Updated paths for new folder structure
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_CSV = os.path.join(BASE_DIR, "data", "final_subgroup_performance.csv")
TRAINING_DATA_FILE = os.path.join(BASE_DIR, "data", "student_health_data.csv")
PIPELINE_SCRIPT = os.path.join(BASE_DIR, "code", "health_prediction_pipeline.py")
HEALTH_RF_SCRIPT = os.path.join(BASE_DIR, "code", "health.py")

st.set_page_config(page_title="Health Equality Dashboard", layout="wide")

# Tabs for different functionalities
tab1, tab2 = st.tabs(["📊 Model Analysis", "🔮 Predict New Data"])

# Helper: check file exists
def file_exists(path):
    return os.path.exists(path) and os.path.isfile(path)

# Helper: load trained model (ensure defined before use)
def load_trained_model():
    """Load the trained stacking ensemble model."""
    import pickle
    model_path = os.path.join(BASE_DIR, "models", "stacking_ensemble.pkl")
    if file_exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    return None

# TAB 1: Model Analysis (existing functionality)
with tab1:
    st.title("Health Equality Dashboard")
    st.caption("Model performance and fairness across subgroups (Gender × Sleep)")

    st.markdown("---")

    # Training results section (only on Tab 1)
    csv_status = file_exists(DATA_CSV)
    if not csv_status:
        st.warning(f"'{DATA_CSV}' not found. Use 'Regenerate outputs' or run your pipeline to produce it.")
    else:
        df = pd.read_csv(DATA_CSV)
        st.subheader("Subgroup Performance (Training Results)")
        st.dataframe(df, width='stretch')

        st.subheader("Summary Metrics")
        mean_f1 = df["F1_Score"].mean() if "F1_Score" in df.columns else None
        mean_acc = df["Accuracy"].mean() if "Accuracy" in df.columns else None
        std_f1 = df["F1_Score"].std() if "F1_Score" in df.columns else None

        cols = st.columns(3)
        cols[0].metric("Mean F1", f"{mean_f1:.2f}" if mean_f1 is not None else "-")
        cols[1].metric("Mean Accuracy", f"{mean_acc:.2f}" if mean_acc is not None else "-")
        cols[2].metric("Std F1 (fairness)", f"{std_f1:.3f}" if std_f1 is not None else "-")

        st.subheader("Live Visualizations (from training data)")
        c1, c2 = st.columns(2)
        with c1:
            st.caption("F1 by Subgroup")
            try:
                fig, ax = plt.subplots(figsize=(5, 3))
                plot_df = df.copy()
                required_cols = ["Subgroup", "F1_Score"]
                missing = [c for c in required_cols if c not in plot_df.columns]
                if missing:
                    raise ValueError(f"Missing columns: {', '.join(missing)}")
                plot_df = plot_df.dropna(subset=["Subgroup", "F1_Score"]) 
                plot_df["Subgroup_label"] = plot_df["Subgroup"].str.replace("_", "\n")
                colors = ["#2ecc71" if f >= 0.8 else "#e74c3c" if f < 0.6 else "#f39c12" for f in plot_df["F1_Score"]]
                ax.barh(plot_df["Subgroup_label"], plot_df["F1_Score"], color=colors, alpha=0.8)
                ax.set_xlim(0, 1.05)
                ax.set_xlabel("F1-Score")
                ax.grid(axis="x", alpha=0.3)
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.error(f"Failed to render F1 bar chart: {e}")
        with c2:
            st.caption("Sample Size vs F1")
            try:
                fig, ax = plt.subplots(figsize=(5, 3))
                required_cols = ["Sample_Size", "F1_Score"]
                missing = [c for c in required_cols if c not in df.columns]
                if missing:
                    raise ValueError(f"Missing columns: {', '.join(missing)}")
                plot_df2 = df.dropna(subset=["Sample_Size", "F1_Score"]).copy()
                ax.scatter(plot_df2["Sample_Size"], plot_df2["F1_Score"], s=100, alpha=0.6, c="#3498db", edgecolors="black", linewidth=1)
                ax.set_xlabel("n (Sample Size)")
                ax.set_ylabel("F1-Score")
                ax.set_ylim(0, 1.05)
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.error(f"Failed to render size vs F1 chart: {e}")

        st.subheader("Additional Analysis")
        c3, c4 = st.columns(2)
        with c3:
            st.caption("Accuracy by Subgroup")
            try:
                fig, ax = plt.subplots(figsize=(5, 3))
                plot_df = df.copy()
                required_cols = ["Subgroup", "Accuracy"]
                missing = [c for c in required_cols if c not in plot_df.columns]
                if missing:
                    raise ValueError(f"Missing columns: {', '.join(missing)}")
                plot_df = plot_df.dropna(subset=["Subgroup", "Accuracy"]) 
                plot_df["Subgroup_label"] = plot_df["Subgroup"].str.replace("_", "\n")
                ax.barh(plot_df["Subgroup_label"], plot_df["Accuracy"], color="#9b59b6", alpha=0.7)
                ax.set_xlim(0, 1.05)
                ax.set_xlabel("Accuracy")
                ax.grid(axis="x", alpha=0.3)
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.error(f"Failed to render accuracy chart: {e}")

        with c4:
            st.caption("F1 vs Accuracy")
            try:
                fig, ax = plt.subplots(figsize=(5, 3))
                required_cols = ["Accuracy", "F1_Score", "Subgroup"]
                missing = [c for c in required_cols if c not in df.columns]
                if missing:
                    raise ValueError(f"Missing columns: {', '.join(missing)}")
                plot_df3 = df.dropna(subset=["Accuracy", "F1_Score"]).copy()
                ax.scatter(plot_df3["Accuracy"], plot_df3["F1_Score"], s=100, alpha=0.6, c="#e67e22", edgecolors="black", linewidth=1)
                for idx, row in plot_df3.iterrows():
                    ax.annotate(row["Subgroup"].replace("_", "\n"), 
                               (row["Accuracy"], row["F1_Score"]), 
                               fontsize=6, ha='center', alpha=0.7)
                ax.set_xlabel("Accuracy")
                ax.set_ylabel("F1-Score")
                ax.set_xlim(0, 1.05)
                ax.set_ylim(0, 1.05)
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.error(f"Failed to render F1 vs Accuracy: {e}")

        # Probability calibration on validation split (trained data analysis)
        st.subheader("Probability Calibration (Validation)")
        st.caption("Reliability curve: predicted probability vs observed frequency on validation data")
        if st.button("Compute Calibration Curve"):
            try:
                # Load training data and model
                if not file_exists(TRAINING_DATA_FILE):
                    raise FileNotFoundError("Training data file not found.")
                model_bundle = load_trained_model()
                if model_bundle is None:
                    raise FileNotFoundError("Model bundle not found. Run the pipeline first.")
                meta_model = model_bundle['model']
                base_models = model_bundle.get('base_models', {})
                scaler = model_bundle['scaler']
                feature_columns = model_bundle['feature_columns']

                # Minimal feature engineering to match training
                df_full = pd.read_csv(TRAINING_DATA_FILE)
                df_full['Date and Time'] = pd.to_datetime(df_full['Date and Time'])
                df_full['target'] = (df_full['Overall Health Score'] > 80).astype(int)
                df_full = df_full.sort_values('Date and Time').reset_index(drop=True)
                window_size = 6
                df_full['Heart_Rate_rolling_mean_30min'] = df_full['Heart Rate (bpm)'].rolling(window=window_size, min_periods=1).mean()
                df_full['Heart_Rate_rolling_std_30min'] = df_full['Heart Rate (bpm)'].rolling(window=window_size, min_periods=1).std()
                df_full['Physical_Activity_rolling_mean_30min'] = df_full['Physical Activity Level (METs)'].rolling(window=window_size, min_periods=1).mean()
                df_full['Physical_Activity_rolling_std_30min'] = df_full['Physical Activity Level (METs)'].rolling(window=window_size, min_periods=1).std()
                num_cols_full = df_full.select_dtypes(include=[np.number]).columns
                df_full[num_cols_full] = df_full[num_cols_full].ffill()
                sleep_bins = [0, 6, 8, np.inf]
                sleep_labels = ['Short', 'Normal', 'Long']
                df_full['Sleep_Category'] = pd.cut(df_full['Sleep Duration (hours)'], bins=sleep_bins, labels=sleep_labels, right=False)
                df_full['Gender_Sleep_Group'] = df_full['Gender'].astype(str) + '_' + df_full['Sleep_Category'].astype(str)

                # Chronological subgroup-aware split (60/10/30)
                train_parts, val_parts, test_parts = [], [], []
                for group, gdf in df_full.sort_values('Date and Time').groupby('Gender_Sleep_Group', sort=False):
                    n = len(gdf)
                    if n == 0:
                        continue
                    train_end = int(n * 0.6)
                    val_end = int(n * 0.7)
                    train_parts.append(gdf.iloc[:train_end])
                    val_parts.append(gdf.iloc[train_end:val_end])
                    test_parts.append(gdf.iloc[val_end:])
                val_df = pd.concat(val_parts).sort_values('Date and Time').reset_index(drop=True)

                # Prepare features for val
                cols_to_drop = ['Date and Time', 'Overall Health Score', 'target', 'Gender_Sleep_Group', 'Sleep_Category']
                X_val = val_df.drop(columns=cols_to_drop)
                y_val = val_df['target']
                cat_cols = X_val.select_dtypes(include=['object','category']).columns.tolist()
                if cat_cols:
                    X_val = pd.get_dummies(X_val, columns=cat_cols, drop_first=True)
                for col in feature_columns:
                    if col not in X_val.columns:
                        X_val[col] = 0
                X_val = X_val[feature_columns]
                X_val_scaled = scaler.transform(X_val)

                # Level-0 probs
                ordered_keys = [k for k in ['rf','svm','mlp','hgb'] if k in base_models]
                lvl0_parts = []
                for key in ordered_keys:
                    proba = base_models[key].predict_proba(X_val_scaled)
                    lvl0_parts.append(proba[:, 1] if proba.shape[1] > 1 else proba[:, 0])
                lvl0_val = np.column_stack(lvl0_parts)
                val_probs = meta_model.predict_proba(lvl0_val)[:, 1]

                # Reliability curve
                frac_pos, mean_pred = calibration_curve(y_val, val_probs, n_bins=8, strategy='uniform')
                fig, ax = plt.subplots(figsize=(5,3))
                ax.plot(mean_pred, frac_pos, marker='o', label='Validation')
                ax.plot([0,1],[0,1], linestyle='--', color='gray', label='Perfect')
                ax.set_xlabel('Predicted probability')
                ax.set_ylabel('Observed frequency')
                ax.set_title('Reliability Curve (Calibration)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.error(f"Calibration computation failed: {e}")

        # Training data preview + download
        st.subheader("Training Data (CSV)")
        if file_exists(TRAINING_DATA_FILE):
            try:
                train_df_preview = pd.read_csv(TRAINING_DATA_FILE).head(15)
                st.dataframe(train_df_preview, width='stretch')
                with open(TRAINING_DATA_FILE, 'rb') as f:
                    st.download_button(
                        label="📥 Download Training Data",
                        data=f.read(),
                        file_name="student_health_data.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.error(f"Failed to load training data: {e}")
        else:
            st.info("Training data file not found at 'data/student_health_data.csv'.")

        # Model details (RF, SVM, MLP, HGB, Logistic Regression meta-model)
        st.subheader("Stacking Ensemble Model (4 Base Models)")
        st.info("🎯 Ensemble uses **ALL 4 MODELS**: RF + SVM + MLP + HGB → Logistic Regression → Final Prediction")
        model_info = load_trained_model()
        if model_info is None:
            st.info("Model file not found in 'models/stacking_ensemble.pkl'. Run the pipeline to train and save the model.")
        else:
            meta_model = model_info.get('model')
            base_models = model_info.get('base_models', {})
            st.markdown("**Meta-Model:** Logistic Regression (stacks predictions from all 4 base models)")
            with st.expander("Meta-Model Parameters", expanded=False):
                try:
                    st.json(meta_model.get_params())
                except Exception:
                    st.write(meta_model)
            st.markdown("**4 Base Models in Ensemble:**")
            cols_models = st.columns(4)
            # RF
            with cols_models[0]:
                st.caption("RandomForestClassifier")
                rf = base_models.get('rf')
                with st.expander("RF Parameters", expanded=False):
                    if rf is not None:
                        try:
                            st.json(rf.get_params())
                        except Exception:
                            st.write(rf)
                    else:
                        st.write("-")
            # SVM
            with cols_models[1]:
                st.caption("SVM (Calibrated)")
                svm = base_models.get('svm')
                with st.expander("SVM Parameters", expanded=False):
                    if svm is not None:
                        try:
                            st.json(svm.get_params())
                        except Exception:
                            st.write(svm)
                    else:
                        st.write("-")
            # MLP
            with cols_models[2]:
                st.caption("MLPClassifier")
                mlp = base_models.get('mlp')
                with st.expander("MLP Parameters", expanded=False):
                    if mlp is not None:
                        try:
                            st.json(mlp.get_params())
                        except Exception:
                            st.write(mlp)
                    else:
                        st.write("-")
            # HGB
            with cols_models[3]:
                st.caption("HistGradientBoostingClassifier")
                hgb = base_models.get('hgb')
                with st.expander("HGB Parameters", expanded=False):
                    if hgb is not None:
                        try:
                            st.json(hgb.get_params())
                        except Exception:
                            st.write(hgb)
                    else:
                        st.write("-")
            # Threshold (prefer attribute saved on meta-model)
            thr = getattr(meta_model, 'best_threshold_', model_info.get('threshold', 0.5))
            st.metric("Selected Threshold", f"{thr:.3f}")
            st.success("✅ All 4 models (RF + SVM + MLP + HGB) work together in the ensemble for best prediction!")

    # Regenerate outputs button for Page 1 - at the bottom
    st.markdown("---")
    st.subheader("Regenerate Pipeline")
    if st.button("🔄 Regenerate outputs (run pipeline on trained data)", key="regen_page1", type="primary", use_container_width=True):
        with st.spinner("Running health_prediction_pipeline.py on training data..."):
            python_cmd = "/usr/local/bin/python3"
            try:
                result = subprocess.run([python_cmd, PIPELINE_SCRIPT], capture_output=True, text=True, cwd=BASE_DIR, timeout=300)
                st.code(result.stdout or "(no stdout)", language="bash")
                if result.stderr:
                    st.warning("Warnings/Errors:\n" + result.stderr)
                st.success("✅ Pipeline finished. Refresh the page to see updated results.")
            except subprocess.TimeoutExpired:
                st.error("Pipeline took too long (>5 minutes). Check your data.")
            except Exception as e:
                st.error(f"Failed to run pipeline: {e}")

# TAB 2: Predict New Data
with tab2:
    st.title("🔮 Predict Health Score for New Data")
    st.caption("Upload a CSV file with new student data to get health predictions from the ensemble (RF + SVM + MLP + HGB)")
    
    st.markdown("---")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Read uploaded file
            new_data = pd.read_csv(uploaded_file)
            st.subheader("Uploaded Data Preview")
            st.dataframe(new_data.head(10), width='stretch')
            
            # New Data Pipeline Summary (for uploaded CSV only)
            with st.expander("📊 New Data Pipeline Summary", expanded=False):
                try:
                    rows_up, cols_up = new_data.shape
                    st.markdown(f"**Raw data:** {rows_up} rows × {cols_up} columns")
                    
                    # Apply feature engineering steps to uploaded data
                    tmp_data = new_data.copy()
                    window_size = 6
                    if 'Heart Rate (bpm)' in tmp_data.columns:
                        tmp_data['Heart_Rate_rolling_mean_30min'] = tmp_data['Heart Rate (bpm)'].rolling(window=window_size, min_periods=1).mean()
                        tmp_data['Heart_Rate_rolling_std_30min'] = tmp_data['Heart Rate (bpm)'].rolling(window=window_size, min_periods=1).std()
                    if 'Physical Activity Level (METs)' in tmp_data.columns:
                        tmp_data['Physical_Activity_rolling_mean_30min'] = tmp_data['Physical Activity Level (METs)'].rolling(window=window_size, min_periods=1).mean()
                        tmp_data['Physical_Activity_rolling_std_30min'] = tmp_data['Physical Activity Level (METs)'].rolling(window=window_size, min_periods=1).std()
                    num_cols_tmp = tmp_data.select_dtypes(include=[np.number]).columns
                    if len(num_cols_tmp) > 0:
                        tmp_data[num_cols_tmp] = tmp_data[num_cols_tmp].bfill().ffill()
                    if 'Sleep Duration (hours)' in tmp_data.columns:
                        sleep_bins = [0, 6, 8, np.inf]
                        sleep_labels = ['Short', 'Normal', 'Long']
                        tmp_data['Sleep_Category'] = pd.cut(tmp_data['Sleep Duration (hours)'], bins=sleep_bins, labels=sleep_labels, right=False)
                    if 'Gender' in tmp_data.columns and 'Sleep_Category' in tmp_data.columns:
                        tmp_data['Gender_Sleep_Group'] = tmp_data['Gender'].astype(str) + '_' + tmp_data['Sleep_Category'].astype(str)
                        sg_unique = tmp_data['Gender_Sleep_Group'].nunique()
                        sg_counts = tmp_data['Gender_Sleep_Group'].value_counts().to_dict()
                        st.markdown(f"**Subgroups (Gender × Sleep):** {sg_unique} unique groups")
                        st.markdown("**Subgroup distribution:**")
                        st.write(pd.Series(sg_counts))
                    engineered_cols = [c for c in tmp_data.columns if c not in new_data.columns]
                    st.markdown(f"**Engineered features added:** {len(engineered_cols)}")
                    st.markdown(f"**Final feature count for prediction:** ~{len(engineered_cols) + cols_up}")
                except Exception as e:
                    st.info(f"Pipeline summary unavailable: {e}")
            
            # Optional: decision threshold override for predictions
            default_thr = 0.5
            model_info_for_thr = load_trained_model()
            if model_info_for_thr is not None:
                try:
                    default_thr = getattr(model_info_for_thr.get('model'), 'best_threshold_', model_info_for_thr.get('threshold', 0.5))
                except Exception:
                    default_thr = model_info_for_thr.get('threshold', 0.5)
            st.caption("Decision threshold used to classify High vs Low health.")
            user_threshold = st.slider("Decision Threshold", min_value=0.1, max_value=0.9, value=float(default_thr), step=0.02)
            
            # Check required columns (make Body Temperature optional)
            required_cols = ["Heart Rate (bpm)", "Physical Activity Level (METs)", 
                           "Stress Level (1-10)", "Sleep Duration (hours)", "Gender"]
            optional_cols = ["Body Temperature (°C)"]
            missing_cols = [col for col in required_cols if col not in new_data.columns]
            optional_missing = [col for col in optional_cols if col not in new_data.columns]
            
            if missing_cols:
                st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                st.info("Required: Heart Rate (bpm), Physical Activity Level (METs), Stress Level (1-10), Sleep Duration (hours), Gender. Optional: Body Temperature (°C)")
            else:
                # Predict button
                if st.button("🚀 Run Predictions", type="primary"):
                    with st.spinner("Processing predictions..."):
                        # Inline ensemble predictions (RF + SVM + MLP + HGB) without writing files
                        try:
                            import pickle
                            model_path = os.path.join(BASE_DIR, "models", "stacking_ensemble.pkl")
                            if not file_exists(model_path):
                                st.error("Model file not found. Please run health_prediction_pipeline.py to train the ensemble.")
                                st.stop()
                            with open(model_path, 'rb') as f:
                                model_info = pickle.load(f)
                            meta_model = model_info['model']
                            base_models = model_info.get('base_models', {})
                            scaler = model_info['scaler']
                            feature_columns = model_info['feature_columns']
                            # Use user-selected threshold for classification
                            threshold = float(user_threshold)

                            # Feature engineering (match training)
                            df_pred = new_data.copy()
                            window_size = 6
                            df_pred['Heart_Rate_rolling_mean_30min'] = df_pred['Heart Rate (bpm)'].rolling(window=window_size, min_periods=1).mean()
                            df_pred['Heart_Rate_rolling_std_30min'] = df_pred['Heart Rate (bpm)'].rolling(window=window_size, min_periods=1).std()
                            df_pred['Physical_Activity_rolling_mean_30min'] = df_pred['Physical Activity Level (METs)'].rolling(window=window_size, min_periods=1).mean()
                            df_pred['Physical_Activity_rolling_std_30min'] = df_pred['Physical Activity Level (METs)'].rolling(window=window_size, min_periods=1).std()
                            num_cols = df_pred.select_dtypes(include=[np.number]).columns
                            df_pred[num_cols] = df_pred[num_cols].bfill().ffill()
                            if 'Sleep Duration (hours)' in df_pred.columns:
                                sleep_bins = [0, 6, 8, np.inf]
                                sleep_labels = ['Short', 'Normal', 'Long']
                                df_pred['Sleep_Category'] = pd.cut(df_pred['Sleep Duration (hours)'], bins=sleep_bins, labels=sleep_labels, right=False)

                            # One-hot encode categoricals and align columns
                            cat_cols = df_pred.select_dtypes(include=['object','category']).columns.tolist()
                            if cat_cols:
                                df_enc = pd.get_dummies(df_pred, columns=cat_cols, drop_first=True)
                            else:
                                df_enc = df_pred.copy()
                            for col in feature_columns:
                                if col not in df_enc.columns:
                                    df_enc[col] = 0
                            X_new = df_enc[feature_columns]

                            # Scale original features (same scaler)
                            try:
                                X_new_scaled = scaler.transform(X_new)
                            except Exception as e:
                                st.error(f"Scaler feature mismatch: {e}\nColumns expected: {len(feature_columns)}; provided: {X_new.shape[1]}")
                                st.write({
                                    'expected_columns': feature_columns,
                                    'provided_columns': X_new.columns.tolist()
                                })
                                st.stop()

                            # Build level-0 probabilities in training order: rf, svm, mlp, hgb
                            ordered_keys = [k for k in ['rf','svm','mlp','hgb'] if k in base_models]
                            lvl0_parts = []
                            for key in ordered_keys:
                                clf = base_models[key]
                                try:
                                    proba = clf.predict_proba(X_new_scaled)
                                except Exception as e:
                                    st.error(f"Model '{key}' input mismatch: {e}")
                                    st.info("Tip: Regenerate the ensemble via 'Regenerate outputs' to update models to the latest feature engineering.")
                                    st.stop()
                                lvl0_parts.append(proba[:, 1] if proba.shape[1] > 1 else proba[:, 0])
                            lvl0 = np.column_stack(lvl0_parts)
                            probabilities = meta_model.predict_proba(lvl0)[:, 1]
                            predictions_bin = (probabilities >= threshold).astype(int)

                            predictions = new_data.copy()
                            predictions['Predicted_Health'] = np.where(predictions_bin == 1, 'High', 'Low')
                            predictions['Confidence'] = np.where(predictions['Predicted_Health'] == 'High', probabilities, 1 - probabilities).round(3)
                        except Exception as e:
                            st.error(f"Prediction error: {e}")
                            st.stop()

                        st.success("✅ Predictions completed using ensemble (RF + SVM + MLP + HGB)!")
                        # Add 'Overall health score' if absent (predicted)
                        # Define score as probability of High health in [0,100], regardless of class
                        if 'Overall health score' not in predictions.columns:
                            try:
                                predictions['Overall health score (predicted)'] = (probabilities * 100.0).round(1)
                            except Exception:
                                predictions['Overall health score (predicted)'] = np.nan

                        # Add raw probability column for clarity (0-1)
                        predictions['Probability of Healthy (0-1)'] = probabilities.round(3)
                        st.subheader("Prediction Results")
                        st.dataframe(predictions, width='stretch')

                        # Summary statistics
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Samples", len(predictions))
                        col2.metric("Predicted Healthy", len(predictions[predictions['Predicted_Health'] == 'High']))
                        col3.metric("Predicted Unhealthy", len(predictions[predictions['Predicted_Health'] == 'Low']))

                        # Overall score quick stats
                        if 'Overall health score' in predictions.columns:
                            overall_col = 'Overall health score'
                        elif 'Overall health score (predicted)' in predictions.columns:
                            overall_col = 'Overall health score (predicted)'
                        else:
                            overall_col = None
                        if overall_col:
                            st.caption("Overall health score summary")
                            m1, m2, m3 = st.columns(3)
                            m1.metric("Mean", f"{predictions[overall_col].mean():.1f}")
                            m2.metric("Median", f"{predictions[overall_col].median():.1f}")
                            m3.metric("Std", f"{predictions[overall_col].std():.1f}")

                        # Build subgroup from Gender + Sleep Duration
                        st.subheader("Predictions Analysis (New Data)")
                        preds = predictions.copy()
                        sleep_bins = [0, 6, 8, float('inf')]
                        sleep_labels = ['Short', 'Normal', 'Long']
                        if 'Sleep Duration (hours)' in preds.columns:
                            preds['Sleep_Category'] = pd.cut(
                                preds['Sleep Duration (hours)'],
                                bins=sleep_bins,
                                labels=sleep_labels,
                                right=False
                            )
                        else:
                            preds['Sleep_Category'] = 'Unknown'
                        if 'Gender' not in preds.columns:
                            preds['Gender'] = 'Unknown'
                        preds['Gender_Sleep_Group'] = preds['Gender'].astype(str) + '_' + preds['Sleep_Category'].astype(str)

                        cc1, cc2 = st.columns(2)
                        with cc1:
                            st.caption("Predicted Health by Subgroup")
                            try:
                                fig, ax = plt.subplots(figsize=(5,3))
                                group_cols = ['Gender_Sleep_Group', 'Predicted_Health']
                                count_df = preds.groupby(group_cols).size().reset_index(name='Count')
                                pivot_df = count_df.pivot(index='Gender_Sleep_Group', columns='Predicted_Health', values='Count').fillna(0)
                                pivot_df = pivot_df.sort_index()
                                pivot_df.plot(kind='bar', stacked=True, ax=ax, color={'High':'#2ecc71','Low':'#e74c3c'})
                                ax.set_ylabel('Count')
                                ax.set_xlabel('Subgroup')
                                ax.legend(title='Predicted')
                                ax.grid(axis='y', alpha=0.3)
                                st.pyplot(fig)
                            except Exception as e:
                                st.error(f"Failed to render predicted counts: {e}")
                        with cc2:
                            st.caption("Mean Confidence by Subgroup")
                            try:
                                fig, ax = plt.subplots(figsize=(5,3))
                                conf_df = preds.groupby('Gender_Sleep_Group')['Confidence'].mean().reset_index()
                                conf_df = conf_df.dropna(subset=['Confidence'])
                                ax.barh(conf_df['Gender_Sleep_Group'], conf_df['Confidence'], color="#3498db", alpha=0.8)
                                ax.set_xlabel('Mean Confidence')
                                ax.set_xlim(0,1.05)
                                ax.grid(axis='x', alpha=0.3)
                                st.pyplot(fig)
                            except Exception as e:
                                st.error(f"Failed to render confidence chart: {e}")

                        st.subheader("Health Score Analysis")
                        hs1, hs2 = st.columns(2)
                        overall_col = None
                        if 'Overall health score' in preds.columns:
                            overall_col = 'Overall health score'
                        elif 'Overall health score (predicted)' in preds.columns:
                            overall_col = 'Overall health score (predicted)'
                        if overall_col:
                            with hs1:
                                st.caption("Distribution of Overall Health Score")
                                try:
                                    fig, ax = plt.subplots(figsize=(5,3))
                                    vals = preds[overall_col].dropna()
                                    ax.hist(vals, bins=10, color="#16a085", alpha=0.8, edgecolor="black")
                                    ax.set_xlabel("Overall Health Score (0–100)")
                                    ax.set_ylabel("Count")
                                    ax.set_xlim(0, 100)
                                    ax.axvline(80, color='red', linestyle='--', linewidth=1)
                                    ax.grid(axis='y', alpha=0.3)
                                    st.pyplot(fig)
                                except Exception as e:
                                    st.error(f"Failed to render score histogram: {e}")
                            with hs2:
                                st.caption("Score by Gender × Sleep Group")
                                try:
                                    fig, ax = plt.subplots(figsize=(5,3))
                                    plot_df = preds.copy()
                                    plot_df = plot_df.dropna(subset=[overall_col])
                                    if 'Gender_Sleep_Group' not in plot_df.columns:
                                        plot_df['Gender_Sleep_Group'] = plot_df['Gender'].astype(str)
                                    sns.boxplot(data=plot_df, x='Gender_Sleep_Group', y=overall_col, ax=ax, color="#2980b9")
                                    ax.set_xlabel("Subgroup")
                                    ax.set_ylabel("Overall Health Score")
                                    ax.tick_params(axis='x', rotation=30)
                                    ax.set_ylim(0, 100)
                                    ax.axhline(80, color='red', linestyle='--', linewidth=1)
                                    counts = plot_df['Gender_Sleep_Group'].value_counts()
                                    xlabels = [lbl.get_text() for lbl in ax.get_xticklabels()]
                                    ax.set_xticklabels([f"{lbl} (n={counts.get(lbl, 0)})" for lbl in xlabels])
                                    ax.grid(axis='y', alpha=0.3)
                                    st.pyplot(fig)
                                except Exception as e:
                                    st.error(f"Failed to render subgroup boxplot: {e}")
                        else:
                            st.info("No overall health score column found in predictions.")

                        csv = predictions.to_csv(index=False)
                        st.download_button(label="📥 Download Predictions", data=csv, file_name="health_predictions.csv", mime="text/csv")
        
        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        st.info("👆 Upload a CSV file to get started")
        
        # Show example format
        st.subheader("Example CSV Format")
        example_data = pd.DataFrame({
            "Heart Rate (bpm)": [75, 82],
            "Physical Activity Level (METs)": [5.2, 3.8],
            "Stress Level (1-10)": [6, 8],
            "Sleep Duration (hours)": [7.5, 5.5],
            "Gender": ["Male", "Female"],
            "Body Temperature (°C)": [37.0, 36.8]
        })
        st.dataframe(example_data, width='stretch')

    # Regenerate outputs button for Page 2 - at the bottom (works ONLY with uploaded data)
    st.markdown("---")
    st.subheader("Regenerate Pipeline on Uploaded Data")
    if uploaded_file is not None:
        # Check if uploaded file has required columns for training
        required_training_cols = ['Date and Time', 'Heart Rate (bpm)', 'Physical Activity Level (METs)', 
                                 'Stress Level (1-10)', 'Sleep Duration (hours)', 'Gender', 'Overall Health Score']
        missing_for_training = [col for col in required_training_cols if col not in new_data.columns]
        
        if missing_for_training:
            st.warning(f"⚠️ Cannot train pipeline on this data. Missing required columns for training: {', '.join(missing_for_training)}")
            st.info("💡 To train the pipeline, your CSV must include: Date and Time, Overall Health Score, and all health metrics.")
        else:
            # Save uploaded file temporarily
            temp_upload_path = os.path.join(BASE_DIR, "data", "temp_upload.csv")
            # Read from new_data which was already loaded above
            new_data.to_csv(temp_upload_path, index=False)
            
            if st.button("🔄 Train pipeline on uploaded data", key="regen_page2", type="primary", use_container_width=True):
                with st.spinner("Training pipeline on your uploaded data..."):
                    python_cmd = "/usr/local/bin/python3"
                    try:
                        # Modify environment to use temp_upload.csv instead of student_health_data.csv
                        env = os.environ.copy()
                        env['TRAINING_DATA_FILE'] = temp_upload_path
                        result = subprocess.run([python_cmd, PIPELINE_SCRIPT], 
                                              capture_output=True, 
                                              text=True, 
                                              cwd=BASE_DIR, 
                                              timeout=300,
                                              env=env)
                        st.code(result.stdout or "(no stdout)", language="bash")
                        if result.stderr:
                            st.warning("Warnings/Errors:\n" + result.stderr)
                        st.success("✅ Pipeline trained on your uploaded data! Models updated.")
                    except subprocess.TimeoutExpired:
                        st.error("Pipeline took too long (>5 minutes). Check your data.")
                    except Exception as e:
                        st.error(f"Failed to run pipeline: {e}")
    else:
        st.info("Upload a CSV file first to train the pipeline on new data.")
