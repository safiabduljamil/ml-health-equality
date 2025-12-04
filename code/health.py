# =============================================================================
# Import der notwendigen Bibliotheken
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from collections import Counter

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score

# =============================================================================
# Konfiguration und globale Einstellungen
# =============================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

RANDOM_SEED = 42
TEST_SIZE = 0.3
N_SPLITS_CV = 5

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, "data", "student_health_data.csv")
MODEL_FILE = None  # No longer persisting standalone RF to avoid duplication
RF_RESULTS_CSV = None  # Skip saving RF-only CSV

# =============================================================================
# Schritt 1: Daten laden und validieren
# =============================================================================
logging.info("Schritt 1: Daten laden und validieren")
try:
    df = pd.read_csv(DATA_FILE)
    logging.info(f"Daten erfolgreich aus '{DATA_FILE}' geladen. Shape: {df.shape}")
except FileNotFoundError:
    logging.error(f"FEHLER: Die Datei '{DATA_FILE}' wurde nicht gefunden. Bitte Pfad überprüfen.")
    exit()
except Exception as e:
    logging.error(f"Ein unerwarteter Fehler ist beim Laden der Daten aufgetreten: {e}")
    exit()

# =============================================================================
# Schritt 2: Forschungsfrage und Datenaufbereitung
# =============================================================================
logging.info("Schritt 2: Forschungsfrage und Datenaufbereitung")
# Forschungsfrage: Hat das Klassifikationsmodell eine gleichwertige Vorhersageleistung für den 
# 'Overall Health Score' in den verschiedenen Subgruppen, definiert durch die Kombination 
# von Geschlecht und kategorisierter Schlafdauer?

# --- Datenaufbereitung für die Klassifikation und Subgruppenanalyse ---

# 1. Zielvariable 'Overall Health Score' kategorisieren
def categorize_health_score(score):
    if score <= 75:
        return "Low"
    elif score <= 90:
        return "Medium"
    else:
        return "High"

df['Health Score Category'] = df['Overall Health Score'].apply(categorize_health_score)

# 2. 'Sleep Duration' für die Subgruppenbildung kategorisieren
def categorize_sleep(hours):
    if hours < 6:
        return "Short"
    elif hours <= 8:
        return "Normal"
    else:
        return "Long"

df['Sleep Category'] = df['Sleep Duration (hours)'].apply(categorize_sleep)

# Definition von Features und Target
numerical_features = ["Physical Activity Level (METs)", "Stress Level (1-10)", "Heart Rate (bpm)", "Body Temperature (°C)"]
categorical_features = ["Gender"]
features = numerical_features + categorical_features
target = "Health Score Category"

# Überprüfung, ob alle benötigten Spalten im DataFrame vorhanden sind
required_cols = features + ["Overall Health Score", "Sleep Duration (hours)"]
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    logging.error(f"FEHLER: Die folgenden Spalten fehlen in der Datei: {missing_cols}")
    exit()

# =============================================================================
# Schritt 3: Explorative Datenanalyse (EDA)
# =============================================================================
logging.info("Schritt 3: Explorative Datenanalyse (EDA)")

# Verteilung der neuen Zielvariable
plt.figure(figsize=(8, 5))
sns.countplot(x=target, data=df, order=['Low', 'Medium', 'High'])
plt.title("Verteilung der Gesundheitskategorien")
plt.savefig(os.path.join(BASE_DIR, "visualizations", "health_category_distribution.png"))
plt.close()

# Verteilung der Gesundheitskategorien pro Subgruppe (Geschlecht & Schlafdauer)
plt.figure(figsize=(12, 7))
sns.catplot(data=df, x='Sleep Category', hue='Gender', col=target, kind='count', 
            col_wrap=3, order=['Short', 'Normal', 'Long'], height=4, aspect=0.8)
plt.suptitle("Verteilung der Gesundheitskategorien nach Subgruppen", y=1.02)
plt.savefig(os.path.join(BASE_DIR, "visualizations", "subgroup_distribution.png"))
plt.close()

# =============================================================================
# Schritt 4: Datenvorverarbeitung für das Modell
# =============================================================================
logging.info("Schritt 4: Datenvorverarbeitung für das Modell")

# Entfernen von Zeilen mit fehlenden Werten
data = df[required_cols + [target, 'Sleep Category']].dropna()
logging.info(f"{df.shape[0] - data.shape[0]} Zeilen mit fehlenden Werten wurden entfernt.")

X = data[features]
y = data[target]

# Aufteilung in Trainings- und Testdaten (mit Stratifizierung, um die Verteilung der Zielvariable beizubehalten)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y)
logging.info(f"Daten aufgeteilt: {X_train.shape[0]} Trainings-Samples, {X_test.shape[0]} Test-Samples.")

# Erstellen des Preprocessing-Pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# =============================================================================
# Schritt 5: Modellerstellung und Training
# =============================================================================
logging.info("Schritt 5: Modellerstellung und Training")

# Wir wählen RandomForest als unser primäres Modell, da es oft eine gute Leistung zeigt.
pipeline_rf = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=RANDOM_SEED, class_weight='balanced'))
])

# Hyperparameter-Optimierung für RandomForest
param_grid_rf = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10, 20],
    'classifier__min_samples_leaf': [2, 4]
}

cv_strategy = KFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_SEED)
grid_rf = GridSearchCV(pipeline_rf, param_grid_rf, cv=cv_strategy, scoring='f1_weighted', n_jobs=-1)

logging.info("Starte Hyperparameter-Optimierung für den Random Forest Classifier...")
grid_rf.fit(X_train, y_train)
best_model = grid_rf.best_estimator_
logging.info(f"Beste Parameter für Random Forest: {grid_rf.best_params_}")

# =============================================================================
# Schritt 6: Evaluation der Gesamtleistung des Modells
# =============================================================================
logging.info("Schritt 6: Evaluation der Gesamtleistung des Modells")

y_pred_total = best_model.predict(X_test)

print("\n" + "="*60)
print("Gesamtleistung des Modells auf dem Test-Set")
print("="*60)
print(classification_report(y_test, y_pred_total))
print("="*60)

# =============================================================================
# Schritt 7: Subgruppenanalyse
# =============================================================================
logging.info("Schritt 7: Subgruppenanalyse")

# Füge die Subgruppen-Informationen zum Test-Set hinzu
X_test_subgroups = X_test.copy()
X_test_subgroups['Sleep Category'] = data.loc[X_test.index, 'Sleep Category']
X_test_subgroups['Actual Health Category'] = y_test

# Definiere die Subgruppen
subgroups = X_test_subgroups.groupby(['Gender', 'Sleep Category'])

subgroup_results = []

print("\n" + "="*80)
print("Analyse der Modellleistung auf Subgruppen")
print("="*80)
print(f"{'Gender':<10} | {'Sleep Category':<15} | {'Anzahl Samples':<15} | {'Accuracy':>10} | {'F1-Score (gew.)':>18}")
print("-"*80)

for (gender, sleep_cat), group_df in subgroups:
    if len(group_df) == 0:
        continue

    X_sub = group_df[features]
    y_sub_true = group_df['Actual Health Category']
    
    # Vorhersagen für die Subgruppe
    y_sub_pred = best_model.predict(X_sub)
    
    # Metriken berechnen
    accuracy = accuracy_score(y_sub_true, y_sub_pred)
    f1 = f1_score(y_sub_true, y_sub_pred, average='weighted', zero_division=0)
    
    # Ergebnisse speichern
    subgroup_results.append({
        'Gender': gender,
        'Sleep Category': sleep_cat,
        'Sample Count': len(group_df),
        'Accuracy': accuracy,
        'F1-Score (weighted)': f1
    })
    
    print(f"{gender:<10} | {sleep_cat:<15} | {len(group_df):<15} | {accuracy:>10.2f} | {f1:>18.2f}")

print("="*80)

# Umwandlung der Ergebnisse in ein DataFrame für die Visualisierung
results_df = pd.DataFrame(subgroup_results)

# =============================================================================
# Schritt 8: Visualisierung der Subgruppen-Ergebnisse
# =============================================================================
logging.info("Schritt 8: Visualisierung der Subgruppen-Ergebnisse")

if not results_df.empty:
    # Ensure visualizations directory exists
    os.makedirs(os.path.join(BASE_DIR, "visualizations"), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    fig.suptitle('Vergleich der Modellleistung über Subgruppen', fontsize=16)

    # Plot für Accuracy
    sns.barplot(data=results_df, x='Accuracy', y='Sleep Category', hue='Gender', ax=axes[0])
    axes[0].set_title('Accuracy pro Subgruppe')
    axes[0].set_xlabel('Accuracy')
    axes[0].set_ylabel('Schlafkategorie')
    axes[0].set_xlim(0, 1)
    axes[0].legend(title='Geschlecht')

    # Plot für F1-Score
    sns.barplot(data=results_df, x='F1-Score (weighted)', y='Sleep Category', hue='Gender', ax=axes[1])
    axes[1].set_title('Gewichteter F1-Score pro Subgruppe')
    axes[1].set_xlabel('F1-Score (weighted)')
    axes[1].set_ylabel('')
    axes[1].set_xlim(0, 1)
    axes[1].legend(title='Geschlecht')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(BASE_DIR, "visualizations", "subgroup_performance_comparison.png"))
    plt.close()
else:
    logging.warning("Keine Subgruppen-Ergebnisse zum Visualisieren vorhanden.")

# =============================================================================
# Schritt 9: Fazit und Dokumentation der Erkenntnisse
# =============================================================================
# Die Ergebnisse der Subgruppenanalyse (Tabelle und Visualisierung) geben direkt
# Antwort auf die Forschungsfrage. Man kann nun diskutieren, ob die Leistung
# (z.B. F1-Score) über die Gruppen hinweg als "gleichwertig" betrachtet werden kann.
# Unterschiede in der Leistung können auf eine geringe Anzahl von Samples in einer
# Subgruppe oder auf tatsächliche Muster in den Daten zurückzuführen sein, bei denen
# das Modell Schwierigkeiten hat, für eine bestimmte Gruppe zu generalisieren.
# Diese Erkenntnisse sind zentral für das Kapitel "Diskussion" im Bericht.
logging.info("Prozess abgeschlossen. Ergebnisse können nun für den Bericht verwendet werden.")

# =============================================================================
# Schritt 10: Ergebnisse speichern (RF Modell und Subgruppenleistung)
# =============================================================================
try:
    # Skip persistence to keep project lean
    logging.info("RF persistence skipped (no model/CSV saved).")
except Exception as e:
    logging.error(f"Fehler beim Überspringen der Speicherung: {e}")


