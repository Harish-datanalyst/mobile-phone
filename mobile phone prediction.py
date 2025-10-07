import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# ---- Config ----

DATA_PATH =r"D:\\intern projects uni\\mobile phone\\dataset.csv"   # change if necessary
OUTPUT_DIR = "/mnt/data/output_models"
TEST_SIZE = 0.20
RANDOM_STATE = 42

# ---- Load ----

df = pd.read_csv(DATA_PATH)
print("Loaded:", DATA_PATH)
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# Detect target column (robust to case)

target_candidates = [c for c in df.columns if 'price' in c.lower()]
if not target_candidates:
  raise RuntimeError("No target column found (looking for 'price' in column names).")
target_col = target_candidates[0]
print("Using target column:", target_col)

# ---- Quick EDA ----

print("\nClass distribution:")
print(df[target_col].value_counts().sort_index())

# ---- Prepare data ----

X = df.drop(columns=[target_col])
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE,
stratify=y, random_state=RANDOM_STATE)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ---- Models to try ----

models = {
"LogisticRegression": LogisticRegression(multi_class="multinomial", solver="saga", max_iter=5000, random_state=RANDOM_STATE),
"RandomForest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
"GradientBoosting": GradientBoostingClassifier(n_estimators=200, random_state=RANDOM_STATE)
}

# ---- Train & evaluate ----

results = {}
for name, model in models.items():
  model.fit(X_train_s, y_train)
preds = model.predict(X_test_s)
acc = accuracy_score(y_test, preds)
cv_scores = cross_val_score(model, scaler.transform(X), y, cv=5, scoring='accuracy')
print(f"{name} -> test_acc={acc:.4f}, cv_mean={cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
results[name] = {
"model": model, "test_acc": acc, "cv_mean": cv_scores.mean(),
"classification_report": classification_report(y_test, preds, digits=4, output_dict=True),
"conf_mat": confusion_matrix(y_test, preds)
}

# ---- Choose best by test accuracy ----

best_name = max(results.keys(), key=lambda k: results[k]["test_acc"])
best_model = results[best_name]["model"]
print("\nBest model selected:", best_name)

# Print classification report for best model

print("\nClassification report (best model):")
print(classification_report(y_test, best_model.predict(X_test_s), digits=4))

# Confusion matrix

cm = results[best_name]["conf_mat"]
print("\nConfusion matrix:\n", cm)

# ---- Optional: plot confusion matrix ----

plt.figure(figsize=(5,4))
plt.imshow(cm, interpolation='nearest')
plt.title(f'Confusion Matrix - {best_name}')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.colorbar()
for i in range(cm.shape[0]):
 for j in range(cm.shape[1]):
   plt.text(j, i, cm[i,j], ha='center', va='center')
plt.tight_layout()
plt.show()

# ---- Feature importances if applicable ----

if hasattr(best_model, "feature_importances_"):
  importances = best_model.feature_importances_
idx = np.argsort(importances)[::-1]
top_n = min(12, len(importances))
print("\nTop features by importance:")
for i in range(top_n):
  print(f"{i+1}. {X.columns[idx[i]]}: {importances[idx[i]]:.4f}")

# ---- Save scaler & model ----

os.makedirs(OUTPUT_DIR, exist_ok=True)
joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.joblib"))
joblib.dump(best_model, os.path.join(OUTPUT_DIR, f"{best_name}.joblib"))
print(f"\nSaved scaler and best model to {OUTPUT_DIR}")

# ---- Example usage: load and predict ----

# s = joblib.load("scaler.joblib")

# m = joblib.load("LogisticRegression.joblib")

# sample = X_test.iloc[[0]]              # DataFrame single row

# sample_s = s.transform(sample)

# print("predicted:", m.predict(sample_s))
