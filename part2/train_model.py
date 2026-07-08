import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, classification_report, roc_curve, roc_auc_score, precision_score, recall_score, f1_score
from sympy import fps

### first task: load the clearn csv file and we define labels
df = pd.read_csv("cleaned_data.csv")
if "customerID" in df.columns:
    df = df.drop(columns=["customerID"])
X = df.drop(columns=["MonthlyCharges", "Churn"])
y_reg = df["MonthlyCharges"]
y_clf = (df["Churn"] == "yes").astype(int)
print(f"Features shape: {X.shape} | Regression Target shape: {y_reg.shape} | Classification Target shape: {y_clf.shape}")

### second task: Encode categorical columns
## we use one hot encoding
categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
print(f"Categorical columns to one-hot encode: {categorical_cols}")
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
for col in X_encoded.columns:
    if X_encoded[col].dtype == "bool":
        X_encoded[col] = X_encoded[col].astype(int)

### third task: Leak-free train-test split and scaling
X_train_reg, X_test_reg, y_reg_train, y_reg_test = train_test_split(X_encoded, y_reg, test_size=0.2, random_state=42)
X_train_clf, X_test_clf, y_clf_train, y_clf_test = train_test_split(X_encoded, y_clf, test_size=0.2, random_state=42)
scaler_reg =StandardScaler()
X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
X_test_reg_scaled = scaler_reg.transform(X_test_reg)
scaler_clf = StandardScaler()
X_train_clf_scaled = scaler_clf.fit_transform(X_train_clf)
X_test_clf_scaled = scaler_clf.transform(X_test_clf)

### fourth task: Regression model 
lr = LinearRegression()
lr.fit(X_train_reg_scaled, y_reg_train)
y_pred_reg = lr.predict(X_test_reg_scaled)
mse_lr = mean_squared_error(y_reg_test, y_pred_reg)
r2_lr = r2_score(y_reg_test, y_pred_reg)
print(f"OLS Linear Regression -> MSE: {mse_lr:.4f} | R²: {r2_lr:.4f}")
coef_df = pd.DataFrame({"Feature": X_encoded.columns, "Coefficient": lr.coef_})
coef_df["Abs_Coef"] = coef_df["Coefficient"].abs()
coef_df = coef_df.sort_values(by="Abs_Coef", ascending=False)
print("\nTop Features by Weight Profile:\n", coef_df)
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_reg_scaled, y_reg_train)
y_pred_ridge = ridge.predict(X_test_reg_scaled)
mse_ridge = mean_squared_error(y_reg_test, y_pred_ridge)
r2_ridge = r2_score(y_reg_test, y_pred_ridge)
print(f"Ridge Regression -> MSE: {mse_ridge:.4f} | R²: {r2_ridge:.4f}")

### fifth task: Classification model Logistic Regression and we use also ROC and AUC curve
print("Before Class Balance Check:\n", y_clf_train.value_counts(normalize=True))
clf_model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
clf_model.fit(X_train_clf_scaled, y_clf_train)
y_pred_clf = clf_model.predict(X_test_clf_scaled)
y_prob_clf = clf_model.predict_proba(X_test_clf_scaled)[:, 1]
print("\nConfusion Matrix:\n", confusion_matrix(y_clf_test, y_pred_clf))
print("\nClassification Report:\n", classification_report(y_clf_test, y_pred_clf))
fpr, tpr, thresholds = roc_curve(y_clf_test, y_prob_clf)
auc_val = roc_auc_score(y_clf_test, y_prob_clf)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {auc_val:.3f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Postive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.close()
print(f"ROC Curve saved to part2/roc_curve.png. AUC: {auc_val:.4f}")

### b) Decision threshold sensitivity
threshold_list = [0.30, 0.40, 0.50, 0.60, 0.70]
print(f"{"Threshold":<10} | {"Precision":<10} | {"Recall":<10} | {"F1-Score":<10}")
for t in threshold_list:
    preds = (y_prob_clf >=t).astype(int)
    p = precision_score(y_clf_test, preds, zero_division=0)
    r = recall_score(y_clf_test, preds, zero_division=0)
    f1 = f1_score(y_clf_test, preds, zero_division=0)
    print(f"{t:<10.2f} | {p:<10.4f} | {r:<10.4} | {f1:<10.4f}")

### sixth task: Regularization experiment on Logistic Regression
clf_weak = LogisticRegression(max_iter=1000, C=0.01, class_weight="balanced", random_state=42)
clf_weak.fit(X_train_clf_scaled, y_clf_train)
y_prob_weak = clf_weak.predict_proba(X_test_clf_scaled)[:, 1]
auc_weak = roc_auc_score(y_clf_test, y_prob_weak)
print(f"Baseline Model (C=1.0) AUC: {auc_val:.4f}")
print(f"Regularized Model (C=0.01) AUC: {auc_weak:.4f}")

### siventh task: Bootstrap confidence interval for AUC difference:
np.random.seed(42)
bootstrap_diffs = []
n_iterations = 500
n_samples = len(y_clf_test)
y_clf_test_arr = np.array(y_clf_test)
for _ in range(n_iterations):
    indices = np.random.choice(n_samples, size=n_samples, replace=True)
    boot_auc_baseline = roc_auc_score(y_clf_test_arr[indices], y_prob_clf[indices])
    boot_auc_weak = roc_auc_score(y_clf_test_arr[indices], y_prob_weak[indices])
    bootstrap_diffs.append(boot_auc_baseline - boot_auc_weak)
mean_diff = np.mean(bootstrap_diffs)
lower_ci = np.percentile(bootstrap_diffs, 2.5)
upper_ci = np.percentile(bootstrap_diffs, 97.5)
print(f"Mean AUC Difference: {mean_diff:.4f}")
print(f"95% Bootstrap Confidence Interval: [{lower_ci:.4f}, {upper_ci:.4f}]")