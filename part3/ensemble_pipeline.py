import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

### before the start we nned to read the clearned csv and train and teat split
df = pd.read_csv("cleaned_data.csv")
if "customerID" in df.columns:
    df = df.drop(columns=["customerID"])
X = df.drop(columns=["MonthlyCharges", "Churn"])  
y_clf = (df["Churn"] == "yes").astype(int)

X_encoded = pd.get_dummies(X, drop_first=True)
for col in X_encoded.columns:
    if X_encoded[col].dtype == "bool":
        X_encoded[col] = X_encoded[col].astype(int)
X_train, X_test, y_clf_train, y_clf_test = train_test_split(
    X_encoded, y_clf, test_size=0.2, random_state=42, stratify=y_clf

)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

### first task: Decision Tree baseline
dt_unconstrained = DecisionTreeClassifier(max_depth=None, random_state=42)
dt_unconstrained.fit(X_train_scaled, y_clf_train)
train_acc_dt1 = accuracy_score(y_clf_train, dt_unconstrained.predict(X_train_scaled))
test_acc_dt1 = accuracy_score(y_clf_test, dt_unconstrained.predict(X_test_scaled))
print(f"Unconstrained Tree -> Train Acc: {train_acc_dt1:.4f} | Test Acc: {test_acc_dt1:.4f}")

### second task: Controlled Decision Tree:
dt_controlled = DecisionTreeClassifier(max_depth=5, min_samples_split=20, random_state=42)
dt_controlled.fit(X_train_scaled, y_clf_train)
train_acc_dt2 = accuracy_score(y_clf_train, dt_controlled.predict(X_train_scaled))
test_acc_dt2 = accuracy_score(y_clf_test, dt_controlled.predict(X_test_scaled))
print(f"Controlled Tree -> Train Acc: {train_acc_dt2:.4f} | Test Acc: {test_acc_dt2:.4f}")

### third task: Gini vs Entropy comparison
dt_gini = DecisionTreeClassifier(max_depth=5, criterion="gini", random_state=42)
dt_gini.fit(X_train_scaled, y_clf_train)
test_acc_gini = accuracy_score(y_clf_test, dt_gini.predict(X_test_scaled))
dt_entropy = DecisionTreeClassifier(max_depth=5, criterion="entropy", random_state=42)
dt_entropy.fit(X_train_scaled, y_clf_train)
test_acc_entropy =accuracy_score(y_clf_test, dt_entropy.predict(X_test_scaled))
print(f"Gini (max_depth=5) Test Acc: {test_acc_gini:.4f}")
print(f"Entropy (max_depth=5) Test Acc: {test_acc_entropy:.4f}")

### fourth task: Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train_scaled, y_clf_train)
train_acc_rf = accuracy_score(y_clf_train, rf.predict(X_train_scaled))
test_acc_rf = accuracy_score(y_clf_test, rf.predict(X_test_scaled))
auc_rf = roc_auc_score(y_clf_test, rf.predict_proba(X_test_scaled)[:, 1])
print(f"Random Forest -> Train Acc: {train_acc_rf:.4f} | Test Acc: {test_acc_rf:.4f} | ROC-AUC: {auc_rf:.4f}")

importances = rf.feature_importances_
feat_imp = pd.DataFrame({"Feature": X_encoded.columns, "Importance": importances})
feat_imp = feat_imp.sort_values(by="Importance", ascending=False)
print("\nTop 5 features by andom Forest Importance:")
print(feat_imp.head(5).to_string(index=False))

### fourth task : a).gradient boosting
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb.fit(X_train_scaled, y_clf_train)
train_acc_gb = accuracy_score(y_clf_train, gb.predict(X_train_scaled))
test_acc_gb = accuracy_score(y_clf_test, gb.predict(X_test_scaled))
auc_gb = roc_auc_score(y_clf_test, gb.predict_proba(X_test_scaled)[:, 1])
print(f"Gradient boosting -> Train Acc: {train_acc_gb:.4f} | Test Acc: {test_acc_gb:.4f} | ROC-AUC: {auc_gb:.4f}")

### fourth task: b). Feature ablation study
lowest_5_features = feat_imp.tail(5)["Feature"].tolist()
print(f"5 Lowest Importance Features Removed: {lowest_5_features}")
X_train_reduced = X_train_scaled[:, ~X_encoded.columns.isin(lowest_5_features)]
X_test_reduced = X_test_scaled[:, ~X_encoded.columns.isin(lowest_5_features)]
rf_reduced = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_reduced.fit(X_train_reduced, y_clf_train)
auc_rf_full = auc_rf
auc_rf_reduced = roc_auc_score(y_clf_test, rf_reduced.predict_proba(X_test_reduced)[:, 1])
print(f"Full Features Set ROC-AUC: {auc_rf_full:.4f}")
print(f"Reduced Features Set (r removed) ROC-AUC: {auc_rf_reduced:.4f}")

### fifth task: Cross-validated comparison
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
models_cv = {
    "Logistic Regression" : LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    "Controlled Decision Tree" : DecisionTreeClassifier(max_depth=5, min_samples_split=20, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
}
cv_results ={}
for name, model in models_cv.items():
    scores = cross_val_score(model, X_train_scaled, y_clf_train, cv=cv, scoring="roc_auc")
    cv_results[name] = (scores.mean(), scores.std())
    print(f"{name:<25} -> 5-Fold CV AUC: Mean = {scores.mean():.4f}, Std ={scores.std():.4f}")

### sixth task: Hyperparameter tuning with GridSearchCV
pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    RandomForestClassifier(random_state=42)
)
param_grid= {
    "randomforestclassifier__n_estimators": [50, 100, 200],
    "randomforestclassifier__max_depth": [5, 10, None],
    "randomforestclassifier__min_samples_leaf": [1, 5]

}
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring="roc_auc" ,
    n_jobs=1
)
grid_search.fit(X_train, y_clf_train)
print(f"Best Hyperparameters Found: {grid_search.best_params_}")
print(f"Best CV ROC-AUC Score: {grid_search.best_score_:.4f}")
best_pipeline = grid_search.best_estimator_

### seventh task: Manual learning curve
fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
print(f"{"Training Fractions":<18} | {"Training AUC" :<15} | {"Test AUC":<15}")
for f in fractions:
    subset_size = int(f * len(X_train))
    X_sub = X_train.iloc[:subset_size]
    y_sub = y_clf_train.iloc[:subset_size]
    best_pipeline.fit(X_sub, y_sub)
    train_probs = best_pipeline.predict_proba(X_sub)[:, 1]
    test_probs = best_pipeline.predict_proba(X_test)[:, 1]
    train_auc_sub = roc_auc_score(y_sub, train_probs)
    test_auc_sub = roc_auc_score(y_clf_test, test_probs)
    print(f"{f*100:<17.0f}% | {train_auc_sub:<15.4f}  | {test_auc_sub:<15.4f}")

### eight task: Serialize the best model
best_pipeline.fit(X_train, y_clf_train)
model_filepath = "best_model.pkl"
joblib.dump(best_pipeline, model_filepath)
print(f"Successfully saved best pipeline to '{model_filepath}'")
loaded_model = joblib.load(model_filepath)
sample_rows = X_test.iloc[:2]
sample_predictions = loaded_model.predict(sample_rows)
sample_probabilities = loaded_model.predict_proba(sample_rows)[:, 1]
print("\nReload-and-Predict Verification Output:")
print("Sample Input Features:\n", sample_rows)
print("Class Predictions:", sample_predictions)
print("Churn Probabilities:", sample_probabilities)
print("Model reloaded and successfully verified!")