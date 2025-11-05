## 1. Summary: Motivation & Limitations of Accuracy

**What I learned / key takeaways**

* Accuracy is simple: proportion of correct predictions.
* But it **hides** how many positive cases the model missed (false negatives), or how many negatives it falsely flagged (false positives).
* Especially in **imbalanced datasets**, a model that predicts “negative” always can still get very high accuracy but be useless.
* Thus we need **richer metrics** that reflect types of errors and trade-offs.

**New / useful functions / code ideas**

* `sklearn.metrics.accuracy_score(y_true, y_pred)` - baseline metric.

* `balanced_accuracy_score(y_true, y_pred)` - adjusts for class imbalance by averaging sensitivity and specificity.

* Use `scoring='balanced_accuracy'` in `cross_val_score` or `GridSearchCV` so that cross-validation optimizes a better metric than plain accuracy.

* Also, use dummy classifiers (e.g. `DummyClassifier(strategy="most_frequent")`) to compute a baseline “always predict majority class” accuracy for comparison.

---

## 2. Summary: Confusion Matrix & Basic Rates (Precision, Recall, FPR, TPR)

**What I learned / key takeaways**

* The *confusion matrix* gives counts: TN, FP, FN, TP.
* From it we define:
    - **Recall** (TPR) = TP / (TP + FN) - how many actual positives we catch.
    - **FPR** = FP / (FP + TN) - proportion of negatives incorrectly labeled positive.
    - **Precision** = TP / (TP + FP) - of predicted positives, how many are correct.
    - **F1 score** = harmonic mean of precision & recall - a single number reflecting the balance.
* Changing the decision threshold moves these values around: increasing threshold usually increases precision but reduces recall, and vice versa.

**New / useful functions / code ideas**

* `sklearn.metrics.confusion_matrix(y_true, y_pred)` → returns 2×2 matrix. Use `.ravel()` to get `tn, fp, fn, tp`.

* `precision_score(y_true, y_pred)`, `recall_score(y_true, y_pred)`

* `f1_score(y_true, y_pred)`

* `precision_recall_fscore_support(y_true, y_pred)` - gives precision, recall, f1, and support in one go for each class.

* If you want **custom metrics** (e.g. cost-weighted), you can wrap a function and pass via `sklearn.metrics.make_scorer(...)` for use in cross-validation / grid search.

---

## 3. Summary: ROC Curve, AUC, and Threshold Tuning

**What I learned / key takeaways**

* Many models output *scores* or *probabilities*, not just class labels. We convert those to labels via a **threshold**.
* The **ROC curve** plots TPR vs FPR across all possible thresholds.
* **AUC (Area Under the Curve)** summarizes the ROC into one scalar - how well the model ranks positives above negatives overall.
* But high AUC doesn’t guarantee that the threshold you choose will give acceptable precision or recall in your actual use case.
* It’s also useful to pick a **“best threshold”** by heuristics (e.g. maximize F1, maximize (TPR – FPR), or minimize specific cost).

**New / useful functions / code ideas**

* `sklearn.metrics.roc_curve(y_true, y_scores)` → gives arrays: `fpr, tpr, thresholds`

* `sklearn.metrics.auc(fpr, tpr)` or `roc_auc_score(y_true, y_scores)`

* `sklearn.metrics.precision_recall_curve(y_true, y_scores)` - for precision vs recall curves

* In code: loop over thresholds:

  ```python
  thresholds = np.linspace(0, 1, 101)
  results = []
  for thr in thresholds:
      y_pred_thr = (y_scores >= thr).astype(int)
      prec = precision_score(y_true, y_pred_thr)
      rec = recall_score(y_true, y_pred_thr)
      results.append((thr, prec, rec))
  ```

* You can compute **Youden’s J statistic** = TPR - FPR and pick the threshold maximizing it.

* Use `sklearn.metrics.make_scorer(..., needs_proba=True)` so that grid search or cross validation can optimize based on predicted probabilities instead of hard labels.

---

## 4. Summary: Handling Imbalanced Classes

**What I learned / key takeaways**

* When the positive class is rare, accuracy is deceptive; models tend to favor the majority class.
* Remedies include: **class weights**, **oversampling**, **undersampling**, and synthetic oversampling (SMOTE).
* These methods shift the decision boundary or adjust data so that the minority class has more influence.
* After these techniques, metrics change - perhaps recall improves, but possibly precision drops or FPR rises, so you still need to inspect trade-offs.

**New / useful functions / code ideas**

* In scikit-learn estimators (e.g. `LogisticRegression`, `RandomForestClassifier`), use parameter `class_weight='balanced'` or supply custom weights `{0: w0, 1: w1}`.

* Use `imblearn` library (from `imbalanced-learn`) for sampling strategies:

  * `from imblearn.over_sampling import SMOTE`
  * `from imblearn.under_sampling import RandomUnderSampler`
  * `from imblearn.pipeline import Pipeline` to combine sampling + model in one pipeline

* Example pipeline:

  ```python
  from imblearn.pipeline import Pipeline
  from imblearn.over_sampling import SMOTE
  from sklearn.linear_model import LogisticRegression

  pipeline = Pipeline([
    ('smote', SMOTE(sampling_strategy=0.5)),
    ('clf', LogisticRegression())
  ])
  pipeline.fit(X_train, y_train)
  ```

* After applying sampling/weighting, use `cross_val_score` or `cross_validate` to see how metrics behave across folds.

---

## 5. Summary: Cross-Validation & Stability of Metrics

**What I learned / key takeaways**

* Instead of relying on a single train/validation split, use **k-fold cross-validation** to get more robust estimates of performance.
* Compute metrics (AUC, F1, etc.) across folds and examine their mean and standard deviation to see how stable the model is.
* Choose hyperparameter settings that balance **good mean performance** and **low variance** across folds (stability).
* Visualizing metrics across folds (e.g. box plots) helps to see if a model sometimes fails badly.

**New / useful functions / code ideas**

* `sklearn.model_selection.cross_val_score(estimator, X, y, cv=5, scoring='roc_auc')`

* `sklearn.model_selection.cross_validate` allows multiple metrics at once, returning dicts of test scores.

* You can pass custom scorers to `cross_validate`, e.g.:

  ```python
  from sklearn.metrics import make_scorer, f1_score
  f1_scorer = make_scorer(f1_score)
  cv_results = cross_validate(model, X, y, cv=5,
                              scoring={'auc': 'roc_auc', 'f1': f1_scorer})
  ```

* Use `StratifiedKFold` to maintain class proportions in each fold:

  ```python
  from sklearn.model_selection import StratifiedKFold
  skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
  ```

* After cross-validation, you can plot boxplots:

  ```python
  import seaborn as sns
  sns.boxplot(data=[cv_results['test_auc'], cv_results['test_f1']])
  ```


