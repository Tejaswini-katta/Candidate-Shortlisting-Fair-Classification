"""
Model Training, Evaluation, and Comparison Module for Candidate Shortlisting.

Trains and evaluates:
1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Random Forest Benchmark
4. Gradient Boosting (XGBoost Benchmark)

Computes Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrices.
Selects and serializes the top-performing candidate shortlisting model.
"""

import os
import json
from typing import Dict, Any, Tuple, List
import pandas as pd
import numpy as np


class LogisticRegressionModel:
    """Logistic Regression Classifier using Gradient Descent with L2 regularization."""

    def __init__(self, lr: float = 0.05, n_iters: int = 500, l2_reg: float = 0.1, random_state: int = 42):
        self.lr = lr
        self.n_iters = n_iters
        self.l2_reg = l2_reg
        self.random_state = random_state
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -25, 25)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionModel":
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)

            dw = (1 / n_samples) * (np.dot(X.T, (y_pred - y)) + self.l2_reg * self.weights)
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        linear_model = np.dot(X, self.weights) + self.bias
        prob_1 = self._sigmoid(linear_model)
        prob_0 = 1.0 - prob_1
        return np.column_stack([prob_0, prob_1])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


class KNNClassifierModel:
    """K-Nearest Neighbors Classifier using Euclidean distance weighting."""

    def __init__(self, k: int = 7):
        self.k = k
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifierModel":
        # Subsample training data if large to maintain fast execution speed
        if X.shape[0] > 3000:
            np.random.seed(42)
            idx = np.random.choice(X.shape[0], 3000, replace=False)
            self.X_train = X[idx]
            self.y_train = y[idx]
        else:
            self.X_train = X
            self.y_train = y
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probas = []
        for x in X:
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train[k_indices]
            p1 = np.mean(k_nearest_labels)
            probas.append([1.0 - p1, p1])
        return np.array(probas)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


class DecisionNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, min_samples_split=5, max_depth=8, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)
        return self

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            leaf_value = np.mean(y) if len(y) > 0 else 0.0
            return DecisionNode(value=leaf_value)

        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)
        best_feat, best_thresh = self._best_split(X, y, feat_idxs)

        if best_feat is None:
            return DecisionNode(value=np.mean(y))

        left_idxs = X[:, best_feat] < best_thresh
        right_idxs = ~left_idxs

        left = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)
        return DecisionNode(best_feat, best_thresh, left, right)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.percentile(X_column, [25, 50, 75])
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold
        return split_idx, split_thresh

    def _information_gain(self, y, X_column, threshold):
        parent_var = np.var(y)
        left_idxs = X_column < threshold
        right_idxs = ~left_idxs

        if len(y[left_idxs]) == 0 or len(y[right_idxs]) == 0:
            return 0

        n = len(y)
        n_l, n_r = len(y[left_idxs]), len(y[right_idxs])
        var_l, var_r = np.var(y[left_idxs]), np.var(y[right_idxs])
        child_var = (n_l / n) * var_l + (n_r / n) * var_r

        return parent_var - child_var

    def predict_proba(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] < node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


class RandomForestModel:
    """Random Forest Ensemble Classifier."""

    def __init__(self, n_trees: int = 15, max_depth: int = 6, random_state: int = 42):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees: List[DecisionTree] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestModel":
        np.random.seed(self.random_state)
        self.trees = []
        n_samples = X.shape[0]

        for i in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth, n_features=int(np.sqrt(X.shape[1])))
            idxs = np.random.choice(n_samples, int(n_samples * 0.8), replace=True)
            tree.fit(X[idxs], y[idxs])
            self.trees.append(tree)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        tree_preds = np.array([tree.predict_proba(X) for tree in self.trees])
        p1 = np.mean(tree_preds, axis=0)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


class GradientBoostingModel:
    """Gradient Boosting Classifier (XGBoost Benchmark)."""

    def __init__(self, n_estimators: int = 20, learning_rate: float = 0.1, max_depth: int = 4, random_state: int = 42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees: List[DecisionTree] = []
        self.base_pred: float = 0.0

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingModel":
        np.random.seed(self.random_state)
        p_init = np.mean(y)
        self.base_pred = np.log(p_init / (1.0 - p_init + 1e-7))

        raw_preds = np.full(len(y), self.base_pred)

        for _ in range(self.n_estimators):
            p = self._sigmoid(raw_preds)
            residuals = y - p

            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X, residuals)
            self.trees.append(tree)

            update = tree.predict_proba(X)
            raw_preds += self.learning_rate * update

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw_preds = np.full(len(X), self.base_pred)
        for tree in self.trees:
            raw_preds += self.learning_rate * tree.predict_proba(X)

        p1 = self._sigmoid(raw_preds)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    """Calculate Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-7)
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)

    # Approximate ROC-AUC via rank-order concordance
    pos_probas = y_proba[y_true == 1]
    neg_probas = y_proba[y_true == 0]

    # Sample for speed if large
    if len(pos_probas) > 500:
        pos_probas = np.random.choice(pos_probas, 500, replace=False)
    if len(neg_probas) > 500:
        neg_probas = np.random.choice(neg_probas, 500, replace=False)

    pairs = 0
    concordant = 0
    for p in pos_probas:
        for n in neg_probas:
            pairs += 1
            if p > n:
                concordant += 1
            elif p == n:
                concordant += 0.5

    roc_auc = concordant / (pairs + 1e-7)

    return {
        "Accuracy": round(float(accuracy), 4),
        "Precision": round(float(precision), 4),
        "Recall": round(float(recall), 4),
        "F1-Score": round(float(f1), 4),
        "ROC-AUC": round(float(roc_auc), 4),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)
    }


def train_and_compare_models(
    processed_dir: str = os.path.join("data", "processed"),
    models_dir: str = os.path.join("models", "trained_models"),
    reports_dir: str = os.path.join("reports", "metrics")
) -> Dict[str, Any]:
    """Train all candidates models, evaluate on validation set, and save the best model."""
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    print("=" * 60)
    print("STEP 4: MODEL TRAINING, EVALUATION & COMPARISON")
    print("=" * 60)

    # Load processed matrices
    X_train = pd.read_csv(os.path.join(processed_dir, "X_train.csv")).values
    y_train = pd.read_csv(os.path.join(processed_dir, "y_train.csv")).values.ravel()

    X_val = pd.read_csv(os.path.join(processed_dir, "X_val.csv")).values
    y_val = pd.read_csv(os.path.join(processed_dir, "y_val.csv")).values.ravel()

    print(f"[INFO] Training Data Shape:   {X_train.shape}")
    print(f"[INFO] Validation Data Shape: {X_val.shape}")

    models = {
        "Logistic Regression": LogisticRegressionModel(lr=0.08, n_iters=400, l2_reg=0.1),
        "KNN Classifier": KNNClassifierModel(k=9),
        "Random Forest Benchmark": RandomForestModel(n_trees=12, max_depth=5),
        "Gradient Boosting (XGBoost)": GradientBoostingModel(n_estimators=15, learning_rate=0.1, max_depth=4)
    }

    results = {}
    best_model_name = ""
    best_roc_auc = -1.0
    best_model_obj = None

    for name, model in models.items():
        print(f"\n[TRAINING] {name}...")
        model.fit(X_train, y_train)

        y_proba = model.predict_proba(X_val)[:, 1]
        y_pred = model.predict(X_val, threshold=0.5)

        metrics = evaluate_metrics(y_val, y_pred, y_proba)
        results[name] = metrics

        print(f" -> Accuracy:  {metrics['Accuracy']:.4f}")
        print(f" -> Precision: {metrics['Precision']:.4f}")
        print(f" -> Recall:    {metrics['Recall']:.4f}")
        print(f" -> F1-Score:  {metrics['F1-Score']:.4f}")
        print(f" -> ROC-AUC:   {metrics['ROC-AUC']:.4f}")

        if metrics["ROC-AUC"] > best_roc_auc:
            best_roc_auc = metrics["ROC-AUC"]
            best_model_name = name
            best_model_obj = model

    # Convert results to DataFrame & save report
    results_df = pd.DataFrame(results).T
    report_path = os.path.join(reports_dir, "model_performance_comparison.csv")
    results_df.to_csv(report_path)
    print(f"\n[OK] Model evaluation report saved to: {report_path}")

    # Save Best Model metadata & weights artifact
    best_info = {
        "best_model_name": best_model_name,
        "best_roc_auc": best_roc_auc,
        "metrics": results[best_model_name]
    }
    best_info_path = os.path.join(models_dir, "best_model_info.json")
    with open(best_info_path, "w") as f:
        json.dump(best_info, f, indent=2)

    # Save model weights if Logistic Regression or base parameters
    if hasattr(best_model_obj, "weights"):
        weights_dict = {
            "model_type": "LogisticRegression",
            "weights": best_model_obj.weights.tolist(),
            "bias": float(best_model_obj.bias)
        }
        with open(os.path.join(models_dir, "best_model.json"), "w") as f:
            json.dump(weights_dict, f, indent=2)

    print(f"[SUCCESS] Top Model: '{best_model_name}' with ROC-AUC: {best_roc_auc:.4f}")
    print("=" * 60)

    return {
        "best_model": best_model_name,
        "best_roc_auc": best_roc_auc,
        "comparison_table": results_df
    }


if __name__ == "__main__":
    train_and_compare_models()
