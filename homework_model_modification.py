# homework_model_modification.py
# Задание 1.1: Модифицированная линейная регрессия с L1/L2 регуляризацией и early stopping

import numpy as np

class LinearRegressionModified:
    """
    Линейная регрессия с поддержкой L1/L2 регуляризации и early stopping.
    """
    def __init__(self, l1=0.0, l2=0.0, early_stopping=False, patience=5, tol=1e-4, max_iter=1000, lr=0.01):
        self.l1 = l1
        self.l2 = l2
        self.early_stopping = early_stopping
        self.patience = patience
        self.tol = tol
        self.max_iter = max_iter
        self.lr = lr
        self.weights = None
        self.bias = None
    
    def fit(self, X, y, X_val=None, y_val=None):
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        best_loss = float('inf')
        patience_counter = 0
        for i in range(self.max_iter):
            y_pred = X @ self.weights + self.bias
            error = y_pred - y
            # Градиенты с учетом регуляризации
            grad_w = (2/n_samples) * (X.T @ error) + self.l1 * np.sign(self.weights) + 2 * self.l2 * self.weights
            grad_b = (2/n_samples) * np.sum(error)
            # Обновление параметров
            self.weights -= self.lr * grad_w
            self.bias -= self.lr * grad_b
            # Early stopping
            if self.early_stopping and X_val is not None and y_val is not None:
                val_pred = np.dot(X_val, self.weights) + self.bias
                val_loss = np.mean((val_pred - y_val) ** 2)
                if val_loss + self.tol < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        break
    
    def predict(self, X):
        X = np.array(X)
        if self.weights is None or self.bias is None:
            raise ValueError("Модель не обучена. Сначала вызовите fit().")
        return X @ self.weights + self.bias

# Задание 1.2: Модифицированная логистическая регрессия с многоклассовой поддержкой и метриками

import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix

class LogisticRegressionModified:
    """
    Логистическая регрессия с поддержкой многоклассовой классификации, метриками и визуализацией confusion matrix.
    """
    def __init__(self, multi_class=True, max_iter=1000, lr=0.01):
        self.multi_class = multi_class
        self.max_iter = max_iter
        self.lr = lr
        self.weights = None
        self.bias = None
        self.classes_ = None
    
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)
        if self.multi_class and n_classes > 2:
            # One-vs-rest
            self.weights = np.zeros((n_classes, n_features))
            self.bias = np.zeros(n_classes)
            for idx, cls in enumerate(self.classes_):
                y_bin = (y == cls).astype(float)
                w = np.zeros(n_features)
                b = 0.0
                for _ in range(self.max_iter):
                    linear = X @ w + b
                    y_pred = self._sigmoid(linear)
                    error = y_pred - y_bin
                    grad_w = X.T @ error / n_samples
                    grad_b = np.sum(error) / n_samples
                    w -= self.lr * grad_w
                    b -= self.lr * grad_b
                self.weights[idx] = w
                self.bias[idx] = b
        else:
            # Бинарная классификация
            self.weights = np.zeros(n_features)
            self.bias = 0.0
            for _ in range(self.max_iter):
                linear = X @ self.weights + self.bias
                y_pred = self._sigmoid(linear)
                error = y_pred - y
                grad_w = X.T @ error / n_samples
                grad_b = np.sum(error) / n_samples
                self.weights -= self.lr * grad_w
                self.bias -= self.lr * grad_b
    
    def predict_proba(self, X):
        X = np.array(X)
        if self.weights is None or self.bias is None:
            raise ValueError("Модель не обучена. Сначала вызовите fit().")
        if self.multi_class and self.classes_ is not None and len(self.classes_) > 2:
            logits = X @ self.weights.T + self.bias
            probs = self._sigmoid(logits)
            return probs
        else:
            linear = X @ self.weights + self.bias
            return self._sigmoid(linear)
    
    def predict(self, X):
        probs = self.predict_proba(X)
        if self.multi_class and self.classes_ is not None and len(self.classes_) > 2:
            return self.classes_[np.argmax(probs, axis=1)]
        else:
            return (probs >= 0.5).astype(int)
    
    def precision(self, y_true, y_pred, average='binary'):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        if average == 'binary' or (not self.multi_class or len(np.unique(y_true)) == 2):
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            return tp / (tp + fp + 1e-8)
        else:
            precisions = []
            for cls in np.unique(y_true):
                tp = np.sum((y_true == cls) & (y_pred == cls))
                fp = np.sum((y_true != cls) & (y_pred == cls))
                precisions.append(tp / (tp + fp + 1e-8))
            return np.mean(precisions)
    
    def recall(self, y_true, y_pred, average='binary'):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        if average == 'binary' or (not self.multi_class or len(np.unique(y_true)) == 2):
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            return tp / (tp + fn + 1e-8)
        else:
            recalls = []
            for cls in np.unique(y_true):
                tp = np.sum((y_true == cls) & (y_pred == cls))
                fn = np.sum((y_true == cls) & (y_pred != cls))
                recalls.append(tp / (tp + fn + 1e-8))
            return np.mean(recalls)
    
    def f1_score(self, y_true, y_pred, average='binary'):
        p = self.precision(y_true, y_pred, average)
        r = self.recall(y_true, y_pred, average)
        return 2 * p * r / (p + r + 1e-8)
    
    def roc_auc(self, y_true, y_score, average='macro'):
        # Для многоклассового случая используем sklearn.metrics.roc_auc_score
        return roc_auc_score(y_true, y_score, average=average, multi_class='ovr' if self.multi_class else 'raise')
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
        plt.title('Confusion Matrix')
        plt.colorbar()
        unique_labels = [str(l) for l in np.unique(y_true)]
        tick_marks = np.arange(len(unique_labels))
        plt.xticks(tick_marks, unique_labels)
        plt.yticks(tick_marks, unique_labels)
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'), ha='center', va='center', color='red')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show() 