# homework_experiments.py
# Задание 3.1: Исследование гиперпараметров
# TODO: Провести эксперименты с learning rate, batch size, оптимизаторами (SGD, Adam, RMSprop)
# TODO: Визуализировать результаты (графики/таблицы)

# Задание 3.2: Feature Engineering
# TODO: Создать новые признаки (полиномиальные, взаимодействия, статистические)
# TODO: Сравнить качество с базовой моделью 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from homework_model_modification import LogisticRegressionModified
from homework_datasets import CustomDataset
from sklearn.preprocessing import PolynomialFeatures

# --- Гиперпараметры ---
LEARNING_RATES = [0.001, 0.01, 0.1]
BATCH_SIZES = [16, 32, 64]
OPTIMIZERS = ['sgd', 'adam', 'rmsprop']

# --- Реализация простых оптимизаторов ---
def get_optimizer(name, lr):
    if name == 'sgd':
        def step_sgd(w, grad, v=None, m=None, t=1):
            return w - lr * grad, v, m
        return step_sgd
    elif name == 'adam':
        def step_adam(w, grad, v, m, t):
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            v = beta1 * v + (1 - beta1) * grad
            m = beta2 * m + (1 - beta2) * (grad ** 2)
            v_hat = v / (1 - beta1 ** t)
            m_hat = m / (1 - beta2 ** t)
            w = w - lr * v_hat / (np.sqrt(m_hat) + eps)
            return w, v, m
        return step_adam
    elif name == 'rmsprop':
        def step_rmsprop(w, grad, v, m, t):
            beta, eps = 0.9, 1e-8
            m = beta * m + (1 - beta) * (grad ** 2)
            w = w - lr * grad / (np.sqrt(m) + eps)
            return w, v, m
        return step_rmsprop
    else:
        raise ValueError('Unknown optimizer')

# --- Модифицированная логистическая регрессия с поддержкой оптимизаторов и батчей ---
class LogisticRegressionExp(LogisticRegressionModified):
    def fit(self, X, y, batch_size=32, optimizer='sgd', lr=0.01, max_iter=100):
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        step = get_optimizer(optimizer, lr)
        v = np.zeros(n_features)
        m = np.zeros(n_features)
        for t in range(1, max_iter+1):
            idx = np.random.permutation(n_samples)
            X_shuffled, y_shuffled = X[idx], y[idx]
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                linear = X_batch @ self.weights + self.bias
                y_pred = 1 / (1 + np.exp(-linear))
                error = y_pred - y_batch
                grad_w = X_batch.T @ error / len(y_batch)
                grad_b = np.sum(error) / len(y_batch)
                self.weights, v, m = step(self.weights, grad_w, v, m, t)
                self.bias -= lr * grad_b

if __name__ == "__main__":
    # Загрузка и подготовка данных
    bc_columns = [
        "id", "clump_thickness", "cell_size_uniformity", "cell_shape_uniformity", "marginal_adhesion",
        "single_epithelial_cell_size", "bare_nuclei", "bland_chromatin", "normal_nucleoli", "mitoses", "target"
    ]
    bc = pd.read_csv("data/breast_cancer.csv", names=bc_columns)
    bc = bc.replace('?', np.nan).dropna()
    bc["bare_nuclei"] = bc["bare_nuclei"].astype(int)
    bc["target"] = (bc["target"] == 4).astype(int)
    bc.to_csv("data/breast_cancer_clean.csv", index=False)
    breast = CustomDataset("data/breast_cancer_clean.csv", target_column="target")
    breast.preprocess(normalize=True, encode_categorical=True)
    X, y = breast.get_features_and_target()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = []
    for opt in OPTIMIZERS:
        for lr in LEARNING_RATES:
            for batch in BATCH_SIZES:
                model = LogisticRegressionExp(multi_class=False, max_iter=1, lr=lr)
                model.fit(X_train, y_train, batch_size=batch, optimizer=opt, lr=lr, max_iter=50)
                y_pred = model.predict(X_test)
                acc = np.mean(y_pred == y_test)
                results.append({"optimizer": opt, "lr": lr, "batch": batch, "accuracy": acc})
                print(f"opt={opt}, lr={lr}, batch={batch} => acc={acc:.4f}")

    # Визуализация
    df = pd.DataFrame(results)
    plt.figure(figsize=(10, 6))
    for opt in OPTIMIZERS:
        subset = df[df["optimizer"] == opt]
        plt.plot(subset["lr"].astype(str) + "/" + subset["batch"].astype(str), subset["accuracy"], label=opt, marker='o')
    plt.xlabel("learning_rate/batch_size")
    plt.ylabel("Accuracy")
    plt.title("Гиперпараметры: точность на Breast Cancer")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("plots/hyperparams_breast_cancer.png")
    plt.show()

    # --- Feature Engineering ---
    print("\n--- Feature Engineering: Breast Cancer ---")
    # Базовая модель
    model_base = LogisticRegressionExp(multi_class=False, max_iter=1, lr=0.01)
    model_base.fit(X_train, y_train, batch_size=32, optimizer='sgd', lr=0.01, max_iter=50)
    acc_base = np.mean(model_base.predict(X_test) == y_test)
    print(f"Базовая точность: {acc_base:.4f}")

    # Полиномиальные признаки (2-й степени)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    model_poly = LogisticRegressionExp(multi_class=False, max_iter=1, lr=0.01)
    model_poly.fit(X_train_poly, y_train, batch_size=32, optimizer='sgd', lr=0.01, max_iter=50)
    acc_poly = np.mean(model_poly.predict(X_test_poly) == y_test)
    print(f"Точность с полиномиальными признаками: {acc_poly:.4f}")

    # Взаимодействия между признаками (только cross terms)
    poly_inter = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
    X_train_inter = poly_inter.fit_transform(X_train)
    X_test_inter = poly_inter.transform(X_test)
    model_inter = LogisticRegressionExp(multi_class=False, max_iter=1, lr=0.01)
    model_inter.fit(X_train_inter, y_train, batch_size=32, optimizer='sgd', lr=0.01, max_iter=50)
    acc_inter = np.mean(model_inter.predict(X_test_inter) == y_test)
    print(f"Точность с взаимодействиями: {acc_inter:.4f}")

    # Статистические признаки (mean, std, min, max по строкам)
    def add_stat_features(X):
        return np.hstack([
            X,
            X.mean(axis=1, keepdims=True),
            X.std(axis=1, keepdims=True),
            X.min(axis=1, keepdims=True),
            X.max(axis=1, keepdims=True)
        ])
    X_train_stat = add_stat_features(X_train)
    X_test_stat = add_stat_features(X_test)
    model_stat = LogisticRegressionExp(multi_class=False, max_iter=1, lr=0.01)
    model_stat.fit(X_train_stat, y_train, batch_size=32, optimizer='sgd', lr=0.01, max_iter=50)
    acc_stat = np.mean(model_stat.predict(X_test_stat) == y_test)
    print(f"Точность с статистическими признаками: {acc_stat:.4f}")

    # Сравнение
    plt.figure(figsize=(7,4))
    accs = [acc_base, acc_poly, acc_inter, acc_stat]
    labels = ["Базовая", "Полиномиальные", "Взаимодействия", "Статистические"]
    plt.bar(labels, accs, color=['gray','blue','green','orange'])
    plt.ylabel("Accuracy")
    plt.title("Feature Engineering: Breast Cancer")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("plots/feature_engineering_breast_cancer.png")
    plt.show() 