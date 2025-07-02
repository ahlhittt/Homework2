# homework_datasets.py
# Задание 2.1: Кастомный Dataset класс для работы с CSV

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class CustomDataset:
    """
    Кастомный класс для работы с CSV файлами.
    Поддержка загрузки, предобработки (нормализация, кодирование категорий), различных форматов данных.
    """
    def __init__(self, csv_path, target_column=None):
        self.data = pd.read_csv(csv_path)
        self.target_column = target_column
        self.X = None
        self.y = None
        self.scaler = None
        self.encoder = None
        self.feature_names = None
        self.categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
        self.numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        if target_column and target_column in self.categorical_cols:
            self.categorical_cols.remove(target_column)
        if target_column and target_column in self.numeric_cols:
            self.numeric_cols.remove(target_column)
    
    def preprocess(self, normalize=True, encode_categorical=True):
        X_df = self.data.drop(columns=[self.target_column]) if self.target_column else self.data.copy()
        feature_names = list(X_df.columns)
        # Кодирование категориальных признаков
        if encode_categorical and self.categorical_cols:
            self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            X_cat = self.encoder.fit_transform(X_df[self.categorical_cols])
            cat_feature_names = self.encoder.get_feature_names_out(self.categorical_cols)
            X_num = X_df.drop(columns=self.categorical_cols).values
            X = np.hstack([X_num, X_cat])
            feature_names = list(X_df.drop(columns=self.categorical_cols).columns) + list(cat_feature_names)
        else:
            X = X_df.values
        # Нормализация числовых признаков
        if normalize and self.numeric_cols:
            self.scaler = StandardScaler()
            X[:, :len(self.numeric_cols)] = self.scaler.fit_transform(X[:, :len(self.numeric_cols)])
        self.X = X
        self.feature_names = feature_names
        if self.target_column:
            self.y = self.data[self.target_column].values
        else:
            self.y = None
    
    def get_features_and_target(self):
        return self.X, self.y

# Задание 2.2: Эксперименты с датасетами
# TODO: Найти датасеты для регрессии и бинарной классификации, обучить линейную и логистическую регрессию 

if __name__ == "__main__":
    from homework_model_modification import LinearRegressionModified, LogisticRegressionModified
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    import warnings
    warnings.filterwarnings("ignore")

    # --- Регрессия: Boston Housing ---
    print("\n--- Boston Housing: Linear Regression ---")
    boston = CustomDataset("data/boston.csv", target_column="medv")
    boston.preprocess(normalize=True, encode_categorical=True)
    X, y = boston.get_features_and_target()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_lr = LinearRegressionModified(l1=0.01, l2=0.01, early_stopping=True, patience=10, max_iter=1000, lr=0.01)
    model_lr.fit(X_train, y_train, X_val=X_test, y_val=y_test)
    y_pred = model_lr.predict(X_test)
    print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")

    # --- Классификация: Breast Cancer ---
    print("\n--- Breast Cancer: Logistic Regression ---")
    # В датасете нет заголовков, добавим их вручную
    bc_columns = [
        "id", "clump_thickness", "cell_size_uniformity", "cell_shape_uniformity", "marginal_adhesion",
        "single_epithelial_cell_size", "bare_nuclei", "bland_chromatin", "normal_nucleoli", "mitoses", "target"
    ]
    bc = pd.read_csv("data/breast_cancer.csv", names=bc_columns)
    # Заменим '?' на NaN и удалим такие строки
    bc = bc.replace('?', np.nan).dropna()
    bc["bare_nuclei"] = bc["bare_nuclei"].astype(int)
    # Преобразуем целевую переменную к 0/1
    bc["target"] = (bc["target"] == 4).astype(int)
    bc.to_csv("data/breast_cancer_clean.csv", index=False)
    breast = CustomDataset("data/breast_cancer_clean.csv", target_column="target")
    breast.preprocess(normalize=True, encode_categorical=True)
    X, y = breast.get_features_and_target()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_logr = LogisticRegressionModified(multi_class=False, max_iter=1000, lr=0.01)
    model_logr.fit(X_train, y_train)
    y_pred = model_logr.predict(X_test)
    y_proba = model_logr.predict_proba(X_test)
    print(f"Precision: {model_logr.precision(y_test, y_pred):.4f}")
    print(f"Recall: {model_logr.recall(y_test, y_pred):.4f}")
    print(f"F1-score: {model_logr.f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC: {model_logr.roc_auc(y_test, y_proba):.4f}")
    model_logr.plot_confusion_matrix(y_test, y_pred) 