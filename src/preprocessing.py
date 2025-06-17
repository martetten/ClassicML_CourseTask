# стандартные
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# виджеты из-за нехватки памяти
import ipywidgets as widgets
from IPython.display import display

# предобработка
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from sklearn.decomposition import PCA

# создание sklearn-трансформеров и пайплайна для обработки выборок
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import joblib

def create_features(df):
    """Функция для создания кастомных переменных"""
    df = df.copy()
    df['HAcceptors_HDonors_ratio'] = df['NumHAcceptors'] / df['NumHDonors'].replace(0, np.nan)
    df['MolWt_TPSA_ratio'] = df['MolWt'] / df['TPSA'].replace(0, np.nan)
    df['RotatableBonds_HeavyAtom_ratio'] = df['NumRotatableBonds'] / df['HeavyAtomCount'].replace(0, np.nan)
    df['Hydrogen_Bond_Total'] = df['NumHAcceptors'] + df['NumHDonors'].replace(0, np.nan)
    df['PolarSurfaceArea_per_Atom'] = df['TPSA'] / df['HeavyAtomCount'].replace(0, np.nan)
    df['LogP_per_Atom'] = df['MolLogP'] / df['HeavyAtomCount'].replace(0, np.nan)

    return df.replace([np.inf, -np.inf], np.nan)

def binarize_fr(df):
    """"Бинаризация fr_* признаков"""
    fr_cols = [col for col in df.columns if col.startswith('fr_')]
    for col in fr_cols:
        df[col] = (df[col] > 0).astype(int)
    return df

class ConstantRemover:
    """"Удаление констант и почти константных признаков"""
    def __init__(self, threshold=0.95):
        self.threshold = threshold
        self.to_drop_ = []

    def fit(self, df):
        self.to_drop_ = []
        for col in df.columns:
            n_unique = df[col].nunique()
            if n_unique == 1:
                self.to_drop_.append(col)
            elif df[col].value_counts(normalize=True).max() > self.threshold:
                self.to_drop_.append(col)
        return self

    def transform(self, df):
        return df.drop(columns=self.to_drop_)
    
class OutlierClipper:
    """"Обработка выбросов"""
    def fit(self, df):
        self.bounds_ = {}
        for col in df.columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            self.bounds_[col] = {
                'lower': q1 - 3 * iqr,
                'upper': q3 + 15 * iqr
            }
        return self

    def transform(self, df):
        df = df.copy()
        for col in df.columns:
            if col in self.bounds_:
                lb = self.bounds_[col]['lower']
                ub = self.bounds_[col]['upper']
                df[col] = np.clip(df[col], lb, ub)
        return df
    
class DataFrameImputer(SimpleImputer):
    """"SimpleImputer с сохранением DataFrame"""
    def transform(self, df):
        arr = super().transform(df)
        return pd.DataFrame(arr, columns=df.columns, index=df.index)
    
class SafeYeoJohnson:
    """Power transformer Yeo-Johnson с сохранением DataFrame"""
    def __init__(self):
        self.transformer = PowerTransformer(method='yeo-johnson')
        self.cols = None

    def fit(self, df):
        self.cols = df.columns
        self.transformer.fit(df)
        return self

    def transform(self, df):
        arr = self.transformer.transform(df)
        return pd.DataFrame(arr, columns=self.cols, index=df.index)
    
class SafeScaler:
    """Scaler с сохранением DataFrame"""
    def __init__(self):
        self.transformer = StandardScaler()
        self.cols = None

    def fit(self, df):
        self.cols = df.columns
        self.transformer.fit(df)
        return self

    def transform(self, df):
        arr = self.transformer.transform(df)
        return pd.DataFrame(arr, columns=self.cols, index=df.index)
    
class GroupPCA:
    """Создание PCA-переменных на основе групп"""
    def __init__(self, group_config):
        self.group_config = group_config
        self.pca_models = {}

    def fit(self, df):
        for group, features in self.group_config.items():
            valid_features = [f for f in features if f in df.columns]
            if len(valid_features) >= 2:
                pca = PCA(n_components=0.8)
                pca.fit(df[valid_features])
                self.pca_models[group] = (pca, valid_features)
        return self

    def transform(self, df):
        df = df.copy()
        for group, (pca, features) in self.pca_models.items():
            valid_features = [f for f in features if f in df.columns]
            if valid_features:
                pca_result = pca.transform(df[valid_features])
                for i in range(pca.n_components_):
                    df[f'{group}_PCA{i+1}'] = pca_result[:, i]
                df = df.drop(columns=valid_features)
        return df
    
class CorrelationRemover:
    """Попарная проверка мультиколлинеарности"""
    def __init__(self, threshold=0.9):
        self.threshold = threshold
        self.to_drop_ = []

    def fit(self, df):
        self.to_drop_ = []
        corr_matrix = df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        for col in upper.columns:
            high_corr = upper.index[upper[col] > self.threshold]
            self.to_drop_.extend(high_corr)

        # Удаляем дубликаты
        self.to_drop_ = list(set(self.to_drop_))
        return self

    def transform(self, df):
        return df.drop(columns=self.to_drop_)
    
class ProcessingPipeline:
    """Обёртка в pipeline предобработки"""
    def __init__(self, group_config, corr_threshold=0.9):
        self.group_config = group_config
        self.corr_threshold = corr_threshold

        # Общие компоненты для всех признаков
        self.common_components = [
            ('feature_creator', lambda df: create_features(df)),
            ('fr_binarizer', lambda df: binarize_fr(df)),
            ('constant_remover', ConstantRemover(threshold=0.95))
        ]

        # Компоненты только для числовых признаков
        self.num_components = [
            ('outlier_clipper', OutlierClipper()),
            ('imputer', DataFrameImputer(strategy='median')),
            ('yeojohnson', SafeYeoJohnson()),
            ('scaler', SafeScaler()),
            ('grouppca', GroupPCA(group_config)),
            ('corr_remover', CorrelationRemover(threshold=corr_threshold))
        ]

        # Компоненты только для бинарных признаков
        self.bin_components = [
            ('bin_imputer', DataFrameImputer(strategy='most_frequent'))
        ]

    def fit(self, X, y=None):
        # Применяем общие компоненты
        df = X.copy()
        for name, transformer in self.common_components:
            if hasattr(transformer, 'fit'):
                df = transformer.fit(df).transform(df)
            else:
                df = transformer(df)

        # Разделяем данные на числовые и бинарные признаки
        self.binary_cols_ = [col for col in df.columns if col.startswith('fr_')]
        self.numeric_cols_ = [col for col in df.columns if col not in self.binary_cols_]

        # Обработка числовых признаков
        df_num = df[self.numeric_cols_]
        for name, transformer in self.num_components:
            if hasattr(transformer, 'fit'):
                df_num = transformer.fit(df_num).transform(df_num)
            else:
                df_num = transformer(df_num)

        # Обработка бинарных признаков
        df_bin = df[self.binary_cols_]
        for name, transformer in self.bin_components:
            if hasattr(transformer, 'fit'):
                df_bin = transformer.fit(df_bin).transform(df_bin)
            else:
                df_bin = transformer(df_bin)

        # Сохраняем состояние для transform
        self.fitted_ = True
        return self

    def transform(self, X):
        if not hasattr(self, 'fitted_'):
            raise RuntimeError("Pipeline not fitted yet")

        # Применяем общие компоненты
        df = X.copy()
        for name, transformer in self.common_components:
            if hasattr(transformer, 'transform'):
                df = transformer.transform(df)
            else:
                df = transformer(df)

        # Разделение данных
        df_num = df[self.numeric_cols_]
        df_bin = df[self.binary_cols_]

        # Трансформация числовых признаков
        for name, transformer in self.num_components:
            df_num = transformer.transform(df_num)

        # Трансформация бинарных признаков
        for name, transformer in self.bin_components:
            df_bin = transformer.transform(df_bin)

        # Объединяем результаты
        return pd.concat([df_num, df_bin], axis=1)

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
