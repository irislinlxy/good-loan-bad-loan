import logging
import sys
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


sys.path.append("..")

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore")
logger = logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)s] [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def create_duration_features(
    df_raw: pd.DataFrame,
    today_date: datetime,
    base_feature_name: str = "earliest_cr_line",
) -> pd.DataFrame:
    """
    Create a new column with duration from start time to today

    Args:
        raw_data (pd.DataFrame): A pandas dataframe containing the raw data
        categorical_column_names_list (List[str]): List of categorical column names

    """
    df_raw["today"] = today_date

    df_raw[base_feature_name] = pd.to_datetime(
        df_raw[base_feature_name], format="%Y-%m-%d %H:%M:%S", errors="coerce"
    )
    df_raw["mths_since_first_cr_line"] = round(
        (df_raw["today"] - df_raw[base_feature_name]).dt.days / 30, 1
    )

    return df_raw


def encode_categorical_features(
    raw_data: pd.DataFrame, categorical_column_names_list: List[str]
) -> tuple:
    """Label encodes the categorical features

    Args:
        raw_data (pd.DataFrame): A pandas dataframe containing the raw data
        categorical_column_names_list (List[str]): List of categorical column names

    Returns:
        tuple: A pandas dataframe containing the encoded categorical features and a list
            with names of the encoded columns
    """
    labelencoder = LabelEncoder()
    for categorical_feature in categorical_column_names_list:
        raw_data[f"{categorical_feature} Encoded"] = labelencoder.fit_transform(
            raw_data[categorical_feature]
        )

    categorical_column_names_list_encoded = [
        f"{categorical_feature} Encoded"
        for categorical_feature in categorical_column_names_list
    ]
    logger.info("Finished Encoding the categorical variables")
    return raw_data, categorical_column_names_list_encoded


def impute_with_zero(
    df_raw: pd.DataFrame, impute_zero_column_list: List[str]
) -> pd.DataFrame:
    """
    Impute selected columns with zero based on EDA findings

    Args:
        raw_data (pd.DataFrame): A pandas dataframe containing the raw data
        impute_zero_column_list (List[str]): List of column names

    """
    df_raw[impute_zero_column_list] = df_raw[impute_zero_column_list].fillna(0)

    return df_raw


def impute_missing_value(
    df: pd.DataFrame,
    impute_features_column_names_list: List[str],
    categorical_column_names_list_encoded: List[str],
    impute_method: str = "median",
) -> pd.DataFrame:
    """
    Impute selected columns

    Args:
        df (pd.DataFrame): A pandas dataframe containing the data
        impute_features_column_names_list (List[str]): List of column names to impute
        impute_method (str): method for imputation. default is column median

    """
    if impute_method == "median":
        for col in impute_features_column_names_list:
            impute_median = df[col].median()
            df[col] = df[col].fillna(impute_median)

    elif impute_method == "mean":
        for col in impute_features_column_names_list:
            impute_mean = df[col].mean()
            df[col] = df[col].fillna(impute_mean)

    df["installment"] = np.where(
        df["installment"].isnull(), df["funded_amnt"] / df["term"], df["installment"]
    )
    df["installment"] = df["installment"].fillna(df["installment"].median())

    for col in categorical_column_names_list_encoded:
        impute_mode = df[col].mode()[0]
        df[col] = df[col].fillna(impute_mode)

    return df


def feature_selection(
    df: pd.DataFrame,
    categorical_column_names_list_encoded: List[str],
    num_feat_list: List[str],
    features_to_drop: List[str],
    target_var_name: str = "predicted_loan_status",
) -> Tuple:
    """
    Drop features based on high multicolinearity and select final features for model training

    Args:
        df (pd.DataFrame): A pandas dataframe containing the input data
        categorical_column_names_list_encoded (List[str]): List of encoded categorical column names
        num_feat_list (List[str]): List of numberical column names
        features_to_drop (List[str]): List of column names to drop from training
        target_var_name (str): name of the target variable

    """
    all_feature_list = (
        categorical_column_names_list_encoded + num_feat_list + [target_var_name]
    )

    df = df[all_feature_list]

    df.drop(features_to_drop, axis=1, inplace=True)

    num_feat_list = [col for col in num_feat_list if col not in features_to_drop]

    return df, num_feat_list


def feature_selection_infer(
    df: pd.DataFrame,
    categorical_column_names_list_encoded: List[str],
    num_feat_list: List[str],
) -> pd.DataFrame:
    """
    Drop features based on high multicolinearity and select final columns for inference

    Args:
        df (pd.DataFrame): A pandas dataframe containing the test data
        categorical_column_names_list_encoded (List[str]): List of encoded categorical column names
        num_feat_list (List[str]): List of numberical column names
        features_to_drop (List[str]): List of column names to drop from training
        target_var_name (str): name of the target variable

    """
    all_feature_list = ["id"] + categorical_column_names_list_encoded + num_feat_list

    df = df[all_feature_list]

    return df
