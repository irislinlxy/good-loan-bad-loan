import logging
import sys
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

sys.path.append("..")

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore")

logger = logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)s] [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def create_target_variable(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    create target variable based on loan status.

    Args:
    df_raw: raw input dataframe.

    """
    logger.info("Generating target variable.")

    good_status = ["Fully Paid"]
    default_status = ["Default", "Charged Off"]

    df_raw["predicted_loan_status"] = np.where(
        df_raw["loan_status"].isin(good_status),
        "good",
        np.where(df_raw["loan_status"].isin(default_status), "bad", "inference"),
    )

    return df_raw


def seperate_train_inference_data(df_raw: pd.DataFrame) -> Tuple:
    """
    Output sperate data for model training and making inference based on created target variable.

    Args:
    df_raw: raw input dataframe.
    """
    df_input = df_raw[df_raw["predicted_loan_status"].isin(["good", "bad"])]
    df_inference = df_raw[df_raw["predicted_loan_status"] == "inference"]

    logger.info(f"Identified {len(df_input)} data points for model training")
    logger.info(f"Identified {len(df_inference)} data points for making inferences")

    return (df_input, df_inference)


def correct_data_types(
    raw_data: pd.DataFrame, numerical_column_names_list: List[str]
) -> pd.DataFrame:
    """Correct the data types for various columns and returns the dataframe

    Args:
        raw_data (pd.DataFrame): A pandas dataframe containing the raw data
        numerical_column_names_list(List[str]): List of numerical column names

    Returns:
        pd.DataFrame: A pandas dataframe containing the raw data with corrected data
            types
    """
    raw_data["term"] = np.where(
        raw_data["term"] == " 36 months",
        36,
        np.where(raw_data["term"] == " 60 months", 60, np.nan),
    )

    raw_data[numerical_column_names_list] = raw_data[
        numerical_column_names_list
    ].astype(float)

    logger.info("Corrected data types")

    return raw_data
