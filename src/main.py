import logging
import sys
import os
import warnings
import datetime
import numpy as np
import pandas as pd

from typing import Dict, List, Tuple

sys.path.append("..")
sys.path.append(os.path.realpath(".."))
dir = os.path.dirname(__file__)

warnings.filterwarnings("ignore")
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

logger = logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)s] [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

from preprocessing import (
    create_target_variable,
    correct_data_types,
    seperate_train_inference_data,
)
from feature_engineering import (
    create_duration_features,
    encode_categorical_features,
    impute_with_zero,
    impute_missing_value,
    feature_selection,
    feature_selection_infer,
)
from model_pipeline import train_loan_model, return_lightgbm_prediction

from config import generate_model_predictions, cat_feat_list, num_feat_list


def main(
    raw_data: pd.DataFrame,
    today_date: datetime,
    cat_feat_list: List[str],
    num_feat_list: List[str],
    impute_zero_column_list: List[str],
    impute_features_column_names_list: List[str],
    features_to_drop: List[str],
    train_data_file_path: str,
    inference_data_file_path: str,
    generate_model_predictions: str,
) -> Tuple:
    """
    Run modeling pipeline end to end

    """
    logger.info("starting loan prediction model")

    raw_data = create_target_variable(raw_data)

    raw_data = create_duration_features(raw_data, today_date)

    raw_data, categorical_column_names_list_encoded = encode_categorical_features(
        raw_data, cat_feat_list
    )

    raw_data = correct_data_types(raw_data, num_feat_list)

    raw_data = impute_with_zero(raw_data, impute_zero_column_list)

    input_data, inference_data = seperate_train_inference_data(raw_data)

    input_data = impute_missing_value(
        input_data,
        impute_features_column_names_list,
        categorical_column_names_list_encoded,
    )

    inference_data = impute_missing_value(
        inference_data,
        impute_features_column_names_list,
        categorical_column_names_list_encoded,
    )

    input_data, final_num_feature_list = feature_selection(
        input_data,
        categorical_column_names_list_encoded,
        num_feat_list,
        features_to_drop,
    )

    input_data_good = input_data[input_data["predicted_loan_status"] == "good"].sample(
        900
    )
    input_data_bad = input_data[input_data["predicted_loan_status"] == "bad"]
    input_data_balanced = pd.concat([input_data_good, input_data_bad])

    inference_data = feature_selection_infer(
        inference_data, categorical_column_names_list_encoded, num_feat_list
    )

    input_data_balanced.to_csv(train_data_file_path)
    logger.info("Writing model input data")
    inference_data.to_csv(inference_data_file_path)
    logger.info("Writing model prediction data")

    final_feature_list = final_num_feature_list + categorical_column_names_list_encoded

    loan_model, feature_importances_df, metrics_df = train_loan_model(
        input_data_balanced, final_feature_list, categorical_column_names_list_encoded
    )

    if generate_model_predictions:
        predictions_df = return_lightgbm_prediction(
            loan_model, inference_data, final_feature_list
        )
    else:
        predictions_df = None

    return predictions_df, feature_importances_df, metrics_df


if __name__ == "__main__":
    today_date = datetime.datetime.today()

    input_data_file_path = os.path.join(dir, "../data/loan_data.csv")
    train_data_file_path = os.path.join(dir, "../data/training_data.csv")
    inference_data_file_path = os.path.join(dir, "../data/inference_data.csv")
    raw_data = pd.read_csv(input_data_file_path)

    # list of features for missing value imputation
    impute_features_column_names_list = [
        "loan_amnt",
        "funded_amnt",
        "int_rate",
        "annual_inc",
        "dti",
        "delinq_2yrs",
        "mths_since_last_delinq",
        "open_acc",
        "mths_since_first_cr_line",
        "term",
    ]

    # list of features for missing value imputation with zero
    impute_zero_column_list = ["mths_since_last_delinq", "mths_since_first_cr_line"]

    # feature to drop based on high multi-collinearity
    features_to_drop = ["funded_amnt"]

    predictions_data, feature_importances_table, validation_metrics = main(
        raw_data,
        today_date,
        cat_feat_list,
        num_feat_list,
        impute_zero_column_list,
        impute_features_column_names_list,
        features_to_drop,
        train_data_file_path,
        inference_data_file_path,
        generate_model_predictions,
    )

    if generate_model_predictions:
        predictions_data.to_csv(
            os.path.join(dir, "../data/output/predictions_data.csv")
        )
        logger.info("Writing prediction data to output folder")

    feature_importances_table.to_csv(
        os.path.join(dir, "../data/evaluation/feature_importances_table.csv")
    )
    logger.info("Writing feature importance table")

    validation_metrics.to_csv(
        os.path.join(dir, "../data/evaluation/validation_metrics.csv")
    )
    logger.info("Writing validation metrics")
