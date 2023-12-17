import logging
import sys
import warnings
import datetime
import numpy as np
import pandas as pd

from typing import Dict, List, Tuple
import lightgbm as lgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

sys.path.append("..")

warnings.filterwarnings("ignore")

logger = logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)s] [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def train_lightgbm_model(
    X_train: pd.DataFrame,
    Y_train: pd.DataFrame,
    feature_name: List[str],
    categorical_feature: List[str],
    label: str,
) -> lgb.Booster:
    """Trains a LightGBM model, outputs the model object.

    Args:
        data (pd.DataFrame): Training data
        feature_name (list[str]): Names of the features in the raw data
        label (str): Name of the target/label column
        categorical_feature (optional): Names of the encoded categorical
        features.

    Returns:
        lgb.Booster: A LightGBM model object
    """
    logger.info("Training LightGBM")
    params = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
    }

    logger.info("Defined model hyperparameters")

    logger.info("Starting model training")

    model_object = lgb.LGBMClassifier(**params)
    model_object.fit(
        X_train,
        Y_train,
        feature_name=feature_name,
        categorical_feature=categorical_feature,
    )

    logger.info("Finished training the model")

    return model_object


def return_lightgbm_prediction(
    model_object: lgb.Booster,
    inference_features: pd.DataFrame,
    final_feature_list: List[str],
) -> pd.DataFrame:
    """Returns the prediction from the model object on each observation in the
    inference dataset.
    Args:
        model_object (lgb.Booster): A lightGBM model object.
        inference_features (pd.DataFrame): A dataframe with the features for the
        inference dataset.

    Returns:
        pd.DataFrame: A dataframe with point predictions for the inference dataset
    """
    logger.info("Starting predictions on the inference dataset")
    predictions = model_object.predict(inference_features[final_feature_list])
    probabilities = model_object.predict_proba(inference_features[final_feature_list])
    predictions_df = pd.DataFrame(
        {
            "ID": inference_features["id"].to_numpy().tolist(),
            "funded_loan_amount": inference_features["funded_amnt"].to_numpy().tolist(),
            "prediction": predictions,
            "default_probability": probabilities[:, 0],
        }
    )
    logger.info("Predictions generated successfully")

    return predictions_df


def train_loan_model(
    df: pd.DataFrame,
    final_feature_list: List[str],
    categorical_column_names_list: List[str],
    target_column_name: str = "predicted_loan_status",
) -> Tuple:
    df.dropna(subset=[target_column_name], inplace=True)
    df.reset_index(drop=True)

    features = df.drop(target_column_name, axis=1)
    X_train, X_val, Y_train, Y_val = train_test_split(
        features, df[target_column_name], random_state=1, test_size=0.3
    )

    model_object_lightgbm = train_lightgbm_model(
        X_train,
        Y_train,
        final_feature_list,
        categorical_column_names_list,
        target_column_name,
    )

    predictions = model_object_lightgbm.predict(X_val)

    accuracy = accuracy_score(Y_val, predictions)
    logger.info(f"Accuracy score from validation set: {accuracy}")

    logger.info(
        f"Validation Metrics below:\n{classification_report(Y_val, predictions)}"
    )
    metrics_df = classification_report(Y_val, predictions, output_dict=True)
    metrics_df = pd.DataFrame(metrics_df).transpose()

    logger.info(f"Confusion Matrix:\n{confusion_matrix(Y_val, predictions)}")

    feature_importances = model_object_lightgbm.feature_importances_
    gain_importance_df = pd.DataFrame(
        {"Feature": final_feature_list, "Gain": feature_importances}
    )
    gain_importance_df = gain_importance_df.sort_values(by="Gain", ascending=False)

    return model_object_lightgbm, gain_importance_df, metrics_df
