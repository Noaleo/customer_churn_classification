from typing import List
import functools
import json
import os
import time

from sklearn.metrics import precision_score, recall_score, classification_report
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import joblib
import matplotlib.pyplot as plt
import pandas as pd


def log_step(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"\U0001f680 Starting: {func.__name__}")
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"✅ Completed: {func.__name__} in {elapsed_time:.2f} seconds\n")
        return result

    return wrapper


def load_data():
    df = pd.read_csv("data/churn_data.csv", parse_dates=["date"])
    return df


def run(n_shift: int):
    df = load_data()
    df = preprocess_data(df, n_shift)
    x_train, y_train, x_val, y_val, x_test, df_test = train_test_valid_split(
        df, n_shift=n_shift
    )
    model = train_model(x_train, y_train, x_val, y_val)

    metrics = evaluate_model(model, x_val, y_val)
    print("Metrics: ")
    print(metrics)
    print("***************")

    metrics_path = f"output/metrics_{n_shift}M.json"
    with open(metrics_path, "w") as f:
        json.dump({f"{n_shift}M": metrics}, f)

    model_path = f"output/model_{n_shift}M.pkl"
    save_model(model, model_path)

    x_predictions = generate_predictions(model, x_test, df_test)
    x_final_predictions = generate_final_predictions(x_predictions)
    df = df.merge(x_final_predictions, on=["customer_id", "date"])
    df.to_csv("output/predictions.csv", index=False)
    print("🎉 Pipeline complete: Models trained, evaluated, and predictions saved.")


def fill_nan(df: pd.DataFrame) -> pd.DataFrame:
    """
    fill missing value for transaction amount & plan_type:
    1. for transaction amount fill the mean with each customer_id & plan_type
    2. for plan_type --> fill the mode per customer_id
    """
    relevant_group = df.groupby(["customer_id", "plan_type"])["transaction_amount"]
    df["transaction_amount"] = relevant_group.transform(lambda x: x.fillna(x.mean()))
    df["plan_type"] = df.groupby("customer_id")["plan_type"].transform(
        lambda x: x.fillna(x.mode()[0] if not x.mode().empty else "Unknown")
    )
    return df


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    create date features
    """
    df["quarter"] = df["date"].dt.quarter
    df["month"] = df["date"].dt.month

    return df


def encode_categorical_features(
    df: pd.DataFrame, cols_to_encode: List[str]
) -> pd.DataFrame:
    """
    encode categorical features --> we have only 1 with less than 5 values so no need to implement the OHE
    """
    for col in cols_to_encode:
        if df[col].nunique() < 5:
            df[col] = df[col].astype("category").cat.codes
        else:
            # todo OHE if needed to
            pass

    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = create_time_features(df)
    for i in range(3, 8):
        col_name = f"transaction_{i}m_avg"
        df[col_name] = df.groupby("customer_id")["transaction_amount"].transform(
            lambda x: x.rolling(window=i, min_periods=1).mean()
        )
    # todo consider calculate this feature after the train-test-val-split?
    df["plan_type_mode"] = df.groupby("customer_id", sort=False)["plan_type"].agg(
        lambda x: x.mode()[0]
    )
    plan_counts = (
        df.groupby(["customer_id", "plan_type"], sort=False)["date"]
        .nunique()
        .unstack(fill_value=0)
    )
    plan_counts.columns = [
        f"plan_{col}_months" for col in plan_counts.columns
    ]
    df = df.merge(plan_counts, on="customer_id")

    return df


def shift_label(df: pd.DataFrame, n_shift: int) -> pd.DataFrame:
    """
    shift the label n_shift backward so that each month predict the label n_shift month forward
    """
    new_label_name = f"churn_{n_shift}M"
    df[new_label_name] = df.groupby("customer_id")["churn"].shift(-n_shift)

    return df


def preprocess_data(df: pd.DataFrame, n_shift: int) -> pd.DataFrame:
    """
    fill nan, feature engineering, shift the churn column & encodings
    """
    df = shift_label(df, n_shift=n_shift)
    df = fill_nan(df)
    cols_to_encode = df.select_dtypes("object")
    df = encode_categorical_features(df, cols_to_encode)
    df = feature_engineering(df)

    return df


def train_test_valid_split(df: pd.DataFrame, n_shift: int):
    """
    split the data to train-test-valid based on n_shift (1|2):
        - train until september/october
        - valid contain october/november
        - test contain november/december
    :return: X_train, X_val, X_test, y_train, y_val, y_test
    """
    churn_col = f"churn_{n_shift}M"
    if n_shift == 1:
        df_train = df[df["date"] < "2023-11-01"]
        df_val = df[(df["date"] >= "2023-11-01") & (df["date"] < "2023-12-01")]
        df_test = df[df["date"] >= "2023-12-01"]
    elif n_shift == 2:
        df_train = df[df["date"] < "2023-10-01"]
        df_val = df[(df["date"] >= "2023-10-01") & (df["date"] < "2023-11-01")]
        df_test = df[df["date"] >= "2023-11-01"]
    else:
        raise ValueError(f"n_shift must be 1 or 2. Got {n_shift}")
    cols_to_drop = ["customer_id", "date", "churn", churn_col]
    (
        x_train,
        y_train,
    ) = (
        df_train.drop(columns=cols_to_drop),
        df_train[churn_col],
    )
    x_val, y_val = df_val.drop(columns=cols_to_drop), df_val[churn_col]
    x_test = df_test.drop(columns=cols_to_drop)
    return x_train, y_train, x_val, y_val, x_test, df_test


def train_model(x_train, y_train, x_val, y_val):
    param_grid = {
        "n_estimators": [82, 83, 84],
        "learning_rate": [0.26, 0.27, 0.28],
        "max_depth": [5, 6, 7],
    }
    model = XGBClassifier(
        use_label_encoder=False, eval_metric="logloss", random_state=42
    )
    grid_search = GridSearchCV(model, param_grid, cv=10, scoring="recall", verbose=1)
    grid_search.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=True)
    print(f"Best Parameters: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_

    return best_model


def evaluate_model(model, x_val, y_val):
    y_pred_ = model.predict(x_val)
    precision = precision_score(y_val, y_pred_)
    recall = recall_score(y_val, y_pred_)
    # class_report = classification_report(y_val, y_pred_, output_dict=True)

    return {
        "precision": precision,
        "recall": recall,
        # 'classification_report': class_report
    }


def save_model(model, path):
    joblib.dump(model, path)


def generate_predictions(model, x_test, df_test):
    df_test.loc[:, "prediction"] = model.predict(x_test)

    return df_test[["customer_id", "date", "prediction"]]


def plot_feature_importance(model, feature_names, title):
    importance = model.feature_importances_
    feature_importance_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": importance}
    )
    feature_importance_df = feature_importance_df.sort_values(
        by="Importance", ascending=False
    )
    plt.figure(figsize=(10, 6))
    plt.barh(
        feature_importance_df["Feature"],
        feature_importance_df["Importance"],
        color="skyblue",
    )
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.savefig("output/feature_importance", bbox_inches="tight")
    plt.show()


def generate_final_predictions(X_pred):
    """
    Generates final prediction --> if churn is 1 in any of the dates --> then churn
    # todo: next step, take approximate value of % of churn and take the k most % to leave?
    """
    final_pred = X_pred.groupby("customer_id", sort=False)["prediction"].max()
    X_pred.merge(final_pred, on="customer_id")

    return X_pred


if __name__ == "__main__":
    # os.chdir("..")
    os.makedirs("output", exist_ok=True)
    functions_to_metric = [
        run, preprocess_data, train_test_valid_split, train_model
    ]
    for function in functions_to_metric:
        locals()[function.__name__] = log_step(function)
    # X_pred_1 = run(n_shift=1)
    run(n_shift=2)
