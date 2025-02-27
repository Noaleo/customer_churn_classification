import functools
import time
import os
import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import json
from sklearn.metrics import precision_score, recall_score, classification_report
import matplotlib.pyplot as plt

# Paths
os.makedirs("output", exist_ok=True)
file_name = "churn_data.csv"
model_path_1M = "output/model_1M.pkl"
model_path_2M = "output/model_2M.pkl"
predictions_path = "data/predictions.csv"
metrics_path = "output/metrics.json"

def log_step(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"\U0001F680 Starting: {func.__name__}")
        start_time = time.time()  # Start timing
        result = func(*args, **kwargs)
        end_time = time.time()  # End timing
        elapsed_time = end_time - start_time  # Compute time elapsed
        print(f"✅ Completed: {func.__name__} in {elapsed_time:.2f} seconds\n")
        return result
    return wrapper

# Load Data
@log_step
def load_data():
    df = pd.read_csv(file_name, parse_dates=['date'])
    return df

@log_step
def fill_nan(df: pd.DataFrame) -> pd.DataFrame:
    """
    fill missing value for transaction amount & plan_type:
    1. for transaction amount fill the mean of each customer_id & plan_type
    2. for plan_type --> fill the mode per customer_id
    """
    relevant_group = df.groupby(['customer_id', 'plan_type'])['transaction_amount']
    df['transaction_amount'] = relevant_group.transform(lambda x: x.fillna(x.mean()))
    df['plan_type'] = df.groupby('customer_id')['plan_type'].transform(
        lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'Unknown'))

    return df

@log_step
def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    create date features
    """
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month

    return df

@log_step
def encode_categorical_features(df, cols_to_encode: list):
    """

    :param df:
    :param cols_to_encode:
    :return:
    """
    for col in cols_to_encode:
        if df[col].nunique() < 5:
            df[col] = df[col].astype('category').cat.codes
        else:
        # todo OHE if needed to
            pass

    return df

@log_step
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    # TODO check the below and add documentation
    # todo add more features such as mode of plan_type per user how many months he's using each of the plans?
    df['transaction_7d_avg'] = df.groupby('customer_id')['transaction_amount'].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean())
    df['transaction_30d_avg'] = df.groupby('customer_id')['transaction_amount'].transform(
        lambda x: x.rolling(window=30, min_periods=1).mean())
    df['days_since_last_txn'] = df.groupby('customer_id')['date'].transform(lambda x: (x - x.shift()).dt.days.fillna(0))

    return df

@log_step
def shift_churn(df: pd.DataFrame):
    """
    todo make sure to remove of df_churn_1 the first month and for df_churn_2 the first 2 months
    # in the train-test-split?
    todo add documentation -- we shift the target 1/2 months backward so its row contain its 'predicted' label
    todo fix it --> when we shift 1 month then x_val is ok, when we shift 2 months then we need to change x_val to contain different dates
    """
    df['churn_1M'] = df.groupby('customer_id')['churn'].shift(-1)
    df['churn_2M'] = df.groupby('customer_id')['churn'].shift(-2)

    return df

@log_step
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    create date features
    """
    df = fill_nan(df)
    df = create_time_features(df)

    cols_to_encode = df.select_dtypes('object')
    df = encode_categorical_features(df, cols_to_encode)

    df = feature_engineering(df)
    df = shift_churn(df)

    return df

@log_step
def train_test_valid_split(df: pd.DataFrame):
    """
    split the data to train-test-valid:
        - train contains january-october
        - valid contain november
        - test contain december
    :return: X_train, X_val, X_test, y_train, y_val, y_test
    """
    df_train = df[df['date'] < '2023-11-01']
    df_val = df[(df['date'] >= '2023-11-01') & (df['date'] < '2023-12-01')]
    df_test = df[df['date'] >= '2023-12-01']
    cols_to_drop = ['customer_id', 'date', 'churn', 'churn_1M', 'churn_2M']


    X_train, y_train_1M, y_train_2M = df_train.drop(columns=cols_to_drop), df_train['churn_1M'], df_train['churn_2M']
    X_val, y_val_1M, y_val_2M = df_val.drop(columns=cols_to_drop), df_val['churn_1M'], df_val['churn_2M']
    X_test = df_test.drop(columns=cols_to_drop)

    return X_train, y_train_1M, y_train_2M, X_val, y_val_1M, y_val_2M, X_test, df_test


@log_step
def train_model(X_train, y_train, X_val, y_val):
    param_grid = {
        'n_estimators': [70, 75, 85],
        'learning_rate': [0.2, 0.25, 0.3],
        'max_depth': [7, 9, 11]
    }

    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='recall', verbose=1)
    grid_search.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)

    print(f"Best Parameters: {grid_search.best_params_}")

    best_model = grid_search.best_estimator_
    return best_model


@log_step
def evaluate_model(model, X_val, y_val):
    y_pred_ = model.predict(X_val)
    precision = precision_score(y_val, y_pred_)
    recall = recall_score(y_val, y_pred_)
    class_report = classification_report(y_val, y_pred_)

    return {
        'precision': precision,
        'recall': recall,
        'classification_report': class_report
    }


@log_step
def save_model(model, path):
    joblib.dump(model, path)

@log_step
def generate_predictions(model, X_test, df_test):
    df_test['prediction'] = model.predict(X_test)
    df_test.to_csv(predictions_path, index=False)

@log_step
def plot_feature_importance(model, feature_names, title):
    importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.show()


if __name__ == "__main__":
    df = load_data()
    df = preprocess_data(df)
    X_train, y_train_1M, y_train_2M, X_val, y_val_1M, y_val_2M, X_test, df_test = train_test_valid_split(df)

    model_1M = train_model(X_train, y_train_1M, X_val, y_val_1M)
    # model_2M = train_model(X_train, y_train_2M, X_val, y_val_2M)

    # Add this step after training each model
    # plot_feature_importance(model_1M, X_train.columns, "Feature Importance for 1M Churn Model")
    # plot_feature_importance(model_2M, X_train.columns, "Feature Importance for 2M Churn Model")

    metrics_1M = evaluate_model(model_1M, X_val, y_val_1M)
    # metrics_2M = evaluate_model(model_2M, X_val, y_val_2M)

    with open(metrics_path, 'w') as f:
        json.dump({'1M': metrics_1M}, f)#, '2M': metrics_2M}, f)

    save_model(model_1M, model_path_1M)
    # save_model(model_2M, model_path_2M)

    generate_predictions(model_1M, X_test, df_test, 'prediction_1M')
    # generate_predictions(model_2M, X_test, df_test, 'prediction_2M')

    print("🎉 Pipeline complete: Models trained, evaluated, and predictions saved.")
