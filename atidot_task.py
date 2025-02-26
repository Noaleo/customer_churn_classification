import pandas as pd
import json
# import joblib
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import precision_score, recall_score

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import precision_score, recall_score

# Paths
file_name = "churn_data.csv"
predictions_path = "data/predictions.csv"
model_path = "output/model.pkl"
metrics_path = "output/metrics.json"

# Load Data
def load_data():
    df = pd.read_csv(file_name, parse_dates=['date'])
    return df


def fill_nan(df: pd.DataFrame) -> pd.DataFrame:
    """
    fill missing value for transaction amount & plan_type:
    1. for transaction amount fill the mean of each customer_id & plan_type
    2. for plan_type --> fill the mode per customer_id
    """
    relevant_group = df.groupby(['customer_id', 'plan_type'])['transaction_amount']
    df['transaction_amount'] = relevant_group.transform(lambda x: x.fillna(x.mean()))
    df['plan_mode'] = df.groupby('customer_id')['plan_type'].transform(
        lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'Unknown'))

    return df

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    create date features
    """
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month

    return df

# Preprocess Data
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    create date features
    """
    df = fill_nan(df)
    df = create_time_features(df)

    return df

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

    X_train, y_train = df_train.drop(columns=['customer_id', 'date', 'churn']), df_train['churn']
    X_val, y_val = df_val.drop(columns=['customer_id', 'date', 'churn']), df_val['churn']
    X_test, y_test = df_test.drop(columns=['customer_id', 'date', 'churn']), df_test['churn']

    return X_train, X_val, X_test, y_train, y_val, y_test

# Train Model
# def train_model(df):
#     X = df.drop(columns=['customer_id', 'date', 'churn'])
#     y = df['churn']
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     model = RandomForestClassifier(n_estimators=100, random_state=42)
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#
#     # Evaluate
#     precision = precision_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)
#
#     # Save model and metrics
#     joblib.dump(model, model_path)
#     with open(metrics_path, 'w') as f:
#         json.dump({'precision': precision, 'recall': recall}, f)
#
#     print("Model trained and saved.")


# Predict Churn
# def predict_churn():
#     df = load_data()
#     model = joblib.load(model_path)
#     df = preprocess_data(df)
#     X = df.drop(columns=['customer_id', 'date', 'churn'])
#     df['prediction'] = model.predict(X)
#     df.to_csv(predictions_path, index=False)
#     print("Predictions saved.")


if __name__ == "__main__":
    df = load_data()
    df = preprocess_data(df)
    # train_model(df)
    # predict_churn()
