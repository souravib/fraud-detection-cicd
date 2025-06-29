import os
import argparse
import boto3
import pandas as pd
import joblib
import json
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from xgboost import XGBClassifier

def main():
    # --- Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket', type=str, required=True)
    parser.add_argument('--key', type=str, required=True)
    args = parser.parse_args()

    # --- Load train_data.csv from S3
    print("🔄 Downloading data from S3...")
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=args.bucket, Key=args.key)
    df = pd.read_csv(obj['Body'])

    print(f"✅ Loaded {df.shape[0]} rows")

    # --- Optional cleanup
    df = df[df['amount'] > 0]

    # --- Define features and target
    features = ['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    target = 'isFraud'
    X = df[features]
    y = df[target]

    # --- Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # --- Handle class imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"⚖️ scale_pos_weight = {scale_pos_weight:.2f}")

    # --- Preprocessing pipeline
    numeric_features = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    categorical_features = ['type']

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('xgb', XGBClassifier(
            objective='binary:logistic',
            use_label_encoder=False,
            eval_metric='logloss',
            scale_pos_weight=scale_pos_weight,
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        ))
    ])

    # --- Train the model
    print("🚀 Training model...")
    pipeline.fit(X_train, y_train)
    print("✅ Training complete.")

    # --- Evaluate and save metrics
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }

    os.makedirs("metrics", exist_ok=True)
    with open("metrics/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # --- Save model with Pickle protocol 4 (safe for SageMaker Python 3.6/3.7)
    os.makedirs("model", exist_ok=True)
    joblib.dump(pipeline, "model/model.pkl", protocol=4)

    # --- Log classification report
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))

    print("\n✅ Model saved to model/model.pkl")
    print("✅ Metrics saved to metrics/metrics.json")

if __name__ == "__main__":
    main()
