# train.py

import os
import argparse
import boto3
import pandas as pd
import json
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

def main():
    # Set up argument parser for SageMaker hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket', type=str, required=True, help='S3 bucket containing the data')
    parser.add_argument('--key', type=str, required=True, help='S3 key for the data file')
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    args = parser.parse_args()

    # 1. Read the file content from S3
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=args.bucket, Key=args.key)
    content = response['Body'].read().decode('utf-8')

    # 2. Parse lines and extract records
    records = []
    for line in content.strip().split('\n'):
        parsed = json.loads(line)
        for header, values in parsed.items():
            columns = header.split('\t')
            row = values.split('\t')
            records.append(dict(zip(columns, row)))

    # 3. Convert to DataFrame
    df = pd.DataFrame(records)

    # 4. Clean and preprocess
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
    df['is_fraud'] = pd.to_numeric(df['is_fraud'], errors='coerce')
    df['hour'] = df['timestamp'].dt.hour
    df = df.dropna()

    # 5. Train/test split
    X = df[['amount', 'hour']]
    y = df['is_fraud']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    # 6. Build and train pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegression())
    ])
    pipeline.fit(X_train, y_train)

    # 7. Save model
    model_path = os.path.join(args.model_dir, 'model.joblib')
    joblib.dump(pipeline, model_path)

    # 8. (Optional) Print classification report
    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred)
    print(report)

if __name__ == '__main__':
    main()
