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
    # Set up arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket', type=str, required=True)
    parser.add_argument('--key', type=str, required=True)
    args = parser.parse_args()

    # Load file from S3
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=args.bucket, Key=args.key)
    content = response['Body'].read().decode('utf-8')

    # Parse lines
    records = []
    for line in content.strip().split('\n'):
        parsed = json.loads(line)
        for header, values in parsed.items():
            columns = header.split('\t')
            row = values.split('\t')
            records.append(dict(zip(columns, row)))

    df = pd.DataFrame(records)
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
    df['is_fraud'] = pd.to_numeric(df['is_fraud'], errors='coerce')
    df['hour'] = df['timestamp'].dt.hour
    df.dropna(inplace=True)

    X = df[['amount', 'hour']]
    y = df['is_fraud']
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegression())
    ])
    pipeline.fit(X_train, y_train)

    # Save model
    os.makedirs("model", exist_ok=True)
    joblib.dump(pipeline, "model/model.joblib")

    # Optional: log performance
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    main()
