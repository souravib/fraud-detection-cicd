# train.py

import os
import sys
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket', type=str, required=True, help='S3 bucket containing the data')
    parser.add_argument('--key', type=str, required=True, help='S3 key for the data file')
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    args = parser.parse_args()

    try:
        print(f"📥 Downloading data from S3: s3://{args.bucket}/{args.key}")
        s3 = boto3.client('s3')
        response = s3.get_object(Bucket=args.bucket, Key=args.key)
        content = response['Body'].read().decode('utf-8')

        print("🔄 Parsing records...")
        records = []
        for line in content.strip().split('\n'):
            parsed = json.loads(line)
            for header, values in parsed.items():
                columns = header.split('\t')
                row = values.split('\t')
                records.append(dict(zip(columns, row)))

        print("📊 Converting to DataFrame")
        df = pd.DataFrame(records)
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
        df['is_fraud'] = pd.to_numeric(df['is_fraud'], errors='coerce')
        df['hour'] = df['timestamp'].dt.hour
        df = df.dropna()

        print("✂️ Splitting data...")
        X = df[['amount', 'hour']]
        y = df['is_fraud']
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

        print("🤖 Training model...")
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('logreg', LogisticRegression())
        ])
        pipeline.fit(X_train, y_train)

        print("💾 Saving model...")
        model_path = os.path.join(args.model_dir, 'model.joblib')
        joblib.dump(pipeline, model_path)
        print(f"✅ Model saved to {model_path}")

        print("📈 Evaluating model...")
        y_pred = pipeline.predict(X_test)
        report = classification_report(y_test, y_pred)
        print(report)

    except Exception as e:
        print(f"❌ Error during training: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
