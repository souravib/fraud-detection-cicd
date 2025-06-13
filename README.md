# Fraud Detection ML Pipeline (CI/CD with SageMaker)

This project automates training and deployment of a fraud detection model using AWS SageMaker, CodeBuild, and CodePipeline.

## Project Files

- `train.py`: Training logic
- `buildspec.yml`: CodeBuild config
- `requirements.txt`: Python dependencies
- `deploy.py`: Deploy model to SageMaker endpoint

## Setup Steps

1. Push this repo to GitHub
2. Create a CodeBuild project pointing to it
3. Create a CodePipeline with GitHub as source and CodeBuild as build step
