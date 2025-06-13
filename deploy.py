import boto3

sagemaker = boto3.client("sagemaker")

model_name = "fraud-model-v1"
endpoint_name = "fraud-detection-endpoint"
role = "arn:aws:iam::377632750099:role/datazone_usr_role_5ih12zk69oqyvq_ce0yt0fsdqn7o6"

# Create model
sagemaker.create_model(
    ModelName=model_name,
    PrimaryContainer={
        "Image": "985815980388.dkr.ecr.eu-west-1.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3",
        "ModelDataUrl": "s3://model-output-1306/fraud-detection-job-2025-06-13-05-25-15-297/output/model.tar.gz"
    },
    ExecutionRoleArn=role
)

# Create endpoint config
sagemaker.create_endpoint_config(
    EndpointConfigName=endpoint_name + "-config",
    ProductionVariants=[{
        "InstanceType": "ml.m5.large",
        "InitialInstanceCount": 1,
        "ModelName": model_name,
        "VariantName": "AllTraffic"
    }]
)

# Create endpoint
sagemaker.create_endpoint(
    EndpointName=endpoint_name,
    EndpointConfigName=endpoint_name + "-config"
)
