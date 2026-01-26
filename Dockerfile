# syntax=docker/dockerfile:1.4
FROM python:slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --no-cache-dir -e .

# Use AWS creds securely during build (so boto3 can pull from S3)
RUN --mount=type=secret,id=awscreds,target=/root/.aws/credentials \
    --mount=type=secret,id=awsconfig,target=/root/.aws/config \
    python pipeline/training_pipeline.py

EXPOSE 5000
CMD ["python", "application.py"]
