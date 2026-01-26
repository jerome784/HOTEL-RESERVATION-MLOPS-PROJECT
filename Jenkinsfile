pipeline {
  agent any

  environment {
    AWS_REGION = "us-east-1"
    AWS_ACCOUNT_ID = "784167813131"
    ECR_REPO_NAME = "hotel-reservation-mlops"
    ECR_REPO = "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO_NAME}"

    EB_APP = "hotel-reservation-mlops-app"
    EB_ENV = "hotel-reservation-mlops-env"

    // version label can be anything unique per build
    VERSION_LABEL = "build-${BUILD_NUMBER}"
  }

  stages {

    stage('Checkout Code') {
      steps {
        checkout scm
      }
    }

    stage('Build Docker Image (needs S3 creds for training)') {
      steps {
        withCredentials([usernamePassword(
          credentialsId: 'aws-s3-creds',
          usernameVariable: 'S3_AWS_ACCESS_KEY_ID',
          passwordVariable: 'S3_AWS_SECRET_ACCESS_KEY'
        )]) {
          sh '''
            set -e

            # Create temporary AWS credential files for BuildKit secrets (NOT committed, NOT baked)
            mkdir -p .aws_tmp

            cat > .aws_tmp/credentials <<EOF
[default]
aws_access_key_id=${S3_AWS_ACCESS_KEY_ID}
aws_secret_access_key=${S3_AWS_SECRET_ACCESS_KEY}
EOF

            cat > .aws_tmp/config <<EOF
[default]
region=${AWS_REGION}
output=json
EOF

            # Enable BuildKit for secrets support
            export DOCKER_BUILDKIT=1

            docker build \
              --secret id=awscreds,src=.aws_tmp/credentials \
              --secret id=awsconfig,src=.aws_tmp/config \
              -t ${ECR_REPO_NAME}:latest .

            # cleanup temp files
            rm -rf .aws_tmp
          '''
        }
      }
    }

    stage('Login to ECR (CICD creds)') {
      steps {
        withCredentials([[
          $class: 'AmazonWebServicesCredentialsBinding',
          credentialsId: 'aws_credentials'
        ]]) {
          sh '''
            set -e
            aws --version
            aws ecr get-login-password --region ${AWS_REGION} \
              | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com
          '''
        }
      }
    }

    stage('Tag & Push to ECR') {
      steps {
        sh '''
          set -e
          docker tag ${ECR_REPO_NAME}:latest ${ECR_REPO}:latest
          docker push ${ECR_REPO}:latest
        '''
      }
    }

    stage('Deploy to Elastic Beanstalk (Docker pulls from ECR)') {
      steps {
        withCredentials([[
          $class: 'AmazonWebServicesCredentialsBinding',
          credentialsId: 'aws_credentials'
        ]]) {
          sh '''
            set -e

            # 1) Create zip for EB (must include Dockerrun.aws.json at root)
            rm -f deploy.zip
            zip -r deploy.zip Dockerrun.aws.json

            # 2) Create EB application if not exists (ignore error if exists)
            aws elasticbeanstalk create-application --application-name "${EB_APP}" \
              || echo "EB application already exists"

            # 3) Create a new application version (uploads to EB-managed S3 internally)
            aws elasticbeanstalk create-application-version \
              --application-name "${EB_APP}" \
              --version-label "${VERSION_LABEL}" \
              --source-bundle S3Bucket="$(aws elasticbeanstalk create-storage-location --query S3Bucket --output text)",S3Key="deploy-${VERSION_LABEL}.zip" \
              --auto-create-application

            # Upload the zip to that bucket/key
            BUCKET="$(aws elasticbeanstalk create-storage-location --query S3Bucket --output text)"
            aws s3 cp deploy.zip "s3://${BUCKET}/deploy-${VERSION_LABEL}.zip"

            # 4) Update the environment to the new version
            aws elasticbeanstalk update-environment \
              --application-name "${EB_APP}" \
              --environment-name "${EB_ENV}" \
              --version-label "${VERSION_LABEL}"
          '''
        }
      }
    }
  }
}
