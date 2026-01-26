pipeline {
    agent any

    environment {
        AWS_REGION = "us-east-1"
        ECR_REPO = "784167813131.dkr.ecr.us-east-1.amazonaws.com/hotel-reservation-mlops"
    }

    stages {

        stage('Checkout Code') {
            steps {
                checkout scm
            }
        }

        stage('Login to ECR') {
            steps {
                withCredentials([[
                    $class: 'AmazonWebServicesCredentialsBinding',
                    credentialsId: 'aws_credentials'
                ]]) {
                    sh '''
                    aws --version
                    aws ecr get-login-password --region $AWS_REGION \
                    | docker login --username AWS --password-stdin $ECR_REPO
                    '''
                }
            }
        }

        stage('Build Docker Image') {
            steps {
                sh '''
                docker build -t hotel-reservation-mlops .
                docker tag hotel-reservation-mlops:latest $ECR_REPO:latest
                '''
            }
        }

        stage('Push to ECR') {
            steps {
                sh '''
                docker push $ECR_REPO:latest
                '''
            }
        }
    }
}
