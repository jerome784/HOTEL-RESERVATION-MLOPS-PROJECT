pipeline{
    agent any

    environment {
        VENV_DIR = 'venv'
    }

    stages{
        stage('Cloning github repo to Jenkins'){
            steps{
                script{
                    echo 'Cloning Github repo to Jenkins ......'
                    checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'github-token', url: 'https://github.com/jerome784/HOTEL-RESERVATION-MLOPS-PROJECT.git']])
                }
            }
        }

        stage('Setting up virtual enironment and installing dependencies'){
            steps{
                script{
                    echo 'Cloning Github repo to Jenkins ......'
                    sh '''
                    python -m venv ${VENV_DIR}
                    . ${VENV_DIR}/bin/activate
                    pip install -upgrade pip
                    pip install -e .
                    '''
                }
            }
        }
    }
}