

# MLOps Hotel Reservation Prediction

An end-to-end **MLOps pipeline** for predicting hotel reservation cancellations using **LightGBM**, **MLflow**, and a clean modular architecture.

This project covers the **complete ML lifecycle**:

* Data ingestion from AWS S3
* Data preprocessing & feature engineering
* Model training with hyperparameter tuning
* Experiment tracking using MLflow
* Artifact management and logging

---

## Project Structure

```text
MLOPS-Hotel-Reservation/
│
├── artifacts/
│   ├── processed/
│   └── models/
│
├── config/
│   ├── config.yaml
│   ├── model_params.py
│   └── paths_config.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── pipeline/
│   └── training_pipeline.py
│
├── src/
│   ├── data_ingestion.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── logger.py
│   └── custom_exception.py
│
├── utils/
│   └── common_functions.py
│
├── logs/
│
├── mlruns/
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Tech Stack

* **Python 3.9+**
* **LightGBM**
* **Scikit-learn**
* **MLflow**
* **Pandas / NumPy**
* **AWS S3 (Data ingestion)**
* **SMOTE (Class imbalance handling)**

---

## Pipeline Flow

### 1️⃣ Data Ingestion

* Downloads dataset from **AWS S3**
* Splits into train & test datasets
* Saves to `data/raw/`

### 2️⃣ Data Preprocessing

* Handles missing values
* Label encoding for categorical features
* Skewness correction
* SMOTE for class imbalance
* Feature selection
* Saves processed data to `artifacts/processed/`

### 3️⃣ Model Training

* LightGBM classifier
* Hyperparameter tuning via `RandomizedSearchCV`
* Model evaluation (Accuracy, Precision, Recall, F1)
* Model saved to `artifacts/models/`
* Full experiment tracking with **MLflow**

---

##  Experiment Tracking (MLflow)

MLflow tracks:

*  Metrics
*  Hyperparameters
*  Trained model
*  Datasets used in training

To start MLflow UI:

```bash
mlflow ui
```

Then open:

```
http://localhost:5000
```

---

##  Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/MLOPS-Hotel-Reservation.git
cd MLOPS-Hotel-Reservation
```

### 2️⃣ Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

##  Run the Full Pipeline

```bash
python pipeline/training_pipeline.py
```

This will:

* Ingest data
* Preprocess data
* Train model
* Save model
* Log experiments to MLflow

---

##  Configuration

### `config/config.yaml`

Controls:

* S3 bucket info
* Train-test split ratio
* Feature lists
* Preprocessing parameters

### `config/model_params.py`

Defines:

* LightGBM hyperparameter search space
* RandomizedSearchCV configuration

---

##  Logging & Error Handling

* Centralized logging via `logger.py`
* Custom exceptions via `CustomException`
* Logs stored in `logs/app.log`

---

##  Files Ignored in GitHub

The following are excluded via `.gitignore`:

* `artifacts/`
* `mlruns/`
* `logs/`
* `*.pkl`
* `data/raw/`
* `data/processed/`
* `venv/`

---

##  Future Improvements

* Dockerization
* CI/CD with GitHub Actions
* MLflow model registry
* API deployment (FastAPI)
* Drift detection & monitoring

---

##  Author

**Jerome Philip John**
B.Tech Computer Science | ML & MLOps Enthusiast

