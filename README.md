# Network Security: Phishing Data Detection

This project implements a machine learning pipeline to detect phishing attempts based on network security data. It includes data ingestion, validation, transformation, model training, and a web interface for real-time predictions.

## Features

* **Automated Training Pipeline**: Orchestrates data ingestion, validation, transformation, and model training.
* **MongoDB Integration**: Supports pushing raw CSV data to MongoDB for centralized storage.
* **FastAPI Web Interface**: Provides endpoints for triggering the training pipeline and performing batch predictions via CSV upload.
* **Extensible Architecture**: Built with modular components for easy maintenance and scaling.

## Tech Stack

* **Language**: Python
* **Web Framework**: FastAPI, Uvicorn
* **Database**: MongoDB
* **ML Libraries**: Scikit-learn, Pandas, Numpy
* **Tracking & DevOps**: MLflow, Dagshub

## Installation

1. **Clone the repository.**
2. **Install dependencies**:
```bash
pip install -r requirements.txt

```


3. **Setup the package**:
```bash
python setup.py install

```



## Usage

### 1. Data Ingestion

To push your raw data (e.g., `phisingData.csv`) to MongoDB, use the `push_data.py` script:

```bash
python push_data.py

```

### 2. Training

You can trigger the training pipeline via the FastAPI endpoint `/train` or by running the `main.py` script.

### 3. Prediction

Start the FastAPI server:

```bash
python app.py

```

Navigate to the `/docs` or `/predict` route to upload a CSV file and receive predictions.

## Project Structure

* `app.py`: FastAPI application for training and prediction routes.
* `main.py`: Entry point for executing the full training pipeline.
* `push_data.py`: Utility to convert CSV data to JSON and insert it into MongoDB.
* `networksecurity/`: Core package containing components, entities, pipelines, and utilities.
* `requirements.txt`: List of necessary Python packages.
* `setup.py`: Configuration for package installation.

## Author

**Reeya Borikar** - [reeyaborikar02@gmail.com](mailto:reeyaborikar02@gmail.com)