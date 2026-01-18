import os
import sys
import certifi
import pandas as pd
import pymongo

from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from starlette.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from networksecurity.logging.logger import logging
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.pipeline.training_pipeline import TrainingPipeline
from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import load_object
from networksecurity.constant.training_pipeline import (
    DATA_INGESTION_COLLECTION_NAME,
    DATA_INGESTION_DATABASE_NAME,
)

# --- Path Configurations ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
MODEL_DIR = os.path.join(BASE_DIR, "final_models")

# --- Lifespan: Handles Startup and Shutdown Logic ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logging.info("Application startup initiated")

        # 1. Establish MongoDB Connection
        mongo_db_url = os.getenv("MONGO_DB_URL")
        if not mongo_db_url:
            raise ValueError("MONGO_DB_URL environment variable not set")

        ca = certifi.where()
        mongo_client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)

        app.state.mongo_client = mongo_client
        app.state.database = mongo_client[DATA_INGESTION_DATABASE_NAME]
        app.state.collection = app.state.database[DATA_INGESTION_COLLECTION_NAME]

        logging.info("MongoDB connection established")

        # 2. Load ML Artifacts (Preprocessor and Model)
        preprocessor_path = os.path.join(MODEL_DIR, "preprocessor.pkl")
        model_path = os.path.join(MODEL_DIR, "model.pkl")

        if not os.path.exists(preprocessor_path) or not os.path.exists(model_path):
            logging.warning("Model artifacts not found. Please run /train route first.")
        else:
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)
            # Initialize the NetworkModel estimator
            app.state.network_model = NetworkModel(preprocessor=preprocessor, model=model)
            logging.info("ML model and preprocessor loaded into app state")

        yield  # App is now running and serving requests

    except Exception as e:
        logging.error(f"Application startup failed: {str(e)}")
        raise NetworkSecurityException(e, sys)
    finally:
        # Cleanup on shutdown
        if hasattr(app.state, "mongo_client"):
            app.state.mongo_client.close()
            logging.info("MongoDB connection closed")

app = FastAPI(
    title="Network Security API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory=TEMPLATE_DIR)

# --- GET Route: Root ---
@app.get("/")
def root():
    """Redirects to the interactive Swagger UI documentation"""
    return RedirectResponse(url="/docs")

# --- GET Route: Training ---
@app.get("/train")
async def train_route():
    """Triggers the Training Pipeline"""
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        return Response("Training completed successfully")
    except Exception as e:
        raise NetworkSecurityException(e, sys)

# --- POST Route: Prediction ---
@app.post("/predict")
async def predict_route(request: Request, file: UploadFile = File(...)):
    """
    POST method to upload a CSV and receive predictions displayed in an HTML table.
    """
    try:
        # Read the uploaded CSV file
        df = pd.read_csv(file.file)

        # Access the pre-loaded model from app state
        if not hasattr(request.app.state, "network_model"):
            return Response("Model not loaded. Please ensure models exist in final_models/ and restart.")

        network_model = request.app.state.network_model
        
        # Perform prediction
        predictions = network_model.predict(df)
        df["prediction"] = predictions

        # Save result locally
        output_dir = os.path.join(BASE_DIR, "prediction_output")
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(os.path.join(output_dir, "output.csv"), index=False)

        # Convert dataframe to HTML for the response template
        table_html = df.to_html(classes="table table-striped", index=False)

        return templates.TemplateResponse(
            "table.html", 
            {"request": request, "table": table_html}
        )

    except Exception as e:
        raise NetworkSecurityException(e, sys)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)