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

# ------------------------------------------------------------------
# Paths (Docker / Hugging Face safe)
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
MODEL_DIR = os.path.join(BASE_DIR, "final_models")

# ------------------------------------------------------------------
# Lifespan (Startup / Shutdown)
# ------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logging.info("Application startup initiated")

        # ---------------- MongoDB ----------------
        mongo_db_url = os.getenv("MONGO_DB_URL")
        if not mongo_db_url:
            raise ValueError("MONGO_DB_URL environment variable not set")

        ca = certifi.where()
        mongo_client = pymongo.MongoClient(
            mongo_db_url,
            tlsCAFile=ca
        )

        database = mongo_client[DATA_INGESTION_DATABASE_NAME]
        collection = database[DATA_INGESTION_COLLECTION_NAME]

        app.state.mongo_client = mongo_client
        app.state.database = database
        app.state.collection = collection

        logging.info("MongoDB connection established")

        # ---------------- Load ML Models ----------------
        preprocessor_path = os.path.join(MODEL_DIR, "preprocessor.pkl")
        model_path = os.path.join(MODEL_DIR, "model.pkl")

        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        preprocessor = load_object(preprocessor_path)
        model = load_object(model_path)

        app.state.network_model = NetworkModel(
            preprocessor=preprocessor,
            model=model
        )

        logging.info("ML model and preprocessor loaded successfully")

        yield  # ---- App is running ----

    except Exception as e:
        logging.error("Application startup failed")
        raise NetworkSecurityException(e, sys)

    finally:
        mongo_client = getattr(app.state, "mongo_client", None)
        if mongo_client:
            mongo_client.close()
            logging.info("MongoDB connection closed")

# ------------------------------------------------------------------
# FastAPI App
# ------------------------------------------------------------------
app = FastAPI(
    title="Network Security App",
    description="Phishing Detection & Network Security ML API",
    version="1.0.0",
    lifespan=lifespan
)

# ------------------------------------------------------------------
# Middleware
# ------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# Templates
# ------------------------------------------------------------------
templates = Jinja2Templates(directory=TEMPLATE_DIR)

# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------
@app.get("/")
def root():
    """Hugging Face health check"""
    return RedirectResponse(url="/docs")


@app.get("/train")
async def train_route():
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        return Response("Training completed successfully")
    except Exception as e:
        raise NetworkSecurityException(e, sys)


@app.post("/predict")
async def predict_route(request: Request, file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)

        network_model = request.app.state.network_model
        predictions = network_model.predict(df)

        df["prediction"] = predictions

        output_dir = os.path.join(BASE_DIR, "prediction_output")
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, "output.csv")
        df.to_csv(output_path, index=False)

        table_html = df.to_html(classes="table table-striped", index=False)

        return templates.TemplateResponse(
            "table.html",
            {
                "request": request,
                "table": table_html
            }
        )

    except Exception as e:
        raise NetworkSecurityException(e, sys)
