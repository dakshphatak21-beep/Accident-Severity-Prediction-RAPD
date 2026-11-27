
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import pandas as pd
import joblib
import traceback


# Load trained model
MODEL_PATH = Path(__file__).resolve().parent / "severity_model.pkl"

try:
    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    features = saved["features"]
    print("Model loaded successfully")
    print("Number of model features:", len(features))

except Exception as e:
    print("MODEL LOADING ERROR:", repr(e))
    raise


# Create FastAPI application
app = FastAPI(title="Accident Severity Prediction")


# Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Input data
class AccidentData(BaseModel):

    Start_Lat: float
    Start_Lng: float

    Temperature_F: float
    Humidity: float
    Pressure: float
    Visibility: float
    Wind_Speed: float
    Precipitation: float

    Amenity: bool = False
    Bump: bool = False
    Crossing: bool = False
    Give_Way: bool = False
    Junction: bool = False
    No_Exit: bool = False
    Railway: bool = False
    Roundabout: bool = False
    Station: bool = False
    Stop: bool = False
    Traffic_Calming: bool = False
    Traffic_Signal: bool = False

    Hour: int
    DayOfWeek: int
    Month: int

    Night: bool
    Adverse_Weather: bool

    Distance: float = 0.0


# Homepage
@app.get("/")
def home():
    return {
        "message": "Accident Severity API is running"
    }


# Health check
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "feature_count": len(features)
    }


# Prediction endpoint
@app.post("/predict")
def predict(data: AccidentData):

    try:

        input_data = {

            "Start_Lat": data.Start_Lat,
            "Start_Lng": data.Start_Lng,

            "Temperature(F)": data.Temperature_F,
            "Humidity(%)": data.Humidity,
            "Pressure(in)": data.Pressure,
            "Visibility(mi)": data.Visibility,
            "Wind_Speed(mph)": data.Wind_Speed,
            "Precipitation(in)": data.Precipitation,

            "Amenity": int(data.Amenity),
            "Bump": int(data.Bump),
            "Crossing": int(data.Crossing),
            "Give_Way": int(data.Give_Way),
            "Junction": int(data.Junction),
            "No_Exit": int(data.No_Exit),
            "Railway": int(data.Railway),
            "Roundabout": int(data.Roundabout),
            "Station": int(data.Station),
            "Stop": int(data.Stop),
            "Traffic_Calming": int(data.Traffic_Calming),
            "Traffic_Signal": int(data.Traffic_Signal),

            "Hour": data.Hour,
            "DayOfWeek": data.DayOfWeek,
            "Month": data.Month,

            "Night": int(data.Night),
            "Adverse_Weather": int(data.Adverse_Weather),

            "Distance(mi)": data.Distance
        }

        # Convert input into DataFrame
        df = pd.DataFrame([input_data])

        # Check that all required model features exist
        missing_features = [
            feature for feature in features
            if feature not in df.columns
        ]

        if missing_features:
            raise ValueError(
                f"Missing model features: {missing_features}"
            )

        # Arrange columns in the exact training order
        df = df[features]

        # Make prediction
        prediction = int(model.predict(df)[0]) + 1

        probabilities = model.predict_proba(df)[0]

        # Prepare response
        result = {
            "predicted_severity": prediction,
            "severity_label": f"Severity {prediction}",
            "probabilities": {
                f"Severity {i+1}": round(float(p), 4)
                for i, p in enumerate(probabilities)
            }
        }

        return result

    except Exception as e:

        print("PREDICTION ERROR:")
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
