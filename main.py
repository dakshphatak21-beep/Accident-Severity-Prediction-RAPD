
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Load trained model
saved = joblib.load("severity_model.pkl")
model = saved["model"]
features = saved["features"]

app = FastAPI(title="Accident Severity Prediction")


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


@app.get("/")
def home():
    return {"message": "Accident Severity API is running"}


@app.post("/predict")
def predict(data: AccidentData):

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

    df = pd.DataFrame([input_data])
    df = df[features]

    prediction = int(model.predict(df)[0]) + 1
    probabilities = model.predict_proba(df)[0]

    return {
        "predicted_severity": prediction,
        "severity_label": f"Severity {prediction}",
        "probabilities": {
            f"Severity {i+1}": round(float(p), 4)
            for i, p in enumerate(probabilities)
        }
    }
