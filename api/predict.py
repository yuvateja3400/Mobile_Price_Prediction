from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

class MobileFeatures(BaseModel):
    battery_power: float
    blue: int
    clock_speed: float
    dual_sim: int
    fc: int
    four_g: int
    int_memory: int
    m_dep: float
    mobile_wt: int
    n_cores: int
    pc: int
    px_height: int
    px_width: int
    ram: int
    screen_area: int
    talk_time: int
    three_g: int
    touch_screen: int
    wifi: int

model = joblib.load("models/best_model.pkl")

@app.post("/predict")
def predict_price(features: MobileFeatures):
    data = np.array([[
        features.battery_power,
        features.blue,
        features.clock_speed,
        features.dual_sim,
        features.fc,
        features.four_g,
        features.int_memory,
        features.m_dep,
        features.mobile_wt,
        features.n_cores,
        features.pc,
        features.px_height,
        features.px_width,
        features.ram,
        features.talk_time,
        features.three_g,
        features.touch_screen,
        features.wifi,
        features.screen_area  # <- use directly, don't compute
    ]])

    prediction = model.predict(data)[0]
    return {"predicted_price_range": int(prediction)}

@app.get("/")
def read_root():
    return {"message": "Welcome to the Mobile Price Prediction API. Use POST /predict"}
