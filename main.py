from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load model & encoders
model = joblib.load("fitness_model.pkl")
encoders = joblib.load("encoders.pkl")

feature_encoders = encoders["features"]
label_encoders = encoders["labels"]

app = FastAPI(
    title="Nizcare Fitness Plan API",
    description="Generate personalized Exercise & Meal Plans using ML",
    version="1.0"
)

# Request body
class InputData(BaseModel):
    gender: str
    goal: str
    bmi: str

@app.post("/predict")
def predict_plan(data: InputData):

    # Encode input
    encoded = [
        feature_encoders["Gender"].transform([data.gender])[0],
        feature_encoders["Goal"].transform([data.goal])[0],
        feature_encoders["BMI Category"].transform([data.bmi])[0]
    ]

    # Predict
    result = model.predict([encoded])[0]

    # Decode results
    exercise = label_encoders["Exercise Schedule"].inverse_transform([result[0]])[0]
    meal = label_encoders["Meal Plan"].inverse_transform([result[1]])[0]

    return {
        "exercise_plan": exercise,
        "meal_plan": meal,
        "status": "success"
    }

@app.get("/")
def home():
    return {"message": "Nizcare Fitness Recommendation API is running!"}