from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pickle
import os

app = FastAPI(title="Iris Classification")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "model.pkl"


# Model Training

iris = load_iris()
X = iris.data
y = iris.target

model = None
accuracy = None


def train_model():
    global model, accuracy

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

    modle = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)


def load_model():
    global model

    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    else:
        train_model()

load_model()


class IrisInput(BaseModel):
    sepal_length: float = Field(..., gt=0)
    sepal_width: float = Field(..., gt=0)
    petal_length: float = Field(..., gt=0)
    petal_width: float = Field(..., gt=0)



# FastAPI Routes

@app.get("/")
def home():
    return {"message": "Iris API Running"}


@app.get("/model-info")
def model_info():
    return {
        "model": "RandomForestClassifier",
        "dataset": "Iris Flowers",
        "classes": iris.target_names.tolist(),
        "accuracy": accuracy,
    }

@app.get("/health")
def health():
    return {"status": "OK"}



@app.post("/predict")
def predict(data: IrisInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    input_data = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0].max()

    return {"prediction": iris.target_names[pred], "confidence": round(float(prob), 3)}


@app.post("/predict-batch")
def predict_batch(data: list[IrisInput]):
    res = []

    for item in data:
        input_data=[[
            item.sepal_length,
            item.sepal_width,
            item.petal_length,
            item.petal_width,
        ]]

        pred = model.predict(input_data)[0]
        res.append(iris.target_names[pred])
    return {"predictions": res}


@app.get("/retrain")
def retrain():
    train_model()
    return {"message": "Model RETRAINED", "accuracy": accuracy}









