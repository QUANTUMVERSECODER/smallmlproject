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








