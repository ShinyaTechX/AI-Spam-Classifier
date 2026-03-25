from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import predict
from src.bert_model import predict_bert

app = FastAPI()

class Email(BaseModel):
    text: str

@app.post("/predict/ml")
def classify_ml(email: Email):
    return predict(email.text)

@app.post("/predict/bert")
def classify_bert(email: Email):
    return predict_bert(email.text)
