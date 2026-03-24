from fastapi import FastAPI
from pydantic import BaseModel
from predict import predict

app = FastAPI()

class Email(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Spam Classifier API running"}

@app.post("/predict")
def classify(email: Email):
    return predict(email.text)

@app.post("/batch")
def batch_classify(emails: list[str]):
    return [predict(e) for e in emails]