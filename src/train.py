import pandas as pd
from model import train_model

data = pd.read_csv("../data/spam.csv", encoding="latin-1")

train_model(data["v2"], data["v1"])
print("Model trained and saved!")