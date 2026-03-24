import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

vectorizer = TfidfVectorizer()
model = LogisticRegression()

def train_model(X, y):
    X_vec = vectorizer.fit_transform(x)
    model.fit(X_vec, y)

    with open("model.pkl", "wb") as f:
        pickle.dump((vectorizer, model), f)

def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)