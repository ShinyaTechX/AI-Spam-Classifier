from model import load_model

vectorizer, model = load_model()

def predict(msg):
    X = vectorizer.transform([msg])
    label = model.predict(X)[0]
    prob = model.predict_proba(X)[0].max()

    return 
    {
        "label": label,
        "confidence": float(prob)
    }