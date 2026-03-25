from transformers import pipeline

# Load pretrained model
classifier = pipeline("text-classification")
#pipeline("sentiment-analysis")
#pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def predict_bert(text):
    result = classifier(text)[0]
    label = result["label"]
    score = float(result["score"])
    # Convert to spam/ham style
    if label.lower() in ["spam", "label_1", "positive"]:
        label = "spam"
    else:
        label = "ham"
    return 
    {
        "label": label,
        "confidence": score
    }