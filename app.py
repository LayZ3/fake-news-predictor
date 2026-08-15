from flask import Flask, render_template, request
import joblib
import re
from pathlib import Path

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

model = joblib.load(MODEL_DIR / "fake_news_model.pkl")
vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")
MODEL_ACCURACY = "98.71%"


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", model_accuracy=MODEL_ACCURACY)


@app.route("/predict", methods=["POST"])
def predict():
    news_text = request.form.get("news_text", "").strip()
    if not news_text:
        return render_template("index.html", error="Please enter a news article.", model_accuracy=MODEL_ACCURACY)

    cleaned = clean_text(news_text)
    transformed = vectorizer.transform([cleaned])
    prediction = int(model.predict(transformed)[0])
    confidence = float(max(model.predict_proba(transformed)[0]) * 100)

    return render_template(
        "index.html",
        prediction="REAL" if prediction == 1 else "FAKE",
        confidence=f"{confidence:.2f}",
        submitted_text=news_text,
        model_accuracy=MODEL_ACCURACY,
    )


if __name__ == "__main__":
    app.run(debug=True)
