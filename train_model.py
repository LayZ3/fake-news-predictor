import os
import re
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")
fake["label"] = 0
true["label"] = 1
data = pd.concat([fake, true], ignore_index=True)
data["title"] = data["title"].fillna("")
data["text"] = data["text"].fillna("")
data["content"] = (data["title"] + " " + data["text"]).apply(clean_text)
data = data[data["content"].str.len() > 0]

X_train, X_test, y_train, y_test = train_test_split(
    data["content"], data["label"], test_size=0.20, random_state=42, stratify=data["label"]
)

vectorizer = TfidfVectorizer(
    stop_words="english", max_df=0.7, min_df=2,
    max_features=50000, ngram_range=(1, 1), sublinear_tf=True
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)
pred = model.predict(X_test_tfidf)

print(f"Accuracy: {accuracy_score(y_test, pred) * 100:.2f}%")
print(classification_report(y_test, pred, target_names=["Fake", "Real"]))

os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/fake_news_model.pkl")
joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")
print("Model saved successfully.")
