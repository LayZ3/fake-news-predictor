# Fake News Predictor

A Flask-based Fake News Predictor using Natural Language Processing (NLP), TF-IDF feature extraction, and Logistic Regression.

## Model Performance

- **Accuracy: 98.71%** on the held-out test set
- **Classifier:** Logistic Regression
- **Feature extraction:** TF-IDF
- **Dataset:** ISOT Fake & Real News Dataset

## Features

- 🟢 REAL NEWS result indicator
- 🔴 FAKE NEWS result indicator
- Confidence percentage and progress bar
- Animated prediction result
- Character and word counter
- Clear button
- Example News button
- Loading animation
- Responsive web interface
- Model accuracy display
- About, workflow, and dataset sections
- Fact-checking disclaimer

## Project Structure

```text
fake-news-predictor/
├── data/
│   ├── Fake.csv                 # download separately
│   └── True.csv                 # download separately
├── model/
│   ├── fake_news_model.pkl      # generated locally
│   └── tfidf_vectorizer.pkl     # generated locally
├── templates/
│   └── index.html
├── static/
│   ├── style.css
│   └── script.js
├── app.py
├── train_model.py               # add your training script
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

Place the trained files in `model/`:

```text
model/fake_news_model.pkl
model/tfidf_vectorizer.pkl
```

Then run:

```bash
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

## Dataset

This project was trained with the ISOT Fake & Real News Dataset. Keep the original CSV files locally in `data/` and run your training script to recreate the model files.

## Important Disclaimer

This application is an educational machine-learning classifier. The **98.71%** figure is the measured test accuracy from the training run and does not guarantee that every future article will be classified correctly. The application is **not a definitive fact-checking service**. Verify important claims using multiple trusted sources.

## Technologies

Python · Flask · Pandas · NumPy · scikit-learn · Joblib · HTML · CSS · JavaScript
