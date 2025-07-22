from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_precision_recall_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from joblib import dump
from preprocess import clean_text

categories = ['sci.space', 'rec.sport.baseball', 'comp.graphics', 'talk.politics.misc']

def train_and_save_model():
    # Veri setini al
    data = fetch_20newsgroups(subset='train', categories=categories)
    cleaned_texts = [clean_text(text) for text in data.data]

    # Vectorizer ve model eğit
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(cleaned_texts)
    y = data.target

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    # Kaydet
    dump(model, "model.joblib")
    dump(vectorizer, "vectorizer.joblib")

    print("Model ve vectorizer başarıyla kaydedildi.")

def evaluate_model():
    # Precision-Recall grafiği çizmek istersen
    data = fetch_20newsgroups(subset='test', categories=categories)
    cleaned_texts = [clean_text(text) for text in data.data]

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(cleaned_texts)
    y = data.target

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    # PR Curve
    plot_precision_recall_curve(model, X, y)
    plt.title("Precision-Recall Curve")
    plt.show()

if __name__ == "__main__":
    train_and_save_model()
    # evaluate_model()  # İstersen yorumdan çıkar
