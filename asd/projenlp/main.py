
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from preprocess import clean_text

# 1. Veri setini yükle
categories = ['sci.space', 'rec.sport.baseball', 'comp.graphics', 'talk.politics.misc']
data = fetch_20newsgroups(subset='all', categories=categories, shuffle=True, random_state=42)

# 2. Metinleri temizle
print("Metinler temizleniyor...")
cleaned_data = [clean_text(text) for text in data.data]

# 3. TF-IDF ile özellik çıkarımı
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(cleaned_data)
y = data.target

# 4. Veriyi eğitim ve test olarak ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Naive Bayes 
model = MultinomialNB()
model.fit(X_train, y_train)

#Test
y_pred = model.predict(X_test)

print("\n✔ Doğruluk Oranı:", accuracy_score(y_test, y_pred))
print("\n📊 Sınıflandırma Raporu:\n")
print(classification_report(y_test, y_pred, target_names=data.target_names))

from visualize import plot_confusion
plot_confusion(y_test, y_pred, data.target_names)

while True:
    print("\n🗞️ Yeni bir haber metni girin (çıkmak için 'q' yazın):")
    user_input = input(">>> ")

    if user_input.lower() == 'q':
        print("Çıkılıyor...")
        break

    #Metni temizleme
    cleaned_input = clean_text(user_input)

    #TF-IDF vektörleştirme
    input_vec = vectorizer.transform([cleaned_input])

    #Tahmin yap ve olasılıkları alma
    probabilities = model.predict_proba(input_vec)[0]
    pred_index = probabilities.argmax()
    pred_label = data.target_names[pred_index]
    pred_prob = probabilities[pred_index] * 100

    print(f"\n📌 Bu haber büyük olasılıkla şu kategoriye ait: **{pred_label}**")
    print(f"🔢 Tahmin olasılığı: %{pred_prob:.2f}")
