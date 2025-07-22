import tkinter as tk
from tkinter import ttk, messagebox
from joblib import load
from preprocess import clean_text
from sklearn.datasets import fetch_20newsgroups

# Model ve vectorizer'ı yükle
try:
    model = load("model.joblib")
    vectorizer = load("vectorizer.joblib")
except Exception as e:
    messagebox.showerror("Yükleme Hatası", f"Model veya vektörleştirici yüklenemedi:\n{e}")
    exit()

categories = ['sci.space', 'rec.sport.baseball', 'comp.graphics', 'talk.politics.misc']
data = fetch_20newsgroups(subset='train', categories=['sci.space', 'rec.sport.baseball', 'comp.graphics', 'talk.politics.misc'])

def predict_category():
    user_text = text_entry.get("1.0", tk.END).strip()
    if not user_text:
        messagebox.showwarning("Uyarı", "Lütfen bir metin girin.")
        return

    try:
        cleaned = clean_text(user_text)
        vec = vectorizer.transform([cleaned])
        probs = model.predict_proba(vec)[0]
        pred_idx = probs.argmax()
        label = data.target_names[pred_idx]
        prob = probs[pred_idx] * 100
        result_var.set(f"Kategori: {label}\nOlasılık: %{prob:.2f}")
    except Exception as e:
        messagebox.showerror("Tahmin Hatası", f"Tahmin yapılırken hata oluştu:\n{e}")



# ARAYÜZ TASARIMI 
# Ana pencere
root = tk.Tk()
root.title("News Article Classifier")
root.geometry("700x450")
root.configure(bg="#f3f3f5")
root.resizable(False, False)  # Pencere boyutunu sabitler

# Stil
style = ttk.Style()
style.configure("TLabel", background="#f3f3f5", font=("Calibri", 14))
style.configure("TButton", font=("Calibri", 12), padding=6)

# Sonuç göstermek için değişken
result_var = tk.StringVar()

# Başlık Label (ortalanmış)
title_label = ttk.Label(root, text="Haber Metnini Girin", font=("Calibri", 16, "bold"))
title_label.grid(row=0, column=0, pady=(20,10), sticky="n", padx=20)
root.grid_columnconfigure(0, weight=1)

# Text alanı
text_entry = tk.Text(root, height=10, width=70, font=("Calibri", 12))
text_entry.grid(row=1, column=0, padx=20)

# Buton
btn = ttk.Button(root, text="Tahmin Et", command=predict_category)
btn.grid(row=2, column=0, pady=15)

# Sonuç Label
result_label = ttk.Label(root, textvariable=result_var, foreground="#0b6623", font=("Calibri", 13, "bold"))
result_label.grid(row=3, column=0, pady=10)

root.mainloop()



