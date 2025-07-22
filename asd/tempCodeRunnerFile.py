from sklearn.datasets import fetch_20newsgroups
newsgroups_data = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
print(f"Toplam belge sayısı: {len(newsgroups_data.data)}")
print(f"Etiket sayısı: {len(set(newsgroups_data.target))}")
print("Örnek bir metin:\n")
print(newsgroups_data.data[0])