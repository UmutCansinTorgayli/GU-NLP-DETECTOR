from sklearn.datasets import fetch_20newsgroups

categories = ['alt.atheism', 'soc.religion.christian',
              'comp.graphics', 'sci.med']

data = fetch_20newsgroups(
    subset='all',
    categories=categories,
    shuffle=True,
    random_state=42,
    data_home=r'C:\Users\btorg\OneDrive\Belgeler\python\asd\projenlp'
)

print("Veri başarıyla indirildi.")