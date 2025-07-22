import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import string
# Gerekli NLTK verilerini indir (ilk seferde çalıştır)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    # Küçük harfe çevir
    text = text.lower()
    
    # Özel karakterleri ve sayıları temizle
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Tokenize et
    tokens = text.split()
    
    # Stopword'leri kaldır
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization uygula
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    
    # Tekrar metne çevir
    return ' '.join(lemmatized_tokens)