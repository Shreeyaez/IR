import pandas as pd
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

df = pd.read_csv("C:/Users/Acer/Documents/ProductReviewProject/1429_1.csv")

df = df[['reviews.text']].rename(columns={'reviews.text': 'review'})

df.dropna(inplace=True)

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    cleaned = " ".join (tokens)
    return cleaned

df['cleaned_review'] = df['review'].apply(clean_text)

df.to_csv("cleaned_reviews.csv", index = False)

print("Preprocessing complete. Cleaned reviews saved to 'cleaned_reviews.csv'")
print(df.head())