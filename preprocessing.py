import pandas as pd
import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required resources
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")

# Load datasets
fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")

# Add labels
fake["label"] = 0
true["label"] = 1

# Merge datasets
data = pd.concat([fake, true], ignore_index=True)

# Keep only required columns
data = data[["text", "label"]]

# Initialize NLP tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\\S+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    tokens = word_tokenize(text)

    words = []
    for word in tokens:
        if word not in stop_words:
            words.append(lemmatizer.lemmatize(word))

    return " ".join(words)

# Apply preprocessing
data["text"] = data["text"].apply(clean_text)

# Shuffle dataset
data = data.sample(frac=1, random_state=42)

# Save cleaned dataset
data.to_csv("dataset/clean_news.csv", index=False)

print("Preprocessing completed successfully.")
print(data.head())