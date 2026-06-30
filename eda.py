import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Load cleaned dataset
data = pd.read_csv("dataset/clean_news.csv")

# ------------------------
# 1. Bar Chart
# ------------------------
plt.figure(figsize=(6,4))
data["label"].value_counts().plot(kind="bar")
plt.title("Fake vs Real News")
plt.xlabel("Label (0 = Fake, 1 = Real)")
plt.ylabel("Count")
plt.savefig("bar_chart.png")
plt.show()

# ------------------------
# 2. Pie Chart
# ------------------------
plt.figure(figsize=(6,6))
data["label"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%",
    labels=["Fake", "Real"]
)
plt.ylabel("")
plt.title("News Distribution")
plt.savefig("pie_chart.png")
plt.show()

# ------------------------
# 3. Word Cloud
# ------------------------
# Remove missing values
data = data.dropna(subset=["text"])

# Convert text column to string
data["text"] = data["text"].astype(str)

text = " ".join(data["text"].tolist())

wordcloud = WordCloud(
    width=800,
    height=400,
    background_color="white"
).generate(text)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud")
plt.savefig("wordcloud.png")
plt.show()

print("EDA completed successfully!")