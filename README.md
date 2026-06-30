# 📰 Fake News Detection System

## About the Project

This project is a Fake News Detection System developed using Python, Machine Learning, and Flask. It predicts whether a news article is Real or Fake by analyzing its text. The project uses Natural Language Processing (NLP) techniques to clean and process news content before making predictions.

The main goal of this project is to reduce the spread of misinformation by providing an easy-to-use web application where users can check the authenticity of news articles.

## Features

- Predicts whether news is Real or Fake
- Uses NLP for text preprocessing
- TF-IDF Vectorization for feature extraction
- Compares Logistic Regression, Naive Bayes, and Random Forest models
- Displays Accuracy, Precision, Recall, and F1-Score
- Generates a Confusion Matrix
- Includes EDA with Bar Chart, Pie Chart, and Word Cloud
- User-friendly web interface built with Flask, HTML, and CSS
- Displays prediction confidence

## Technologies Used

- Python
- Flask
- HTML
- CSS
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Matplotlib
- WordCloud
- Joblib

## Project Structure

```
Fake-News-Detection/
│── dataset/
│── static/
│── templates/
│── preprocessing.py
│── train_model.py
│── evaluation.py
│── eda.py
│── predict.py
│── app.py
│── model.pkl
│── vectorizer.pkl
│── requirements.txt
│── README.md
```

## How to Run the Project

1. Clone or download the project.
2. Install all required libraries:
   ```
   pip install -r requirements.txt
   ```
3. Train the model:
   ```
   python train_model.py
   ```
4. Run the Flask application:
   ```
   python app.py
   ```
5. Open your browser and visit:
   ```
   http://127.0.0.1:5000
   ```

## Results

The system classifies a news article as **Real News** or **Fake News** and also displays the confidence score. The trained models are evaluated using Accuracy, Precision, Recall, F1-Score, and a Confusion Matrix to identify the best-performing model.

## Future Improvements

- Improve prediction accuracy using deep learning models.
- Add support for news articles in multiple languages.
- Deploy the application on a cloud platform for public access.

## Important Note

The following files are not included in this repository because they are large generated files and can be recreated by running the project:

- `dataset/Fake.csv`
- `dataset/True.csv`
- `dataset/clean_news.csv`
- `model.pkl`
- `vectorizer.pkl`

### How to Run the Project

1. Download the **Fake and Real News Dataset**.
2. Place `Fake.csv` and `True.csv` inside the `dataset/` folder.
3. Run the following commands:

```bash
python preprocessing.py
python train_model.py
python app.py
```

These commands will automatically generate:

- `clean_news.csv`
- `model.pkl`
- `vectorizer.pkl`


## Author

**Anita Mishra**