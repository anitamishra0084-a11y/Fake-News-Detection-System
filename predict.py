from flask import Flask, render_template, request
from predict import predict_news

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    news = request.form["news"]

    result, confidence = predict_news(news)

    return render_template(
        "index.html",
        prediction_text=result,
        confidence=confidence
    )

if __name__ == "__main__":
    app.run(debug=True)