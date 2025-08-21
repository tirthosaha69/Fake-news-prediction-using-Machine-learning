from flask import Flask, render_template, request
import re
import joblib
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load model and vectorizer
model = joblib.load("Models/model.pkl")
vectorizer = joblib.load("Models/vectorizer.pkl")

# Initialize Flask app
app = Flask(__name__)

# Preprocessing (same as training)
port_stem = PorterStemmer()
def stemming(content):
    steammed_content = re.sub('[^a-zA-Z]', ' ', content)
    steammed_content = steammed_content.lower()
    steammed_content = steammed_content.split()
    steammed_content = [port_stem.stem(word) for word in steammed_content if not word in stopwords.words('english')]
    steammed_content = ' '.join(steammed_content)
    return steammed_content

# Home route
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""
    if request.method == "POST":
        news_text = request.form["news"]
        processed_text = stemming(news_text)
        vectorized_text = vectorizer.transform([processed_text])
        result = model.predict(vectorized_text)[0]
        prediction = "✅ Real News" if result == 1 else "❌ Fake News"
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
