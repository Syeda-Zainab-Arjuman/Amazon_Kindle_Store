import numpy as np
from flask import Flask, request, render_template
import joblib
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf
import re
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Clear any existing Keras/TensorFlow sessions
tf.keras.backend.clear_session()

# Load the pre-trained model
model = load_model("sentiment_model.h5")

# Load the vectorizer
vectorizer = joblib.load('vectorizer.save')

# Label mapping
label_map = {0: 'Negative review', 1: 'Neutral review', 2: 'Positive review'}

# Preprocessing function
def preprocess_text(text):
    wordnet = WordNetLemmatizer()
    temp = re.sub('[^a-zA-Z]', ' ', text)
    temp = temp.lower().split()
    temp = [wordnet.lemmatize(word) for word in temp if word not in set(stopwords.words('english'))]
    return ' '.join(temp)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/y_predict', methods=['POST'])
def y_predict():
    try:
        # Get input sentence from the form
        d = request.form['Sentence']
        print("Input sentence:", d)

        # Preprocess input
        processed = preprocess_text(d)

        # Transform input and make prediction
        transformed_input = vectorizer.transform([processed]).toarray()
        result = model.predict(transformed_input)
        pred_label = np.argmax(result, axis=1)[0]
        output = label_map[pred_label]

        return render_template('index.html', prediction_text='{}'.format(output))
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)