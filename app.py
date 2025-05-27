from flask import Flask, request, render_template
import joblib
import pandas as pd

# Load model and encoder
model = joblib.load('course_recommender_model.pkl')
encoder = joblib.load('course_recommender_encoder.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    goal = request.form['goal']
    hobby = request.form['hobby']

    # Create DataFrame from user input
    input_df = pd.DataFrame({'goal': [goal], 'hobby': [hobby]})

    # Encode input
    encoded_input = encoder.transform(input_df)

    # Predict
    prediction = model.predict(encoded_input)

    return render_template('index.html', prediction_text=f"Recommended Course: {prediction[0]}")

if __name__ == '__main__':
    app.run(debug=True)
