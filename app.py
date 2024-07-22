from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load your trained model
model = pickle.load(open('gas.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/y_predict', methods=['POST'])
def y_predict():
    try:
        # Extract user input
        year = int(request.form['Year'])
        month = int(request.form['Month'])
        day = int(request.form['Day'])

        # Validate input
        if not (1 <= month <= 12):
            raise ValueError("Month must be between 1 and 12")
        if not (1 <= day <= 31):
            raise ValueError("Day must be between 1 and 31")

        # Prepare data for prediction
        data = np.array([year, month, day])  # Example: create a NumPy array
        data = data.reshape(1, -1)  # Reshape for single prediction

        # Make prediction using your loaded model
        prediction = model.predict(data)  # Predict
        predicted_price = np.round(prediction[0], 2)  # Assuming prediction is a single value

        return render_template('predict.html', prediction_text=f"Predicted Natural Gas Price: ${predicted_price}")

    except ValueError as e:
        return render_template('predict.html', prediction_text=f"Error: {str(e)}")
    except Exception as e:
        return render_template('predict.html', prediction_text=f"An error occurred: {str(e)}")

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(debug=True)

