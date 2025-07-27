
import pickle
import numpy as np
from flask import Flask, render_template, request, abort


app = Flask(__name__)

def load_model(path='./training/gwp.pkl'):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

model = load_model()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            features = [
                float(request.form.get('quarter', 0)),
                float(request.form.get('department', 0)),
                float(request.form.get('day', 0)),
                float(request.form.get('team', 0)),
                float(request.form.get('targeted_productivity', 0)),
                float(request.form.get('smv', 0)),
                float(request.form.get('over_time', 0)),
                float(request.form.get('incentive', 0)),
                float(request.form.get('idle_time', 0)),
                float(request.form.get('idle_men', 0)),
                float(request.form.get('no_of_style_change', 0)),
                float(request.form.get('no_of_workers', 0)),
                float(request.form.get('month', 0))
            ]
            if model is None:
                abort(500, description="Model not loaded.")
            final_features = np.array(features).reshape(1, -1)
            prediction = model.predict(final_features)[0]
            prediction = round(float(prediction), 4)
            return render_template('submit.html', prediction=prediction)
        except Exception as e:
            print(f"Prediction error: {e}")
            abort(400, description="Invalid input or prediction error.")
    return render_template('predict.html')

if __name__ == "__main__":
    app.run(debug=True)