import pickle
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open('code/model.pkl', 'rb'))
scaler = pickle.load(open('code/scaler.pkl', 'rb'))


# Mapping functions
gender_map = {'male': 0, 'female': 1}
bool_map = {'true': 1, 'false': 0}
career_labels = {
    0: 'Lawyer', 1: 'Doctor', 2: 'Biotechnology', 3: 'Biomedical', 4: 'Civil', 5: 'Engineer' }

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/recommend')
def recommend():
    return render_template('recommend.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect form data
        gender = 0 if request.form['gender'] == 'male' else 1
        extracurricular = 1 if request.form.get('extracurricular_activities') == 'true' else 0
        weekly_self_study_hours = float(request.form.get('weekly_self_study_hours', 0))

        math = float(request.form.get('math_score', 0))
        history = float(request.form.get('history_score', 0))
        physics = float(request.form.get('physics_score', 0))
        chemistry = float(request.form.get('chemistry_score', 0))
        biology = float(request.form.get('biology_score', 0))
        english = float(request.form.get('english_score', 0))

         # Derived features
        total_score = math + history + physics + chemistry + biology + english
        avg_score = total_score / 6

        # Prepare input for model
        features = np.array([[gender, extracurricular, weekly_self_study_hours,
                              math, history, physics, chemistry, biology, english,
                              total_score, avg_score]])
        scaled_input = scaler.transform(features)
        probabilities = model.predict_proba(scaled_input)[0]
        
        career_labels = ['Lawyer', 'Doctor', 'Civil', 'Biotechnology', 'Software Engineering', 'Biomedical']

        # Top 5 recommendations
        top5 = sorted(
            [(career_labels[i], round(prob, 4)) for i, prob in enumerate(probabilities)],
            key=lambda x: x[1], reverse=True)[:5]

        return render_template('results.html', recommendations=top5)

    except Exception as e:
        return f"❌ Error: {e}"
    #else:
        #return render_template('recommend.html')
if __name__ == '__main__':
        app.run(debug=True)
