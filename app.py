from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import os

# Load the trained model and symptom columns
model = joblib.load(os.path.join('model', 'disease_prediction_model.pkl'))
symptom_columns = joblib.load(os.path.join('model', 'symptom_columns.pkl'))

app = Flask(__name__)

@app.route('/')
def home():
    symptoms = [
        "memory_loss", "confusion", "difficulty_recognizing_people", 
        "poor_judgment", "cognitive_decline", "tremors", "slowness", 
        "stooped_posture", "small_handwriting", "drooling", 
        "muscle_weakness", "speech_difficulty", "difficulty_swallowing", 
        "muscle_cramps", "difficulty_holding_objects", "chorea", 
        "mood_swings", "seizures", "personality_changes"
    ]
    return render_template('symptom_form.html', symptoms=symptoms)

@app.route('/predict', methods=['POST'])
def predict():
    selected_symptoms = request.form.getlist('symptoms')
    prediction = predict_disease(selected_symptoms)
    return render_template('result.html', predicted_disease=prediction)

def predict_disease(symptoms):
    symptom_dict = {col: 0 for col in symptom_columns}
    for symptom in symptoms:
        if symptom in symptom_dict:
            symptom_dict[symptom] = 1
    input_data = pd.DataFrame([symptom_dict])
    prediction = model.predict(input_data)
    return prediction[0]

if __name__ == '__main__':
    app.run(debug=True)
