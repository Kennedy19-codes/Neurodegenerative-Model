import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Define symptoms for each disease (1 means the symptom is present, 0 means absent)
diseases = ['Alzheimer’s', 'Parkinson’s', 'ALS', 'Huntington’s']

# Define the symptoms for each disease profile
symptoms_profile = {
    'Alzheimer’s': ['memory_loss', 'confusion', 'difficulty_recognizing_people', 'poor_judgment', 'cognitive_decline'],
    'Parkinson’s': ['tremors', 'slowness', 'stooped_posture', 'small_handwriting', 'drooling'],
    'ALS': ['muscle_weakness', 'speech_difficulty', 'difficulty_swallowing', 'muscle_cramps', 'difficulty_holding_objects'],
    'Huntington’s': ['chorea', 'mood_swings', 'seizures', 'cognitive_decline', 'personality_changes']
}

# Create a function to generate synthetic patients
def generate_synthetic_data(num_patients=50):
    data = []
    for _ in range(num_patients):
        # Randomly select a disease
        disease = random.choice(diseases)
        # Generate symptom data based on disease profile
        patient_data = {symptom: 1 if symptom in symptoms_profile[disease] else 0 for symptom in set(sum(symptoms_profile.values(), []))}
        # Append the disease label
        patient_data['disease'] = disease
        data.append(patient_data)
    return data

# Generate synthetic data for 50 patients
synthetic_data = generate_synthetic_data(num_patients=50)

# Convert to DataFrame
df = pd.DataFrame(synthetic_data)

# Features (symptoms)
X = df.drop(columns=['disease'])

# Target (disease)
y = df['disease']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a simple Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Save the trained model and symptom columns for later use
joblib.dump(model, 'disease_prediction_model.pkl')
joblib.dump(list(X.columns), 'symptom_columns.pkl')  # Save the symptom columns

