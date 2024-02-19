from flask import Flask, request, render_template
import pandas as pd
import pickle

app = Flask(__name__)
with open("best_xgb_model_pipeline.pkl", "rb") as file:
    pipeline = pickle.load(file)

# Categorical feature choices
feature_choices = {
    'gender': ['Male', 'Female'],
    'parental_education_level': ['Some high school', 'High school', 'Some college', "Associate's degree", "Bachelor's degree", "Master's degree"],
    'lunch': ['Standard', 'Free/reduced'],
    'test_prep_course': ['None', 'Completed']
}

@app.route('/')
def home():
    # Pass feature names and choices to the template
    return render_template('index.html', feature_choices=feature_choices)

@app.route('/predict', methods=['POST'])
def predict():
    # Extract features from form inputs
    features = {feature: request.form[feature] for feature in feature_choices}
    for feature in ['math_score', 'reading_score', 'writing_score']:
        features[feature] = float(request.form[feature])
    
    # Convert features to DataFrame
    import pandas as pd
    features_df = pd.DataFrame([features])
    
    # Prediction
    prediction = pipeline.predict(features_df)
    
    # Map prediction to named class
    mapping = {0: 'Group A', 1: 'Group B', 2: 'Group C', 3: 'Group D', 4: 'Group E'}
    output = mapping.get(prediction[0], "Unknown Group")

    return render_template('index.html', feature_choices=feature_choices, prediction_text=f'Predicted Group: {output}')

if __name__ == "__main__":
    app.run(debug=True)