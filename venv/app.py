import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

app = Flask(__name__)

dataset = pd.read_csv('survey lung cancer.csv')


le_gender = LabelEncoder()
le_lung_cancer = LabelEncoder()


dataset['GENDER'] = le_gender.fit_transform(dataset['GENDER'])
dataset['LUNG_CANCER'] = le_lung_cancer.fit_transform(dataset['LUNG_CANCER'])

X = dataset.iloc[:, :15].values 
y = dataset.iloc[:, -1].values   

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Standardize the features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Initialize and train the SVC model
classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(X_train, y_train)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract form data
    gender = request.form['gender']
    age = int(request.form['age'])
    smoking = int(request.form['smoking'])
    yellow_fingers = int(request.form['yellow_fingers'])
    anxiety = int(request.form['anxiety'])
    peer_pressure = int(request.form['peer_pressure'])
    chronic_disease = int(request.form['chronic_disease'])
    fatigue = int(request.form['fatigue'])
    allergy = int(request.form['allergy'])
    wheezing = int(request.form['wheezing'])
    alcohol_consuming = int(request.form['alcohol_consuming'])
    coughing = int(request.form['coughing'])
    shortness_of_breath = int(request.form['shortness_of_breath'])
    swallowing_difficulty = int(request.form['swallowing_difficulty'])
    chest_pain = int(request.form['chest_pain'])

    # Map gender to numerical values if needed
    gender_numeric = 1 if gender == 'M' else 0  # Assuming Male (M) is 1, Female (F) is 0

    # Create feature array
    features = [gender_numeric, age, smoking, yellow_fingers, anxiety, peer_pressure,
                chronic_disease, fatigue, allergy, wheezing, alcohol_consuming,
                coughing, shortness_of_breath, swallowing_difficulty, chest_pain]

    # Reshape and predict
    final_features = np.array(features).reshape(1, -1)
    
    # Print features to check input values
    print("Input Features:", final_features)

    # Predict
    prediction = classifier.predict(sc.transform(final_features))

    output = prediction[0]
    prediction_text = 'YES' if output == 1 else 'NO'
    
    # Print prediction to check model output
    print("Prediction:", prediction_text)

    return render_template('index.html', prediction_text='Lung Cancer Prediction: {}'.format(prediction_text))


@app.route('/results', methods=['POST'])
def results():
    # Extract JSON data
    data = request.get_json(force=True)
    
    # Assuming the JSON keys are mapped to the correct order of features
    features = [int(data['gender']), int(data['age']), int(data['smoking']), int(data['yellow_fingers']),
                int(data['anxiety']), int(data['peer_pressure']), int(data['chronic_disease']),
                int(data['fatigue']), int(data['allergy']), int(data['wheezing']), int(data['alcohol_consuming']),
                int(data['coughing']), int(data['shortness_of_breath']), int(data['swallowing_difficulty']),
                int(data['chest_pain'])]

    # Reshape and predict
    final_features = np.array(features).reshape(1, -1)
    prediction = classifier.predict(sc.transform(final_features))

    output = prediction[0]
    return jsonify(output)


if __name__ == "__main__":
    app.run(debug=True)
