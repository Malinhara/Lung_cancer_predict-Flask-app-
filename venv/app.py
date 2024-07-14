import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import mysql.connector
from dotenv import load_dotenv
import os

app = Flask(__name__)
load_dotenv()

# Get MySQL connection details from environment variables
db_config = {
    'host': os.getenv('DB_HOST'),
    'user': os.getenv('DB_USER'),
    'port': os.getenv('DB_PORT'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_NAME')
}

# Connect to the MySQL database
try:
    connection = mysql.connector.connect(**db_config)
    if connection.is_connected():
        print("Successfully connected to the database")
except mysql.connector.Error as err:
    print(f"Error: {err}")

# Read dataset from MySQL into a Pandas DataFrame
try:
    dataset_query = "SELECT * FROM survey_lung_cancer"
    dataset = pd.read_sql(dataset_query, con=connection)
except mysql.connector.Error as err:
    print(f"Error reading dataset from MySQL: {err}")
    dataset = pd.DataFrame()  # Empty DataFrame if there's an error

# Data preprocessing
dataset.columns = dataset.columns.str.strip()
le_gender = LabelEncoder()
le_lung_cancer = LabelEncoder()

dataset['GENDER'] = le_gender.fit_transform(dataset['GENDER'])
dataset['LUNG_CANCER'] = le_lung_cancer.fit_transform(dataset['LUNG_CANCER'])

# Print dataset after preprocessing
print("Dataset after preprocessing:\n", dataset)

# Check class balance
print("Class distribution in dataset:\n", dataset['LUNG_CANCER'].value_counts())

X = dataset.iloc[:, :15].values
y = dataset.iloc[:, -1].values

# Apply SMOTE to balance the dataset
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

# Split the resampled dataset
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.25, random_state=0)

# Standardize the features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Initialize and train the Random Forest model with class weights
classifier = RandomForestClassifier(class_weight='balanced', random_state=0)
classifier.fit(X_train, y_train)

# Evaluate model performance
train_score = classifier.score(X_train, y_train)
test_score = classifier.score(X_test, y_test)
y_pred = classifier.predict(X_test)

print(f"Training accuracy: {train_score}")
print(f"Test accuracy: {test_score}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

def predict_lung_cancer(model, scaler, features):
    features = np.array(features).reshape(1, -1)
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)
    return prediction[0]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract form data and preprocess
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

    gender_numeric = 1 if gender == 'M' else 0
    features = [gender_numeric, age, smoking, yellow_fingers, anxiety, peer_pressure,
                chronic_disease, fatigue, allergy, wheezing, alcohol_consuming,
                coughing, shortness_of_breath, swallowing_difficulty, chest_pain]

    output = predict_lung_cancer(classifier,sc,features)
    prediction_text = 'YES' if output == 1 else 'NO'

    return render_template('index.html', prediction_text='Lung Cancer Prediction: {}'.format(prediction_text))

@app.route('/results', methods=['POST'])
def results():
    data = request.get_json(force=True)
    features = [int(data['gender']), int(data['age']), int(data['smoking']), int(data['yellow_fingers']),
                int(data['anxiety']), int(data['peer_pressure']), int(data['chronic_disease']),
                int(data['fatigue']), int(data['allergy']), int(data['wheezing']), int(data['alcohol_consuming']),
                int(data['coughing']), int(data['shortness_of_breath']), int(data['swallowing_difficulty']),
                int(data['chest_pain'])]

    output = predict_lung_cancer(classifier, sc, features)
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
