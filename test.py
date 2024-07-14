import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import mysql.connector
from dotenv import load_dotenv
import os
from imblearn.over_sampling import SMOTE

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
print("Class distribution in original dataset:\n", dataset['LUNG_CANCER'].value_counts())

X = dataset.iloc[:, :15].values
y = dataset.iloc[:, -1].values

# Balance the dataset
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

# Print class distribution after SMOTE
print("Class distribution after SMOTE:\n", pd.Series(y_res).value_counts())

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.25, random_state=0)

# Standardize the features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Initialize and train the SVC model
classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(X_train, y_train)

# Evaluate model performance
train_score = classifier.score(X_train, y_train)
test_score = classifier.score(X_test, y_test)
print(f"Training accuracy after balancing: {train_score}")
print(f"Test accuracy after balancing: {test_score}")

# Inspect the predictions on the test set
y_pred = classifier.predict(X_test)
print("Test set predictions:\n", y_pred)
print("Actual test set labels:\n", y_test)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract form data and print each field
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

    print(f"Gender: {gender}")
    print(f"Age: {age}")
    print(f"Smoking: {smoking}")
    print(f"Yellow Fingers: {yellow_fingers}")
    print(f"Anxiety: {anxiety}")
    print(f"Peer Pressure: {peer_pressure}")
    print(f"Chronic Disease: {chronic_disease}")
    print(f"Fatigue: {fatigue}")
    print(f"Allergy: {allergy}")
    print(f"Wheezing: {wheezing}")
    print(f"Alcohol Consuming: {alcohol_consuming}")
    print(f"Coughing: {coughing}")
    print(f"Shortness of Breath: {shortness_of_breath}")
    print(f"Swallowing Difficulty: {swallowing_difficulty}")
    print(f"Chest Pain: {chest_pain}")

    # Map gender to numerical values if needed
    gender_numeric = 1 if gender == 'M' else 0  # Assuming Male (M) is 1, Female (F) is 0

    # Create feature array
    features = [gender_numeric, age, smoking, yellow_fingers, anxiety, peer_pressure,
                chronic_disease, fatigue, allergy, wheezing, alcohol_consuming,
                coughing, shortness_of_breath, swallowing_difficulty, chest_pain]

    # Reshape and predict
    final_features = np.array(features).reshape(1, -1)
    prediction = classifier.predict(sc.transform(final_features))

    print(f"Features: {final_features}")
    print(f"Prediction: {prediction}")

    output = prediction[0]
    prediction_text = 'YES' if output == 1 else 'NO'

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
