from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global variables to hold the model and data
best_model = None
dataset = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    global dataset
    file = request.files['file']
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Load the dataset
    dataset = pd.read_csv(filepath)
    return jsonify({"message": "File uploaded successfully"}), 200

@app.route('/train', methods=['POST'])
def train():
    global best_model, dataset

    if dataset is None:
        return jsonify({"error": "No dataset uploaded"}), 400

    # Preprocessing
    dataset['Downtime_Flag'] = dataset['Downtime_Flag'].map({'Yes': 1, 'No': 0})
    X = dataset[['Temperature', 'Run_Time']]
    y = dataset['Downtime_Flag']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the model
    model = DecisionTreeClassifier(random_state=42, class_weight='balanced')
    param_grid = {
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    }

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Best model
    best_model = grid_search.best_estimator_

    # Evaluate the model
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    return jsonify({"accuracy": accuracy, "classification_report": report}), 200

@app.route('/predict', methods=['POST'])
def predict():
    global best_model

    if best_model is None:
        return jsonify({"error": "Model not trained"}), 400

    data = request.json
    temperature = data.get('Temperature')
    run_time = data.get('Run_Time')

    if temperature is None or run_time is None:
        return jsonify({"error": "Invalid input"}), 400

    input_data = pd.DataFrame({'Temperature': [temperature], 'Run_Time': [run_time]})
    prediction = best_model.predict(input_data)
    prediction_proba = best_model.predict_proba(input_data)

    downtime = 'Yes' if prediction[0] == 1 else 'No'
    confidence = prediction_proba[0][prediction[0]]

    return jsonify({"downtime": downtime, "confidence": confidence}), 200

if __name__ == '__main__':
    app.run(debug=True)
