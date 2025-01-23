import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv('manufacturing_data.csv')

# Preprocessing: Convert Downtime_Flag to numeric (Yes=1, No=0)
df['Downtime_Flag'] = df['Downtime_Flag'].map({'Yes': 1, 'No': 0})

# Feature columns and target variable
X = df[['Temperature', 'Run_Time']]
y = df['Downtime_Flag']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Decision Tree model
model = DecisionTreeClassifier(random_state=42, class_weight='balanced')

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best model from grid search
best_model = grid_search.best_estimator_

# Make predictions on the test set
y_pred = best_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(report)

# Function to predict Downtime based on Temperature and Run_Time
def predict_downtime(temperature, run_time):
    # Prepare the input data as a DataFrame
    input_data = pd.DataFrame({'Temperature': [temperature], 'Run_Time': [run_time]})
    
    # Use the trained model to predict the downtime
    prediction = best_model.predict(input_data)
    prediction_proba = best_model.predict_proba(input_data)
    
    # Get the probability of the predicted class (Downtime: Yes or No)
    confidence = prediction_proba[0][prediction[0]]  # Confidence for the predicted class
    
    # Output the result
    downtime = 'Yes' if prediction[0] == 1 else 'No'
    print(f"Downtime: {downtime}, Confidence: {confidence:.2f}")

# Test the function with an example input (you can change the values here)
temperature_input = float(input("Enter Temperature: "))
run_time_input = float(input("Enter Run Time: "))
predict_downtime(temperature_input, run_time_input)
