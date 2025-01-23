# Manufacturing Downtime Prediction

**Website:** [Manufacturing Downtime Prediction](https://manufacturing-downtime-prediction.onrender.com/)

## Overview

This project is a Flask-based web application for predicting manufacturing downtime. It allows users to upload a dataset, train a machine learning model, and make predictions based on the trained model.

## Features

- Upload a CSV dataset for training.
- Train a Decision Tree model on the uploaded dataset.
- Make predictions based on the trained model.
- Download a sample dataset for testing.

## Setup Instructions

### Prerequisites

- Python 3.x
- Flask
- Pandas
- Scikit-learn
- Joblib
- Flask-CORS

### Installation

1. **Clone the Repository**

    ```sh
    git clone https://github.com/your-username/manufacturing-downtime-prediction.git
    cd manufacturing-downtime-prediction
    ```

2. **Create a Virtual Environment**

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install Dependencies**

    ```sh
    pip install -r requirements.txt
    ```

4. **Run the Application**

    ```sh
    python app.py
    ```

    The application will be available at `http://127.0.0.1:5000`.

### Deployment on Render

1. **Create a Render Account**

    Sign up for a Render account at [Render](https://render.com/).

2. **Create a New Web Service**

    - Go to the Render dashboard.
    - Click on "New" and select "Web Service".
    - Connect your GitHub repository.
    - Select the repository and branch containing your Flask application.
    - Set the build command to `pip install -r requirements.txt`.
    - Set the start command to `python app.py`.

3. **Deploy**

    Click on "Create Web Service" to deploy your application.

## Usage Instructions

1. **Access the Application**

    Open your web browser and navigate to the URL provided by Render (e.g., [Website link](https://manufacturing-downtime-prediction.onrender.com)).

2. **Download Sample Dataset**

    Click on the "Download Sample Dataset" button to download a sample CSV file for testing.

3. **Upload Training Data**

    - Click on the "Upload Training Data" button.
    - Select a CSV file from your local machine and click "Upload File".

4. **Train Prediction Model**

    - Click on the "Train Model" button to train the model using the uploaded dataset.

5. **Make Prediction**

    - Enter the temperature and run time values.
    - Click on the "Predict" button to get the prediction result.

## Architecture

For more details on the architecture of the application, please refer to the [architecture link](https://drive.google.com/file/d/1Kb4lxV7lxlFMn81gche2_-SOZTOx6Dvj/view?usp=sharing).

## Video Demonstration

Watch the video demonstration of the application [here](https://drive.google.com/file/d/1lbqWgnYwOw3A4L42knt8aipZsrEPTGXN/view?usp=sharing).

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Contact

For any questions or support, please contact patilshruti7273@gmail.com.
