# AI-Powered Disease Prediction System

This is a Flask-based web application that uses machine learning to predict the risk of several common diseases based on user-inputted health data. It provides a user-friendly interface for patients to assess their risk for Diabetes, Heart Disease, Liver Disease, Kidney Disease, and Stroke.

## Features

-   **Multi-Disease Prediction**: Supports prediction for 5 different diseases.
-   **User-Friendly Forms**: Dynamically renders specific input forms for each disease.
-   **Personalized Recommendations**: Generates tailored lifestyle, diet, and medical advice based on the predicted risk level.
-   **Risk Visualization**: Displays a risk probability gauge chart for easy interpretation of results.
-   **PDF & CSV Reports**: Allows users to download a detailed PDF report of their prediction and a CSV file of their input data.
-   **Single-File Application**: The entire application is contained within a single `app.py` file for simplicity.
-   **Downloadable Source Code**: Users can download the `app.py` source file directly from the web interface.

## Project Structure

The application is self-contained in a single Python file:

-   `app.py`: The main Flask application file. It contains:
    -   Flask setup and routing.
    -   Machine learning model training (using dummy data).
    -   HTML templates embedded as strings.
    -   Prediction logic.
    -   PDF and CSV report generation.
-   `requirements.txt`: A file listing all the Python dependencies required to run the application.
-   `reports/`: A directory that is automatically created to store the generated PDF and CSV files.

## Setup and Installation

To run this application locally, follow these steps:

1.  **Clone the repository or download the source code.**

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**
    -   On Windows:
        ```bash
        venv\Scripts\activate
        ```
    -   On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```

4.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run the Application

1.  Make sure you have completed the setup steps above.

2.  Run the Flask application:
    ```bash
    python app.py
    ```

3.  Open your web browser and navigate to:
    ```
    http://127.0.0.1:5000
    ```

You should now see the home page of the AI Disease Prediction System.

## How It Works

1.  **Select a Disease**: From the home page, choose one of the diseases you want to get a prediction for.
2.  **Enter Data**: Fill in the required health parameters in the form.
3.  **AI Analysis**: The machine learning model analyzes the data you provided.
4.  **Get Results**: The application displays the prediction result (Positive/Negative), risk probability, and personalized recommendations.
5.  **Download Report**: You can download the results as a PDF report or a CSV file.

## Disclaimer

This tool is for educational and informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
