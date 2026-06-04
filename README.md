# Iris Species Prediction Dashboard

This project provides an interactive Streamlit dashboard for predicting Iris species based on their sepal and petal measurements. It allows users to select between different machine learning models, perform real-time predictions, upload CSV files for batch predictions, and explore the Iris dataset through various visualizations.

## Features

-   **Multiple ML Model Selection**: Choose between Logistic Regression and Random Forest Classifier for predictions.
-   **Real-time Prediction**: Input Iris features using sliders and get instant species predictions with probability scores.
-   **Batch Prediction**: Upload a CSV file containing multiple Iris entries to get predictions for all of them.
-   **Model Accuracy Display**: View the accuracy of the selected model on a test set.
-   **Data Exploration**: Visualize feature distributions (histograms) and relationships between features (pair plots) of the Iris dataset.
-   **Styled UI Components**: Enhanced user interface using Streamlit's styling capabilities.
-   **ngrok Deployment**: Easily deploy and access the Streamlit application from anywhere using ngrok.

## Setup and Installation

1.  **Clone the Repository (if applicable) or Open in Google Colab**:

    If running in Colab, ensure you have the notebook loaded.

2.  **Install Dependencies**:

    The project uses `pip` for package management. Install all required libraries using the `requirements.txt` file:

    ```bash
    pip install -r requirements.txt
    ```

    *(In Google Colab, these are typically installed via initial cells.)*

3.  **ngrok Authtoken**:

    To deploy the Streamlit app using `ngrok`, you'll need an ngrok authtoken. 

    *   Sign up for a free account at [ngrok.com](https://ngrok.com/).
    *   Obtain your authtoken from your ngrok dashboard.
    *   In the Colab notebook, locate the cell where `ngrok.connect` is called (or where `conf.get_default().auth_token` is set) and replace `"YOUR_NGROK_AUTHTOKEN"` with your actual token.

## How to Run the Application

1.  **Execute Colab Cells Sequentially**:

    Run all the cells in the Google Colab notebook from top to bottom.

    *   This will install necessary libraries.
    *   Train and save the Logistic Regression and Random Forest models (`model.pkl` and `random_forest_model.pkl`).
    *   Create the `app.py` file containing the Streamlit application code.

2.  **Launch Streamlit with ngrok**:

    Locate and execute the cell that starts the Streamlit app and the ngrok tunnel. This cell will typically look like this:

    ```python
    # ... ngrok and streamlit commands ...
    public_url = ngrok.connect(addr="8501", proto="http")
    print(f"Streamlit App URL: {public_url}")
    ```

3.  **Access the Dashboard**:

    Once the cell finishes execution, a public URL will be printed (e.g., `https://xxxxxx.ngrok-free.dev`). Click this URL to open the Streamlit dashboard in your web browser.

## How to Stop the Application

To stop the Streamlit app and the ngrok tunnel, go to the Colab notebook and interrupt the kernel. You can do this by going to `Runtime -> Interrupt execution` in the Colab menu.
