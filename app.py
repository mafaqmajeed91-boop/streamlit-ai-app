
import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
# --- Load Models ---
# Load the Logistic Regression model
log_reg_model = joblib.load('model.pkl')
# Load the Random Forest model
rf_model = joblib.load('random_forest_model.pkl')

# Define a dictionary of models
models = {
    'Logistic Regression': log_reg_model,
    'Random Forest Classifier': rf_model
}

# Define the class names for the Iris dataset
iris_species = ['Setosa', 'Versicolor', 'Virginica']

st.set_page_config(layout="wide") # Set page width to wide

st.title('🌸 Iris Species Prediction Dashboard')
st.markdown("Welcome to the AI Model Deployment Dashboard! Use this tool to predict Iris species and explore model performance.")

# --- Model Selection ---
st.sidebar.header('Model Configuration')
selected_model_name = st.sidebar.selectbox('Choose a Machine Learning Model:', list(models.keys()))

model = models[selected_model_name]
st.sidebar.info(f"Current model selected: **{selected_model_name}**")

# --- Load Iris Data for accuracy and exploration ---
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# --- Model Performance Section ---
st.header('📊 Model Performance')
st.write("Here you can see the performance metrics of the selected model.")

# Re-split the data to calculate accuracy on a test set within the app
X_app_train, X_app_test, y_app_train, y_app_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_accuracy = model.score(X_app_test, y_app_test)
st.metric(label=f"Accuracy ({selected_model_name})", value=f"{model_accuracy:.2f}")

# --- Input Fields for Prediction ---
st.header('🎯 Predict Iris Species')
st.markdown("Adjust the sliders below to input the characteristics of an Iris flower and get a real-time prediction.")

feature_names = data.feature_names

# Create input fields dynamically based on feature names
input_values = {}
col1, col2, col3, col4 = st.columns(4)

with col1:
    input_values[feature_names[0]] = st.slider(f'{feature_names[0].replace("_", " ").title()} (cm)', 4.0, 8.0, 5.4, help="Length of the sepal in cm")
with col2:
    input_values[feature_names[1]] = st.slider(f'{feature_names[1].replace("_", " ").title()} (cm)', 2.0, 4.5, 3.4, help="Width of the sepal in cm")
with col3:
    input_values[feature_names[2]] = st.slider(f'{feature_names[2].replace("_", " ").title()} (cm)', 1.0, 7.0, 1.3, help="Length of the petal in cm")
with col4:
    input_values[feature_names[3]] = st.slider(f'{feature_names[3].replace("_", " ").title()} (cm)', 0.1, 2.5, 0.2, help="Width of the petal in cm")

# Prepare the input features for prediction
features = np.array([[input_values[f] for f in feature_names]])

if st.button('Predict Species', help="Click to get the prediction for the entered values."):
    prediction = model.predict(features)
    prediction_proba = model.predict_proba(features)

    st.subheader('Prediction Result:')
    predicted_species = iris_species[prediction[0]]
    st.success(f'The predicted Iris species is: **{predicted_species}**')

    st.subheader('Prediction Probability:')
    for i, species in enumerate(iris_species):
        st.write(f'**{species}**: {prediction_proba[0][i]*100:.2f}%')

# --- Batch Prediction Section ---
st.header('⬆️ Batch Prediction from CSV')
st.markdown("Upload a CSV file with Iris features to get predictions for multiple entries.")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    batch_df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data Preview:")
    st.dataframe(batch_df.head())

    # Ensure the uploaded CSV has the correct feature columns
    if all(f in batch_df.columns for f in feature_names):
        with st.spinner('Making batch predictions...'):
            # Make predictions
            batch_predictions = model.predict(batch_df[feature_names])
            batch_prediction_proba = model.predict_proba(batch_df[feature_names])

            # Add predictions and probabilities to the DataFrame
            batch_df['Predicted_Species'] = [iris_species[p] for p in batch_predictions]
            for i, species in enumerate(iris_species):
                batch_df[f'Probability_{species}'] = batch_prediction_proba[:, i]

            st.subheader('Batch Predictions Results:')
            st.dataframe(batch_df)
        st.success("Batch predictions completed successfully!")
    else:
        st.error("Uploaded CSV file must contain the following columns for prediction: " + ", ".join(feature_names))
        st.warning("Please ensure your CSV column names match: " + ", ".join([f'**{f}**' for f in feature_names]))

# --- Data Exploration Section ---
st.header('🔍 Data Exploration')
st.markdown("Visualize the distribution of features and relationships within the Iris dataset.")

with st.expander("View Feature Distributions"):
    st.subheader('Feature Distributions')
    # Plot distributions of each feature
    for i, feature in enumerate(feature_names):
        fig, ax = plt.subplots()
        sns.histplot(X[feature], kde=True, ax=ax)
        ax.set_title(f'Distribution of {feature.replace("_", " ").title()}')
        st.pyplot(fig)
        plt.close(fig)

with st.expander("View Pair Plot"):
    st.subheader('Pair Plot of Features')
    # Create a pair plot to visualize relationships between features
    fig_pair = sns.pairplot(X.assign(Species=y.map({0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'})), hue='Species')
    st.pyplot(fig_pair)
    plt.close(fig_pair)

st.sidebar.markdown("--- Developed by your friendly AI assistant --- ")
