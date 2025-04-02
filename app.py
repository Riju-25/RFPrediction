import streamlit as st
import pickle
import pandas as pd

# Load the trained model
with open("rainfall_prediction_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)  # Directly load the model

# Ensure the model is correctly loaded
if model is None or not hasattr(model, "predict"):
    raise ValueError("Loaded object is not a valid model. Ensure the .pkl file is correctly saved.")

# Set Streamlit page config
st.set_page_config(page_title="Rainfall Prediction App", page_icon="☔️", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Predict Rainfall", "About Model"])

# Home Page
if page == "Home":
    st.title("⛆️ Welcome to the Rainfall Prediction App")
    st.write(
        "This application uses a **Random Forest Classifier** model to predict rainfall "
        "based on meteorological parameters like pressure, humidity, cloud cover, wind conditions, and more."
    )
    st.image("https://source.unsplash.com/800x400/?rain,clouds", use_container_width=True)
    st.markdown("---")
    st.subheader("How It Works?")
    st.write("Simply navigate to the **Predict Rainfall** section, enter the required parameters, and get an instant rainfall prediction.")

# Rainfall Prediction Page
elif page == "Predict Rainfall":
    st.title("☁️ Rainfall Prediction")
    st.write("Enter the following weather parameters to predict rainfall:")
    
    # User input fields
    col1, col2 = st.columns(2)
    with col1:
        pressure = st.number_input("Pressure (hPa)", value=1013.25)
        dewpoint = st.number_input("Dew Point (°C)", value=10.0)
        humidity = st.number_input("Humidity (%)", value=70.0)
        cloud = st.number_input("Cloud Cover (%)", value=50.0)
    with col2:
        sunshine = st.number_input("Sunshine Duration (hours)", value=5.0)
        winddirection = st.number_input("Wind Direction (degrees)", value=180.0)
        windspeed = st.number_input("Wind Speed (km/h)", value=10.0)
    
    # Predict button
    if st.button("☔️ Predict Rainfall", key="predict"):
        # Prepare input data as a pandas DataFrame
        input_df = pd.DataFrame([[pressure, dewpoint, humidity, cloud, sunshine, winddirection, windspeed]],
                                columns=['pressure', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed'])
        
        # Make prediction
        prediction = model.predict(input_df)
        
        # Display result
        if prediction[0] > 0:
            st.success(f"☔️ Expected Rainfall: {prediction[0]:.2f} mm")
        else:
            st.info("☀️ No significant rainfall expected.")

# About Model Page
elif page == "About Model":
    st.title("🤖 About the Random Forest Model")
    st.write(
        "The Random Forest Classifier used in this app is an ensemble learning method "
        "that builds multiple decision trees and combines their results for a more accurate and robust prediction."
    )
    
    st.subheader("🔍 How Random Forest Works?")
    st.write("- Creates multiple decision trees from random subsets of data.\n"
             "- Each tree votes on the outcome, and the most common prediction is selected.\n"
             "- Helps in reducing overfitting and improves accuracy.")
    
    st.subheader("📊 Model Parameters")
    st.code(
        """
        RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )
        """,
        language="python"
    )
    
    st.subheader("✨ Advantages of Random Forest")
    st.write("✔️ Handles large datasets efficiently\n"
             "✔️ Works well with missing data and noisy inputs\n"
             "✔️ Provides feature importance insights")
    
    st.image("https://source.unsplash.com/800x400/?forest,trees", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.write("Developed with ❤️ using Streamlit")
