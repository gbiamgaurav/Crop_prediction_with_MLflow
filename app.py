import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# Load the model
model = joblib.load("final_model/model.joblib")

# Load the preprocessor
preprocessor = joblib.load("final_model/preprocessor.joblib")

# Define a function to display the image as a banner
def display_banner_image(image_path):
    st.image(image_path, use_column_width=True, output_format="auto")

# Define a dictionary to map numeric labels to crop names, their respective plant emojis, and plant photos
crop_mapping = {
    1: ("rice", "🌾", "images/rice.jpeg"),
    2: ("maize", "🌽", "images/maize.jpeg"),
    3: ("jute", "🌿", "images/jute.jpeg"),
    4: ("cotton", "🌱", "images/cotton.jpeg"),
    5: ("coconut", "🥥", "images/coconut.jpeg"),
    6: ("papaya", "🌱", "images/papaya.jpeg"),
    7: ("orange", "🍊", "images/orange.jpeg"),
    8: ("apple", "🍎", "images/apple.jpeg"),
    9: ("muskmelon", "🍈", "images/muskmelon.jpeg"),
    10: ("watermelon", "🍉", "images/watermelon.jpeg"),
    11: ("grapes", "🍇", "images/grapes.jpeg"),
    12: ("mango", "🥭", "images/mango.jpeg"),
    13: ("banana", "🍌", "images/banana.jpeg"),
    14: ("pomegranate", "🍑", "images/pomegranate.jpeg"),
    15: ("lentil", "🌱", "images/lentil.jpeg"),
    16: ("blackgram", "🌱", "images/blackgram.jpeg"),
    17: ("mungbean", "🌱", "images/mungbean.jpeg"),
    18: ("mothbeans", "🌱", "images/mothbeans.jpeg"),
    19: ("pigeonpeas", "🌱", "images/pigeonpeas.jpeg"),
    20: ("kidneybean", "🌱", "images/kidneybean.jpeg"),
    21: ("chickpea", "🌱", "images/chickpea.jpeg"),
    22: ("coffee", "☕", "images/coffee.jpeg")
}

def main():

    st.title('Crop Prediction App')

    # Display the image at the top of the app
    display_banner_image("images/images.jpeg")

    st.markdown("Please Enter the inputs to check which crop or fruit to Grow")

    # Set default values for sliders or use None for initial values
    N = st.slider('N', 0, 140, step=1, value=None)
    P = st.slider('P', 5, 145, step=1, value=None)
    K = st.slider('K', 5, 205, step=1, value=None)
    temperature = st.slider('Temperature in degrees', 8.1, 44.9, step=0.01, value=None)
    humidity = st.slider('Humidity in farenhite', 14.1, 99.9, step=0.01, value=None)
    ph = st.slider('PH level', 3.5, 9.9, step=0.01, value=None)
    rainfall = st.slider('Rainfall in mm', 20.2, 298.23, step=0.01, value=None)

    # Define the form
    with st.form("Crop Prediction"):
        # Add submit button inside the form
        submit_button = st.form_submit_button("Submit")

    if submit_button:
        try:
            # Check if any of the sliders are None (user hasn't selected a value)
            if any(value is None for value in [N, P, K, temperature, humidity, ph, rainfall]):
                st.warning("Please select values for all input parameters.")
            else:
                # Create a dataframe
                input_data = pd.DataFrame({
                    "N": [N],
                    "P": [P],
                    "K": [K],
                    "temperature": [temperature],
                    "humidity": [humidity],
                    "ph": [ph],
                    "rainfall": [rainfall]
                })

                # Preprocess the input data
                X_transformed = preprocessor.transform(input_data)

                # Make the prediction
                prediction = model.predict(X_transformed)[0]

                # Map the numeric label to the crop name, plant emoji, and plant photo
                predicted_crop, plant_emoji, plant_photo = crop_mapping.get(
                    prediction, ("Unknown Crop", "🌱", "images/unknown.jpg")
                )

                # Display the plant photo, prediction, and plant emoji to the user
                st.image(plant_photo, use_column_width=True)
                st.write(f"You Should Grow {plant_emoji} {predicted_crop} on Your Farm")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
