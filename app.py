import streamlit as st
import pandas as pd
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

# Load the saved model
loaded_model = load_model('ensemble_model.h5')

# Load the labels.csv file
labels_df = pd.read_csv('Dataset/labels.csv')

# Function to preprocess the uploaded image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.  # Preprocess the image as done during training
    return img_array

# Function to make predictions
def predict_count(image_array):
    prediction = loaded_model.predict(image_array)
    return np.floor(prediction[0][0])  # Round down to the nearest integer

# Main Streamlit app
def main():
    st.title('Crowd Count Prediction')
    st.write('Upload an image and get the predicted and actual crowd count.')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Predicting...")

        # Preprocess the uploaded image
        img_array = preprocess_image(uploaded_file)

        # Make predictions
        prediction = predict_count(img_array)

        # Get the corresponding actual count
        image_name = uploaded_file.name.split('.')[0]
        actual_count = labels_df.loc[labels_df['id'] == int(image_name.split('_')[1]), 'count'].values[0]

        st.write(f'Predicted crowd count: {prediction:.0f}')  # Display predicted count as integer
        st.write(f'Actual crowd count: {actual_count}')


main()