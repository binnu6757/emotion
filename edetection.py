import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# Display title
st.image(r"C:\Users\reddy\OneDrive\Desktop\new logo.jpg")  # Replace with your actual PNG image file path


# Display the PNG image
st.title("Emotional Analysis Distribution ")
st.write('This app performs emotional analysis Distribution.')


try:
    model = pickle.load(open(r"C:\Users\reddy\machine learning\ss.pkl",'rb'))
    bow = pickle.load(open(r"C:\Users\reddy\machine learning\bow1.pkl",'rb'))
except Exception as e:
    st.error(f"An error occurred: {e}")



# Text input for user statement
user_input = st.text_area("Enter a emotional analysis statement:")

if user_input:
    # Transform the user input using the loaded CountVectorizer
    user_input_transformed = bow.transform([user_input])
    
    # Predict the sentiment using the loaded model
    prediction = model.predict(user_input_transformed)[0]
    

if st.button("Submit"):
    if prediction == 0:
        st.write("Sad")
        st.image(r"C:\Users\reddy\OneDrive\Desktop\sad.jpeg")
    elif prediction == 1:
        st.write("Joy")
        st.image(r"C:\Users\reddy\OneDrive\Desktop\joy.jpeg")
    elif prediction == 2:
        st.write("Love")
        st.image(r"C:\Users\reddy\OneDrive\Desktop\love.jpeg")
    elif prediction == 3:
        st.write("Anger")
        st.image(r"C:\Users\reddy\OneDrive\Desktop\anger.png")
    elif prediction == 4:
        st.write("Fear")
        st.image(r"C:\Users\reddy\OneDrive\Desktop\fear.jpeg")
    elif prediction == 5:
        st.write("Surprise")
        st.image(r"C:\Users\reddy\OneDrive\Desktop\surprised.png")
