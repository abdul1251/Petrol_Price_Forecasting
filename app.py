import streamlit as st
import numpy as np
import pickle
import pandas as pd

def load_model(modelfile):
	loaded_model = pickle.load(open(modelfile, 'rb'))
	return loaded_model


st.title("Predicting Whether a user will Bounce or Not")
data = st.file_uploader("Upload an File", type=["csv"])

if st.button('Predict'):

    # read dataset
    df = pd.read_csv(data)

    df = df.dropna()
    # Convert 'Date' column to datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    # Feature Engineering: Extract additional features from the 'Date' column
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df = df.drop("Date", axis=1)
    st.dataframe(df)

    # loading Model
    loaded_model = load_model('model.pkl')


    result = loaded_model.predict(df)

    df['User_Bounce'] = result
    st.dataframe(df)
