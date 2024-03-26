import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt

def load_model(modelfile):
	loaded_model = pickle.load(open(modelfile, 'rb'))
	return loaded_model


st.title("Petrol Price Forecasting")
data = st.file_uploader("Upload an File", type=["csv"])

if st.button('Predict'):

    # read dataset
    df = pd.read_csv(data)
    train_df = pd.read_csv('training_data.csv')

    df = df.dropna()
    # Convert 'Date' column to datetime format
    df['Date'] = pd.to_datetime(df['Date'])
    train_df['Date'] = pd.to_datetime(train_df['Date'])

    # Feature Engineering: Extract additional features from the 'Date' column
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day

    # loading Model
    loaded_model = pickle.load(open('model.pkl', 'rb'))
    # Results DataFrame
    result = loaded_model.predict(df[['Year','Month','Day']])

    df['Prediction'] = result

    # Create a Matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the petrol prices over time from train_df
    ax.plot(train_df['Date'], train_df['Petrol (USD)'], label='Actual Petrol Price')

    # Plot the predicted petrol prices over time from df
    ax.plot(df['Date'], df['Prediction'], label='Predicted Petrol Price')

    # Set title and labels
    ax.set_title('Petrol Prices Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Petrol Price (USD)')
    ax.tick_params(axis='x', rotation=45)

    # Add legend
    ax.legend()

    # Show plot in Streamlit
    st.pyplot(fig)
