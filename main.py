#Import Libraries 
import pandas as pd
import numpy as np
import streamlit as st
from datetime import date
import pandas as pd
import numpy as np
import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

#Specify dates
START = "2018-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock and Cypto Prediction')

stocks_and_crypto = ('AMZN', 'AAPL', 'GOOG', 'BTC-CAD', 'ETH-CAD','LTC-CAD')
selected_stock = st.selectbox('Select the Stock/Cyrpto for prediction', stocks_and_crypto)

n_years = st.slider('Select the number of year/years for price prediction?', 1, 3)
period = n_years * 365


@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

	
data_load_state = st.text('Loading data...')
data = load_data(selected_stock)

st.subheader('Raw data')
st.write(data.tail())


# Predict forecast with Prophet.
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.tail())
    
st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)
