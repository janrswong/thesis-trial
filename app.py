import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
# from statsmodels.tsa.arima_model import ARIMA
# from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error,mean_absolute_error
import numpy as np
# import plotly.graph_objects as go
import plotly.express as px
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, 

# page expands to full width
st.set_page_config(page_title="LSTM vs ARIMA", layout='wide')

# PAGE LAYOUT
# heading
st.title("Crude Oil Benchmark Stock Price Prediction LSTM and ARIMA Models")
st.write("""© Castillon, Ignas, Wong""")

# sidebar
# Sidebar - Specify parameter settings
with st.sidebar.header('Set Data Split'):
  # PARAMETERS min,max,default,skip
    st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80,5)

# model selection
modSelect = st.selectbox("Select Model for Prediction:",("ARIMA & LSTM","LSTM", "ARIMA"))

# //show option selected
# st.write(modSelect) 

# select time interval
interv = st.select_slider('Select Time Series Data Interval for Prediction', options=['Weekly', 'Monthly','Quarterly','Yearly'])

# st.write(interv[0])

# Function to convert time series to interval
def getInterval(argument):
    switcher = {
        "W": "1wk",
        "M": "1mo",
        "Q": "3mo",
        "Y": "1d"
    }
    return switcher.get(argument, "1d")


# show raw data
st.write("Raw Data")
# using button
# if st.button('Press to see Brent Crude Oil Raw Data'):
df = yf.download('BZ=F', interval= getInterval(interv[0]))
df
  
# model

# graph visualization
st.write("Visualizations")

# TODO: find better line graphs for visualization
st.line_chart(data=df['Close'], width=0, height=0, use_container_width=True,)


# or plot the time series 
fig = px.line(df, x=df.index, y=["Close","Open"], 
    title="BRENT CRUDE OIL PRICES", width=1000)
st.plotly_chart(fig, use_container_width=True)

# predicted data
st.write("Predicted Data")
# accuracy metrics
st.write("Accuracy Metrics")