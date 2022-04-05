import streamlit as st
import pandas as pd
import yfinance as yf
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, 

# page expands to full width
st.set_page_config(page_title="LSTM vs ARIMA", layout='wide')

# PAGE LAYOUT
# heading
st.title("Brent Crude Oil Benchmark")
st.write("""© Castillon, Ignas, Wong""")

# sidebar
# Sidebar - Specify parameter settings
with st.sidebar.header('Set Data Split'):
  # PARAMETERS min,max,default,skip
    st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80,5)

# model selection
options = st.selectbox("Model:",("ARIMA & LSTM","LSTM", "ARIMA"))

# st.write(options) //show option selected

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
  
