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
st.write("""Yaharooo""")

# sidebar
# Sidebar - Specify parameter settings
with st.sidebar.header('2. Set Parameters'):
    # split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 50, 60, 70, 90, 80)
  # PARAMETERS min,max,default,skip
    st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80,5)

# model selection
options = st.selectbox("Model:",("ARIMA & LSTM","LSTM", "ARIMA"))

# st.write(options) //show option selected

# select time interval
interv = st.select_slider('Select Time Series Data Interval for Prediction', options=['Weekly', 'Monthly','Quarterly','Yearly'])

# st.write(interv[0])


# Function to convert number into string
# Switcher is dictionary data type here
def getInterval(argument):
    switcher = {
        "W": "1wk",
        "M": "1mo",
        "Q": "3mo",
        "Y": "1d"
    }
 
    # get() method of dictionary data type returns
    # value of passed argument if it is present
    # in dictionary otherwise second argument will
    # be assigned as default value of passed argument
    return switcher.get(argument, "1d")


# show raw data using button
st.write("Raw Data")
if st.button('Press to see Brent Crude Oil Raw Data'):
  df = yf.download('BZ=F', interval= getInterval(interv[0]))
  df
  
# model
  
