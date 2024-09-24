#Importing Modules
import math
import streamlit as st
import numpy as np
import csv as c
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import plotly.express as px
from streamlit_plotly_events import plotly_events
from bokeh.plotting import figure

#Defining Webpage
st.set_page_config(page_title="Power Factor", page_icon=":flag-pf:", layout="wide")
with st.container():
    st.subheader("Hi, We are :red[THUNDERS] :zap:")
    st.title(':red[Power Factor Corrector]')
    st.write("\nTo detect the Power factor from the line and Correcting the Power Factor\n\n")
#Defining Function

#Required Capacitance

def ReqCapacitance(ReactPower,volt):
    return(ReactPower/(math.pi*2*50*(volt*volt)))

#To save entered data into csv
 
def writecsv(pf,v,i,n192,n96,pw):
    data=list()
    data.append(pf)
    data.append(v)
    data.append(i)
    data.append(n192)
    data.append(n96)
    data.append(pw)
    #st.write(data)
    f="c:/Users/Nandhakishore/Desktop/Python/pf.csv"
    with open(f,'a') as csvfile: 
        csvwriter=c.writer(csvfile)
        csvwriter.writerow(data)


#Defining Power Factor

PFfixed=0.97

#data set to be added
f=pd.read_csv("c:/Users/Nandhakishore/Desktop/Python/pf.csv")

#Dictionary to store the value temp

#Getting Power from Supply Line

with st.form("my_form"):

    volt=st.number_input("Max Voltage in the Transmission Line : ")
    if volt:
        try:
            volt=float(volt)
        except:
            st.write("Enter the valid Voltage")
    amp=st.number_input("Max Current in the Transmission Line : ")
    if amp:
        try:
            amp=float(amp)
        except:
            st.write("Enter the valid Current")
    submit_button = st.form_submit_button("Submit")


if submit_button:

    #Finding Phase Angle

    # Loading dataset
    data = pd.read_csv('c:/Users/Nandhakishore/Desktop/Python/pf.csv')

    #Assuming the Value of X and Y
    X = data.drop(['pf','n192','n96'],axis=1)
    y = data['pf']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=22)

    # Create a linear regression model
    model = LinearRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    #st.write(mse,r2)
    #Calculating Power
    power=volt*amp

    #Predicting Power Factor
    sam=[[volt,amp,power]]
    PFline=model.predict(sam)+mse

    PhAngfixed=(math.acos(PFfixed))
    PhAngline=(math.acos(PFline))

#Finding Required Reactive Power

    ReactPower=(volt*amp*(math.tan(PhAngline)-math.tan(PhAngfixed)))
    #print(ReactPower)


#Required Capacitance to Maintain Power Factor

    ReqCap=ReqCapacitance(ReactPower,volt)
    #print('\n',ReqCap)

#Finding the Number of Capacitance to be used

    count192,count96=0,0
    if PFline<0.97:
        count192=ReactPower//1920
        ReactPower=ReactPower%1920
        count96=ReactPower//960
        ReactPower=ReactPower%960
        if ReactPower!=0:
            count96+=1
    writecsv(PFline[0],volt,amp,count192,count96,power)


#Sending Output to Capacitive Switch
    st.subheader("**:orange[PREDICTED] POWER FACTOR = :orange[{}]**".format(PFline[0]))
    st.subheader("**To Achieve :red[POWER FACTOR] = :red[0.97]**")
    st.write("\n\nNumber of 1.92kVAR capacitance to be Turned ON : :red[**{}**]".format(count192))
    st.write("\n\nNumber of 0.96kVAR capacitance to be Turned ON : :red[**{}**]".format(count96))
    #st.dataframe(f)
    #st.scatter_chart(data, x='pw', y='pf', x_label="POWER", y_label="POWER FACTOR", color=None, stack=None, width=None, height=None, use_container_width=True)
    st.header("***:orange[GRAPHICAL REPRESENTATION]***")
    st.subheader("**:red[INPUT GRAPH]**")
    st.write("Power VS Power Factor")
    st.scatter_chart(data, x='pw', y='pf', x_label="POWER", y_label="POWER FACTOR", color='#FF0000', size=None, width=None, height=None, use_container_width=True)
    #with st.expander('Plot'):
        #fig = px.line(x=[power], y=[PFline[0]])
        #selected_points = plotly_events(fig)

    #selected_points = plotly_events(fig, click_event=False, hover_event=True)
    st.subheader("**:red[OUTPUT GRAPH]**")
    st.write("Capacitance VS Power Factor")
    a=[0.0,ReqCap]
    b=[PFline[0],0.97]
    p = figure(title="OUTPUT", x_axis_label="CAPACITOR IN F", y_axis_label="POWER FACTOR")
    p.line(a, b, legend_label="OUTCOME", line_width=2)
    st.bokeh_chart(p, use_container_width=True)
    bt=st.button("**CLOSE**:fire:")
    if bt:
        pass