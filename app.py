import pandas as pd
from flask import Flask, render_template, request
import pickle
import base64
from cProfile import label
from io import BytesIO
from turtle import color
import pickle
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from matplotlib.pyplot import axes
from matplotlib.widgets import Button
import numpy as np

app = Flask(__name__)
data = pd.read_csv('Cleaned_data.csv')
pipe = pickle.load(open("RidgeModel.pkl", 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/house')
def index():

    locations = sorted(data['location'].unique())
    return render_template('house.html', locations=locations)


@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk = request.form.get('bhk')
    bath = request.form.get('bath')
    sqft = request.form.get('total_sqft')

    print(location, bhk, bath, sqft)
    input = pd.DataFrame([[location, sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])
    prediction = pipe.predict(input)[0]
    if int(sqft)>=1000 and int(sqft)<=50000:    
        if prediction > 100:
            prediction = round(prediction/100, 2)
            prediction = str(prediction) + ' Crore'
            return str(prediction)
        else:
            prediction = str(prediction) + ' Lakhs'
            return str(prediction)
    else:
        return "Value must be present between 1000sqft and 50000sqft"
    return str(prediction)


@app.route('/getData')
def getData():
    arr = np.genfromtxt("data1.csv", delimiter=",") 
    val=-10
    y=[]
    for i in range(10):
        y.append(arr[val][1])
        val+=1
    return arr.tostring()

@app.route('/gold')
def hello():
    # Generate the figure **without using pyplot**.
    arr = np.genfromtxt("data1.csv", delimiter=",")
    val=-10
    x=[1,2,3,4,5,6,7,8,9,10]
    y=[]
    for i in range(10):
        y.append(arr[val][1])
        val+=1
    fig = plt.figure()
    #ax=plt.plot([168.69909961, 82.06289995 ,116.13640032 ,127.68450062 ,120.66540098],color='green',label='Predicted Value')
    plt.plot(x,y)
    plt.xticks(x)
    plt.title('Predicted Price')
    plt.xlabel('Number of days')
    plt.ylabel('GLD Price')


    # Save it to a temporary buffer.a
    buf = BytesIO()
    fig.savefig('./static/plot.png', format="png")
    return render_template('gold.html',data=y)
    # Embed the result in the html output.
    #data = base64.b64encode(buf.getbuffer()).decode("ascii")
    #return "<center><body bgcolor='pink'><b><h1 align='center' color='white'>Gold Prediction</h1></b>\n"+f"\n<img src='data:image/png;base64,{data}'/></body><br><br>"+"\n<a href='http://127.0.0.1:5000/house'>Back</a></center>"








