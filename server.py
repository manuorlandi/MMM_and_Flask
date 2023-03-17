from flask import Flask, jsonify, request, render_template
import plotly
import plotly.express as px
import pandas as pd
import pickle as pkl
import os
from model_fit import train_me
from plotting import plot_me
import json
app = Flask(__name__)

# api esposta con path /predict, accedibile con metodo POST
# il codice in do_prediction è quello che viene eseguito in caso di chiamate REST a 'http://localhost:5000/predict'
@app.route("/train", methods=['GET'])
def train():
    train_me()
    return "done"

@app.route("/plotme", methods=['GET'])
def plotme():

   graphJSON = plot_me()
   return render_template('plotme.html', graphJSON=graphJSON)


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)


@app.route('/results')
def results():
   
   # Students data available in a list of list
    students = [['Akash', 34, 'Sydney', 'Australia'],
                ['Rithika', 30, 'Coimbatore', 'India'],
                ['Priya', 31, 'Coimbatore', 'India'],
                ['Sandy', 32, 'Tokyo', 'Japan'],
                ['Praneeth', 16, 'New York', 'US'],
                ['Praveen', 17, 'Toronto', 'Canada']]
     
    # Convert list to dataframe and assign column values
    df = pd.DataFrame(students,
                      columns=['Name', 'Age', 'City', 'Country'],
                      index=['a', 'b', 'c', 'd', 'e', 'f'])
     
    # Create Bar chart
    fig = px.bar(df, x='Name', y='Age', color='City', barmode='group')
     
    # Create graphJSON
    graphJSON = plot_me()
     
    # Use render_template to pass graphJSON to html
    return render_template('template.html', graphJSON=graphJSON)
 
