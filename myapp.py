import numpy as np
from flask import Flask, request, render_template
app = Flask(__name__)

import pandas as pd
import warnings
from sklearn import linear_model

df=pd.read_csv('salary.csv')

reg=linear_model.LinearRegression()
reg.fit(df[['YearsExperience','mcq','ftof']],df.Salary)

@app.route("/")
def hello():
    return render_template('echo.html')

@app.route("/", methods=['POST'])
def echo(): 

    value1=request.form['Exp']
    value2=request.form['Mcq']
    value3=request.form['Ftof']

    test=reg.predict([[int(value1),int(value2),int(value3)]])

    return 'Expected Salary would be : {}'.format(test)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
