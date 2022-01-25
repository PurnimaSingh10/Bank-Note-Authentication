# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 12:22:30 2022

@author: so
"""

from flask import Flask,request,render_template,redirect
import pandas as pd
import numpy as np
import pickle
import flasgger
from flasgger import Swagger

app=Flask(__name__)
pickle_in=open('classifier.pkl','rb')
classifier=pickle.load(pickle_in)


@app.route('/')
def welcome():
    return render_template('index.html')
 
   
@app.route('/predict',methods=['POST'])   
def predict():
    if request.method == 'POST':
     var=request.form['variance']
     skewn=request.form['skewness']
     curt=request.form['curtosis']
     entr=request.form['entropy']
     my_prediction=classifier.predict([[var,skewn,curt,entr]])
     return render_template('index.html', glon=my_prediction)
 




if __name__=='__main__':
        app.run()