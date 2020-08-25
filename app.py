from flask import Flask, request, jsonify, render_template
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

import numpy as np
from PIL import Image
import requests, os, glob, time
from fastai import *
from fastai.vision import *
import fastai
from io import BytesIO

from sklearn.metrics import cohen_kappa_score
def quadratic_kappa(y_hat, y):
    return torch.tensor(cohen_kappa_score(torch.argmax(y_hat,1), y, weights='quadratic'))

def model_fn(model_dir):
    learn = load_learner(path = model_dir, fname = 'restnet50.pkl')
    return learn

model = model_fn('models')

app = Flask(__name__)

                   
@app.route('/')
def home(): 
    return render_template('index.html')

@app.route('/about/')
def about():
    return render_template('about.html')

@app.route('/learnmore/')
def learnmore():
    return render_template('learnmore.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    if request.method == 'POST':
        file = request.files['file']
        img = open_image(file)
        category = model.predict(img)
            
        result = {0 : 'You are completely fine!', 1 : 'You have mild symptoms of Diabetic Retinopathy', 2 : 'You have moderate symptoms of Diabetic Retionpathy', 3 : 'You have severe symptoms of Diabetic Retinopathy', 4 : 'You have symptoms of Proliferative Diabetic Retinopathy'}
        return render_template('index.html', prediction_text="{}".format(str(result[int(category[0])])))

if __name__ == '__main__':
    app.run(debug=True)
    
                   
                   
