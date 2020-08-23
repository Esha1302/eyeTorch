from flask import Flask, request, jsonify, render_template
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

import numpy as np
from PIL import Image
import requests, os, glob, time


model2 = torchvision.models.AlexNet(num_classes = 5)
check = torch.load('models/eyeTorch_model_1.pth.tar', map_location=torch.device('cpu'))
model2.load_state_dict(check['state_dict'])
model2.eval()

def image_loader(loader, image_name):
    image = Image.open(image_name)
    image = loader(image).float()
    image = torch.tensor(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image

data_transforms = transforms.Compose([
    transforms.Resize(512),
    transforms.ToTensor()
])


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
        category = np.argmax(model2(image_loader(data_transforms, file)).detach().numpy())
        result = {0 : 'You are completely fine!', 1 : 'You have mild symptoms of Diabetic Retinopathy', 2 : 'You have moderate symptoms of Diabetic Retionpathy', 3 : 'You have severe symptoms of Diabetic Retinopathy', 4 : 'You have symptoms of Proliferative Diabetic Retinopathy'}
        return render_template('index.html', prediction_text="{}".format(str(result[category])))

if __name__ == '__main__':
    app.run(debug=True)
    
                   
                   
