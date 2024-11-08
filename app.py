from flask import Flask, render_template, Response
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import io
import base64
from torch import nn
from torchvision import transforms, utils, datasets

app = Flask(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

def imshow(img):
    img = img * 0.5 + 0.5  
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def zScoreScaling(tensor):
    mean = tensor.mean(axis=0, keepdims=True)
    std = tensor.std(axis=0, keepdims=True)
    return (tensor - mean) / std

def minMaxScaling(tensor):
    min = tensor.min(axis=0, keepdims=True).values
    max = tensor.max(axis=0, keepdims=True).values
    return (tensor - min) / (max - min)

@app.route('/')
def home():
    for i, (images, labels) in enumerate(trainloader, 0):
        img_html = imshow(utils.make_grid(images[:8]))
        break

    tensor = torch.tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ])
    zscore = zScoreScaling(tensor)
    min_max = minMaxScaling(tensor)

    class Linear:
        def __init__(self, in_features, out_features):
            self.weight = torch.randn(in_features, out_features)
            self.bias = torch.randn(1, out_features)

        def forward(self, x):
            return x @ self.weight + self.bias

    linear = Linear(3, 2)
    out = linear.forward(torch.tensor([1.0, 2.0, 3.0]))

    result = {
        'zscore': zscore,
        'min_max': min_max,
        'out': out,
        'weight': linear.weight,
        'bias': linear.bias,
        'img_html': img_html
    }

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
