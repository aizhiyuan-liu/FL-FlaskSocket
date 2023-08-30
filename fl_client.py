from flask import Flask, render_template, request
import requests
import json

app = Flask(__name__)

SERVER_URL = "http://127.0.0.1:5000"  # Replace with your server's URL

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/noised', methods=['POST'])
def noised():


@app.route('/update', methods=['POST'])
def update():
    weight = float(request.form['weight'])
    update = list(map(float, request.form['update'].split(',')))

    # Prepare data to send to the server
    data = {
        "weight": weight,
        "update": update
    }

    # Send data to the server
    response = requests.post(f"{SERVER_URL}/receive_update", json=data)

    if response.status_code == 200:
        return "Update submitted successfully"
    else:
        return "Error submitting update"

@app.route('/train_local_model', methods=['POST'])
def train_local_model():
    # Send a request to the server to train the local model
    data = request.form['data']
    epochs = request.form['epoch']
    # update = list(map(float, request.form['update'].split(',')))
    reqData = {
        "data":data,
        "epoch":epochs
    }
    response = requests.post(f"{SERVER_URL}/train_local_model",json=reqData)
    # print(str(response.content.decode('utf-8')))
    if response.status_code == 200:
        return str(response.content.decode('utf-8'))
    else:
        return "Error training local model"

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001)
