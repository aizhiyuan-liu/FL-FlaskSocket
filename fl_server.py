from flask import Flask, request, jsonify
import json
import dataset
import argparse,json
import torch.optim as optim
import torch
app = Flask(__name__)

# Import your FNN model and any necessary libraries for training
import numpy as np
import  models
clients = []
# Initialize your FNN model
global_model = models.getModel()

@app.route('/')
def index():
    return "Federated Learning Server"

@app.route('/receive_update', methods=['POST'])
def receive_update():
    data = request.json

    weight = data['weight']
    update = data['update']
    print(weight)
    print(update)

    # Update global model
    global_model.apply_client_update(update)

    return "Update received and applied successfully"

@app.route('/get_global_model', methods=['GET'])
def get_global_model():
    return jsonify(global_model.get_weights())

@app.route('/register_client', methods=['POST'])
def register_client():
    client_id = len(clients) + 1
    clients.append(client_id)
    return jsonify({"client_id": client_id})

@app.route('/train_local_model', methods=['POST'])
def train_local_model():
    accTest = 0
    losss =0
    trainLoss = 0
    data = request.json
    print(data)
    dataFile = data['data']
    epochs = data['epoch']
    # update = data['update']
    print(dataFile)
    # print(update)
    # Load your dataset
    XTrain,XTest,YTrain,YTest = dataset.getDataset(dataFile)

    # Train the local model on the server
    # global_model.a
    # global_model.train(XTrain, YTrain)  # Implement this function to train your model
    iter = 0
    global_model.train()
    for e in range(int(epochs)):
        # print(f"{iter}")
        lossFunc = torch.nn.BCELoss()
        opt = optim.Adam(global_model.parameters(), lr=0.001) 
        x = XTrain
        lbl = YTrain
        opt.zero_grad()
        output = global_model(XTrain)
        loss = lossFunc(torch.squeeze(output),lbl)
        trainLoss= loss.item()
        loss.backward()
        opt.step()

        iter += 1
        if iter % 1  == 0:
            with torch.no_grad():
                correctTest  = 0
                totalTest = 0
                outputTest = torch.squeeze(global_model(XTest))
                losstest  = lossFunc(outputTest,YTest)

                predTest = outputTest.round().detach().numpy()
                totalTest += YTest.size(0)
                correctTest +=  np.sum(predTest == YTest.detach().numpy())
                accTest = 100 * correctTest / totalTest

                total = 0
                correct = 0
                total += YTrain.size(0)
                correct += np.sum(torch.squeeze(output).round().detach().numpy() == YTrain.detach().numpy())
                acc = 100 * correct / total
                losss  = losstest.item()
                print(f'Iter: TestLoss: {losstest.item()} TestAcc: {accTest}\nIter: TrainLoss: {loss.item()} TrainAcc: {acc}\n')
    local_model_path = 'model.pth'
    torch.save(global_model.state_dict(), local_model_path)
    return f'Iter: TestLoss: {losss} TestAcc: {accTest}\nIter: TrainLoss: {trainLoss} TrainAcc: {acc}\n'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Federated Learning")
    parser.add_argument("-c","--conf",dest="conf")
    args = parser.parse_args()

    with open(args.conf,"r") as f:
        conf = json.load(f)
    app.run(host='127.0.0.1', port=5000)
