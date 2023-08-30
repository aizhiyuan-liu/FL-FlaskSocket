import torch
# from torchvision import models
import math


class  FNN(torch.nn.Module):
    def __init__(self,inputSize,hiddenSize,outputsize):
        super(FNN,self).__init__()
        self.fc1 = torch.nn.Linear(inputSize,hiddenSize)
        self.fc2 = torch.nn.Linear(hiddenSize,outputsize)
        self.relu = torch.nn.ReLU()
        self.sigmoid  = torch.nn.Sigmoid()
    
    def  forward(self,x):
        x = self.relu(self.fc1(x))
        x  = self.fc2(x)
        return self.sigmoid(x)
    
    def apply_client_update(self, update):
        # Convert the update list to a tensor
        update_tensor = torch.tensor(update)

        # Apply the update to the model's parameters
        with torch.no_grad():
            for param, delta in zip(self.parameters(), update_tensor):
                param.add_(delta)


def getModel():
    input_size = 70  
    hidden_size = 64
    output_size = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = FNN(input_size,hidden_size,output_size)
    return model.to(device)

# def getModel(name="vgg16",pretrained=True):
#     if name == "resnet18":
#         model = models.resnet18(pretrained=pretrained)
#     elif name == "vgg16":
#         model = models.vgg16(pretrained=pretrained)
    
#     if torch.cuda.is_available():
#         return model.cuda()
#     else:
#         return model

# def modelNorm(model_1,model_2):
#     squaredSum = 0
#     for name, layer in model_1.named_parameters():
#         squaredSum += torch.sum(torch.pow(layer.data- model_2.state_dict()[name].data,2))
    
#     return math.sqrt(squaredSum)