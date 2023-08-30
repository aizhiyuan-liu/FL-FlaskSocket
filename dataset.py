
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

def getDataset(csvFl):
    df = pd.read_csv(csvFl)
    # df.isnull().sum()[30:60]
    df =df.drop(columns=['vip_cust_id','vip_lvl','idty_typ','ocpn_code','educat_degree_code','cust_star','busi_typ','cmcc_pub_tl_typ_code','rcn_chnl_id'],axis=1)
    # df.isnull().sum()[50:80]
    df= df.drop(columns=['rcn_chnl_typ','rcn_mode','user_star_val'],axis=1)
    X = df.drop(columns=['label','data_sim_and_m2m_user_flag','pretty_num_typ'],axis=1)
    Y =  df['label']
    XTrain,XTest,YTrain,YTest = train_test_split(X,Y,test_size=0.2,random_state=2)
    # print(X.shape,XTrain.shape,XTest.shape)
    XTrain = torch.from_numpy(XTrain.values).float()
    XTest = torch.from_numpy(XTest.values).float()
    YTrain = torch.from_numpy(YTrain.values).float()
    YTest = torch.from_numpy(YTest.values).float()
    return XTrain,XTest,YTrain,YTest
# def getDataset(dir,name):
#     if name =="mnist":
#         trainDataset  = datasets.MNIST(dir,train=True,download=True,transform=transforms.ToTensor())
#         evalDataset = datasets.MNIST(dir,train=False,transform=transforms.ToTensor())

#     elif name == "cifar":
#         transformTrain = transforms.Compose([
#             transforms.RandomCrop(32,padding=4),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
#         ])

#         transformTest = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#         ])

#         trainDataset = datasets.CIFAR10(dir,train=True,download=True,transform=transformTrain)
#         evalDataset = datasets.CIFAR10(dir,train=False,transform=transformTest)

    
#     return trainDataset,evalDataset