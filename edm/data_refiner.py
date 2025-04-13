import torch
import torch.nn as nn
from torch.optim import Adam
import pandas as pd
from train import MedicalDataset
from torch.utils.data import DataLoader,Dataset
from tqdm import trange
import pdb

class RealData(Dataset):
    def __init__(self, file_path):
        self.data_list = []
        with open(file_path, 'r') as file:
            for line in file:
                data = list(map(float, line.strip().split(',')))
                tensor_data = torch.tensor(data, dtype=torch.float32)
                self.data_list.append(tensor_data)
 
    def __len__(self):
        return len(self.data_list)
 
    def __getitem__(self, idx):
        return self.data_list[idx]
    
class DataFetcher(Dataset):
    def __init__(self, tensor1, tensor2):
        self.tensor1 = tensor1
        self.tensor2 = tensor2
 
    def __len__(self):
        return len(self.tensor1)
 
    def __getitem__(self, idx):
        return self.tensor1[idx], self.tensor2[idx]

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.hidden = nn.Linear(6, 32)  # 隐藏层，6个输入特征，32个输出特征
        self.output = nn.Linear(32, 1)  # 输出层，32个输入特征，1个输出特征
    
    def forward(self, x):
        x = torch.relu(self.hidden(x))  # ReLU激活函数
        x = self.output(x)  # 线性输出
        return x
    
def loss_fn(model, model_output, real_data):
    bias = model_output - real_data
    pred_bias = model(bias[:,:-1])
    loss = (pred_bias - bias[:,-1]).abs()
    return loss.sum()

def refiner(model, model_output, con):
    bias = model_output[:,:-1] - con
    with torch.no_grad():
        pred_bias = model(bias)
    refined_labels = model_output[:,-1].unsqueeze(1) - pred_bias
    pdb.set_trace()
    return refined_labels

if __name__ == '__main__':
    # 读取 model_output
    file_path = 'generated_data_2avg.xlsx' 
    df = pd.read_excel(file_path,header=None)
    data_array = df.to_numpy()
    model_output = torch.tensor(data_array, dtype=torch.float32) 
    # 读取 real_data
    file_path = 'train_type123.txt'
    dataset = RealData(file_path)
    dataloader = DataLoader(dataset, batch_size=964, shuffle=False)
    real_data = next(iter(dataloader))

    data = DataFetcher(model_output, real_data)
    data_loader = DataLoader(data, batch_size=64, shuffle=False, num_workers=4)

    model = SimpleNN()
    n_epochs = 1000
    batch_size =  64 
    lr=1e-4 
    device = 'cpu'

    optimizer = Adam(model.parameters(), lr=lr)
    tqdm_epoch = trange(n_epochs)
    for epoch in tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        for modeloutput, realdata in data_loader:
            modeloutput = modeloutput.to(device)
            realdata = realdata.to(device)

            loss = loss_fn(model, modeloutput, realdata)
            optimizer.zero_grad()
            loss.backward()    
            optimizer.step()
            avg_loss += loss.item() * modeloutput.shape[0]
            num_items += modeloutput.shape[0]
        # Print the averaged training loss so far.
        tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
        # Update the checkpoint after each epoch of training.
        torch.save(model.state_dict(), 'refiner_ckpt.pth')

