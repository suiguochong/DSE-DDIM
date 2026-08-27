import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from stage_1 import *


class MLPBlock_MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        return self.block(x)


class Stage2_Network_MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.workload_proj = MLPBlock_MLP(input_dim=6, output_dim=32)
        self.PE_num_proj = MLPBlock_MLP(input_dim=1, output_dim=32)
        self.net = nn.Sequential(
            MLPBlock_MLP(input_dim=64, output_dim=1024),
            MLPBlock_MLP(input_dim=1024, output_dim=1024),
            MLPBlock_MLP(input_dim=1024, output_dim=1024),
            MLPBlock_MLP(input_dim=1024, output_dim=512),
            MLPBlock_MLP(input_dim=512, output_dim=256),
            MLPBlock_MLP(input_dim=256, output_dim=25)
        )

    def forward(self, workload, PE_num):
        # workload: [B, 6]   (K,C,X,Y,R,S)
        # PE_num: [B, 1]
        workload_emb = self.workload_proj(workload)
        # workload_emb: [B, 32]
        PE_num_emb = self.PE_num_proj(PE_num)
        # PE_num_emb: [B, 32]
        x = torch.cat([workload_emb, PE_num_emb], dim=-1)
        # x: [B, 64]

        vector = self.net(x)
        # vector: [B, 25]
        return vector


# Store all data and labels of the dataset
# Each piece of data includes:
# w: (K,C,X,Y,R,S)
# PE_num: Number of PE
# tag: 0 indicates poor performance, 1 indicates good performance
# vector: The vector representation of length 25 obtained after passing through stage_1 for dataflow
class MyDataset_Stage2_MLP(torch.utils.data.Dataset):
    def __init__(self, datasetname='./dataset_file/dataset_preprocess.csv'):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.workload = []
        self.PE_num = []
        self.vector = []

        with open(datasetname, 'r', newline='') as f:
            reader = csv.reader(f)
            next(reader)
            for data in reader:
                cur_workload = [
                    int(data[0]), int(data[1]), int(data[2]),
                    int(data[3]), int(data[4]), int(data[5])
                ]
                cur_PE_num = int(data[6])
                cur_dataflow = ast.literal_eval(data[7])
                cur_tag = int(data[8])

                if cur_tag == 0:
                    continue

                part_1, part_2, part_3, part_4, part_5, part_6 = \
                    dataflow2input(cur_dataflow, cur_PE_num, [cur_workload])
                cur_workload = torch.tensor(cur_workload, dtype=torch.float32).to(device)
                cur_PE_num = torch.tensor([cur_PE_num], dtype=torch.float32).to(device)
                cur_vector = torch.cat([part_1, part_2, part_3, part_4, part_5, part_6], dim=-1)
                cur_vector = cur_vector.to(device)  # [25]

                self.workload.append(cur_workload)
                self.PE_num.append(cur_PE_num)
                self.vector.append(cur_vector)
        
        # print(len(self.workload))
        # print(self.workload[0])
        # print(self.PE_num[0])
        # print(self.vector[0])
        print("dataset preprocessed.")
        
    def __len__(self):
        return len(self.workload)
    
    def __getitem__(self, index):
        return self.workload[index], self.PE_num[index], self.vector[index]




def train_one_epoch_stage2_MLP(model, optimizer, dataloader):
    total_loss = 0
    for _, data in enumerate(dataloader):
        workload, PE_num, vector = data
        # print(workload.shape)   # torch.Size([B, 6])
        # print(PE_num.shape)     # torch.Size([B, 1])
        # print(vector.shape)     # torch.Size([B, 25])
        
        pred_vector = model(workload, PE_num)
        # pred_vector: torch.Size([B, 25])

        loss = F.mse_loss(pred_vector, vector)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def train_stage2_MLP(datasetname='./dataset_file/dataset_preprocess.csv', epochs=1000, \
                     save=True, save_filename='./model_file/stage2_MLP.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Stage2_Network_MLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    dataset = MyDataset_Stage2_MLP(datasetname=datasetname)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    model.train()
    min_loss = float('inf')
    print('stage2_MLP training starts.')
    for epoch in range(epochs):
        loss = train_one_epoch_stage2_MLP(model, optimizer, dataloader)
        print('epoch: ', end=''); print(epoch, end='  '); print('loss: ', end=''); print(loss)
        if loss < min_loss:
            min_loss = loss
            if save == True:
                torch.save(model, save_filename)
                print("model Stage2_MLP_Network saved.")





if __name__ == "__main__":
    train_stage2_MLP(datasetname='./dataset_file/dataset_preprocess.csv', epochs=200, \
                     save=True, save_filename='./model_file/stage2_MLP.pth')

