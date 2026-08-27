import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from stage_1 import *


# Sinusoidal timestep embedding
class TimeEmbedding(nn.Module):
    def __init__(self, time_emb_dim=128, hidden_dim=512):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.time_proj = nn.Linear(time_emb_dim, hidden_dim)

    def forward(self, t):
        # t: [B]
        device = t.device

        half_dim = self.time_emb_dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * (-emb_scale))
        emb = t[:, None] * emb[None, :]
        emb = torch.cat(
            [emb.sin(), emb.cos()],
            dim=-1
        )
        # emb: [B, 128]
        t_emb = self.time_proj(emb)
        # t_emb: [B, 512]
        return t_emb


# MLP Block
class MLPBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        # in: [B, input_dim]
        # out: [B, output_dim]
        return self.block(x)


# Asymmetric U-Net (MLP-based)
class DenoisingUNet(nn.Module):
    def __init__(self, input_dim=1536, output_dim=512):
        super().__init__()

        # Downsampling
        self.down1 = MLPBlock(input_dim, int(input_dim*2/3))
        self.down2 = MLPBlock(int(input_dim*2/3), int(input_dim/3))
        self.down3 = MLPBlock(int(input_dim/3), int(input_dim/6))

        # Bottleneck
        self.middle = MLPBlock(int(input_dim/6), int(input_dim/6))

        # Upsampling
        self.up1 = MLPBlock(int(input_dim/6) + int(input_dim/6), int(input_dim/3))
        self.up2 = MLPBlock(int(input_dim/3) + int(input_dim/3), int(input_dim/3))
        self.up3 = MLPBlock(int(input_dim/3) + int(input_dim*2/3), output_dim)

    def forward(self, x):
        # x: [B, 1536]
        d1 = self.down1(x)
        # d1: [B, 1024]
        d2 = self.down2(d1)
        # d2: [B, 512]
        d3 = self.down3(d2)
        # d3: [B, 256]

        mid = self.middle(d3)
        # d3: [B, 256]

        u1 = self.up1(torch.cat([mid, d3], dim=-1))
        # u1: [B, 256+256] -> [B, 512]
        u2 = self.up2(torch.cat([u1, d2], dim=-1))
        # u2: [B, 512+512] -> [B, 512]
        u3 = self.up3(torch.cat([u2, d1], dim=-1))
        # u3: [B, 512+1024] -> [B, 512]

        return u3


class Stage2_Network(nn.Module):
    def __init__(self, vector_dim=25, hidden_dim=512, time_emb_dim=128):
        super().__init__()
        self.vector_proj = nn.Linear(vector_dim, hidden_dim)

        self.time_embedding = TimeEmbedding(time_emb_dim=time_emb_dim, hidden_dim=hidden_dim)

        self.workload_proj = MLPBlock(input_dim=6, output_dim=32)
        self.PE_num_proj = MLPBlock(input_dim=1, output_dim=32)
        self.tag_proj = MLPBlock(input_dim=1, output_dim=64)
        self.cond_proj = nn.Linear(32 + 32 + 64, hidden_dim)

        self.unet = DenoisingUNet(input_dim=hidden_dim * 3, output_dim=hidden_dim)

        self.final_layer = nn.Linear(hidden_dim, vector_dim)

    def forward(self, vt, t, workload, PE_num, tag):
        # vt: [B, 25]  vector
        # t: [B]   diffusion time step
        # workload: [B, 6]   (K,C,X,Y,R,S)
        # PE_num: [B, 1]   number of PE
        # tag: [B, 1]   0 indicates poor performance, 1 indicates good performance
        
        vt_emb = self.vector_proj(vt)
        # vt_emb: [B, 512]

        t_emb = self.time_embedding(t)
        # t_emb: [B, 512]

        workload_emb = self.workload_proj(workload)
        # workload_emb: [B, 32]
        PE_num_emb = self.PE_num_proj(PE_num)
        # PE_num_emb: [B, 32]
        tag_emb = self.tag_proj(tag)
        # tag_emb: [B, 64]
        cond = torch.cat([workload_emb, PE_num_emb, tag_emb], dim=-1)
        # cond: [B, 128]
        cond = self.cond_proj(cond)
        # cond: [B, 512]

        x = torch.cat([vt_emb, t_emb, cond], dim=-1)
        # x: [B, 1536]

        mid = self.unet(x)
        # mid: [B, 512]

        pred_noise = self.final_layer(mid)
        # pred_noise: [B, 25]

        return pred_noise


# Store all data and labels of the dataset
# Each piece of data includes:
# w: (K,C,X,Y,R,S)
# PE_num: Number of PE
# tag: 0 indicates poor performance, 1 indicates good performance
# vector: The vector representation of length 25 obtained after passing through stage_1 for dataflow
class MyDataset_Stage2(torch.utils.data.Dataset):
    def __init__(self, datasetname='./dataset_file/dataset_preprocess.csv'):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.workload = []
        self.PE_num = []
        self.tag = []
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

                part_1, part_2, part_3, part_4, part_5, part_6 = \
                    dataflow2input(cur_dataflow, cur_PE_num, [cur_workload])
                cur_workload = torch.tensor(cur_workload, dtype=torch.float32).to(device)
                cur_PE_num = torch.tensor([cur_PE_num], dtype=torch.float32).to(device)
                cur_tag = torch.tensor([cur_tag], dtype=torch.float32).to(device)
                cur_vector = torch.cat([part_1, part_2, part_3, part_4, part_5, part_6], dim=-1)
                cur_vector = cur_vector.to(device)  # [25]

                self.workload.append(cur_workload)
                self.PE_num.append(cur_PE_num)
                self.tag.append(cur_tag)
                self.vector.append(cur_vector)
        
        # print(len(self.workload))
        # print(self.workload[0])
        # print(self.PE_num[0])
        # print(self.tag[0])
        # print(self.vector[0])
        print("dataset preprocessed.")
        
    def __len__(self):
        return len(self.workload)
    
    def __getitem__(self, index):
        return self.workload[index], self.PE_num[index], self.tag[index], self.vector[index]



T = 1000
beta = torch.linspace(1e-4, 0.02, T).to("cuda" if torch.cuda.is_available() else "cpu")
alpha = 1 - beta
alpha_bar = torch.cumprod(alpha, dim=0)

def q_sample(x0, t, noise):
    # forward diffusion
    # x0: [B, 25]
    # t: [B]
    # noise: [B, 25]
    sqrt_ab = torch.sqrt(alpha_bar[t])[:, None]
    sqrt_1_ab = torch.sqrt(1 - alpha_bar[t])[:, None]
    return sqrt_ab * x0 + sqrt_1_ab * noise
    

def train_one_epoch_stage2(model, optimizer, device, dataloader):
    total_loss = 0
    for _, data in enumerate(dataloader):
        workload, PE_num, tag, vector = data
        # print(workload.shape)   # torch.Size([B, 6])
        # print(PE_num.shape)     # torch.Size([B, 1])
        # print(tag.shape)        # torch.Size([B, 1])
        # print(vector.shape)     # torch.Size([B, 25])
        
        t = torch.randint(0, T, (vector.size(0),), device=device)
        # t \in [0, 1000)
        noise = torch.randn_like(vector)
        # noise: torch.Size([B, 25])
        vt = q_sample(vector, t, noise)
        # vt: torch.Size([B, 25])
        pred_noise = model(vt, t, workload, PE_num, tag)
        # pred_noise: torch.Size([B, 25])

        loss = F.mse_loss(pred_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def train_stage2(datasetname='./dataset_file/dataset_preprocess.csv', epochs=1000, \
                 save=True, save_filename='./model_file/stage2.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Stage2_Network().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    dataset = MyDataset_Stage2(datasetname=datasetname)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    model.train()
    min_loss = float('inf')
    print('stage2 training starts.')
    for epoch in range(epochs):
        loss = train_one_epoch_stage2(model, optimizer, device, dataloader)
        print('epoch: ', end=''); print(epoch, end='  '); print('loss: ', end=''); print(loss)
        if loss < min_loss:
            min_loss = loss
            if save == True:
                torch.save(model, save_filename)
                print("model Stage2_Network saved.")





if __name__ == "__main__":
    train_stage2(datasetname='./dataset_file/dataset_preprocess.csv', epochs=200, \
                 save=True, save_filename='./model_file/stage2.pth')

