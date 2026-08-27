from stage_2 import *
from stage_2_MLP import *


# Apply DDPM to generate dataflows
@torch.no_grad()
def gen_dataflow_DDPM(workload_list, PE_num, tag, n, \
                      stage2_model_name="./model_file/stage2.pth"):
    # workload: [[K, C, R, S, X, Y], ......]
    # PE_num: 38
    # tag: 0 / 1
    # n: batchsize, the number of dataflows to be generated
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vt = torch.randn(n, 25).to(device)                                  # torch.Size([B, 25])
    workload_0 = torch.tensor(workload_list[0]).float()
    workload_0 = workload_0.unsqueeze(0).repeat(n, 1).to(device)        # torch.Size([B, 6])
    PE_num_tensor = torch.tensor([PE_num]).float()
    PE_num_tensor = PE_num_tensor.unsqueeze(0).repeat(n, 1).to(device)  # torch.Size([B, 1])
    tag_tensor = torch.full((n, 1), tag, device=device).float()         # torch.Size([B, 1])
    stage2_model = torch.load(stage2_model_name).to(device)

    for t in reversed(range(T)):
        t_tensor = torch.full((n,), t, device=device).float()   # torch.Size([B])
        pred_noise = stage2_model(vt, t_tensor, workload_0, PE_num_tensor, tag_tensor) # torch.Size([B, 128])

        alpha_t = alpha[t]
        alpha_bar_t = alpha_bar[t]
        beta_t = beta[t]

        noise = torch.randn_like(vt) if t > 0 else 0
        vt = (1 / torch.sqrt(alpha_t)) * (vt - (beta_t / torch.sqrt(1 - alpha_bar_t)) * pred_noise) + \
             torch.sqrt(beta_t) * noise
    
    part_1, part_2, part_3, part_4, part_5, part_6 = torch.split(vt, [4, 6, 4, 4, 1, 6], dim=1)
    # print(part_1)       # torch.Size([B, 4])
    # print(part_2)       # torch.Size([B, 6])
    # print(part_3)       # torch.Size([B, 4])
    # print(part_4)       # torch.Size([B, 4])
    # print(part_5)       # torch.Size([B, 1])
    # print(part_6)       # torch.Size([B, 6])
    dataflow_list = []
    for i in range(n):
        dataflow = output2dataflow(part_1[i], part_2[i], part_3[i], part_4[i], part_5[i], part_6[i], PE_num, workload_list)
        dataflow_list.append(dataflow)
    return dataflow_list



@torch.no_grad()
def gen_dataflow_DDIM(workload_list, PE_num, tag, n, \
                      stage2_model_name="./model_file/stage2.pth", \
                      DDIM_steps=50, eta=0.0):
    # workload: [[K, C, R, S, X, Y], ......]
    # PE_num: 38
    # tag: 0 / 1
    # n: batchsize, the number of dataflows to be generated
    # DDIM_steps: the number of steps in reverse diffusion of DDIM model
    # eta: eta=0 -> deterministic
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vt = torch.randn(n, 25).to(device)                                  # torch.Size([B, 25])
    workload_0 = torch.tensor(workload_list[0]).float()
    workload_0 = workload_0.unsqueeze(0).repeat(n, 1).to(device)        # torch.Size([B, 6])
    PE_num_tensor = torch.tensor([PE_num]).float()
    PE_num_tensor = PE_num_tensor.unsqueeze(0).repeat(n, 1).to(device)  # torch.Size([B, 1])
    tag_tensor = torch.full((n, 1), tag, device=device).float()         # torch.Size([B, 1])
    stage2_model = torch.load(stage2_model_name).to(device)

    timesteps = torch.linspace(T - 1, 0, DDIM_steps).long().to(device)
    # timesteps: tensor([999, 978, 958, 937, ... , 40,  20,   0])     torch.Size([50])
    for i in range(DDIM_steps):
        t = timesteps[i]
        if i == DDIM_steps - 1:
            t_prev = torch.tensor(0).to(device)
        else:
            t_prev = timesteps[i + 1]
        
        t_tensor = torch.full((n,), t, device=device).float()   # torch.Size([B])
        pred_noise = stage2_model(vt, t_tensor, workload_0, PE_num_tensor, tag_tensor) # torch.Size([B, 25])

        alpha_bar_t = alpha_bar[t]
        alpha_bar_prev = alpha_bar[t_prev]

        v0_pred = (vt - torch.sqrt(1 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)

        # DDIM sigma
        # eta=0 -> sigma=0    deterministic
        sigma = (
            eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t))
            * torch.sqrt(1 - alpha_bar_t / alpha_bar_prev)
        )

        # eta=0 -> noise=0    deterministic
        noise = torch.randn_like(vt) if eta > 0 else 0

        # direction pointing to vt
        dir_vt = torch.sqrt(1 - alpha_bar_prev - sigma**2) * pred_noise

        # DDIM update
        vt = torch.sqrt(alpha_bar_prev) * v0_pred + dir_vt + sigma * noise

    part_1, part_2, part_3, part_4, part_5, part_6 = torch.split(vt, [4, 6, 4, 4, 1, 6], dim=1)
    # print(part_1)       # torch.Size([B, 4])
    # print(part_2)       # torch.Size([B, 6])
    # print(part_3)       # torch.Size([B, 4])
    # print(part_4)       # torch.Size([B, 4])
    # print(part_5)       # torch.Size([B, 1])
    # print(part_6)       # torch.Size([B, 6])
    dataflow_list = []
    for i in range(n):
        dataflow = output2dataflow(part_1[i], part_2[i], part_3[i], part_4[i], part_5[i], part_6[i], PE_num, workload_list)
        dataflow_list.append(dataflow)
    return dataflow_list




@torch.no_grad()
def gen_dataflow_MLP(workload_list, PE_num, n, \
                     stage2_model_name="./model_file/stage2_MLP.pth"):
    # workload: [[K, C, R, S, X, Y], ......]
    # PE_num: 38
    # n: batchsize, the number of dataflows to be generated
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    workload_0 = torch.tensor(workload_list[0]).float()
    workload_0 = workload_0.unsqueeze(0).repeat(n, 1).to(device)        # torch.Size([B, 6])
    PE_num_tensor = torch.tensor([PE_num]).float()
    PE_num_tensor = PE_num_tensor.unsqueeze(0).repeat(n, 1).to(device)  # torch.Size([B, 1])
    stage2_model = torch.load(stage2_model_name).to(device)

    vt = stage2_model(workload_0, PE_num_tensor)

    part_1, part_2, part_3, part_4, part_5, part_6 = torch.split(vt, [4, 6, 4, 4, 1, 6], dim=1)
    # print(part_1.shape)       # torch.Size([B, 4])
    # print(part_2.shape)       # torch.Size([B, 6])
    # print(part_3.shape)       # torch.Size([B, 4])
    # print(part_4.shape)       # torch.Size([B, 4])
    # print(part_5.shape)       # torch.Size([B, 1])
    # print(part_6.shape)       # torch.Size([B, 6])
    dataflow_list = []
    for i in range(n):
        dataflow = output2dataflow(part_1[i], part_2[i], part_3[i], part_4[i], part_5[i], part_6[i], PE_num, workload_list)
        dataflow_list.append(dataflow)
    return dataflow_list



