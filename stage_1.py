import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
import numpy as np
import csv
import ast


def get_factors(n):
    return set(reduce(list.__add__,
                        ([i, n // i] for i in range(1, int(n ** 0.5) + 1) if n % i == 0)))
                    #   ([n // i] for i in range(1, int(n ** 0.5) + 1) if n % i == 0)))

# dimensions = [[96, 32, 7, 7, 3, 3], [32, 64, 14, 14, 3, 3]]
def get_dimension_factors(dimensions):
    dimension_factors = dict()
    for d in dimensions:
        dimension_dict = {"K": d[0], "C": d[1], "Y": d[2], "X": d[3], "R": d[4], "S": d[5]}
        for key, value in dimension_dict.items():
            # "K", 96
            factors = get_factors(value)
            # factors: {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 96}
            # cutout too low parallel level
            minv = int(value ** 0.5)
            factors = set([i for i in factors if i >= minv or i == 1]) 
            # factors: {1, 12, 16, 24, 32, 48, 96}
            if key in dimension_factors.keys():
                dimension_factors[key]["set"] = dimension_factors[key]["set"] & factors
            else:
                dimension_factors[key] = {"set": factors}
    for key in dimension_factors.keys():
        # print(f'{key} {dimension_factors[key]}')
        dimension_factors[key]["array"] = np.array(list(dimension_factors[key]["set"]))

    # print(dimension_factors)
#     {
#     "K": {"set": {1, 16, 32}, "array": array([1, 16, 32])},
#     "C": {"set": {1, 8, 16, 32}, "array": array([1, 8, 16, 32])},
#     "Y": {"set": {1, 7}, "array": array([1, 7])},
#     "X": {"set": {1, 7}, "array": array([1, 7])},
#     "R": {"set": {1, 3}, "array": array([1, 3])},
#     "S": {"set": {1, 3}, "array": array([1, 3])}
#     }
    return dimension_factors

def get_KCXYRS_pos(dataflow, layer):
    # layer: 0 / 1
    pos_dict = {}
    if layer == 0:
        pos_dict[dataflow[1][0]] = 1
        pos_dict[dataflow[2][0]] = 2
        pos_dict[dataflow[3][0]] = 3
        pos_dict[dataflow[4][0]] = 4
        pos_dict[dataflow[5][0]] = 5
        pos_dict[dataflow[6][0]] = 6
    elif layer == 1:
        pos_dict[dataflow[8][0]] = 8
        pos_dict[dataflow[9][0]] = 9
        pos_dict[dataflow[10][0]] = 10
        pos_dict[dataflow[11][0]] = 11
        pos_dict[dataflow[12][0]] = 12
        pos_dict[dataflow[13][0]] = 13
    # print(pos_dict)
    # {'Y': 1, 'R': 2, 'S': 3, 'X': 4, 'C': 5, 'K': 6}
    return pos_dict

# Convert the original dataflow into an input format that can be fed into the network
def dataflow2input(dataflow, PE_num, dimensions):
    # dataflow: [['K', 28], ['Y', 1], ['R', 1], ['S', 1], ['X', 7], ['C', 8], ['K', 32], 
    #            ['X', 5], ['Y', 1], ['R', 1], ['X', 1], ['C', 1], ['K', 1], ['S', 1]]
    # PE_num: 38
    # dimensions: [[96, 32, 7, 7, 3, 3], [32, 64, 14, 14, 3, 3]]
    dimension_factors = get_dimension_factors(dimensions)

    if dataflow[0][0] == 'K':
        part_1 = torch.tensor([1, 0, 0, 0]).float()
    elif dataflow[0][0] == 'C':
        part_1 = torch.tensor([0, 1, 0, 0]).float()
    elif dataflow[0][0] == 'X':
        part_1 = torch.tensor([0, 0, 1, 0]).float()
    elif dataflow[0][0] == 'Y':
        part_1 = torch.tensor([0, 0, 0, 1]).float()
    else:
        print("error")
        exit(-1)

    part_2 = torch.tensor([0, 0, 0, 0, 0, 0]).float()
    # order_dict: Record the corresponding positions of the part_2 vector in KCXYRS order
    order_dict = {'K': 0, 'C': 1, 'X': 2, 'Y': 3, 'R': 4, 'S': 5}
    part_2[order_dict[dataflow[1][0]]] = 0.0
    part_2[order_dict[dataflow[2][0]]] = 0.2
    part_2[order_dict[dataflow[3][0]]] = 0.4
    part_2[order_dict[dataflow[4][0]]] = 0.6
    part_2[order_dict[dataflow[5][0]]] = 0.8
    part_2[order_dict[dataflow[6][0]]] = 1.0

    part_3 = torch.tensor([0, 0, 0, 0]).float()
    pos_dict = get_KCXYRS_pos(dataflow, layer=0)
    # pos_dict: {'Y': 1, 'R': 2, 'S': 3, 'X': 4, 'C': 5, 'K': 6}
    K_choice = np.sort(dimension_factors['K']['array'])
    # K_choice: [ 1 16 32]
    part_3[0] = (np.where(K_choice == dataflow[pos_dict['K']][1])[0][0] + 0.5) / len(K_choice)
    C_choice = np.sort(dimension_factors['C']['array'])
    part_3[1] = (np.where(C_choice == dataflow[pos_dict['C']][1])[0][0] + 0.5) / len(C_choice)
    X_choice = np.sort(dimension_factors['X']['array'])
    part_3[2] = (np.where(X_choice == dataflow[pos_dict['X']][1])[0][0] + 0.5) / len(X_choice)
    Y_choice = np.sort(dimension_factors['Y']['array'])
    part_3[3] = (np.where(Y_choice == dataflow[pos_dict['Y']][1])[0][0] + 0.5) / len(Y_choice)
    
    if dataflow[7][0] == 'K':
        part_4 = torch.tensor([1, 0, 0, 0]).float()
    elif dataflow[7][0] == 'C':
        part_4 = torch.tensor([0, 1, 0, 0]).float()
    elif dataflow[7][0] == 'X':
        part_4 = torch.tensor([0, 0, 1, 0]).float()
    elif dataflow[7][0] == 'Y':
        part_4 = torch.tensor([0, 0, 0, 1]).float()
    else:
        print("error")
        exit(-1)

    # The range of PE_num for the second layer: [1, layer2_PE_num]
    layer2_pos = pos_dict[dataflow[7][0]]
    layer2_PE_num = min(PE_num, dataflow[layer2_pos][1])
    part_5 = torch.tensor([0]).float()
    part_5[0] = (dataflow[7][1] - 0.5) / layer2_PE_num

    part_6 = torch.tensor([0, 0, 0, 0, 0, 0]).float()
    # order_dict: Record the corresponding positions of the part_6 vector in KCXYRS order
    order_dict = {'K': 0, 'C': 1, 'X': 2, 'Y': 3, 'R': 4, 'S': 5}
    part_6[order_dict[dataflow[8][0]]] = 0.0
    part_6[order_dict[dataflow[9][0]]] = 0.2
    part_6[order_dict[dataflow[10][0]]] = 0.4
    part_6[order_dict[dataflow[11][0]]] = 0.6
    part_6[order_dict[dataflow[12][0]]] = 0.8
    part_6[order_dict[dataflow[13][0]]] = 1.0

    return part_1, part_2, part_3, part_4, part_5, part_6


def output2dataflow(part_1, part_2, part_3, part_4, part_5, part_6, PE_num, dimensions):
    # part_1: torch.Size([4])
    # part_2: torch.Size([6])
    # part_3: torch.Size([4])
    # part_4: torch.Size([4])
    # part_5: torch.Size([1])
    # part_6: torch.Size([6])
    # PE_num: 38
    # dimensions: [[96, 32, 7, 7, 3, 3], [32, 64, 14, 14, 3, 3]]
    dataflow = [[None, 666], [None, None], [None, None], [None, None], [None, None], [None, None], [None, None], \
                [None, None], [None, 1], [None, 1], [None, 1], [None, 1], [None, 1], [None, 1]]
    dataflow[0][1] = PE_num
    
    part_1_idx = torch.argmax(part_1).item()
    # print(part_1_idx)
    if part_1_idx == 0:
        dataflow[0][0] = 'K'
    elif part_1_idx == 1:
        dataflow[0][0] = 'C'
    elif part_1_idx == 2:
        dataflow[0][0] = 'X'
    elif part_1_idx == 3:
        dataflow[0][0] = 'Y'

    part_2_idx_list = torch.argsort(part_2)
    order_dict = {0: 'K', 1: 'C', 2: 'X', 3: 'Y', 4: 'R', 5: 'S'}
    # print(part_2_idx_list)
    dataflow[1][0] = order_dict[int(part_2_idx_list[0])]
    dataflow[2][0] = order_dict[int(part_2_idx_list[1])]
    dataflow[3][0] = order_dict[int(part_2_idx_list[2])]
    dataflow[4][0] = order_dict[int(part_2_idx_list[3])]
    dataflow[5][0] = order_dict[int(part_2_idx_list[4])]
    dataflow[6][0] = order_dict[int(part_2_idx_list[5])]

    dimension_factors = get_dimension_factors(dimensions)
    # print(dimension_factors)
    pos_dict = get_KCXYRS_pos(dataflow, layer=0)
    # pos_dict: {'Y': 1, 'R': 2, 'S': 3, 'X': 4, 'C': 5, 'K': 6}
    K_choice = np.sort(dimension_factors['K']['array'])
    # K_choice: [ 1 16 32]
    part_3_K_idx = max(min(int(part_3[0] * len(K_choice)), len(K_choice) - 1), 0)
    dataflow[pos_dict['K']][1] = K_choice[part_3_K_idx]
    C_choice = np.sort(dimension_factors['C']['array'])
    part_3_C_idx = max(min(int(part_3[1] * len(C_choice)), len(C_choice) - 1), 0)
    dataflow[pos_dict['C']][1] = C_choice[part_3_C_idx]
    X_choice = np.sort(dimension_factors['X']['array'])
    part_3_X_idx = max(min(int(part_3[2] * len(X_choice)), len(X_choice) - 1), 0)
    dataflow[pos_dict['X']][1] = X_choice[part_3_X_idx]
    Y_choice = np.sort(dimension_factors['Y']['array'])
    part_3_Y_idx = max(min(int(part_3[3] * len(Y_choice)), len(Y_choice) - 1), 0)
    dataflow[pos_dict['Y']][1] = Y_choice[part_3_Y_idx]
    dataflow[pos_dict['R']][1] = 1
    dataflow[pos_dict['S']][1] = 1

    part_4_idx = torch.argmax(part_4).item()
    # print(part_4_idx)
    if part_4_idx == 0:
        dataflow[7][0] = 'K'
    elif part_4_idx == 1:
        dataflow[7][0] = 'C'
    elif part_4_idx == 2:
        dataflow[7][0] = 'X'
    elif part_4_idx == 3:
        dataflow[7][0] = 'Y'

    layer2_pos = pos_dict[dataflow[7][0]]
    layer2_PE_num = min(PE_num, dataflow[layer2_pos][1])
    dataflow[7][1] = max(min(int(part_5[0] * layer2_PE_num) + 1, layer2_PE_num), 1)

    part_6_idx_list = torch.argsort(part_6)
    order_dict = {0: 'K', 1: 'C', 2: 'X', 3: 'Y', 4: 'R', 5: 'S'}
    # print(part_6_idx_list)
    dataflow[8][0] = order_dict[int(part_6_idx_list[0])]
    dataflow[9][0] = order_dict[int(part_6_idx_list[1])]
    dataflow[10][0] = order_dict[int(part_6_idx_list[2])]
    dataflow[11][0] = order_dict[int(part_6_idx_list[3])]
    dataflow[12][0] = order_dict[int(part_6_idx_list[4])]
    dataflow[13][0] = order_dict[int(part_6_idx_list[5])]

    return dataflow




