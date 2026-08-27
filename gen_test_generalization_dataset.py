import torch
import numpy as np
import random
from hwmetrics.multicore import MultiCore
from multiprocessing.pool import Pool
from multiprocessing import cpu_count

import csv

PE_RANGE = [[10, 50], [51, 100], [101, 150], [151, 200], [201, 250], [251, 300], [301, 350], [351, 400], [401, 450], [451, 500]]

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# The network structure to be measured: dataset_network.csv
# Return an array of 470 networks
def read_network(file_name='./dataset_file/dataset_network.csv'):
    with open(file_name, 'r', newline='') as f:
        dims = [row for row in csv.reader(f)]
        del dims[0]
        for dim in dims:
            dim[0] = int(dim[0])
            dim[1] = int(dim[1])
            dim[2] = int(dim[2])
            dim[3] = int(dim[3])
            dim[4] = int(dim[4])
            dim[5] = int(dim[5])
    # print('len(dims)=' + str(len(dims)))
    return dims



# Each test sample is run 20 times with GA_1arge and considered as an ideal EDP (ref EDP).
# Afterwards, the dataset was fixed.
def gen_dataset(dims, gen_file_name='dataset_test.csv', GA_num=20):
    # dims: all network structures to be measured
    pool = Pool(min(cpu_count(), 16))
    with open(gen_file_name, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['K', 'C', 'X', 'Y', 'R', 'S', 'PE', 'ref_EDP'])
        for i in range(len(dims)):
            setup_seed(i+123)
            dim = dims[i]
            print('current idx: ' + str(i))
            print('current dim: ' + str(dim))
            K = dim[0]
            C = dim[1]
            X = dim[2]
            Y = dim[3]
            R = dim[4]
            S = dim[5]
            types = ['CONV']

            for cur_range in PE_RANGE:
                cur_PE = random.randint(cur_range[0], cur_range[1])
                print('PE_num: ' + str(cur_PE), end='  ')
                min_EDP = float('inf')
                j = 0
                while j < GA_num:
                    env = MultiCore(dimensions=[dim], types=types, model_file="tmp_model.m", fitness='EDP', l1_size=-1, l2_size=-1, \
                                    num_pe=cur_PE)
                    # ckpt = env.explore()
                    ckpt = env.explore(num_population=128, num_generations=64, elite_ratio=0.05, parents_ratio=0.4, ratio_decay=1, 
                                       pool=pool)
                    best_sol = ckpt["best_sol"]
                    realres = env.get_indiv_info(best_sol, num_pe=cur_PE)
                    if realres is not None:
                        runtime, throughput, energy, area, l1_size, l2_size, mac, power, num_pe = realres[0]
                        cur_min_EDP = energy * runtime
                        if cur_min_EDP > 0.1 and cur_min_EDP < min_EDP:
                            min_EDP = cur_min_EDP
                    j += 1
                print("min_EDP: ", min_EDP)
                data = [K, C, X, Y, R, S, cur_PE, min_EDP]
                writer.writerow(data)
            print()
    pool.close()


# Construct a test dataset consisting of 470 workloads * 10 random PE_num, totaling 4700 test samples
def gen_dataset_test(gen_file_name='dataset_test.csv', GA_num=20):
    dims = read_network()
    # dims: 470个网络的数组
    gen_dataset(dims, gen_file_name=gen_file_name, GA_num=GA_num)


# Construct a generalization dataset consisting of 112 workloads * 10 random PE_num, totaling 1120 test samples
def gen_dataset_generalization(gen_file_name='dataset_generalization.csv', GA_num=20):
    dims = [[32, 3, 224, 224, 3, 3], [1, 32, 112, 112, 3, 3], [8, 32, 1, 1, 1, 1], [32, 8, 1, 1, 1, 1], [16, 32, 112, 112, 1, 1], [96, 16, 112, 112, 1, 1], [4, 96, 1, 1, 1, 1], [96, 4, 1, 1, 1, 1], [24, 96, 56, 56, 1, 1], [144, 24, 56, 56, 1, 1], [1, 144, 56, 56, 3, 3], [6, 144, 1, 1, 1, 1], [144, 6, 1, 1, 1, 1], [24, 144, 56, 56, 1, 1], [1, 144, 56, 56, 5, 5], [40, 144, 28, 28, 1, 1], [240, 40, 28, 28, 1, 1], [1, 240, 28, 28, 5, 5], [10, 240, 1, 1, 1, 1], [240, 10, 1, 1, 1, 1], [40, 240, 28, 28, 1, 1], [1, 240, 28, 28, 3, 3], [80, 240, 14, 14, 1, 1], [480, 80, 14, 14, 1, 1], [1, 480, 14, 14, 3, 3], [20, 480, 1, 1, 1, 1], [480, 20, 1, 1, 1, 1], [80, 480, 14, 14, 1, 1], [1, 480, 14, 14, 5, 5], [112, 480, 14, 14, 1, 1], [28, 672, 1, 1, 1,1], [672, 28, 1, 1, 1, 1], [192, 672, 7, 7, 1, 1], [1152, 192, 7, 7, 1, 1], [1, 1152, 7, 7, 5, 5], [48, 1152, 1, 1, 1, 1], [1152, 48, 1, 1, 1, 1], [192, 1152, 7, 7, 1, 1], [1, 1152, 7, 7, 3, 3], [320, 1152, 7, 7, 1, 1], [1280, 320, 7, 7, 1, 1], [1000, 1280, 1, 1, 1, 1], [64, 32, 112, 112, 1, 1], [64, 64, 112, 112, 3, 3], [8, 64, 1, 1, 1, 1], [64, 8, 1, 1, 1, 1], [64, 64, 56, 56, 1, 1], [128, 64, 56, 56, 1, 1], [128, 128, 56, 56, 3, 3], [16, 128, 1, 1, 1, 1], [128, 16, 1, 1, 1, 1], [128, 128, 28, 28, 1, 1], [128, 128, 28, 28, 3, 3], [32, 128, 1, 1, 1, 1], [128, 32, 1, 1, 1, 1], [320, 128, 28, 28, 1, 1], [320, 320, 28, 28, 3, 3], [32, 320, 1, 1, 1, 1], [320, 32, 1, 1, 1, 1], [320, 320, 14, 14, 1, 1], [320, 320, 14, 14, 3, 3], [80, 320, 1, 1, 1, 1], [320, 80, 1, 1, 1, 1], [768, 320, 14, 14, 1, 1], [768, 768, 14, 14, 3, 3], [80, 768, 1, 1, 1, 1], [768, 80, 1, 1, 1, 1], [768, 768, 7, 7, 1, 1], [768, 768, 7, 7, 3, 3], [192, 768, 1, 1, 1, 1], [768, 192, 1, 1, 1, 1], [1000, 768, 1, 1, 1, 1], [96, 3, 224, 224, 4, 4], [384, 96, 1, 1, 1, 1], [96, 384, 1, 1, 1, 1], [192, 96, 56, 56, 2, 2], [384, 192, 28, 28, 2, 2], [1536, 384, 1, 1, 1, 1], [384, 1536, 1, 1, 1, 1], [768, 384, 14, 14, 2, 2], [3072, 768, 1, 1, 1, 1], [768, 3072, 1, 1, 1, 1], [64, 3, 224, 224, 1, 1], [64, 3, 224, 224, 3, 3], [128, 64, 112, 112, 1, 1], [128, 64, 112, 112, 3, 3], [128, 128, 56, 56, 1, 1], [256, 128, 56, 56, 1, 1], [256, 128, 56, 56, 3, 3], [256, 256, 28, 28, 1, 1], [256, 256, 28, 28, 3, 3], [512, 256, 28, 28, 1, 1], [512, 256, 28, 28, 3, 3], [512, 512, 14, 14, 1, 1], [512, 512, 14, 14, 3, 3], [2048, 512, 14, 14, 1, 1], [2048, 512, 14, 14, 3, 3], [1000, 2048, 1, 1, 1, 1], [64, 3, 224, 224, 7, 7], [64, 64, 56, 56, 3, 3], [256, 64, 56, 56, 1, 1], [64, 256, 56, 56, 1, 1], [128, 256, 56, 56, 1, 1], [512, 256, 56, 56, 1, 1], [256, 512, 28, 28, 1, 1], [1024, 512, 28, 28, 1, 1], [256, 256, 14, 14, 3, 3], [512, 1024, 14, 14, 1, 1], [2048, 512, 7, 7, 1, 1], [2048, 1024, 14, 14, 1, 1], [512, 2048, 7, 7, 1, 1], [512, 512, 7, 7, 3, 3]]
    gen_dataset(dims, gen_file_name=gen_file_name, GA_num=GA_num)




if __name__=='__main__':
    gen_dataset_test(gen_file_name='dataset_test.csv', GA_num=20)
    # gen_dataset_generalization(gen_file_name='dataset_generalization.csv', GA_num=20)








