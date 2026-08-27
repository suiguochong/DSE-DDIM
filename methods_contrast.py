from stage_2 import *
from stage_2_MLP import *
from gen_dataflow import *

import torch
import argparse
import numpy as np
import random
import time
from hwmetrics.multicore import MultiCore
from multiprocessing.pool import Pool
from multiprocessing import cpu_count


import csv


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate_N", type=int, default=64)
    parser.add_argument("--method", type=str, default='DDIM', \
                        choices=['DDIM', 'DDPM', 'Random', 'GA', 'GA_small', 'GA_large', 'MLP'])
    return parser.parse_args()


# test set: dataset_test.csv
# Return an array of 4700 networks, with each element shaped as: [[K,C,X,Y,R,S], PE_num, ref_EDP]
def read_dataset_test(file_name='./dataset_file/dataset_test.csv'):
    with open(file_name, 'r', newline='') as f:
        dataset = [row for row in csv.reader(f)]
        del dataset[0]
        for i in range(len(dataset)):
            data = dataset[i]
            data = [[int(data[0]), int(data[1]), int(data[2]), int(data[3]), int(data[4]), int(data[5])], \
                    int(data[6]), float(data[7])]
            dataset[i] = data
    # print('len(dataset)=' + str(len(dataset)))
    return dataset



def method_DDIM(workload, PE_num, type, batchsize, pool):
    env = MultiCore(dimensions=[workload], types=[type], model_file="tmp_model.m", fitness='EDP', l1_size=-1, l2_size=-1, \
                    num_pe=PE_num)
    env.set_args(num_population=32, num_generations=16, elite_ratio=0.05, parents_ratio=0.4, ratio_decay=1)
    dataflow_list = gen_dataflow_DDIM(workload_list=[workload], PE_num=PE_num, tag=1, n=batchsize, \
                                        stage2_model_name="./model_file/stage2.pth", \
                                        DDIM_steps=50, eta=0.0)
    # print(dataflow_list)
    PE_num_list = [PE_num] * batchsize
    realres_list = pool.map(env.get_indiv_info_pool, list(zip(dataflow_list, PE_num_list)))
    min_EDP_DDIM = float('inf')
    for realres in realres_list:
        if realres is not None:
            runtime, throughput, energy, area, l1_size, l2_size, mac, power, num_pe = realres[0]
            EDP = energy * runtime
            if EDP > 0.1 and EDP < min_EDP_DDIM:
                min_EDP_DDIM = EDP
    return min_EDP_DDIM


def method_DDPM(workload, PE_num, type, batchsize, pool):
    env = MultiCore(dimensions=[workload], types=[type], model_file="tmp_model.m", fitness='EDP', l1_size=-1, l2_size=-1, \
                    num_pe=PE_num)
    env.set_args(num_population=32, num_generations=16, elite_ratio=0.05, parents_ratio=0.4, ratio_decay=1)
    dataflow_list = gen_dataflow_DDPM(workload_list=[workload], PE_num=PE_num, tag=1, n=batchsize, \
                                      stage2_model_name="./model_file/stage2.pth")
    # print(dataflow_list)
    PE_num_list = [PE_num] * batchsize
    realres_list = pool.map(env.get_indiv_info_pool, list(zip(dataflow_list, PE_num_list)))
    min_EDP_DDM = float('inf')
    for realres in realres_list:
        if realres is not None:
            runtime, throughput, energy, area, l1_size, l2_size, mac, power, num_pe = realres[0]
            EDP = energy * runtime
            if EDP > 0.1 and EDP < min_EDP_DDM:
                min_EDP_DDM = EDP
    return min_EDP_DDM


def method_GA(workload, PE_num, type, pool):
    min_EDP_GA = 0.0
    while min_EDP_GA < 0.1:
        env = MultiCore(dimensions=[workload], types=[type], model_file="tmp_model.m", fitness='EDP', l1_size=-1, l2_size=-1, \
                        num_pe=PE_num)
        ckpt = env.explore(pool=pool)
        best_sol = ckpt["best_sol"]
        realres = env.get_indiv_info(best_sol, num_pe=PE_num)
        if realres is not None:
            runtime, throughput, energy, area, l1_size, l2_size, mac, power, num_pe = realres[0]
            min_EDP_GA = energy * runtime
    return min_EDP_GA


def method_GA_small(workload, PE_num, type, pool):
    min_EDP_GA_small = 0.0
    while min_EDP_GA_small < 0.1:
        env = MultiCore(dimensions=[workload], types=[type], model_file="tmp_model.m", fitness='EDP', l1_size=-1, l2_size=-1, \
                        num_pe=PE_num)
        ckpt = env.explore(num_population=8, num_generations=4, elite_ratio=0.2, parents_ratio=0.4, ratio_decay=1, 
                            pool=pool)
        best_sol = ckpt["best_sol"]
        realres = env.get_indiv_info(best_sol, num_pe=PE_num)
        if realres is not None:
            runtime, throughput, energy, area, l1_size, l2_size, mac, power, num_pe = realres[0]
            min_EDP_GA_small = energy * runtime
    return min_EDP_GA_small


def method_GA_large(workload, PE_num, type, pool):
    min_EDP_GA_small = 0.0
    while min_EDP_GA_small < 0.1:
        env = MultiCore(dimensions=[workload], types=[type], model_file="tmp_model.m", fitness='EDP', l1_size=-1, l2_size=-1, \
                        num_pe=PE_num)
        ckpt = env.explore(num_population=128, num_generations=64, elite_ratio=0.05, parents_ratio=0.4, ratio_decay=1, 
                            pool=pool)
        best_sol = ckpt["best_sol"]
        realres = env.get_indiv_info(best_sol, num_pe=PE_num)
        if realres is not None:
            runtime, throughput, energy, area, l1_size, l2_size, mac, power, num_pe = realres[0]
            min_EDP_GA_small = energy * runtime
    return min_EDP_GA_small


def method_MLP(workload, PE_num, type, batchsize, pool):
    env = MultiCore(dimensions=[workload], types=[type], model_file="tmp_model.m", fitness='EDP', l1_size=-1, l2_size=-1, \
                    num_pe=PE_num)
    env.set_args(num_population=32, num_generations=16, elite_ratio=0.05, parents_ratio=0.4, ratio_decay=1)
    dataflow_list = gen_dataflow_MLP(workload_list=[workload], PE_num=PE_num, n=batchsize, \
                                     stage2_model_name="./model_file/stage2_MLP.pth")
    # print(dataflow_list)
    PE_num_list = [PE_num] * batchsize
    realres_list = pool.map(env.get_indiv_info_pool, list(zip(dataflow_list, PE_num_list)))
    min_EDP_MLP = float('inf')
    for realres in realres_list:
        if realres is not None:
            runtime, throughput, energy, area, l1_size, l2_size, mac, power, num_pe = realres[0]
            EDP = energy * runtime
            if EDP > 0.1 and EDP < min_EDP_MLP:
                min_EDP_MLP = EDP
    return min_EDP_MLP


def method_Random(workload, PE_num, type, batchsize, pool):
    env = MultiCore(dimensions=[workload], types=[type], model_file="tmp_model.m", fitness='EDP', l1_size=-1, l2_size=-1, \
                    num_pe=PE_num)
    env.set_args(num_population=32, num_generations=16, elite_ratio=0.05, parents_ratio=0.4, ratio_decay=1)
    dataflow_list = []
    for j in range(batchsize):
        rand_idv = env.create_genome()
        dataflow_list.append(rand_idv)
    PE_num_list = [PE_num] * batchsize
    realres_list = pool.map(env.get_indiv_info_pool, list(zip(dataflow_list, PE_num_list)))
    min_EDP_Random = float('inf')
    for realres in realres_list:
        if realres is not None:
            runtime, throughput, energy, area, l1_size, l2_size, mac, power, num_pe = realres[0]
            EDP = energy * runtime
            if EDP > 0.1 and EDP < min_EDP_Random:
                min_EDP_Random = EDP
    return min_EDP_Random




if __name__=='__main__':
    args = setup_args()
    batchsize = args.candidate_N
    method = args.method
    # method：'DDIM', 'DDPM', 'GA', 'GA_small', 'GA_large', 'MLP', 'Random'

    
    setup_seed(123)
    print('candidate_N: ' + str(batchsize))
    print('method: ' + method)
    print()

    dataset_list = read_dataset_test()
    # dataset_list = read_dataset_test(file_name='./dataset_file/dataset_generalization.csv')
    # dataset_list: an array of 4700/1120 networks, with each element shaped as: [[K,C,X,Y,R,S], PE_num, ref_EDP]


    time_method = 0.0
    method_vs_ref_EDP_ratio = 0     # the ratio of min_EDP to ref_EDP for different methods
    method_percent_105_num = 0      # the number of samples with min_EDP within 105% ref_EDP
    method_percent_120_num = 0      # the number of samples with min_EDP within 120% ref_EDP
    method_percent_200_num = 0      # the number of samples with min_EDP within 200% ref_EDP
    method_percent_300_num = 0      # the number of samples with min_EDP within 300% ref_EDP


    pool = Pool(min(batchsize, cpu_count(), 16))

    for i in range(len(dataset_list)):
        workload = dataset_list[i][0]
        PE_num = dataset_list[i][1]
        ref_EDP = dataset_list[i][2]
        print('current idx: ' + str(i))
        print('current workload: ' + str(workload))
        print('current PE_num: ' + str(PE_num))
        type = 'CONV'

        start_time = time.time()
        if method == 'DDIM':
            min_EDP_method = method_DDIM(workload, PE_num, type, batchsize, pool)
        elif method == 'DDPM':
            min_EDP_method = method_DDPM(workload, PE_num, type, batchsize, pool)
        elif method == 'GA':
            min_EDP_method = method_GA(workload, PE_num, type, pool)
        elif method == 'GA_small':
            min_EDP_method = method_GA_small(workload, PE_num, type, pool)
        elif method == 'GA_large':
            min_EDP_method = method_GA_large(workload, PE_num, type, pool)
        elif method == 'MLP':
            min_EDP_method = method_MLP(workload, PE_num, type, batchsize, pool)
        elif method == 'Random':
            min_EDP_method = method_Random(workload, PE_num, type, batchsize, pool)
        else:
            print('error')
            exit(0)
        print("min_EDP_" + method + ":", min_EDP_method)
        end_time = time.time()
        time_method += end_time - start_time

        print("ref_EDP:", ref_EDP)

        if min_EDP_method < ref_EDP * 1.05:
            method_percent_105_num += 1
        if min_EDP_method < ref_EDP * 1.2:
            method_percent_120_num += 1
        if min_EDP_method < ref_EDP * 2.0:
            method_percent_200_num += 1
        if min_EDP_method < ref_EDP * 3.0:
            method_percent_300_num += 1
        
        # The ratio of individual data should be controlled between [0.5, 5]
        method_vs_ref_EDP_ratio += max(0.5, min(min_EDP_method/ref_EDP, 5))

        print()
    pool.close()

    time_method = time_method / len(dataset_list)
    method_vs_ref_EDP_ratio = method_vs_ref_EDP_ratio / len(dataset_list)
    print('avg_time_' + method + ': ' + str(time_method) + ' s')
    print(method + '_vs_ref_EDP_ratio: ' + str(method_vs_ref_EDP_ratio))
    print(method + '_percent_105_num: ' + str(method_percent_105_num))
    print(method + '_percent_120_num: ' + str(method_percent_120_num))
    print(method + '_percent_200_num: ' + str(method_percent_200_num))
    print(method + '_percent_300_num: ' + str(method_percent_300_num))









