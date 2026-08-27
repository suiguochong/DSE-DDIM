from .frontend.helpers.torch_maestro_summary import summary
import torch
import argparse
import numpy as np
import random
import torch.nn as nn
import re
import os
import time
from subprocess import Popen, PIPE
import pandas as pd
from hwmetrics.multicore import MultiCore
from .random_partition_layer import Random_partition_layer
from .cluster_partition_layer import Cluster_partition_layer
import copy

MAC_AREA_MAESTRO = 4470
MAC_AREA_INT8 = 282
BUF_AREA_perbit = 0.086
L2BUF_AREA_MAESTRO = 4161.536
L1BUF_AREA_MAESTRO = 4505.1889
L2BUF_UNIT = 32768
L1BUF_UNIT = 64
LOG_INFO = False

# TODO 
# define hardware constrains here
# can be optimized to cmd line
MAX_PE = 96


script_dir = os.path.dirname(__file__)


def hwperf(model, input_shape, batchsize, num_cores, choice='random'):

    dimensions, types, strides = dump2modelf(model, input_shape, batch=batchsize)


    cluster_st = time.time()

    if choice == 'kmeans':
        core_dims, core_types, num_cores, pelimits = Cluster_partition_layer(dimensions=dimensions, types=types, limit_pe=MAX_PE, algorithm='kmeans', num_cores=num_cores).get()
    elif choice == 'hierarchical':
        core_dims, core_types, num_cores, pelimits = Cluster_partition_layer(dimensions=dimensions, types=types, limit_pe=MAX_PE, algorithm='hierarchical', num_cores=num_cores).get()
    elif choice == 'dbscan':
        core_dims, core_types, num_cores, pelimits = Cluster_partition_layer(dimensions=dimensions, types=types, limit_pe=MAX_PE, algorithm='dbscan', num_cores=num_cores, strides=strides).get()
    elif choice == 'random':
        core_dims, core_types, num_cores, pelimits = Random_partition_layer(dimensions=dimensions, types=types, limit_pe=MAX_PE, num_cores=num_cores).get()

    ckpts = []
    runtime_sum = 0
    max_runtime_interval = 0
    total_energy = 0
    total_area = 0

    rerangeidx = Cluster_partition_layer(dimensions=dimensions, types=types, strides=strides).rerange(core_dims, core_types)

    cal = 0
    for cluster in core_dims:
        cal += len(cluster)

    core_dims = [[dim[:6] for dim in cluster] for cluster in core_dims]

    # equal assign computing resources
    for i, (dims, types) in enumerate(zip(core_dims, core_types)):
        start_time = time.time()
        if len(dims) == 0: continue
        pelimit = pelimits[i]
        realres = None

        # incase representitive sol cannot fit all dims
        while realres is None:
            # search in representitive dim
            env = MultiCore(dimensions=dims, types=types, model_file="tmp_model.m", fitness='EDP', l1_size=-1, l2_size=-1,
                            num_pe=pelimit)
            st = time.time()
            ckpt = env.explore()
            best_sol = ckpt["best_sol"]
            
            # evaluate in all dims
            st = time.time()
            realres = env.get_indiv_info(best_sol, num_pe=pelimit)

        best_runtime, best_throughput, best_energy, best_area, best_l1_size, best_l2_size, best_mac, best_power, best_num_pe = realres[0]
        
        ckpts.append({
            "reward": ckpt['best_reward'],
            "best_sol": best_sol,
            "runtime": best_runtime,
            "area": best_area,
            "pe_area_ratio": best_num_pe * MAC_AREA_INT8 / best_area,
            "PE": best_num_pe,
            "PE_area": best_num_pe * MAC_AREA_INT8,
            "L1_area": best_l1_size * best_num_pe * BUF_AREA_perbit * 8,
            "L2_area": best_l2_size * BUF_AREA_perbit * 8,
            "L1_size": best_l1_size,
            "L2_size": best_l2_size,
            "energy": best_energy,
            "EDP": best_runtime * best_energy,
            "throughput": best_throughput
        })

        runtime_sum += best_runtime
        max_runtime_interval = max(max_runtime_interval, best_runtime)
        total_energy += best_energy
        total_area += best_area

    ckpt = {
        "energy": total_energy,
        "runtime": max_runtime_interval,
        "EDP": total_energy * max_runtime_interval,
        "area": total_area
    }

    return ckpt


def compute_area_external(num_pe, l1_size, l2_size):
    MAC_AREA_INT8 = 282
    MAC_AREA_INT32 = 3495
    BUF_AREA_perbit = 0.086
    buf_size = l1_size * num_pe + l2_size
    area = num_pe * MAC_AREA_INT8 + buf_size * BUF_AREA_perbit * 8
    return area


def dump2mapf(df, mapfile=None):
    if mapfile:
        m_file = mapfile
    else:
        m_file = "tmp_mapping.m"

    dsconv = 0
    with open(os.path.join(script_dir, 'frontend/dataflow', df + ".m"), "r") as fd:
        with open(os.path.join(script_dir, 'frontend/dataflow', "dpt.m"), "r") as fdpt:
            with open(os.path.join(script_dir, 'output', m_file), "w") as fo:
                with open(os.path.join(script_dir, 'output', "tmp_model.m"), "r") as fm:
                    for line in fm:
                        if (re.search("DSCONV", line)):
                            dsconv = 1
                        if (re.search("Dimensions", line)):
                            fo.write(line)
                            if (dsconv):
                                fdpt.seek(0)
                                fo.write(fdpt.read())
                            else:
                                fd.seek(0)
                                fo.write(fd.read())
                            dsconv = 0
                        else:
                            fo.write(line)


def dump2modelf(model, input_shape, batch=2) -> list:
    mae_summary = summary(model, input_shape, batch_size=batch)
    # print("mae_summary:------------------------\n", mae_summary)
    dimensions = []
    types = []
    strides = []
    with open(os.path.join(script_dir, 'output', 'tmp_model.m'), "w") as fo:
        fo.write("Network {} {{\n".format(model.__module__))
        for key, val in mae_summary.items():
            pc = re.compile("^Conv")
            pl = re.compile("^Linear")

            fo.write("Layer {} {{\n".format(key))
            type = val["type"]
            fo.write("Type: {}\n".format(type))
            types.append(type)
            if val["stride"]:
                fo.write("Stride {{ X: {}, Y: {} }}\n".format(*val["stride"]))
                strides.append(val["stride"][0])
            else:
                strides.append(1)
            fo.write("Dimensions {{ K: {}, C: {}, R: {}, S: {}, Y: {}, X: {} }}\n".format(
                *val["dimension_ic"][1:]))
            fo.write("}\n")
            dimension = [*val["dimension_ic"][1:]]
            # change to KCYXRS
            dimension[2:4], dimension[4:6] = dimension[4:6], dimension[2:4],
            dimensions.append(dimension)

        fo.write("}")
    return dimensions, types, strides
