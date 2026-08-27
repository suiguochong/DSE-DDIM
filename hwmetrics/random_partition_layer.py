import random

DEBUG = False
MAX_PE = 96

class Random_partition_layer:
    def __init__(self, dimensions, types, num_cores, limit_pe=96):
        self.dimensions = dimensions
        self.types = types
        self.num_cores = num_cores
        self.limit_pe = limit_pe

    def get(self):
        core_dims, core_types = self.partition_layers()
        pelimits = self.partition_pe(core_dims)
        return core_dims, core_types, self.num_cores, pelimits

    def partition_layers(self):
        # 把网络层随机均匀分配到num_cores上
        core_dims = [[] for _ in range(self.num_cores)]
        core_types = [[] for _ in range(self.num_cores)]
        idx = list(range(len(self.dimensions)))
        random.shuffle(idx)
        for i in idx:
            core_dims[i % self.num_cores].append(self.dimensions[i])
            core_types[i % self.num_cores].append(self.types[i])
        return core_dims, core_types


    # 根据计算量来确认PE的数量
    def partition_pe(self, core_dims):
        pelimits = []
        # 根据MACs的比例分配PE, # 如果是最后一个core，直接分配剩余的PE,而且每一个最少一个PE
        macs = [sum([dim[0] * dim[1] * dim[2] * dim[3] for dim in dims]) for dims in core_dims]
        total_macs = sum(macs)
        for idx, dims in enumerate(core_dims):
            mac = macs[idx]
            pelimit = int((MAX_PE - self.num_cores) * mac / total_macs)
            if idx == self.num_cores - 1:
                pelimit = (MAX_PE - self.num_cores) - sum(pelimits)
            pelimits.append(pelimit)
        pelimits = [pe + 1 for pe in pelimits]
        return pelimits