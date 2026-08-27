import os
from multiprocessing import cpu_count
from multiprocessing.pool import Pool
import numpy as np
import copy
import random
from functools import reduce
from collections import defaultdict
from subprocess import Popen, PIPE
import pandas as pd
from math import ceil

script_dir = os.path.dirname(__file__)
workdir = os.path.dirname(os.path.dirname(__file__))
callcnt = 0

class MultiCore:
    def __init__(self, dimensions, types, model_file, num_pe=64, pe_limit=1024, fitness="latency", constraints=dict(),
                 par_RS=False, l1_size=512, l2_size=108000, NocBW=81920000, offchipBW=81920000, slevel_min=2,
                 slevel_max=2, fixedCluster=0, log_level=2, constraint_class=None, external_mem_cstr=None,
                 use_factor=True, uni_base=True) -> None:
        self.dimensions = [dimensions[0]]
        self.evadims = dimensions
        self.types = types
        self.dimension = dimensions[0]
        dimension = dimensions[0]
        self.dimension_dict = {"K": dimension[0], "C": dimension[1], "Y": dimension[2], "X": dimension[3],
                               "R": dimension[4], "S": dimension[5]}
        self.model_file = model_file
        self.backend = os.path.join(script_dir, 'maestro')
        self.num_pe = num_pe
        self.pe_limit = pe_limit
        self.objective = fitness
        self.cluster_space = ["K", "C", "Y", "X"]
        self.dimension_factors = self.get_dimension_factors(dimensions)
        self.slevel_min = slevel_min
        self.slevel_max = slevel_max
        self.use_factor = use_factor
        self.NocBW = NocBW if NocBW > 0 else 2 ** 30
        self.offchipBW = offchipBW if offchipBW > 0 else 2 ** 30
        self.l1_size = l1_size if l1_size > 0 else 2 ** 30
        self.l2_size = l2_size if l2_size > 0 else 2 ** 30
        self.fixedCluster = fixedCluster
        self.history = {}

        # print(f'factors of dimensions: {self.dimension_factors}')

    def get_factors(self, n):
        return set(reduce(list.__add__,
                          ([i, n // i] for i in range(1, int(n ** 0.5) + 1) if n % i == 0)))
                        #   ([n // i] for i in range(1, int(n ** 0.5) + 1) if n % i == 0)))

    def get_dimension_factors(self, dimensions):
        dimension_factors = dict()
        for d in dimensions:
            dimension_dict = {"K": d[0], "C": d[1], "Y": d[2], "X": d[3], "R": d[4], "S": d[5]}
            for key, value in dimension_dict.items():
                factors = self.get_factors(value)
                # cutout too low parallel level
                minv = int(value ** 0.5)
                factors = set([i for i in factors if i >= minv or i == 1]) 
                if key in dimension_factors.keys():
                    dimension_factors[key]["set"] = dimension_factors[key]["set"] & factors
                else:
                    dimension_factors[key] = {"set": factors}
        for key in dimension_factors.keys():
            # print(f'{key} {dimension_factors[key]}')
            dimension_factors[key]["array"] = np.array(list(dimension_factors[key]["set"]))

        return dimension_factors

    def set_args(self, num_population=50, num_generations=25, elite_ratio=0.05, parents_ratio=0.4, ratio_decay=1):
        self.num_generations = num_generations
        self.num_population = num_population
        self.ratio_decay = ratio_decay
        self.num_elite = int(num_population * elite_ratio)
        self.parents_ratio = parents_ratio
        self.ratio_decay = ratio_decay
        self.init_pop = None
        self.best_sol = None
        self.best_reward_list = []
        self.best_reward = -float("Inf")
        self.best_activity = None

    def create_unit_base_pops(self, population, num_all_unit=None):
        if num_all_unit is None:
           num_all_unit = len(population)
        for idx in range(num_all_unit):
            for level in range(len(population[0]) // 7):
                for i in range(1, 7):
                    population[idx][i + level * 7][1] = 1

    def _create_genome(self, uni_base=False, last_cluster_dict=None, l1_bias_template=None):
        if uni_base:
            if l1_bias_template:
                K, C, Y, X, R, S = l1_bias_template
            else:
                K, C, Y, X, R, S = [1] * len(self.dimension)
        else:
            K, C, Y, X, R, S = self.dimension
        if uni_base is False and last_cluster_dict:
            K = last_cluster_dict["K"]
            C = last_cluster_dict["C"]
            Y = last_cluster_dict["Y"]
            X = last_cluster_dict["X"]
            R = last_cluster_dict["R"]
            S = last_cluster_dict["S"]
        sp = random.choice(self.cluster_space)
        lastcluster_sz = last_cluster_dict[sp] if last_cluster_dict else self.dimension_factors[sp]["array"]

        if uni_base == True:
            if self.fixedCluster > 0:
                sp_sz = self.fixedCluster
            else:
                if self.num_pe > 0:
                    sp_sz = random.randint(1, min(lastcluster_sz, self.num_pe))
                else:
                    sp_sz = random.randint(1, lastcluster_sz)
        else:
            sp_sz = random.randint(1, self.num_pe if self.num_pe > 0 else self.pe_limit)
        if self.use_factor and not uni_base:
            df = [["K", np.random.choice(self.dimension_factors["K"]["array"])],
                  ["C", np.random.choice(self.dimension_factors["C"]["array"])],
                  ["Y", np.random.choice(self.dimension_factors["Y"]["array"])],
                  ["X", np.random.choice(self.dimension_factors["X"]["array"])],
                  ["R", np.random.choice(self.dimension_factors["R"]["array"])],
                  ["S", np.random.choice(self.dimension_factors["S"]["array"])]]
        else:
            if uni_base:
                df = [["K", K], ["C", C], ["Y", Y], ["X", X], ["R", R], ["S", S]]
            else:
                df = [["K", random.randint(1, K)], ["C", random.randint(1, C)], ["Y", random.randint(1, Y)],
                      ["X", random.randint(1, X)], ["R", random.randint(1, R)], ["S", random.randint(1, S)]]
        idx = np.random.permutation(len(df))
        indv = [[sp, sp_sz]] + [df[i] for i in idx]
        return indv

    def scan_indv(self, indv):
        last_cluster_dict = defaultdict(str)
        for i in range(len(indv) - 6, len(indv), 1):
            d, d_sz = indv[i]
            last_cluster_dict[d] = d_sz
        return last_cluster_dict

    def born_cluster_ind(self, ind):
        if (len(ind)) // 7 < self.slevel_max:
            last_cluster_dict = self.scan_indv(ind)
            new_ind = ind + self._create_genome(uni_base=True, l1_bias_template=None, last_cluster_dict=last_cluster_dict)
            ind = new_ind
        return ind

    def create_genome(self, bias=None):
        ind = self._create_genome()
        for _ in range(self.slevel_min - 1):
            ind = self.born_cluster_ind(ind)
        return ind

    def init_population(self, pool, num_population, best_sol, init_pop):
        population = [self.create_genome() for _ in range(num_population)]
        self.create_unit_base_pops(population, num_all_unit=2)
        if init_pop is not None:
            population[:10] = init_pop[:10]
        else:
            if best_sol is not None:
                population[0] = best_sol
        self.num_parents = num_population
        self.fitness = np.ones(max(num_population, len(population)), float)
        self.evaluate(pool=pool, population=population)
        return population

    def thread_fun(self, individual):
        indv_hist = self.history.get(str(individual))
        if indv_hist is not None:
            return indv_hist
        reward, activity_count = self.observe_maestro(individual)
        self.history[str(individual)] = [reward, activity_count]
        return [reward, activity_count]

    def write_maestro(self, indv, layer_id=0, m_file=None):
        dimensions = self.dimensions
        with open("{}.m".format(m_file), "w") as fo:
            fo.write("Network {} {{\n".format(layer_id))
            for i in range(len(dimensions)):
                dimension = dimensions[i]
                m_type = self.types[i]
                fo.write("Layer {} {{\n".format(m_type))
                fo.write("Type: {}\n".format(m_type))
                fo.write(
                    "Dimensions {{ K: {:.0f}, C: {:.0f}, Y: {:.0f}, X: {:.0f}, R: {:.0f}, S: {:.0f} }}\n".format(*dimension)
                )
                fo.write("Dataflow {\n")
                for k in range(0, len(indv), 7):
                    for i in range(k, k + 7):
                        if len(indv[i]) == 2:
                            d, d_sz = indv[i]
                        else:
                            d, d_sz, _ = indv[i]

                        # R or S index
                        if d == 'R':
                            d_sz = dimension[4]
                        elif d == 'S':
                            d_sz = dimension[5]

                        if i % 7 == 0:
                            if k != 0:
                                fo.write("Cluster({},P);\n".format(d_sz))
                        else:
                            sp = "SpatialMap" if d == indv[k][0] or (
                                    len(indv[k]) > 2 and d == indv[k][2]) else "TemporalMap"
                            # MAESTRO cannot take K dimension as dataflow file
                            if not (m_type == "DSCONV"):
                                fo.write("{}({},{}) {};\n".format(sp, d_sz, d_sz, self.get_out_repr(d)))
                            else:
                                if self.get_out_repr(d) == "C" and self.get_out_repr(indv[k][0]) == "K":
                                    fo.write("{}({},{}) {};\n".format("SpatialMap", d_sz, d_sz, "C"))
                                else:
                                    if not (self.get_out_repr(d) == "K"):
                                        fo.write("{}({},{}) {};\n".format(sp, d_sz, d_sz, self.get_out_repr(d)))

                fo.write("}\n")
                fo.write("}\n")
            fo.write("}")

    def get_out_repr(self, x):
        if x in {"K", "C", "R", "S"}:
            return x
        else:
            return x + "'"

    def judge(self):
        runtime, throughput, energy, area, l1_size, l2_size, mac, power, num_pe = self.observation[0]

        def get_objective(objective):
            term = objective
            if term == "energy":
                reward = -energy
            elif term == "thrpt_ave":
                reward = throughput
            elif term == "EDP":
                reward = -energy * runtime
            elif term == "LAP":
                reward = -area * runtime
            elif term == "LALP":
                reward = -area * runtime * runtime
            elif term == "EAP":
                reward = -area * energy
            elif term == "thrpt" or term == "thrpt_naive":
                reward = throughput
            elif term == "thrpt_btnk":
                reward = throughput
            elif term == "latency":
                reward = -runtime
            elif term == "area":
                reward = -area
            elif term == "l1_size":
                reward = - l1_size
            elif term == "l2_size":
                reward = -l2_size
            elif term == "power":
                reward = -power
            elif term == "ranking":
                reward = -1
            elif term == "L-PE-L2":
                reward = -runtime * num_pe * l2_size
            elif term == "L-PE":
                reward = -runtime * num_pe
            elif term == "PE":
                reward = -num_pe
            else:
                raise NameError(f'Undefined fitness type: {term}')
            return reward
        values = get_objective(self.objective)
        return values

    def compute_area_external(self, num_pe, l1_size, l2_size):
        MAC_AREA_INT8 = 282
        MAC_AREA_INT32 = 3495
        BUF_AREA_perbit = 0.086
        buf_size = l1_size * num_pe + l2_size
        area = num_pe * MAC_AREA_INT8 + buf_size * BUF_AREA_perbit * 8
        return area

    def observe_maestro(self, indv, num_pe=None, l1_size=None, l2_size=None, NocBW=None, offchipBW=None):

        m_file = "{}".format(random.randint(0, 2 ** 32))
        self.write_maestro(indv, m_file=m_file)
        if num_pe:
            to_use_num_pe = num_pe
        elif self.num_pe < 1:
            to_use_num_pe = indv[0][1]
        else:
            to_use_num_pe = self.num_pe
        csvpath = os.path.join(workdir, m_file + ".csv")
        os.remove(csvpath) if os.path.exists(csvpath) else None
        command = [self.backend,
                   "--Mapping_file={}.m".format(m_file),
                   "--full_buffer=false",
                   "--noc_bw_cstr={}".format(self.NocBW if not NocBW else NocBW),
                   "--noc_hops=1",
                   "--noc_hop_latency=1",
                   "--offchip_bw_cstr={}".format(self.offchipBW if not offchipBW else offchipBW),
                   "--noc_mc_support=true",
                   "--num_pes={}".format(int(to_use_num_pe)),
                   "--num_simd_lanes=1",
                   "--l1_size_cstr={}".format(self.l1_size if not l1_size else l1_size),
                   "--l2_size_cstr={}".format(self.l2_size if not l2_size else l2_size),
                   "--print_res=false",
                   "--print_res_csv_file=true",
                   "--print_log_file=false",
                   "--print_design_space=false",
                   "--msg_print_lv=0"
        ]

        process = Popen(command, stdout=PIPE, stderr=PIPE)
        stdout, stderr = process.communicate()
        process.wait()
        mfilepath = os.path.join(workdir, m_file + ".m")
        os.remove(mfilepath) if os.path.exists(mfilepath) else None
        try:
            df = pd.read_csv(csvpath)
            layer_name = df[" Layer Number"]
            max_pip = np.array(np.max(df[" Runtime (Cycles)"])).reshape(-1, 1)
            runtime = np.array(np.sum(df[" Runtime (Cycles)"])).reshape(-1, 1)
            runtime_series = np.array(df[" Runtime (Cycles)"]).reshape(-1, 1)
            throughput = np.array(df[" Throughput (MACs/Cycle)"]).reshape(-1, 1)
            energy = np.array(np.sum(df[" Activity count-based Energy (nJ)"])).reshape(-1, 1)
            area = np.array(np.mean(df[" Area"])).reshape(-1, 1)
            power = np.array(np.sum(df[" Power"])).reshape(-1, 1)
            l1_size = np.array(np.max(df[" L1 SRAM Size Req (Bytes)"])).reshape(-1, 1)
            l2_size = np.array(np.max(df["  L2 SRAM Size Req (Bytes)"])).reshape(-1, 1)
            l1_size_series = np.array(df[" L1 SRAM Size Req (Bytes)"]).reshape(-1, 1)
            l2_size_series = np.array(df["  L2 SRAM Size Req (Bytes)"]).reshape(-1, 1)
            l1_input_read = np.array(df[" input l1 read"]).reshape(-1, 1)
            l1_input_write = np.array(df[" input l1 write"]).reshape(-1, 1)
            l1_weight_read = np.array(df["filter l1 read"]).reshape(-1, 1)
            l1_weight_write = np.array(df[" filter l1 write"]).reshape(-1, 1)
            l1_output_read = np.array(df["output l1 read"]).reshape(-1, 1)
            l1_output_write = np.array(df[" output l1 write"]).reshape(-1, 1)
            l2_input_read = np.array(df[" input l2 read"]).reshape(-1, 1)
            l2_input_write = np.array(df[" input l2 write"]).reshape(-1, 1)
            l2_weight_read = np.array(df[" filter l2 read"]).reshape(-1, 1)
            l2_weight_write = np.array(df[" filter l2 write"]).reshape(-1, 1)
            l2_output_read = np.array(df[" output l2 read"]).reshape(-1, 1)
            l2_output_write = np.array(df[" output l2 write"]).reshape(-1, 1)
            mac = np.array(df[" Num MACs"]).reshape(-1, 1)
            activity_count = {}
            activity_count["l1_input_read"] = l1_input_read
            activity_count["l1_input_write"] = l1_input_write
            activity_count["l1_weight_read"] = l1_weight_read
            activity_count["l1_weight_write"] = l1_weight_write
            activity_count["l1_output_read"] = l1_output_read
            activity_count["l1_output_write"] = l1_output_write
            activity_count["l2_input_read"] = l2_input_read
            activity_count["l2_input_write"] = l2_input_write
            activity_count["l2_weight_read"] = l2_weight_read
            activity_count["l2_weight_write"] = l2_weight_write
            activity_count["l2_output_read"] = l2_output_read
            activity_count["l2_output_write"] = l2_output_write
            activity_count["mac_activity"] = mac
            os.remove(csvpath) if os.path.exists(csvpath) else None
            logpath = os.path.join(workdir, "log.txt")
            os.remove(logpath) if os.path.exists(logpath) else None
            # if self.external_area_model:
            area = self.compute_area_external(to_use_num_pe, l1_size, l2_size)
            # elif self.area_pebuf_only:
            #     area = self.compute_area_maestro(to_use_num_pe, l1_size, l2_size)
            self.observation = [
                [np.mean(x) for x in [runtime, throughput, energy, area, l1_size, l2_size, mac, power, to_use_num_pe]],
                runtime_series]

            def catch_exception():
                if l1_size > self.l1_size or l2_size > self.l2_size or any(runtime_series < 1) or any(
                        l1_size_series < 1) or any(l2_size_series < 1):
                    return True
                else:
                    return False

            stdout_as_str = stdout.decode("utf-8")
            stdout_as_str = "".join(stdout_as_str.split())
            # if (len(str(stdout))>3 and stdout_as_str[:len("Numpartialsumsislessthan0!")]!="Numpartialsumsislessthan0!") or catch_exception() or not self.validTo_external_mem_cstr(indv, num_pe=to_use_num_pe):
            # if len(str(stdout))>3  or catch_exception() or not self.validTo_external_mem_cstr(indv, num_pe=to_use_num_pe):
            if catch_exception():
                # if  catch_exception():
                return None, None
            # print(f'indv {indv}')
            return self.judge(), activity_count
        except:
            return None, None

    def evaluate(self, pool, population):
        gen_best = -float("Inf")
        gen_best_activity = None
        gen_best_idx = 0
        count_non_valid = 0
        valid = 0
        reward_activ_list = pool.map(self.thread_fun, population)

        self.fitness = np.ones(len(population), float)
        for i in range(len(population)):
            reward, activity_count = reward_activ_list[i]
            if reward is None:
                reward = float("-Inf")
                count_non_valid += 1
            else:
                valid += 1
            judging_reward = reward
            self.fitness[i] = reward
            if gen_best < judging_reward:
                gen_best = judging_reward
                gen_best_activity = activity_count
                gen_best_idx = i
        judging_best_reward = self.best_reward
        # print(f'valid: {valid}, not valid: {count_non_valid}')
        # self.cal_statstics()

        if judging_best_reward < gen_best:
            self.best_reward = copy.deepcopy(self.fitness[gen_best_idx])
            self.best_activity = copy.deepcopy(gen_best_activity)
            self.best_sol = copy.deepcopy(population[gen_best_idx])

        self.best_reward_list.append(self.best_reward)
        self.num_parents = int(self.num_population * self.parents_ratio)
        self.num_parents = min(self.num_parents, len(population) - count_non_valid)
        self.parents_ratio *= self.ratio_decay

        chkpt = {
            "best_activity": self.best_activity,
            "best_reward": self.best_reward,
            "best_reward_list": self.best_reward_list,
            "best_sol": self.best_sol,
            "num_population": self.num_population,
            "num_generations": self.num_generations,
            "fitness_use": self.objective,
            "num_pe": self.num_pe,
            "pe_limit": self.pe_limit,
            "l1_size": self.l1_size,
            "l2_size": self.l2_size,
            "NocBW": self.NocBW,
            "dimension": self.dimension,
        }

        return chkpt

    def select_parents(self, pop, fitness, num_parents, num_population):
        # =====sel unique======================
        pop_set = set()
        to_saved_idx = []
        for i in range(len(pop)):
            cur_cand = tuple([tt for i, t in enumerate(pop[i]) for j, tt in enumerate(t) if (i, j) != (0, 1)])
            if cur_cand not in pop_set:
                pop_set.add(cur_cand)
                to_saved_idx.append(i)
        fitness = fitness[to_saved_idx]
        pop = [pop[i] for i in range(len(pop)) if i in set(to_saved_idx)]

        fitness_list = [tuple([fitness[i]] + [-i]) for i in range(len(fitness))]
        fitness_list = sorted(fitness_list, reverse=True)
        idx = [int(-ar[-1]) for ar in fitness_list]
        new_pop = [pop[i] for i in idx][:num_population]
        new_fitness = fitness[idx][:num_population]
        parents = copy.deepcopy(new_pop[:num_parents])
        return new_pop, new_fitness, parents

    def correctify_tile_dependency(self, pop):
        for i in range(0, len(pop)):
            ind = pop[i]
            cur_cluster = None
            levels = len(ind) // 7
            for i in range(levels):
                last_cluster = copy.deepcopy(cur_cluster)
                cur_cluster = self.scan_indv(ind[7 * i:7 * (i + 1)])
                if i == 0:
                    continue
                else:
                    for idx in range(7 * i + 1, 7 * (i + 1)):
                        d, d_sz = ind[idx]
                        d_sz = min(last_cluster[d], d_sz)
                        ind[idx][1] = d_sz

    def get_indiv_info(self, individual, num_pe=None, l1_size=None, l2_size=None, NocBW=None):
        self.dimensions = self.evadims
        self.observation = None
        self.observe_maestro(individual, num_pe=num_pe, l1_size=l1_size, l2_size=l2_size, NocBW=NocBW)
        return self.observation

    # for multi-thread operations
    def get_indiv_info_pool(self, arg):
        individual, num_pe = arg
        l1_size = None
        l2_size = None
        NocBW = None
        # print(individual, num_pe)
        self.dimensions = self.evadims
        self.observation = None
        self.observe_maestro(individual, num_pe=num_pe, l1_size=l1_size, l2_size=l2_size, NocBW=NocBW)
        return self.observation

    # Set pool as external input
    def explore(self, num_population=32, num_generations=16, elite_ratio=0.05, parents_ratio=0.4, ratio_decay=1, \
                pool = None):
        if pool == None:
            pool_tag = 0
        else:
            pool_tag = 1

        self.set_args(num_population, num_generations, elite_ratio, parents_ratio, ratio_decay)
        if pool_tag == 0:
            pool = Pool(min(self.num_population + self.num_elite, cpu_count()))
        population = self.init_population(pool, self.num_population, self.best_sol, self.init_pop)
        ckpt = None

        for g in range(num_generations):
            while self.num_parents < 1:
                print(f"Reinitialize population: {self.num_parents}")
                population = self.init_population(pool, self.num_population, self.best_sol, self.init_pop)

            population, self.fitness, self.parents = self.select_parents(population, self.fitness, self.num_parents,
                                                                         self.num_population)
            elite = copy.deepcopy(self.parents[:self.num_elite])
            self.elite_fitness = copy.deepcopy(self.fitness[:(len(elite))])

            # TODO multi core constrain
            self.crossover_tile(self.parents, population, alpha=0.57)
            self.swap_order(population, alpha=0.47)
            self.mutate_tile(population, num_mu_loc=3, range_alpha=0.53, alpha=0.53, is_finetune=False)
            self.mutate_pe(population, alpha=1 if g == 0 else 0.5) if self.num_pe < 1 else None
            self.mutate_par(population, alpha=0.1)

            self.born_cluster(population, alpha=0.57)
            self.kill_cluster(population, alpha=0.27)

            self.correctify_tile_dependency(population)
            population = elite + population

            chkpt = self.evaluate(pool=pool, population=population)

            if len(self.best_reward_list) > 3 and self.best_reward_list[-1]-self.best_reward_list[-3] < 1:
                break

        if pool_tag == 0:
            pool.close()
        return chkpt

    def kill_cluster(self, pop, alpha=0.5):
        max_count = len(pop)
        while max_count > 0:
            max_count -= 1
            if random.random() < alpha:
                idx = random.randint(0, len(pop) - 1)
                if (len(pop[idx])) // 7 > self.slevel_min:
                    pop[idx] = pop[idx][:-7]

    def born_cluster(self, pop, alpha=0.1):
        max_count = len(pop)
        while max_count > 0:
            max_count -= 1
            if random.random() < alpha:
                idx = random.randint(0, len(pop) - 1)
                ind = self.born_cluster_ind(pop[idx])
                pop[idx] = ind

    def mutate_par(self, pop, alpha=0.5):
        # if self.map_cstr is not None:
        #     return
        for idx in range(len(pop)):
            if random.random() < alpha:
                # if self.map_cstr is not None:
                #     avail_val = self.num_free_par + self.num_free_order - 1
                # else:
                #     avail_val = len(indv) - 1
                # ##===ad hoc trial=========
                pop[idx][7][0], pop[idx][0][0] = pop[idx][0][0], pop[idx][7][0]
                continue
                # #=========================
                pick = random.randint(0, avail_val)
                pick_level = pick // 7
                pick = int(pick_level * 7)
                if self.map_cstr and "sp" in self.cstr_list[pick_level]:
                    choices = self.cstr_list[pick_level]["sp"]
                else:
                    choices = self.cluster_space
                sp = random.choice(choices)
                if self.map_cstr and "sp_sz" in self.cstr_list[pick_level]:
                    sp_sz = self.self.cstr_list[pick_level]["sp_sz"]
                else:
                    if self.fixedCluster < 1:
                        last_cluster_dict = self.scan_indv(indv[:-7]) if pick != 0 else None
                        lastcluster_sz = last_cluster_dict[sp] if last_cluster_dict else self.dimension_dict[sp]
                        sp_sz = random.randint(1, min(lastcluster_sz, self.num_pe))
                    else:
                        sp_sz = self.fixedCluster
                pop[idx][pick] = [sp, sp_sz]

    def mutate_tile(self, pop, is_finetune=False, num_mu_loc=1, alpha=0.5, range_alpha=0.5, cluster_only=False):
        for idx in range(len(pop)):
            indv = pop[idx]
            for _ in range(num_mu_loc):
                if random.random() < alpha:
                    # if self.map_cstr:
                    #     num_free_tile = self.cstr_list[1]["num_free_tile"]
                    #     if num_free_tile==0:
                    #         pick = random.randint(0, len(indv) - 6 - 1)
                    #     else:
                    #         pick = random.randint(0, len(indv) - 1)
                    # else:
                    pick = random.randint(0, len(indv) - 1)
                    if cluster_only:
                        pick = 7
                    if pick % 7 == 0:
                        # if  self.map_cstr  and "sp" in self.cstr_list[pick // 7]:
                        #     choices = self.cstr_list[pick // 7]["sp"]
                        # else:
                        choices = self.cluster_space
                        sp = random.choice(choices)
                        if pick > 0:
                            # if self.map_cstr  and "sp_sz" in self.cstr_list[pick // 7]:
                            #     sp_sz = self.cstr_list[pick // 7]["sp_sz"]
                            # else:
                            if self.fixedCluster < 1:
                                last_cluster_dict = self.scan_indv(indv[:-7]) if pick != 0 else None
                                lastcluster_sz = last_cluster_dict[sp] if last_cluster_dict else self.dimension_dict[sp]
                                if self.num_pe > 0:
                                    # sp_sz = max(1, random.randint(0, min(lastcluster_sz, self.num_pe)))
                                    sp_sz = max(1,
                                                random.choice(list(self.get_factors(min(lastcluster_sz, self.num_pe)))))
                                else:
                                    # sp_sz = max(1, random.randint(0, min(lastcluster_sz, indv[0][1])))
                                    sp_sz = max(1,
                                                random.choice(list(self.get_factors(min(lastcluster_sz, indv[0][1])))))
                            else:
                                sp_sz = self.fixedCluster
                        else:
                            sp_sz = pop[idx][pick][1]
                        pop[idx][pick] = [sp, sp_sz]
                    else:
                        d, d_sz = indv[pick]
                        if pick > 7:
                            last_cluster_dict = self.scan_indv(indv[:-7])
                            thr = last_cluster_dict[d]
                            if self.use_factor is False:
                                new_d_sz = random.randint(1, thr)
                            else:
                                choices = self.get_factors(thr)
                                new_d_sz = np.random.choice(list(choices))

                        else:
                            if self.use_factor is False:
                                thr = self.dimension_dict[d]
                                new_d_sz = random.randint(1, thr)
                            else:
                                new_d_sz = np.random.choice(self.dimension_factors[d]["array"])
                        if is_finetune:
                            sampling = np.random.uniform(-range_alpha, range_alpha, 1)
                            sampling = int(sampling * thr)
                            new_d_sz = d_sz + sampling
                            new_d_sz = max(1, min(new_d_sz, self.dimension_dict[d]))
                        pop[idx][pick][1] = new_d_sz

    def mutate_pe(self, pop, alpha=0.5, mutate_range_ratio=0.5):
        for idx in range(len(pop)):
            if len(pop[idx]) <= 7:
                if random.random() < alpha:
                    pop[idx][0][1] = random.randint(1, self.pe_limit)
            else:
                sp, sp_sz, *a = pop[idx][7]
                cur_multiplier = pop[idx][0][1] // sp_sz
                if random.random() < alpha:
                    if self.use_factor is False:
                        # ==method 1
                        last_cluster_dict = self.scan_indv(pop[idx][:7])
                        last_cluster_dict_sz = last_cluster_dict[sp]
                        max_multiplier = max(1, self.pe_limit // sp_sz)
                        cur_multiplier = random.randint(1, min(max_multiplier,
                                                               ceil(self.dimension_dict[sp] / last_cluster_dict_sz)))
                        # ====constrained to smaller search space====
                        max_value = min(max_multiplier, ceil(self.dimension_dict[sp] / last_cluster_dict_sz))
                        cur_multiplier = random.randint(max(1, int(max_value * mutate_range_ratio)), max_value)
                        # ============================================
                    else:
                        # method 2
                        factors = self.dimension_factors[sp]["array"]
                        max_multiplier = max(1, self.pe_limit // sp_sz)
                        factors = factors[(factors <= max_multiplier)]
                        cur_multiplier = random.choice(factors)
                        # ====constrained to smaller search space====
                        cur_multiplier = random.choice(factors[int(len(factors) * mutate_range_ratio):])
                        # ============================================
                cur_pe = min(self.pe_limit, cur_multiplier * sp_sz)
                pop[idx][0][1] = cur_pe
                # pop[idx][7][1] = sp_sz

    def swap_order(self, pop, alpha=0.5):
        max_count = len(pop)
        while max_count > 0:
            max_count -= 1
            if random.random() < alpha:
                idx = random.randint(0, len(pop) - 1)
                sel_cluster = random.randint(0, (len(pop[idx]) - 1) // 7)
                swap_id = np.random.randint(1, 6 + 1, (2,)) + sel_cluster * 7
                pop[idx][swap_id[0]], pop[idx][swap_id[1]] = pop[idx][swap_id[1]], pop[idx][swap_id[0]]

    def crossover_tile(self, parents, pop, alpha=0.5):
        if len(parents) == 1:
            for idx in range(len(pop)):
                pop[idx] = copy.deepcopy(parents[0])
        else:
            for idx in range(0, len(pop), 2):
                pick_range = np.random.permutation(np.arange(0, len(parents)))
                dad, mom = parents[pick_range[0]], parents[pick_range[1]]
                # dad, mom = parents[random.randint(0, len(parents)-1)], parents[random.randint(0, len(parents)-1)]
                dad = copy.deepcopy(dad)
                mom = copy.deepcopy(mom)
                length = min(len(dad), len(mom))
                if random.random() < alpha:
                    cross_point = random.choice(["K", "C", "Y", "X", "R", "S"])
                    for k in range(0, length, 7):
                        for i in range(k + 1, k + 7):
                            d, d_sz = dad[i]
                            if d == cross_point:
                                dad_sz = d_sz
                                dad_idx = i
                            d, d_sz = mom[i]
                            if d == cross_point:
                                mom_sz = d_sz
                                mom_idx = i
                        dad[dad_idx][1] = mom_sz
                        mom[mom_idx][1] = dad_sz
                pop[idx] = dad
                if idx + 1 < len(pop):
                    pop[idx + 1] = mom
