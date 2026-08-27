import numpy as np
import torch
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

DEBUG = False
MAX_PE = 512


class Cluster_partition_layer:
    def __init__(self, dimensions, types, strides=None, num_cores=-1, limit_pe=96, algorithm='kmeans'):
        self.dimensions = dimensions
        self.types = types
        self.num_cores = num_cores
        self.strides = strides
        self.limit_pe = limit_pe
        self.algorithm = algorithm  # kmeans, hierarchical, dbscan

        self.layer_features = []

        for i in range(len(self.dimensions)):
            K, C, Y, X, R, S = dimensions[i][:6]
            stride = self.strides[i]
            self.layer_features.append([
                K,C,R,S,X,Y
            ])

    def get(self):
        core_dims, core_types, self.num_cores = self.partition_layers()

        min_samples = len(self.layer_features) // 16
        min_samples = 1
        self.rerange(core_dims, core_types)

        def append_near(d, nd, t, nt):
            distance = []
            for id in nd:
                dis = torch.sum(torch.as_tensor(d[0]) - torch.as_tensor(id[0]))
                if dis != 0:
                    distance.append(dis)
                else:
                    distance.append(1e12)
            i = torch.argmin(torch.as_tensor(distance))
            nd[i] += d
            nt[i] += t
        
        notfit = True
        while notfit:
            notfit = False
            i = 0
            for dims,types in zip(core_dims,core_types):
                if len(dims) < min_samples:
                    notfit = True
                    append_near(dims, core_dims, types, core_types)
                    core_dims.pop(i)
                    core_types.pop(i)
                    break
                i += 1

        self.num_cores = len(core_dims)

        pelimits = self.partition_pe(core_dims)
        return core_dims, core_types, self.num_cores, pelimits

    def partition_layers(self):
        core_dims = []
        core_types = []

        layer_features = np.array(self.layer_features)
        if self.algorithm == 'kmeans':
            layer_assignment, num_cores = self.kmeans_partition_layers(layer_features, num_cores_range=(
                self.num_cores, self.num_cores) if self.num_cores > 0 else (2, 32))
        elif self.algorithm == 'hierarchical':
            layer_assignment, num_cores = self.hierarchical_partition_layers(layer_features, num_cores_range=(
                self.num_cores, self.num_cores) if self.num_cores > 0 else (2, 32))
        elif self.algorithm == 'dbscan':
            layer_assignment, num_cores = self.dbscan_partition_layers(layer_features, eps=0.5)

        # 根据聚类结果进行层划分
        for core in range(num_cores):
            idx = np.where(layer_assignment == core)[0]
            core_dims.append([self.dimensions[item] for item in idx])
            core_types.append([self.types[item] for item in idx])

        return core_dims, core_types, num_cores

    # k-means
    def kmeans_partition_layers(self, layer_features, num_cores_range):
        # 对层特征进行标准化
        scaler = StandardScaler()
        layer_features_normalized = scaler.fit_transform(layer_features)

        silhouette_scores = []
        K_range = range(num_cores_range[0], min(num_cores_range[1], len(layer_features)) + 1, 2)

        for k in K_range:
            print(f'k range: {k}')
            kmeans = KMeans(n_clusters=k).fit(layer_features_normalized)
            score = silhouette_score(layer_features_normalized, kmeans.labels_)
            silhouette_scores.append(score)
            if DEBUG: print(f"k={k}, 轮廓系数={score:.4f}")

        # 选择最佳簇数
        best_k = K_range[np.argmax(silhouette_scores)]
        kmeans = KMeans(n_clusters=best_k, random_state=0).fit(layer_features_normalized)

        return kmeans.labels_, best_k

    # hierarchical clustering
    def hierarchical_partition_layers(self, layer_features, num_cores_range):

        scaler = StandardScaler()
        layer_features_normalized = scaler.fit_transform(layer_features)

        silhouette_scores = []
        K_range = range(num_cores_range[0], min(num_cores_range[1], len(layer_features_normalized)))

        for k in K_range:
            clustering = AgglomerativeClustering(n_clusters=k).fit(layer_features_normalized)
            score = silhouette_score(layer_features_normalized, clustering.labels_)
            silhouette_scores.append(score)
            if DEBUG: print(f"k={k}, 轮廓系数={score:.4f}")

        # 选择最佳簇数
        best_k = K_range[np.argmax(silhouette_scores)]
        clustering = AgglomerativeClustering(n_clusters=best_k).fit(layer_features_normalized)

        return clustering.labels_, best_k

    # DBSCAN
    def dbscan_partition_layers(self, layer_features, eps, min_samples=1):
        # min_samples = len(layer_features)//4
        # scaler = StandardScaler()
        scaler = MinMaxScaler()
        layer_features_normalized = scaler.fit_transform(layer_features)

        dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(layer_features_normalized)
        core_labels = dbscan.labels_

        # 计算轮廓系数
        if len(set(core_labels)) > 1 and len(set(core_labels)) < len(layer_features_normalized):
            score = silhouette_score(layer_features_normalized, core_labels)
            if DEBUG: print(f"DBSCAN 轮廓系数={score:.4f}")
        else:
            if DEBUG: print("DBSCAN 未能形成有效簇")
        # FIXME: 聚类数等于簇的数量（忽略噪声点 -1）
        num_cores = len(set(core_labels)) - (1 if -1 in core_labels else 0)
        return core_labels, num_cores

    # 根据计算量来确认PE的数量
    def partition_pe(self, core_dims):
        pelimits = []
        # 根据MACs的比例分配PE, # 如果是最后一个core，直接分配剩余的PE,而且每一个最少一个PE
        # macs = [sum([dim[0] * dim[1] * dim[2] * dim[3] for dim in dims]) for dims in core_dims]
        macs = [sum([dim[0] * dim[1] * dim[2] * dim[3] * dim[4] * dim[5] for dim in dims]) for dims in core_dims]
        total_macs = sum(macs)
        for idx, dims in enumerate(core_dims):
            mac = macs[idx]
            pelimit = int((MAX_PE - self.num_cores) * mac / total_macs)
            if (mac/total_macs) - int(mac/total_macs)>0.5:
                pelimit += 1
            if idx == self.num_cores - 1:
                pelimit = (MAX_PE - self.num_cores) - sum(pelimits)
            pelimits.append(pelimit)
        pelimits = [pe + 1 for pe in pelimits]
        return pelimits
    
    def rerange(self, core_dims, core_types):
        centers = []
        for i, dim in enumerate(core_dims):
            kmedoids = KMedoids(k=1).fit(np.asarray(dim))
            center = kmedoids[0][0]
            centers.append(center)

            def changeid(listname, idx1, idx2):
                listname[idx1], listname[idx2] = listname[idx2], listname[idx1]

            for outputlist in [core_dims, core_types]:
                changeid(outputlist[i], 0, center)

        return centers


from sklearn.metrics.pairwise import pairwise_distances  
  
class KMedoids:  
    def __init__(self, k=3, max_iterations=100):  
        # 初始化KMedoids类，设置聚类数k和最大迭代次数max_iterations  
        self.k = k  
        self.max_iterations = max_iterations  
  
    def fit(self, X):  
        # 对输入数据X进行聚类  
        n_samples, n_features = X.shape  # 获取数据的样本数和特征数  
  
        # 初始化  
        medoids = np.zeros(self.k, dtype=int)  # 初始化medoids数组，用于存储每个聚类的中心点（代表点）的索引  
        dists = pairwise_distances(X, metric='euclidean')  # 计算数据点之间的欧氏距离矩阵  
  
        # 随机选择medoids  
        medoids = np.random.choice(n_samples, self.k, replace=False)  # 从数据集中随机选择k个点作为初始的medoids  
  
        # 将样本分配给最近的medoid  
        labels = np.argmin(dists[:, medoids], axis=1)  # 根据距离矩阵，将每个样本分配给最近的medoid  
  
        # 迭代更新medoids  
        for i in range(self.max_iterations):  
            # 对每个聚类更新medoid  
            for j in range(self.k):  
                indices = np.where(labels == j)[0]  # 获取当前聚类中所有样本的索引  
                if len(indices) == 0:  
                    continue  # 如果当前聚类中没有样本，则跳过更新medoid  
                costs = dists[indices][:, indices]  # 计算当前聚类中所有样本之间的距离矩阵  
                total_cost = np.sum(costs, axis=1)  # 计算每个样本作为medoid时的总成本（即该点到聚类中其他点的距离之和）  
                new_medoid = indices[np.argmin(total_cost)]  # 选择总成本最小的样本作为新的medoid  
                medoids[j] = new_medoid  # 更新medoid  
  
            # 将样本重新分配给最近的medoid  
            labels = np.argmin(dists[:, medoids], axis=1)  # 根据新的medoids，重新分配样本  
  
        # 获取聚类中心（即medoids对应的样本）  
        centroids = X[medoids]  
          
        # 返回medoids的索引、聚类中心和每个样本的聚类标签  
        return medoids, centroids, labels
