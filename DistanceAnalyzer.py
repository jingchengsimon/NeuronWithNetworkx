import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from sklearn.cluster import SpectralClustering
from CellwithNetworkx import *
import time
from tqdm import tqdm
import random

class DistanceAnalyzer:
    def __init__(self, distance_matrix, type_list, bin_list):
        self.distance_matrix = distance_matrix
        self.type_list = type_list
        self.bin_list = bin_list

        self.initial_guess = [1, 1]
        self.max_iters = 10000
        self.num_epochs = None
        self.shuffle_scale = 10
        self.type_list_clustered = None
        self.percentage_list_clustered = None
        self.error_min_list = None
        self.percentage_list = self._calculate_bin_percentage(self.type_list)

        # for traditional clustering
        self.n_clusters = 2  

        # 指定各个cluster中的点的个数
        self.n_A = 500
        self.n_B = 500
        
    def _create_type_matrix(self, type_list):
        n = len(type_list)
        type_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if self.type_list[i] == type_list[j]:
                    type_matrix[i, j] = 1
                else:
                    type_matrix[i, j] = 0

        return type_matrix
    
    def _calculate_bin_percentage(self, type_list):
        distance_matrix, bin_list = self.distance_matrix, self.bin_list
        type_matrix = self._create_type_matrix(type_list)
        percentage_list = []
        for i, j in zip(bin_list, bin_list[1:]):
            mask_dis = (distance_matrix < j) & (distance_matrix > i)
            mask_dis_type = mask_dis & (type_matrix == 1)
            # distance_matrix_bin  = distance_matrix[(distance_matrix < j) & (distance_matrix > i)]
            count_pairs_all = np.count_nonzero(distance_matrix[mask_dis])
            count_pairs_same = np.count_nonzero(distance_matrix[mask_dis_type])
            if count_pairs_all == 0:
                percentage_list.append(0)
            else:
                percentage_list.append(count_pairs_same / count_pairs_all)

        return percentage_list

    def _exponential_func(self, x, a, b):
        lambd = 7
        return a * np.exp(-x / lambd) + b

    def _calculate_error(self, percentage_list, initial_guess=[1, 1]):
        bin_list = self.bin_list  
        x = np.array(bin_list[1:])
        y_noisy = np.array(percentage_list)
        params, covariance = curve_fit(self._exponential_func, x, y_noisy, p0=initial_guess, maxfev=10000)

        # 获取拟合结果
        a_fit, b_fit = params

        # 计算拟合值
        y_fit = self._exponential_func(x, a_fit, b_fit)

        # 计算拟合误差
        error = np.sqrt(np.mean((y_noisy - y_fit) ** 2))

        return error,[a_fit, b_fit]

    def _partial_shuffle(self, input_list, percentage=10):
        '''Shuffles any n number of values in a list'''
        count = int(len(input_list)*percentage/100)
        indices_to_shuffle = random.sample(range(len(input_list)), k=count)
        to_shuffle = [input_list[i] for i in indices_to_shuffle]
        random.shuffle(to_shuffle)
        for index, value in enumerate(to_shuffle):
            old_index = indices_to_shuffle[index]
            input_list[old_index] = value
        return input_list

    def _custom_spectral_clustering(self):
        distance_matrix, n_clusters, n_A, n_B = self.distance_matrix, self.n_clusters, self.n_A, self.n_B
        # 使用谱聚类将数据点分成两类
        spectral_clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=0)
        labels = spectral_clustering.fit_predict(distance_matrix)

        # 计算初始聚类结果中每个类别的点个数
        n_cluster_A = np.sum(labels == 0)
        n_cluster_B = np.sum(labels == 1)

        # 调整聚类结果，直到满足点个数约束
        while n_cluster_A != n_A or n_cluster_B != n_B:
            if n_cluster_A > n_A:
                # 如果A的个数多于n_A，我们从A类别中移动一些点到B类别中
                indices_to_move = np.where(labels == 0)[0][:n_cluster_A - n_A]
                labels[indices_to_move] = 1

            elif n_cluster_A < n_A:
                # 如果A的个数少于n_A，我们从B类别中移动一些点到A类别中
                indices_to_move = np.where(labels == 1)[0][:n_A - n_cluster_A]
                labels[indices_to_move] = 0

            # 更新每个类别的点个数
            n_cluster_A = np.sum(labels == 0)
            n_cluster_B = np.sum(labels == 1)

        return labels

    def _custom_kmeans(self):
        max_iters=self.max_iters
        distance_matrix, n_clusters, n_A, n_B = self.distance_matrix, self.n_clusters, self.n_A, self.n_B
        # 转换距离矩阵为相似性矩阵（通过取倒数）
        similarity_matrix = 1 / (1 + distance_matrix)

        # 初始化聚类中心（可以随机选择n_clusters个点作为初始中心）
        center_indices = np.random.choice(len(similarity_matrix), size=n_clusters, replace=False)
        centers = similarity_matrix[center_indices]

        n_cluster_wished = np.array([n_A,  n_B])
        # 进行KMeans迭代更新
        for _ in range(max_iters):
            # 计算每个点到聚类中心的距离
            distances = np.linalg.norm(centers[:, np.newaxis] - similarity_matrix, axis=2)

            # 确定每个点的聚类标签
            # In this step, all labels of points are reassigned, 
            # so -1 we got in last iteration will be overwritten
            labels = np.argmin(distances, axis=0)

            n_cluster_prev = np.array([np.sum(labels == i) for i in range(n_clusters)])

            # 根据指定的n_A和n_B来调整聚类中心
            for i in range(n_clusters):
                cluster_indices = np.where(labels == i)[0]
                # n_cluster_A = np.sum(labels[cluster_indices] == 0)
                # n_cluster_B = len(cluster_indices) - n_cluster_A

                if n_cluster_prev[i] > n_cluster_wished[i]:
                    # 如果A的个数多于n_A，我们删除一些A类型的点
                    indices_to_remove = np.random.choice(cluster_indices, size=n_cluster_prev[i] - n_cluster_wished[i], replace=False)
                    labels[indices_to_remove] = -1  # 将删除的点标记为-1

                # if n_cluster_A > n_A:
                #     # 如果A的个数多于n_A，我们删除一些A类型的点
                #     indices_to_remove = np.random.choice(cluster_indices[labels[cluster_indices] == 0], size=n_cluster_A - n_A, replace=False)
                #     labels[indices_to_remove] = -1  # 将删除的点标记为-1

                # if n_cluster_B > n_B:
                #     # 如果B的个数多于n_B，我们删除一些B类型的点
                #     indices_to_remove = np.random.choice(cluster_indices[labels[cluster_indices] == 1], size=n_cluster_B - n_B, replace=False)
                #     labels[indices_to_remove] = -1  # 将删除的点标记为-1

                # 更新聚类中心
                cluster_indices = np.where(labels == i)[0]
                centers[i] = np.mean(similarity_matrix[cluster_indices], axis=0)

            # 如果没有点被删除，说明已经满足要求，停止迭代
            if np.sum(labels == -1) == 0:
                break
            
        # 删除被标记为-1的点
        valid_indices = np.where(labels != -1)[0]
        labels = labels[valid_indices]

        return labels

    def _custom_kmeans_1(self):
        max_iters=self.max_iters
        distance_matrix, n_clusters, n_A, n_B = self.distance_matrix, self.n_clusters, self.n_A, self.n_B
        # 转换距离矩阵为相似性矩阵（通过取倒数）
        similarity_matrix = 1 / (1 + distance_matrix)

        # 初始化聚类中心（可以随机选择n_clusters个点作为初始中心）
        center_indices = np.random.choice(len(similarity_matrix), size=n_clusters, replace=False)
        centers = similarity_matrix[center_indices]

        n_cluster_wished = np.array([n_A,  n_B])
        for _ in range(max_iters):
            # 计算每个点到聚类中心的距离
            distances = np.linalg.norm(centers[:, np.newaxis] - similarity_matrix, axis=2)

            # 确定每个点的聚类标签
            labels = np.argmin(distances, axis=0)

            # 记录当前每个cluster的A类型点个数
            n_cluster_prev = np.array([np.sum(labels == i) for i in range(n_clusters)])

            # 调整聚类中心
            for i in range(n_clusters):
                cluster_indices = np.where(labels == i)[0]

                if n_cluster_prev[i] > n_cluster_wished[i]:
                    # 如果A的个数多于n_A，我们删除一些A类型的点
                    indices_to_remove = np.random.choice(cluster_indices, size=int((n_cluster_prev[i] - n_cluster_wished[i])/2), replace=False)
                    labels[indices_to_remove] = -1  # 将删除的点标记为-1

                # 更新聚类中心
                cluster_indices = np.where(labels == i)[0]
                centers[i] = np.mean(similarity_matrix[cluster_indices], axis=0)

            # 记录每个cluster的A类型点个数
            n_cluster_after = np.array([np.sum(labels == i) for i in range(n_clusters)])

            # 检查每个cluster的A类型点个数是否小于n_A
            for i in range(n_clusters):
                if n_cluster_after[i] < n_cluster_wished[i]:
                    # 如果A的个数少于n_A，我们从其他cluster移动一些B类型的点到该cluster中
                    cluster_indices = np.where(labels == i)[0]
                    other_clusters = np.arange(n_clusters)[n_cluster_after > n_cluster_wished]

                    # 在其他cluster中随机选择一些B类型的点，移动到当前cluster中
                    for j in other_clusters:
                        B_indices = np.where(labels == j)[0]
                        # currently only for 2 clusters, more than 2 than we need a proportion to give points 
                        # for each cluster with more points
                        indices_to_move = np.random.choice(B_indices, size=n_cluster_after[j] - n_cluster_wished[j], replace=False)

                        # 更新聚类标签和聚类中心
                        labels[indices_to_move] = i
                        centers[i] = np.mean(similarity_matrix[cluster_indices], axis=0)

                        # 更新每个cluster的A类型点个数
                        n_cluster_after = np.array([np.sum(labels == k) for k in range(n_clusters)])
                        if np.all(n_cluster_after >= n_cluster_wished):
                            break

            # 如果没有点被删除或移动，说明已经满足要求，停止迭代
            if np.sum(labels == -1) == 0 and np.all(n_cluster_after >= n_cluster_wished):
                break
            
        # 删除被标记为-1的点
        # valid_indices = np.where(labels != -1)[0]
        # labels = labels[valid_indices]

        # Reassign invalid labels to 0 and 1 (temporary solution)
        invalid_indices = np.where(labels == -1)[0]
        indices_to_be_A = np.random.choice(invalid_indices, size=int(len(invalid_indices)/2), replace=False)
        indices_to_be_B = np.random.choice(invalid_indices, size=len(invalid_indices)-len(indices_to_be_A), replace=False)
                
        labels[indices_to_be_A] = 0
        labels[indices_to_be_B] = 1

        return labels

    def cluster_traditional(self):
        # 使用自定义KMeans聚类算法聚类
        labels = self._custom_kmeans_1()

        # 重新得到聚类后的type_list
        type_list_clustered = np.array(['A' if label == 0 else 'B' for label in labels])

        # Use the final type_list and calculate the clustered percentage_list
        self.type_list_clustered = type_list_clustered
        self.percentage_list_clustered = self._calculate_bin_percentage(self.type_list_clustered)

        # 打印调整后的聚类结果
        print("聚类结果:")
        print(np.count_nonzero(labels == 0))
        print(np.count_nonzero(labels == 1))

    def cluster_shuffle(self, num_epochs=100):
        error_min, initial_guess_min = self._calculate_error(self.percentage_list)
        type_list_fin = self.type_list
        self.num_epochs = num_epochs
        error_min_list = []
        for _ in tqdm(range(num_epochs)):
            error_list = [error_min]
            initial_guess_list = [initial_guess_min]
            type_list_list = [type_list_fin]

            for _ in range(self.shuffle_scale):
                type_list_shuffle = self._partial_shuffle(type_list_fin, percentage=10)
                percentage_list = self._calculate_bin_percentage(type_list_shuffle)
                error, initial_guess = self._calculate_error(percentage_list, initial_guess_min)
                error_list.append(error)
                initial_guess_list.append(initial_guess)
                type_list_list.append(type_list_shuffle)

            error_min = min(error_list)
            type_list_fin = type_list_list[error_list.index(error_min)]
            initial_guess = initial_guess_list[error_list.index(error_min)]

            error_min_list.append(error_min)
            # Add a small delay to prevent excessive resource usage
            time.sleep(0.01)
        
        self.error_min_list = error_min_list

        # Use the final type_list and calculate the clustered percentage_list
        self.type_list_clustered = type_list_fin
        self.percentage_list_clustered = self._calculate_bin_percentage(self.type_list_clustered)

    def visualize_learning_curve(self):
        epochs = list(range(1, self.num_epochs + 1))
        plt.figure(figsize=(6, 6))
        plt.plot(epochs, self.error_min_list, marker='o', linestyle='-', color='b', label='Error')
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.legend()
        plt.title('Learning Curve')

    def visualize_results(self):
        plt.figure(figsize=(4, 4))

        # plt.subplot(1, 2, 1)
        plt.plot(self.bin_list[1:], self.percentage_list, label='Percentage before clustered')
        plt.xlabel('Distance (microns)')
        plt.ylabel('Percentage')
        plt.title('Percentage before clustered')

        # plt.subplot(1, 2, 2)
        # plt.plot(self.bin_list[1:], self.percentage_list_clustered, label='Percentage after clustered')
        # plt.xlabel('Distance (microns)')
        # plt.ylabel('Percentage')
        # plt.title('Percentage after clustered')

        # plt.show()



