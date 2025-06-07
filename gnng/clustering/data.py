import torch
from typing import Tuple


def generate_simulated_clustering_data(
        num_clusters: int = 3,
        points_per_cluster: int = 50,
        dimensions: int = 2,
        cluster_std: float = 1.0,
        must_link_ratio: float = 0.1,
        cannot_link_ratio: float = 0.1,
        random_seed: int = 42
):
    """
    生成用于聚类任务的随机模拟数据集，并根据数据点之间的关系生成朋友（must_link）和敌人（cannot_link）。

    参数：
    - num_clusters: 簇的数量。
    - points_per_cluster: 每个簇中的数据点数量。
    - dimensions: 数据的维度。
    - cluster_std: 簇的标准差（决定簇的紧密程度）。
    - must_link_ratio: 生成must_link关系的比例（相对于每个簇中的点数）。
    - cannot_link_ratio: 生成cannot_link关系的比例（相对于所有不同簇点数的乘积）。
    - random_seed: 随机种子，确保结果可复现。

    返回：
    - data: 生成的数据点，形状为 (num_points, dimensions)。
    - labels: Tensor
    - must_link: must_link关系的张量，形状为 (num_must_link, 2)。
    - cannot_link: cannot_link关系的张量，形状为 (num_cannot_link, 2)。
    """
    torch.manual_seed(random_seed)

    # 生成簇中心，扩大分布范围以避免簇之间过于接近
    cluster_centers = torch.randn(num_clusters, dimensions) * 5

    data: list = []
    labels: list = []
    for cluster_idx in range(num_clusters):
        # 为每个簇生成数据点
        points = cluster_centers[cluster_idx] + torch.randn(points_per_cluster, dimensions) * cluster_std
        data.append(points)
        labels += [cluster_idx] * points_per_cluster

    data: torch.Tensor = torch.vstack(data)
    labels: torch.Tensor = torch.tensor(labels)
    num_points = data.shape[0]

    # 创建簇索引列表
    cluster_indices = [torch.nonzero(labels == cluster_idx, as_tuple=False).squeeze() for cluster_idx in
                       range(num_clusters)]

    # 生成 must_link 关系
    must_link = []
    for indices in cluster_indices:
        num_points_in_cluster = indices.numel()
        if num_points_in_cluster < 2:
            continue
        # 计算每个簇需要的 must_link 数量
        num_possible_pairs = num_points_in_cluster * (num_points_in_cluster - 1) // 2
        num_must_link = int(num_possible_pairs * must_link_ratio)
        num_must_link = max(1, num_must_link) if num_possible_pairs > 0 else 0
        num_must_link = min(num_must_link, num_possible_pairs)

        if num_must_link == 0:
            continue

        # 随机选择 must_link 对
        # 选择第一个点
        first = torch.randint(0, num_points_in_cluster, (num_must_link,))
        # 选择第二个点，确保第二点大于第一个点以避免重复
        second = torch.randint(0, num_points_in_cluster, (num_must_link,))
        mask = second > first
        first = first[mask]
        second = second[mask]

        # 如果采样不足，重新采样
        while first.numel() < num_must_link:
            needed = num_must_link - first.numel()
            new_first = torch.randint(0, num_points_in_cluster, (needed,))
            new_second = torch.randint(0, num_points_in_cluster, (needed,))
            new_mask = new_second > new_first
            first = torch.cat([first, new_first[new_mask]], dim=0)
            second = torch.cat([second, new_second[new_mask]], dim=0)

        # 取前 num_must_link 对
        first = first[:num_must_link]
        second = second[:num_must_link]
        pairs = torch.stack([indices[first], indices[second]], dim=1)
        must_link.append(pairs)

    if must_link:
        must_link = torch.cat(must_link, dim=0)
    else:
        must_link = torch.empty((0, 2), dtype=torch.long)

    # 生成 cannot_link 关系
    # 首先获取每个簇的索引
    cluster_indices = [torch.nonzero(labels == cluster_idx, as_tuple=False).squeeze() for cluster_idx in
                       range(num_clusters)]

    # 计算不同簇之间的总可能对数
    total_cannot_link_pairs = 0
    for i in range(num_clusters):
        for j in range(i + 1, num_clusters):
            total_cannot_link_pairs += cluster_indices[i].numel() * cluster_indices[j].numel()

    # 计算需要的 cannot_link 数量
    num_cannot_link = int(total_cannot_link_pairs * cannot_link_ratio)
    num_cannot_link = min(num_cannot_link, total_cannot_link_pairs)

    if num_cannot_link == 0:
        cannot_link = torch.empty((0, 2), dtype=torch.long)
    else:
        # 随机采样 cannot_link 对
        # 首先随机选择 num_cannot_link 对的簇对
        # 每个簇对的概率与其可能对数成正比
        pair_weights = torch.tensor([
            cluster_indices[i].numel() * cluster_indices[j].numel()
            for i in range(num_clusters) for j in range(i + 1, num_clusters)
        ], dtype=torch.float)
        pair_weights /= pair_weights.sum()

        # 多项分布确定每个簇对采样的数量
        sampled_counts = torch.multinomial(pair_weights, num_cannot_link, replacement=True)
        # To ensure counts do not exceed possible, use floor
        sampled_counts = (pair_weights * num_cannot_link).floor().long()

        # 如果总数不足，调整
        current_sum = sampled_counts.sum().item()
        if current_sum < num_cannot_link:
            additional = num_cannot_link - current_sum
            additional_indices = torch.multinomial(pair_weights, additional, replacement=True)
            sampled_counts += torch.bincount(additional_indices, minlength=pair_weights.size(0))

        # 生成 cannot_link 对
        cannot_link = []
        pair_idx = 0
        for i in range(num_clusters):
            for j in range(i + 1, num_clusters):
                count = sampled_counts[pair_idx].item()
                pair_idx += 1
                if count == 0:
                    continue
                indices_i = cluster_indices[i]
                indices_j = cluster_indices[j]
                if indices_i.numel() == 0 or indices_j.numel() == 0:
                    continue
                # 随机选择 count 个点
                chosen_i = indices_i[torch.randint(0, indices_i.numel(), (count,))]
                chosen_j = indices_j[torch.randint(0, indices_j.numel(), (count,))]
                pairs = torch.stack([chosen_i, chosen_j], dim=1)
                cannot_link.append(pairs)

        if cannot_link:
            cannot_link = torch.cat(cannot_link, dim=0)[:num_cannot_link]
        else:
            cannot_link = torch.empty((0, 2), dtype=torch.long)

    return data, labels, must_link, cannot_link
