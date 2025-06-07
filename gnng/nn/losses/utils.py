import torch


def positive_pair_mask(z, n_view=2):
    """
    在重塑后的表示张量中找到所有正样本对。

    Args:
        z (Tensor): 表示张量，形状为 (n_view, batch_size, d)。
        n_view (int): 每个样本的视图数量。

    Returns:
        Tensor: 一个二进制掩码矩阵，形状为 (n_view * batch_size, n_view * batch_size)，
                其中 mask[i, j] = 1 表示样本 i 和样本 j 是正样本对，反之为 0。
    """
    if z.ndim > 2:
        n_view, batch_size = z.shape[:2]
        # 重塑张量为 (n_view * batch_size, d)
        z = z.flatten(0, 1)
    else:
        n_all = z.size(0)
        batch_size = n_all / n_view

    # 创建标签：每个样本根据其在批次中的索引分配一个标签
    # 例如，对于 batch_size=3, n_view=2，标签将是 [0, 1, 2, 0, 1, 2]
    labels = torch.arange(batch_size, device=z.device).tile(n_view)

    # 扩展标签以进行广播比较
    labels = labels.unsqueeze(0)  # 形状: (1, n_view * batch_size)
    # 比较标签，生成正样本掩码
    mask: "torch.BoolTensor" = labels == labels.T  # 形状: (n_view * batch_size, n_view * batch_size)

    # 去除自配对（即 mask[i, i] = 0）
    mask.fill_diagonal_(0)
    return mask, z


# 示例用法
if __name__ == "__main__":
    n_view = 3  # 例如，3个视图
    batch_size = 4  # 例如，批量大小为4
    d = 128  # 嵌入维度为128

    # 生成随机表示张量
    z = torch.randn(n_view, batch_size, d)

    # 找到所有正样本对的掩码
    positive_mask = positive_pair_mask(z, n_view).float()

    print("Positive Mask:")
    print(positive_mask)
