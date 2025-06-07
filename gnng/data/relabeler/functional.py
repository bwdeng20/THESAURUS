import torch


def labels_to_one_hot(yn, num_classes=None):
    if num_classes is None:
        num_classes = yn.max() + 1
    one_hot = torch.zeros(yn.size(0), num_classes, device=yn.device)
    one_hot.scatter_(1, yn.view(-1, 1), 1)
    return one_hot


def target_distribution_from_adjacency(labels, Ay, undirected=False, within_old=None):
    """
    Generates a target distribution for each sample based on a random walk graph adjacency matrix.

    Parameters
    ----------
    labels : torch.Tensor
        Tensor of shape (n,) containing the labels of each sample.
    Ay : torch.Tensor
        A (C, C) adjacency matrix where C is the total number of classes.
        Each row in Ay represents the distribution over classes for a specific class,
        and each row sums to 1.

    Returns
    -------
    torch.Tensor
        A tensor of shape (n, C) where each row is the target distribution
        for the corresponding sample, based on its class label and the adjacency matrix.

    Examples
    --------
    >>> labels = torch.tensor([0, 1, 2])
    >>> Ay = torch.tensor([
    ...     [0.8, 0.1, 0.1],
    ...     [0.2, 0.6, 0.2],
    ...     [0.2, 0.0, 0.6]
    ... ])
    >>> lb=target_distribution_from_adjacency(labels, Ay)
    >>> assert torch.all(lb.sum(dim=1)==1)
    """
    Ay = torch.as_tensor(Ay)
    n = labels.size(0)  # Number of samples
    num_classes = Ay.size(0)  # Total number of classes
    if undirected:
        Ay = symmetrize_adjacency_matrix(Ay)

    row_sum = Ay.sum(dim=1, keepdim=True)
    if torch.any(row_sum != 1):
        Ay = Ay / row_sum

    # Initialize the target distribution tensor
    target_dist = torch.zeros((n, num_classes), device=Ay.device)

    # Assign the target distribution for each sample based on its label
    if within_old is not None:
        for c in range(num_classes):
            if c not in within_old:
                Ay[c] = torch.zeros_like(Ay[c])
                Ay[c][c] = 1.0

    for i, label in enumerate(labels):
        target_dist[i] = Ay[label]

    return target_dist


def target_distribution_from_nbr_dict(
        labels, neighbors_dict, a=0.9, num_classes=None, undirected=False, within_old=None
):
    """
    Constructs a target distribution with specific values for each class and its top-k neighbors.

    Parameters
    ----------
    labels : torch.Tensor
        Tensor of shape (n,) containing the labels of each sample.
    num_classes : int
        Total number of classes in the dataset.
    neighbors_dict : dict
        Dictionary where each key is a class (int) and its value is a list of the top-k neighbor classes.
    a : float, optional
        The target distribution value assigned to the class itself, by default 0.9.
        The remaining (1 - a) is divided equally among the top-k neighbors.
    undirected: bool, optional
        If True, undirected label semantic graph is constructed.
    Returns
    -------
    torch.Tensor
        A tensor of shape (n, num_classes) with the constructed target distribution for each sample.

    Examples
    --------
    >>> labels = torch.tensor([0, 1, 2, 3, 4, 5, 6])
    >>> num_classes = labels.max() + 1
    >>> neighbors_dict = {0: [3], 1: [4], 2: [5]}
    >>> target_distribution_from_nbr_dict(labels, num_classes, neighbors_dict, a=0.9)
    """
    if num_classes is None:
        num_classes = labels.max() + 1

    n = labels.size(0)
    target_dist = torch.zeros((n, num_classes), device=labels.device)

    if undirected:
        neighbors_dict = symmetrize_dol(neighbors_dict)

    for i, label in enumerate(labels):
        # Set the value `a` for the target class `c`
        if within_old is not None and label not in within_old:
            continue  # skip new classes
        target_dist[i, label] = a

        # Get the top-k neighbors for the current label
        neighbors = neighbors_dict.get(label.item(), [])
        k = len(neighbors)

        # Distribute (1 - a) / k among the top-k neighbors
        if k > 0:
            neighbor_value = (1 - a) / k
            target_dist[i, neighbors] = neighbor_value
        else:
            target_dist[i, label] = 1.0
    return target_dist


def symmetrize_adjacency_matrix(A):
    A_sym = (A + A.T) / 2
    return A_sym


def symmetrize_dol(neighbors_dict):
    """
    Converts a directed graph represented as a dictionary of lists into an undirected graph.

    Parameters
    ----------
    neighbors_dict : dict
        A dictionary where each key is a node and the value is a list of neighboring nodes.

    Returns
    -------
    dict
        An undirected graph where all connections are bidirectional.
    """
    undirected_graph = {}

    for node, neighbors in neighbors_dict.items():
        # Ensure each node exists in the undirected graph
        if node not in undirected_graph:
            undirected_graph[node] = set(neighbors)
        else:
            undirected_graph[node].update(neighbors)

        # Make connections bidirectional
        for neighbor in neighbors:
            if neighbor not in undirected_graph:
                undirected_graph[neighbor] = {node}
            else:
                undirected_graph[neighbor].add(node)

    # Convert sets back to lists for the final graph representation
    return {node: list(neigh) for node, neigh in undirected_graph.items()}


if __name__ == "__main__":
    # labels = torch.tensor([0, 1, 2])
    # Ay = torch.tensor([[0.8, 0.1, 0.1], [0.2, 0.6, 0.2], [0.6, 0.0, 0.2]])
    # lb = target_distribution_from_adjacency(labels, Ay)
    # print(lb)
    # assert torch.all(lb.sum(dim=1) == 1)
    labels = torch.tensor([0, 1, 2, 3, 4, 5, 6])
    num_classes = labels.max() + 1
    neighbors_dict = {0: [3], 1: [4], 2: [5]}
    target = target_distribution_from_nbr_dict(labels, num_classes, neighbors_dict, a=0.9)
    print(target)

    target = target_distribution_from_nbr_dict(labels, num_classes, neighbors_dict, a=0.9, undirected=True)
    print(target)
