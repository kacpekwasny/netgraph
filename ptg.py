import random
import torch
import networkx as nx
import matplotlib.pyplot as plt
import torch_geometric as tg

from torch_geometric.data import Data


def create_graph1() -> tuple[Data, nx.Graph]:
    # COO format, shape (2, num_edges)
    edge_index = torch.tensor([[0, 1, 1, 2],
                               [1, 0, 2, 1]], dtype=torch.long)

    x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index)

    g = tg.utils.to_networkx(data, to_undirected=True)

    return data, g


def create_graph2() -> tuple[Data, nx.Graph]:
    g = nx.Graph()
    N = 10
    E = 13

    g.add_nodes_from(range(N))

    while g.number_of_edges() < E:
        u = random.randint(0, N-1)
        v = random.randint(0, N-1)

        if u == v or g.has_edge(u, v):
            continue

        g.add_edge(u, v)

    return tg.utils.from_networkx(g), g


for f in [create_graph1, create_graph2]:
    data, g = f()
    nx.draw(g)
    plt.show()
