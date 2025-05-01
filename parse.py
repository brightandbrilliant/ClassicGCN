import os.path as osp
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import subgraph


class SocialGraphDataset(InMemoryDataset):
    def __init__(self, root, num_clients=5, transform=None):
        self.num_clients = num_clients
        super().__init__(root, transform)

    @property
    def raw_file_names(self):
        return ['nodes.csv', 'groups.csv', 'node_groups.csv', 'edges.csv']

    @property
    def processed_file_names(self):
        return [f'client_{i}.pt' for i in range(self.num_clients)]

    def process_node_features_and_labels(self):
        # 读取节点 ID 列表
        node_ids = pd.read_csv(osp.join(self.raw_dir, 'nodes.csv'))['node_id'].unique()
        node_groups = pd.read_csv(osp.join(self.raw_dir, 'node_groups.csv'))
        group_ids = sorted(pd.read_csv(osp.join(self.raw_dir, 'groups.csv'))['group_id'].unique())

        # one-hot 编码 group 特征
        group_dummies = pd.get_dummies(node_groups, columns=['group_id'],
                                       prefix='', prefix_sep='').groupby('node_id').max()

        # 确保所有节点都在特征矩阵中
        full_index = pd.Index(node_ids, name='node_id')
        group_dummies = group_dummies.reindex(full_index, fill_value=0)

        # 确保所有 group 列都存在
        for g in group_ids:
            if g not in group_dummies.columns:
                group_dummies[g] = 0
        group_dummies = group_dummies[sorted(group_ids)]

        # 特征和标签
        x = torch.tensor(group_dummies.values, dtype=torch.float)
        y = torch.tensor(group_dummies.values.argmax(axis=1), dtype=torch.long)  # 单标签分类

        return x, y

    def process_edges(self):
        edges = pd.read_csv(osp.join(self.raw_dir, 'edges.csv'))
        if not edges['probability'].between(0, 1).all():
            raise ValueError("Edge probabilities must be in [0,1]")

        edge_index = torch.tensor(edges[['source', 'target']].values.T, dtype=torch.long)
        return edge_index

    def split_clients(self, data):
        client_data_list = []
        num_groups = data.x.size(1)

        for client_id in range(self.num_clients):
            main_group = client_id % num_groups
            main_nodes = torch.where(data.x[:, main_group] == 1)[0]
            num_main = min(len(main_nodes), int(0.6 * data.num_nodes / self.num_clients))
            main_sample = main_nodes[torch.randperm(len(main_nodes))[:num_main]]

            other_nodes = torch.where(data.x[:, main_group] != 1)[0]
            num_other = min(len(other_nodes), int(0.4 * data.num_nodes / self.num_clients))
            other_sample = other_nodes[torch.randperm(len(other_nodes))[:num_other]]

            selected_nodes = torch.cat([main_sample, other_sample]).unique()

            edge_mask = (torch.isin(data.edge_index[0], selected_nodes)
                         & torch.isin(data.edge_index[1], selected_nodes))
            edge_index_sub = data.edge_index[:, edge_mask]
            edge_index_sub, _ = subgraph(selected_nodes, edge_index_sub, relabel_nodes=True)

            client_data = Data(
                x=data.x[selected_nodes],
                edge_index=edge_index_sub,
                y=data.y[selected_nodes],
                original_nodes=selected_nodes
            )
            client_data_list.append(client_data)

        return client_data_list

    def process(self):
        x, y = self.process_node_features_and_labels()
        edge_index = self.process_edges()

        full_graph = Data(x=x, edge_index=edge_index, y=y, num_nodes=len(x))
        client_data_list = self.split_clients(full_graph)

        for i, client_data in enumerate(client_data_list):
            torch.save(client_data, osp.join(self.processed_dir, f'client_{i}.pt'))

    def get_client_data(self, client_id):
        return torch.load(osp.join(self.processed_dir, f'client_{client_id}.pt'))

    def len(self):
        return self.num_clients

    def get(self, idx):
        return self.get_client_data(idx)


if __name__ == "__main__":
    dataset = SocialGraphDataset(root='./data', num_clients=3)
    for i in range(len(dataset)):
        client_data = dataset.get_client_data(i)
        print(f"Client {i}:")
        print("  Features:", client_data.x.shape)
        print("  Edges:", client_data.edge_index.shape)
        print("  Labels:", client_data.y.shape)