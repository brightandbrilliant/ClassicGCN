# train_gcn.py
import argparse
import torch
from torch_geometric.loader import DataLoader
import os
from client import GCNClient  # 你自己写的 Client 模块
from parse import SocialGraphDataset  # 你自定义的数据集类
import torch.nn.functional as F


def create_multi_label_targets(data):
    # 构造邻接矩阵式的多标签 one-hot 向量 [num_nodes, num_nodes]
    num_nodes = data.num_nodes
    edge_index = data.edge_index
    labels = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
    labels[edge_index[0], edge_index[1]] = 1.0
    return labels


def main():
    parser = argparse.ArgumentParser(description="Federated Graph Learning Trainer")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--feature_dim', type=int, default=39)  # group 数
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--total_rounds', type=int, default=5)
    parser.add_argument('--client_id', type=int, default=0)
    parser.add_argument('--data_root', type=str, default='./Raw_Dataset/BlogCatalog-dataset')
    args = parser.parse_args()

    # 加载预处理后的 client 数据
    data = torch.load('./Raw_Dataset/BlogCatalog-dataset/processed/client_0.pt')

    # 设置 batch 属性（PyG 要求）
    if not hasattr(data, 'batch') or data.batch is None:
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long)

    # 用 original_nodes 作为 user_ids
    data.user_ids = data.original_nodes

    # 构建多标签 one-hot 目标矩阵
    data.target_labels = create_multi_label_targets(data)

    print(f'Client {args.client_id} data loaded:')
    print('  Features:', data.x.shape)
    print('  Edges:', data.edge_index.shape)
    print('  Labels:', data.target_labels.shape)

    dataloader = DataLoader([data], batch_size=args.batch_size, shuffle=True)

    client = GCNClient(
        in_dim=args.feature_dim,
        gcn_hidden_dim=128,
        gcn_out_dim=128,
        pred_hidden_dim=128,
        out_dim=data.target_labels.shape[1],  # 每一行是长度为 N 的多标签向量
        gcn_layers=3,
        pred_layers=6,
        gcn_dropout=0.4,
        pred_dropout=0.4
    )
    client.create_optimizer(lr=5e-3, weight_decay=1e-5)

    device = torch.device(args.device)
    client = client.to(device)

    checkpoint_dir = 'Checkpoints_Client_0'
    os.makedirs(checkpoint_dir, exist_ok=True)

    for steps in range(1, args.total_rounds + 1):
        client.train()
        for batch in dataloader:
            batch = batch.to(device)
            output, loss = client(batch, batch.target_labels.to(device))
            client.optimizer.zero_grad()
            loss.backward()
            client.optimizer.step()
            print(f"[Round {steps}] Loss: {loss.item():.4f}")

        if (steps+1) % 1500 == 0:
            save_path = os.path.join(checkpoint_dir, f'client_{args.client_id}_round_{steps+1}.pth')
            torch.save(client.state_dict(), save_path)


if __name__ == "__main__":
    main()






