# train_gcn.py
import argparse
from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
import random
import numpy as np
from client import GCNClient  # 你刚才写的那个组合模型文件
import os


def main():
    parser = argparse.ArgumentParser(description="Federated Graph Learning Trainer")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--feature_dim', type=int, default=39)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--total_rounds', type=int, default=5)
    args = parser.parse_args()

    data_path = './Dataset/BlogCatalog/client0_train.pt'
    data = torch.load(data_path)  # 假设是 List[Data] 或 Dataset
    if not hasattr(data, 'batch') or data.batch is None:
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long)

    # user_ids 应为本地节点的真实 ID
    if not hasattr(data, 'user_ids'):
        raise ValueError("Client 0 的数据缺少 `user_ids` 属性")
    max_uid = int(data.user_ids.max().item())
    print('data_loaded')

    dataloader = DataLoader([data], batch_size=args.batch_size, shuffle=True)
    print('loader_create')
    client = GCNClient(
        in_dim=args.feature_dim,
        gcn_hidden_dim=32,
        gcn_out_dim=64,
        pred_hidden_dim=96,
        out_dim=max_uid + 1,
        gcn_layers=3,
        pred_layers=4,
        gcn_dropout=0.2,
        pred_dropout=0.2
    )
    client.create_optimizer(lr=5e-3, weight_decay=1e-5)

    device = torch.device(args.device)
    client = client.to(device)

    checkpoint_dir = 'Check1'
    os.makedirs(checkpoint_dir, exist_ok=True)

    for steps in range(0, args.total_rounds):
        print(f'Learning round: {steps+1}')
        client.train()
        for batch in dataloader:
            batch = batch.to(device)
            _, loss = client(batch, batch.target_labels)
            client.optimizer.zero_grad()
            loss.backward()
            client.optimizer.step()

            print(f"[Round {steps+1}] Loss: {loss.item():.4f}")

        if (steps+1) % 50 == 0:
            save_path = os.path.join(checkpoint_dir, f'client_0_round_{steps+1}.pth')
            torch.save(client.state_dict(), save_path)


if __name__ == "__main__":
    main()





