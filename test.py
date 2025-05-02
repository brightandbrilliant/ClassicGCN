# test.py
import torch
import argparse
import os
from parse import SocialGraphDataset
from client import GCNClient
from torch_geometric.loader import DataLoader
from sklearn.metrics import precision_score, recall_score


def create_multi_label_targets(data):
    num_nodes = data.num_nodes
    edge_index = data.edge_index
    labels = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
    labels[edge_index[0], edge_index[1]] = 1.0
    return labels


def evaluate(model, data, device, threshold=0.05):
    model.eval()
    data = data.to(device)

    with torch.no_grad():
        preds, _ = model(data, data.target_labels.to(device))
        preds = preds.sigmoid()  # 多标签预测用 sigmoid
        preds_binary = (preds > threshold).float()

    y_true = data.target_labels.cpu().numpy()
    y_pred = preds_binary.cpu().numpy()

    # 展平为一维向量评估
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    precision = precision_score(y_true_flat, y_pred_flat, zero_division=0)
    recall = recall_score(y_true_flat, y_pred_flat, zero_division=0)

    return precision, recall


def main():

    device = torch.device('cpu')

    # 加载数据
    data = torch.load('./Raw_Dataset/BlogCatalog-dataset/processed/client_0.pt')
    data.user_ids = data.original_nodes
    data.target_labels = create_multi_label_targets(data)
    if not hasattr(data, 'batch') or data.batch is None:
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long)

    dataloader = DataLoader([data], batch_size=1)

    # 构建模型并加载参数
    model = GCNClient(
        in_dim=39,
        gcn_hidden_dim=128,
        gcn_out_dim=128,
        pred_hidden_dim=128,
        out_dim=data.num_nodes,  # 输出一个 [N, N] 的矩阵
        gcn_layers=3,
        pred_layers=6,
        gcn_dropout=0.4,
        pred_dropout=0.4
    ).to(device)
    model.load_state_dict(torch.load('./Checkpoints_Client_0/client_0_round_6000.pth', map_location=device))

    # 评估
    for batch in dataloader:
        precision, recall = evaluate(model, batch, device)
        print(f"Client {0} Evaluation Results:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")


if __name__ == "__main__":
    main()
