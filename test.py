# eval_gcn.py
import argparse
import torch
from client import GCNClient  # 你写好的组合模型
import os
import random
import numpy as np
from torch_geometric.loader import DataLoader
from sklearn.metrics import precision_score, recall_score


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    for batch in dataloader:
        batch = batch.to(device)

        # 前向传播
        logits, _ = model(batch, batch.target_labels)  # shape: [batch_size, num_classes]
        preds = torch.argmax(logits, dim=1)  # 每个样本预测一个类别索引

        all_preds.append(preds.cpu())
        all_labels.append(batch.target_labels.cpu())

    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    precision = precision_score(all_labels, all_preds, average='micro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='micro', zero_division=0)

    return precision, recall



def main():
    set_seed(42)

    device = torch.device('cpu')

    # === 加载数据 ===
    data_path = './Dataset/BlogCatalog/client0_train.pt'
    data = torch.load(data_path)
    data = data.to(device)
    test_loader = DataLoader([data], batch_size=32, shuffle=False)

    if not hasattr(data, 'user_ids'):
        raise ValueError("Data must contain `user_ids` for output dimension")

    max_uid = int(data.user_ids.max().item())

    # === 创建模型并加载参数 ===
    model = GCNClient(
        in_dim=39,
        gcn_hidden_dim=32,
        gcn_out_dim=64,
        pred_hidden_dim=96,
        out_dim=max_uid + 1,
        gcn_layers=3,
        pred_layers=4,
        gcn_dropout=0.2,
        pred_dropout=0.2
    ).to(device)
    model.create_optimizer(lr=5e-3, weight_decay=1e-5)

    model_path = 'Check1/client_0_round_50.pth'
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)

    train_acc = evaluate(model, test_loader, device)
    val_acc = evaluate(model, test_loader, device)
    test_acc = evaluate(model, test_loader, device)

    print(f"[Eval] Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}")


if __name__ == '__main__':
    main()
