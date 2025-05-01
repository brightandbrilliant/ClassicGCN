import torch
import torch.nn as nn
from torch_geometric.data import Data
from gcn import GCNEmbedding, Predictor  # 确保这两个类在你项目中可导入


class GCNClient(nn.Module):
    def __init__(self, in_dim, gcn_hidden_dim, gcn_out_dim, pred_hidden_dim, out_dim,
                 gcn_layers=2, pred_layers=2, gcn_dropout=0.2, pred_dropout=0.5):
        super(GCNClient, self).__init__()
        self.embedding_model = GCNEmbedding(
            in_dim=in_dim,
            hidden_dim=gcn_hidden_dim,
            out_dim=gcn_out_dim,
            num_layers=gcn_layers,
            dropout=gcn_dropout
        )

        self.predictor = Predictor(
            in_dim=gcn_out_dim,
            hidden_dim=pred_hidden_dim,
            out_dim=out_dim,
            num_layers=pred_layers,
            dropout=pred_dropout
        )
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.optimizer = None

    def forward(self, data: Data, target_labels):
        """
        输入:
            data: PyG 的图对象，包含 x, edge_index
        输出:
            每个节点的 following 多标签预测 (logits)
        """
        x = data.x
        edge_index = data.edge_index

        node_embed = self.embedding_model(x, edge_index)
        pred_logits = self.predictor(node_embed)
        loss = self.loss_fn(pred_logits, target_labels)
        return pred_logits, loss

    def get_node_embedding(self, data: Data):
        """
        仅获取 GCN 输出的节点嵌入，用于中间查看或评估。
        """
        return self.embedding_model(data.x, data.edge_index)

    def create_optimizer(self, lr=1e-3, weight_decay=1e-5):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
