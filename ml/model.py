
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import HeteroConv, SAGEConv


# ========== 1. 模态投影层 ==========

class ModalityProjection(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


# ========== 2. 门控融合层 ==========

class GatedFusion(nn.Module):
    def __init__(self, hidden_dim, num_modalities=2):
        super().__init__()

        self.gate = nn.Linear(
            hidden_dim * num_modalities,
            num_modalities
        )

    def forward(self, features, masks=None):
        # features: 每个模态的 [N, hidden_dim] 向量
        x = torch.stack(features, dim=1)

        combined = torch.cat(features, dim=-1)
        scores = self.gate(combined)

        if masks is not None:
            masks = masks.bool()

            # 每个节点至少有一种可用模态
            if not masks.any(dim=1).all():
                raise ValueError("存在没有可用模态的节点")

            scores = scores.masked_fill(
                ~masks,
                torch.finfo(scores.dtype).min
            )

        weights = F.softmax(scores, dim=-1)

        fused = (
            x * weights.unsqueeze(-1)
        ).sum(dim=1)

        return fused, weights


# ========== 3. 多模态 GraphSAGE 模型 ==========

class MultimodalGraphSAGE(nn.Module):

    def __init__(
    self,
    hidden_dim=128,
    dropout=0.3,
    use_review_text=True,
    use_numeric=True,
    use_product_text=True,
    use_image=True,
    use_gnn=True
):
        super().__init__()

        self.dropout = dropout

        # ---- 评论节点模态投影 ----

        self.review_text_proj = ModalityProjection(
            384, hidden_dim
        )

        self.review_numeric_proj = ModalityProjection(
            4, hidden_dim
        )

        self.review_fusion = GatedFusion(
            hidden_dim, num_modalities=2
        )

        # ---- 商品节点模态投影 ----

        self.product_text_proj = ModalityProjection(
            384, hidden_dim
        )

        self.product_image_proj = ModalityProjection(
            512, hidden_dim
        )

        self.product_fusion = GatedFusion(
            hidden_dim, num_modalities=2
        )

        # ---- 第一层异构 GraphSAGE ----

        self.conv1 = HeteroConv(
            {
                (
                    "review",
                    "belongs_to",
                    "product"
                ): SAGEConv(
                    (-1, -1), hidden_dim
                ),

                (
                    "product",
                    "has_review",
                    "review"
                ): SAGEConv(
                    (-1, -1), hidden_dim
                )
            },
            aggr="sum"
        )

        # ---- 第二层异构 GraphSAGE ----

        self.conv2 = HeteroConv(
            {
                (
                    "review",
                    "belongs_to",
                    "product"
                ): SAGEConv(
                    (-1, -1), hidden_dim
                ),

                (
                    "product",
                    "has_review",
                    "review"
                ): SAGEConv(
                    (-1, -1), hidden_dim
                )
            },
            aggr="sum"
        )

        # ---- 节点特征残差连接 ----

        self.review_norm1 = nn.LayerNorm(hidden_dim)
        self.product_norm1 = nn.LayerNorm(hidden_dim)

        self.review_norm2 = nn.LayerNorm(hidden_dim)
        self.product_norm2 = nn.LayerNorm(hidden_dim)

        # ---- 评论有用性分类头 ----

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, data, return_attention=False):

        # ========== 4.1 评论节点特征 ==========

        review_text = self.review_text_proj(
            data["review"].text
        )

        review_numeric = self.review_numeric_proj(
            data["review"].numeric
        )

        review_x, review_weights = self.review_fusion(
            [review_text, review_numeric]
        )

        # ========== 4.2 商品节点特征 ==========

        product_text = self.product_text_proj(
            data["product"].text
        )

        product_image = self.product_image_proj(
            data["product"].image
        )

        image_mask = (
            data["product"].image_mask > 0
        )

        text_mask = torch.ones_like(
            image_mask,
            dtype=torch.bool
        )

        product_masks = torch.stack(
            [text_mask, image_mask],
            dim=1
        )

        product_x, product_weights = self.product_fusion(
            [product_text, product_image],
            masks=product_masks
        )

        # ========== 4.3 创建节点特征字典 ==========

        x_dict = {
            "review": review_x,
            "product": product_x
        }

        edge_index_dict = data.edge_index_dict

        # ========== 4.4 第一层 GraphSAGE ==========

        h_dict = self.conv1(
            x_dict,
            edge_index_dict
        )

        x_dict = {
            "review": self.review_norm1(
                x_dict["review"] + h_dict["review"]
            ),

            "product": self.product_norm1(
                x_dict["product"] + h_dict["product"]
            )
        }

        x_dict = {
            key: F.dropout(
                F.relu(value),
                p=self.dropout,
                training=self.training
            )
            for key, value in x_dict.items()
        }

        # ========== 4.5 第二层 GraphSAGE ==========

        h_dict = self.conv2(
            x_dict,
            edge_index_dict
        )

        x_dict = {
            "review": self.review_norm2(
                x_dict["review"] + h_dict["review"]
            ),

            "product": self.product_norm2(
                x_dict["product"] + h_dict["product"]
            )
        }

        x_dict = {
            key: F.relu(value)
            for key, value in x_dict.items()
        }

        # ========== 4.6 评论节点分类 ==========

        logits = self.classifier(
            x_dict["review"]
        ).squeeze(-1)

        if return_attention:
            return logits, {
                "review_weights": review_weights,
                "product_weights": product_weights
            }

        return logits
