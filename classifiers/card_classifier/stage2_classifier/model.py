"""
Multi-Head Card Attribute Classifier Model
==========================================
Architecture:
- Shared feature extraction backbone (MobileNetV3-Small / ResNet18)
- 6 task-specific linear heads:
  1. Rank (14 classes: 2..A + Stone) -> Softmax
  2. Suit (5 classes: Hearts, Diamonds, Clubs, Spades, None) -> Softmax
  3. Edition (4 classes: None, Foil, Holo, Polychrome) -> Softmax
  4. Enhancement (9 classes: None, Bonus, Mult, Wild, Glass, Steel, Gold, Lucky, Stone) -> Softmax
  5. Seal (5 classes: None, Gold, Red, Blue, Purple) -> Softmax
  6. Selected (1 binary logit) -> Sigmoid
"""

import torch
import torch.nn as nn
import torchvision.models as models

try:
    from .config import HEAD_CONFIG
except ImportError:
    from config import HEAD_CONFIG


class MultiHeadCardClassifier(nn.Module):
    def __init__(self, backbone_name="mobilenet_v3_small", pretrained=True, dropout=0.2):
        super().__init__()
        self.backbone_name = backbone_name

        if backbone_name == "mobilenet_v3_small":
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            base = models.mobilenet_v3_small(weights=weights)
            self.features = base.features
            self.avgpool = base.avgpool
            self.fc_shared = nn.Sequential(
                base.classifier[0],  # Linear(576, 1024)
                base.classifier[1],  # Hardswish()
                nn.Dropout(p=dropout, inplace=True)
            )
            feat_dim = 1024
        elif backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            base = models.resnet18(weights=weights)
            self.features = nn.Sequential(
                base.conv1, base.bn1, base.relu, base.maxpool,
                base.layer1, base.layer2, base.layer3, base.layer4
            )
            self.avgpool = base.avgpool
            self.fc_shared = nn.Sequential(
                nn.Dropout(p=dropout)
            )
            feat_dim = base.fc.in_features
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # Attribute-specific output classification heads
        self.head_rank = nn.Linear(feat_dim, HEAD_CONFIG["rank"])
        self.head_suit = nn.Linear(feat_dim, HEAD_CONFIG["suit"])
        self.head_edition = nn.Linear(feat_dim, HEAD_CONFIG["edition"])
        self.head_enhancement = nn.Linear(feat_dim, HEAD_CONFIG["enhancement"])
        self.head_seal = nn.Linear(feat_dim, HEAD_CONFIG["seal"])
        self.head_selected = nn.Linear(feat_dim, HEAD_CONFIG["selected"])

    def forward(self, x):
        """
        Forward pass for a batch of card crop images.
        Args:
            x: Tensor of shape (B, 3, CROP_HEIGHT, CROP_WIDTH)
        Returns:
            Dict containing raw logits for each attribute head.
        """
        feats = self.features(x)
        pooled = self.avgpool(feats)
        flattened = torch.flatten(pooled, 1)
        shared = self.fc_shared(flattened)

        return {
            "rank": self.head_rank(shared),
            "suit": self.head_suit(shared),
            "edition": self.head_edition(shared),
            "enhancement": self.head_enhancement(shared),
            "seal": self.head_seal(shared),
            "selected": self.head_selected(shared).squeeze(-1)
        }

    def predict_attributes(self, x):
        """
        Runs inference and converts logits into class indices and confidence scores.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            
            p_rank = torch.softmax(logits["rank"], dim=-1)
            p_suit = torch.softmax(logits["suit"], dim=-1)
            p_edition = torch.softmax(logits["edition"], dim=-1)
            p_enhancement = torch.softmax(logits["enhancement"], dim=-1)
            p_seal = torch.softmax(logits["seal"], dim=-1)
            p_selected = torch.sigmoid(logits["selected"])

            return {
                "rank_idx": p_rank.argmax(dim=-1),
                "rank_conf": p_rank.max(dim=-1).values,
                "suit_idx": p_suit.argmax(dim=-1),
                "suit_conf": p_suit.max(dim=-1).values,
                "edition_idx": p_edition.argmax(dim=-1),
                "edition_conf": p_edition.max(dim=-1).values,
                "enhancement_idx": p_enhancement.argmax(dim=-1),
                "enhancement_conf": p_enhancement.max(dim=-1).values,
                "seal_idx": p_seal.argmax(dim=-1),
                "seal_conf": p_seal.max(dim=-1).values,
                "selected": p_selected > 0.5,
                "selected_conf": p_selected
            }
