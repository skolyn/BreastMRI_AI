import torch
import torch.nn as nn
from torchvision import models


class EfficientNetClassifier(nn.Module):
    """EfficientNet-based classifier for breast MRI case-level classification.

    Supports EfficientNet variants from B0 to B7, with configurable
    input channels (e.g., 4 frames per slice).

    The model first processes each slice independently through EfficientNet,
    then aggregates the slice-level features into a case-level prediction
    using mean pooling across slices.

    Attributes:
        base (nn.Module): The EfficientNet backbone.
    """

    def __init__(self,
                 num_classes: int = 3,
                 in_channels: int = 4,
                 version: str = "b0",
                 pretrained: bool = False):
        """Initialize the EfficientNetClassifier.

        Args:
            num_classes (int): Number of output classes (e.g., benign=0,
                malignant=1, normal=2). Defaults to 3.
            in_channels (int): Number of input channels per image.
                For DCE-MRI with 4 frames, use 4. Defaults to 4.
            version (str): Which EfficientNet variant to use.
                Options: "b0", "b1", ..., "b7". Defaults to "b0".
            pretrained (bool): Whether to load pretrained ImageNet weights.
                If True, weights are loaded and the first conv layer is
                modified to handle `in_channels`. Defaults to False.
        """
        super(EfficientNetClassifier, self).__init__()

        constructor = getattr(models, f"efficientnet_{version}")
        weights = None
        if pretrained:
            weights = models.get_model_weights(f"efficientnet_{version}").DEFAULT

        self.base = constructor(weights=weights)

        first_conv = self.base.features[0][0]
        self.base.features[0][0] = nn.Conv2d(
            in_channels,
            first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=False,
        )

        in_feats = self.base.classifier[1].in_features
        self.base.classifier[1] = nn.Linear(in_feats, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the EfficientNet classifier.

        Args:
            x (torch.Tensor): Input tensor of shape
                (batch_size, num_slices, channels, height, width).

        Returns:
            torch.Tensor: Case-level classification logits of shape
                (batch_size, num_classes).
        """
        B, S, C, H, W = x.shape
        x = x.view(B * S, C, H, W)
        feats = self.base(x)

        feats = feats.view(B, S, -1)
        feats = feats.mean(dim=1)
        return feats
