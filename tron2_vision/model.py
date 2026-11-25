"""A compact pretrained ResNet18 feature-pyramid segmenter; no SAM3 at runtime."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18

from .annotations import sha256

MODEL_CONTRACT = "top-resnet18-fpn-v1"
CACHE_CONTRACT = "top-foreground-student-v1"


def block(input_channels, output_channels=32):
    return nn.Sequential(nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
                         nn.GroupNorm(8, output_channels), nn.ReLU(inplace=True))


class ForegroundNet(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.encoder = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.encoder.avgpool = nn.Identity()
        self.encoder.fc = nn.Identity()
        self.lateral = nn.ModuleList([nn.Conv2d(c, 32, 1) for c in (64, 64, 128, 256, 512)])
        self.fuse = nn.ModuleList([block(32) for _ in range(5)])
        self.head = nn.Sequential(block(32), nn.Conv2d(32, 3, 1))
        self.register_buffer("mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def train(self, mode=True):
        super().train(mode)
        # Preserve pretrained BN statistics for the small annotation batches.
        for module in self.encoder.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        return self

    def forward(self, rgb):
        x = (rgb - self.mean) / self.std
        encoder = self.encoder
        x0 = encoder.relu(encoder.bn1(encoder.conv1(x)))
        x1 = encoder.layer1(encoder.maxpool(x0))
        x2 = encoder.layer2(x1)
        x3 = encoder.layer3(x2)
        x4 = encoder.layer4(x3)
        features = [x0, x1, x2, x3, x4]
        fused = None
        for i in range(4, -1, -1):
            lateral = self.lateral[i](features[i])
            fused = lateral if fused is None else lateral + F.interpolate(fused, size=lateral.shape[-2:], mode="bilinear", align_corners=False)
            fused = self.fuse[i](fused)
        return F.interpolate(self.head(fused), size=rgb.shape[-2:], mode="bilinear", align_corners=False)


class ForegroundPredictor:
    """The same current-RGB API is used by offline export and future live input."""
    def __init__(self, checkpoint, device="cuda"):
        self.device = torch.device(device)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state["contract"] != MODEL_CONTRACT:
            raise ValueError("Unsupported foreground model")
        self.metadata = state["metadata"]
        self.input_size = tuple(self.metadata["input_size"])
        self.model = ForegroundNet().to(self.device).eval()
        self.model.load_state_dict(state["model"])
        self.checkpoint_sha256 = sha256(checkpoint)

    @torch.inference_mode()
    def predict(self, rgb):
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError("Expected uint8 RGB [H,W,3]")
        height, width = rgb.shape[:2]
        x = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).to(self.device).float()[None] / 255
        x = F.interpolate(x, size=self.input_size, mode="bilinear", align_corners=False)
        logits = self.model(x)
        logits = F.interpolate(logits, size=(height, width), mode="bilinear", align_corners=False)
        probability = logits.softmax(1)
        confidence, label = probability.max(1)
        label = label[0].byte().cpu().numpy()
        return {"labels": label, "hand": label == 1, "objects": label == 2,
                "union": label != 0, "rgb": np.where((label != 0)[..., None], rgb, 0),
                "mean_confidence": float(confidence.mean())}
