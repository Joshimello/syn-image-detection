from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

try:
    import open_clip
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("open-clip-torch must be installed to use the model module") from exc

from me26sid.config import ModelConfig


class CLIPIntermediateEncoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.clip = self._create_model(config)
        self.clip.eval()
        for parameter in self.clip.parameters():
            parameter.requires_grad = False

        visual = getattr(self.clip, "visual", None)
        if visual is None or not hasattr(visual, "transformer") or not hasattr(
            visual.transformer, "resblocks"
        ):
            raise ValueError("Configured CLIP model does not expose transformer resblocks")

        self.resblocks: Sequence[nn.Module] = list(visual.transformer.resblocks)
        self.num_blocks = len(self.resblocks)
        self.feature_dim = int(visual.transformer.width)

    def _create_model(self, config: ModelConfig) -> nn.Module:
        try:
            model, _, _ = open_clip.create_model_and_transforms(
                config.model_name,
                pretrained=config.pretrained,
                precision="fp32",
                device="cpu",
            )
        except TypeError:
            model, _, _ = open_clip.create_model_and_transforms(
                config.model_name,
                pretrained=config.pretrained,
            )
        return model.float()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        captures: list[torch.Tensor] = []
        hooks = [
            block.register_forward_hook(self._build_hook(captures, images.shape[0]))
            for block in self.resblocks
        ]
        try:
            _ = self.clip.encode_image(images, normalize=False)
        finally:
            for hook in hooks:
                hook.remove()
        if len(captures) != self.num_blocks:
            raise RuntimeError(f"Expected {self.num_blocks} block captures, found {len(captures)}")
        return torch.stack(captures, dim=1)

    def _build_hook(self, captures: list[torch.Tensor], batch_size: int):
        def hook(
            _module: nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            if isinstance(output, tuple):
                tensor = output[0]
            else:
                tensor = output
            if tensor.ndim != 3:
                raise RuntimeError(
                    f"Expected 3D transformer output, got shape {tuple(tensor.shape)}"
                )
            if tensor.shape[0] == batch_size:
                cls_token = tensor[:, 0, :]
            elif tensor.shape[1] == batch_size:
                cls_token = tensor[0, :, :]
            else:
                raise RuntimeError(f"Cannot infer CLS position from shape {tuple(tensor.shape)}")
            captures.append(cls_token.float())

        return hook


class IntermediateFusionHead(nn.Module):
    def __init__(self, num_blocks: int, feature_dim: int, config: ModelConfig) -> None:
        super().__init__()
        self.projection = nn.Linear(feature_dim, config.projection_dim)
        self.block_logits = nn.Parameter(torch.zeros(num_blocks))
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.projection_dim),
            nn.Dropout(config.dropout),
            nn.Linear(config.projection_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, block_features: torch.Tensor) -> torch.Tensor:
        projected = self.projection(block_features)
        weights = torch.softmax(self.block_logits, dim=0)
        fused = torch.sum(projected * weights.view(1, -1, 1), dim=1)
        logits = self.classifier(fused).squeeze(-1)
        return logits

    def get_block_weights(self) -> torch.Tensor:
        return torch.softmax(self.block_logits.detach().cpu(), dim=0)


class SyntheticImageDetector(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.encoder = CLIPIntermediateEncoder(config)
        self.head = IntermediateFusionHead(
            num_blocks=self.encoder.num_blocks,
            feature_dim=self.encoder.feature_dim,
            config=config,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            block_features = self.encoder(images)
        return self.head(block_features)

    def trainable_parameters(self):
        return self.head.parameters()
