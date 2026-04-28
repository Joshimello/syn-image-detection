from pathlib import Path

import torch
from PIL import Image

from me26sid.config import load_settings
from me26sid.data import EvalTransform
from me26sid.eval import predict_loader


def test_eval_transform_returns_five_crops(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("eval:\n  num_crops: 5\n", encoding="utf-8")
    settings = load_settings(config_path, run_name_override="smoke")
    image = Image.new("RGB", (320, 240), color=(128, 64, 32))

    transformed = EvalTransform(settings)(image)

    assert transformed.shape == (5, 3, settings.data.input_size, settings.data.input_size)


class DummyModel(torch.nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return torch.arange(images.shape[0], dtype=torch.float32, device=images.device)


def test_predict_loader_averages_multi_crop_logits() -> None:
    dataset = [
        (
            torch.zeros(5, 3, 224, 224),
            torch.tensor(1.0),
            "img.jpg",
            torch.tensor(320),
            torch.tensor(240),
        )
    ]
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)

    predictions = predict_loader(
        model=DummyModel(),
        loader=loader,
        device=torch.device("cpu"),
        amp=False,
    )

    assert len(predictions) == 1
    assert float(predictions.loc[0, "logit"]) == 2.0
