import typing

import onnx
import torch
from torchvision.models import resnet18

from src.modules_utils.object_detector import MembraneDetector

if __name__ == '__main__':
    model_state_dict_path: typing.Final[str] = '../models/Legionellas/membrane_detector.pth'
    inference_size: typing.Final[int] = 1024
    output_path = model_state_dict_path.replace('.pth', '.onnx')

    backbone = resnet18(weights=None)
    model = MembraneDetector(backbone, 1)
    model.load_state_dict(torch.load(model_state_dict_path))

    input = torch.randn(1, 3, inference_size, inference_size)

    torch.onnx.export(
        model,
        input,
        output_path,
        verbose=True,
        opset_version=11,
        output_names=['bboxes', 'class_logits', 'x_prob']
    )

    # load exported .onnx model
    onnx_model = onnx.load(output_path)
    # verify that the model can be imported
    onnx.checker.check_model(onnx_model)

