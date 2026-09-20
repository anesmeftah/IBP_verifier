import torch
import torch.nn as nn


class ToyModel(nn.Module):

    def __init__(self):
        super(ToyModel, self).__init__()
        self.fc1 = nn.Linear(1, 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(2, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def export_to_onnx(onnx_filename="models/toy_model.onnx"):
    model = ToyModel()
    model.eval()

    # Exemple d'entrée 1D avec une dimension de batch de 1 (shape: [1, 1])
    dummy_input = torch.tensor([[0.5]], dtype=torch.float32)

    # Exporter au format ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_filename,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=13,
    )

    print(f"Modèle ONNX généré avec succès : {onnx_filename}")


if __name__ == "__main__":
    export_to_onnx()