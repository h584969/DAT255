import torch
import numpy as np
import base64
from core import *
from torch.nn.modules import Module

layer_encodings = {
    torch.nn.Linear: LinearEncoder,
    torch.nn.ReLU: ReLUEncoder,
    torch.nn.Sigmoid: SigmoidEncoder
}

class ModelEncoder:
    def __init__(self) -> None:
        self.encoders = layer_encodings

    def encode(self, model: torch.nn.Module):
        layers = self.encode_module("",model)

        out = []

        for layer in layers:
            params = []
            for param in layer.parameters():
                data = to_bytes(param.tensor())
                print(data)
                value = base64.b64encode(data).decode()
                params.append(f"""
                    {{
                        "name": "{param.name()}",
                        "shape": {str(list(param.shape()))},
                        "value": "{value}"
                    }}
                """)
            
            out.append(f"""
            {{
                "name": "{layer.name()}",
                "kind": "{layer.layer_kind()}",
                "parameters": [
                    {','.join(params)}   
                ]
            }}
            """)

        return f"""{{ "layers": [{','.join(out)}]}}""".replace("\n","").replace("\r","").replace(" ", "")

    def encode_module(self, name: str, module: torch.nn.Module) -> list[LayerEncoder]:
        out = []

        module_class = type(module)

        if module_class in layer_encodings:
            layer_encoder = layer_encodings[module_class](name, module)
            out.append(layer_encoder)
            return out

        if isinstance(module, torch.nn.Sequential):
            for i, x in enumerate(module):
                encoded = self.encode_module(f'{name}_{i}', x)
                out += encoded
            return out

        raise UnknownLayerError(module_class.__name__)
    

def to_bytes(tensor: torch.Tensor):
    arr: np.ndarray = tensor.detach().numpy()
    return arr.tobytes()
    