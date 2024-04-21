import torch

class UnknownLayerError(BaseException):
    def __init__(self, layer_class) -> None:
        super().__init__(f'{layer_class} is not reckognized.')

class Parameter:
    def __init__(self, name: str, data: torch.Tensor) -> None:
        self._name = name
        self._data = data

    def shape(self) -> torch.Size:
        return self._data.shape
    
    def name(self) -> str:
        return self._name
    
    def tensor(self) -> torch.Tensor:
        return self._data

class LayerEncoder:

    def __init__(self, name: str, layer: torch.nn.Module) -> None:
        self._name = name
        self.layer = layer

    def name(self) -> str:
        return self._name
    
    def layer_kind(self) -> str:
        raise NotImplementedError()

    def parameters(self) -> list[Parameter]:
        """The parameters this layer uses.
        
        """
        raise NotImplementedError()

    def input_shape(self) -> torch.Size | None:
        """Returns the input shape for the layer

        Returns
        -------
        `torch.Size`: if this layer has a defined input shape.
        `None`: if this layer's shape is defined by the input layer. Usually for mathematical layers like `torch.nn.ReLU`
        
        """
        raise NotImplementedError()
    
    def output_shape(self) -> torch.Size | None:
        """Returns the output shape for the layer

        Returns
        -------
        `torch.Size`: if this layer has a defined output shape.
        `None`: if this layer's shape is defined by the input layer. Usually for mathematical layers like `torch.nn.ReLU`
        
        """
        raise NotImplementedError()
    
    def layer_kind(self) -> str:
        """What kind of layer this is.
        
        """


class LinearEncoder(LayerEncoder):
    def __init__(self,name: str, layer: torch.nn.Linear) -> None:
        super().__init__(name, layer)
        self.layer = layer

    # 1d shape
    def input_shape(self) -> torch.Size:
        return (self.layer.in_features,)
    
    def output_shape(self) -> torch.Size:
        return (self.layer.out_features,)
    

    def layer_kind(self) -> str:
        return "Linear"
    
    def parameters(self) -> list[Parameter]:
        return [
            Parameter("weight",self.layer.weight),
            Parameter("bias", self.layer.bias)
        ]
    
class ReLUEncoder(LayerEncoder):
    def __init__(self, name: str, layer: torch.nn.ReLU) -> None:
        super().__init__(name, layer)
        self.layer = layer

    def input_shape(self) -> torch.Size:
        return None
    
    def output_shape(self) -> torch.Size | None:
        return None
    
    def parameters(self) -> list[Parameter]:
        return []
    
    def layer_kind(self) -> str:
        return "ReLU"

class SigmoidEncoder(LayerEncoder):
    def __init__(self, name: str, layer: torch.nn.Sigmoid) -> None:
        super().__init__(name, layer)
        self.layer = layer
    
    def input_shape(self) -> torch.Size | None:
        return None
    
    def output_shape(self) -> torch.Size | None:
        return None
    
    def parameters(self) -> list[Parameter]:
        return []
    
    def layer_kind(self) -> str:
        return "Sigmoid"