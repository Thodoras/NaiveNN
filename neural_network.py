import torch

from layers import Layers

class NeuralNetwork(torch.nn.Module):
    
    def __init__(self, layers: Layers):
        super().__init__()  
        self.layers = torch.nn.Sequential(*layers.toSequence())
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)