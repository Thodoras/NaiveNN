from typing import List, Tuple
import torch
import logging

class HiddenLayer:
    
    def __init__(self, layer_size: int, dropout: float = 0.0):
        self.layer_size = layer_size
        self.dropout = dropout
        
    def validate(self) -> bool:
        if self.layer_size <= 0:
            logging.warn("Invalid hidden layer size.")
            return False
            
        if self.dropout < 0.0 or self.dropout > 1.0:
            logging.warn("Invalid dropout value.")
            return False
        
        return True

class Layers:
    
    def __init__(self, input_size: int, output_size: int, hidden_layers: List[HiddenLayer]):
        self._input_size = input_size
        self._output_size = output_size
        self._hidden_layers = hidden_layers
        
    def toSequence(self) -> List:
        if len(self._hidden_layers) == 0:
            return [torch.nn.Linear(self._input_size, self._output_size)]
        
        prev_size = self._input_size
        results = []
        for hidden_layer in self._hidden_layers:
            results.append(torch.nn.Linear(prev_size, hidden_layer.layer_size))
            results.append(torch.nn.ReLU())
            results.append(torch.nn.Dropout(hidden_layer.dropout))
            prev_size = hidden_layer.layer_size
            
        results.append(torch.nn.Linear(prev_size, self._output_size))
        return results
    
    def validate(self) -> bool:
        if self._input_size <= 0:
            logging.warn("Invalid input size.")
            return False
        
        if self._output_size <= 0:
            logging.warn("Invalid output size.")
            return False
        
        for hidden_layer in self._hidden_layers:
            if not hidden_layer.validate():
                return False
        
        return True
    

        
        
            