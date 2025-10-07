import torch

from torch.utils.data import DataLoader

from neural_network import NeuralNetwork

class Trainer:
    
    def __init__(
        self, 
        model: NeuralNetwork,
        train_loader: DataLoader, 
        test_loader: DataLoader, 
        epochs: int, 
        learning_rate: float
    ):
        self._model = model
        self._train_loader = train_loader
        self._test_loader = test_loader
        self._epochs = epochs
        self._criterion = torch.nn.CrossEntropyLoss()
        self._optimizer = torch.optim.SGD(self._model.parameters(), lr=learning_rate)
        
    def train(self):
        train_losses = []
        test_accuracies = []
        
        for epoch in range(self._epochs):
            running_loss = 0.0
            for i, (inputs, targets) in enumerate(self._train_loader):
                outputs = self._model(inputs)
                loss = self._criterion(outputs, targets)
                
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
                
                running_loss += loss.item()
        
            avg_train_loss = running_loss / len(self._train_loader)
            train_losses.append(avg_train_loss)
            
            test_accuracy = self._evaluate_model()
            test_accuracies.append(test_accuracy)
            
            print(f'Epoch [{epoch+1}/{self._epochs}], Train Loss: {avg_train_loss:.4f}, Val Accuracy: {test_accuracy:.2f}%')
        
        return train_losses, test_accuracies
            
    def _evaluate_model(self):
        self._model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in self._test_loader:
                outputs = self._model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.shape[0]
                correct += (predicted == targets).sum().item()
        accuracy = 100 * correct / total
        return accuracy