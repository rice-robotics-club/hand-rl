from abc import ABC, abstractmethod

# abstract neural network base class
class NeuralNetwork(ABC):
    def __init__(self):
        self.optimizer = None
        self.loss = None
        self.optimizer_params = None
        self.model = None
        self.model_params = None
        self.training_history = None
        self.testing_history = None

    @abstractmethod
    def train(self, train_data, val_data=None):
        pass

    @abstractmethod
    def test(self, test_data):
        pass

    @abstractmethod
    def predict(self, data):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass

# rsl_rl neural network implementation
class RslRlNeuralNetwork(NeuralNetwork):
    def __init__(self):
        super().__init__()
        self.optimizer = None
        self.loss = None
        self.optimizer_params = None
        self.model = None
        self.model_params = None
        self.training_history = None
        self.testing_history = None

    def train(self, train_data, val_data=None):
        # RSL RL training loop

    def test(self, test_data):
        pass

    def predict(self, data):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass
