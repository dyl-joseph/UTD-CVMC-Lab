import numpy as np

class FCNet():
    def __init__(self, inp_size=784, hidden_size=128, num_classes=10, num_epohs = 100, batch_size=32):
        self.W1 = np.random.normal(size=(hidden_size,inp_size))
        self.b1 = np.zeros(shape=(hidden_size,))
        self.W2 = np.random.normal(size=(num_classes,hidden_size))
        self.b2 = np.zeros(shape=(num_classes))

    def forward(self, input):
        h_B = np.matmul(self.W1, input) + self.b1
    def backward(self, loss_grad, forward_cache):
        pass

    def train(self, train_images, train_labels):
        # split data into train and val sets

        # iterate through data in batches. In each iteration, call forwrad, backward, and update the parameters. Select batches randomly. Backwards will get you gradients. Save the best sets of parameters
        pass
    def evaluate(self, test_images, test_labels):
        pass
    
    def save_parameters(self, epoch):
        pass

    def load_parameters(self, W1_path, b1_path, W2_path, b2_path):
        pass