import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.1, n_iterations=100, threshold=0.5):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.threshold = threshold
        self.weights = np.zeros(2)
        self.bias = 0

    def activation_function(self, x):
        return 1 if x >= self.threshold else 0

    def predict(self, inputs):
        linear_output = np.dot(inputs, self.weights) + self.bias
        y_predicted = self.activation_function(linear_output)
        return y_predicted

    def train(self, X, y):
        for _ in range(self.n_iterations):
            for x, y_true in zip(X, y):
                y_pred = self.predict(x)
                error = y_true - y_pred
                self.weights += error * self.learning_rate * x
                self.bias += error * self.learning_rate

def test_perceptron(perceptron, X, y, logic_gate):
    print(f"Testing {logic_gate} Gate")
    for x, y_true in zip(X, y):
        y_pred = perceptron.predict(x)
        print(f"Input: {x}, Expected: {y_true}, Predicted: {y_pred}")

if __name__ == "__main__":
    # Define input for logic gates
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    # Define expected output for AND, OR, and NAND gates
    y_and = np.array([0, 0, 0, 1])
    y_or = np.array([0, 1, 1, 1])
    y_nand = np.array([1, 1, 1, 0])

    # Initialize and train perceptrons for AND, OR, and NAND
    perceptron_and = Perceptron()
    perceptron_and.train(X, y_and)
    test_perceptron(perceptron_and, X, y_and, "AND")

    perceptron_or = Perceptron()
    perceptron_or.train(X, y_or)
    test_perceptron(perceptron_or, X, y_or, "OR")

    perceptron_nand = Perceptron()
    perceptron_nand.train(X, y_nand)
    test_perceptron(perceptron_nand, X, y_nand, "NAND")

    # Note on XOR:
    print("\nNote: A single-layer perceptron cannot solve the XOR problem due to its non-linear separability.")
