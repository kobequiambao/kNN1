import torch
import torch.nn as nn
import torch.nn.init


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, num_classes, list_hidden, activation="sigmoid"):
        """Class constructor for NeuralNetwork.

        Arguments:
            input_size {int} -- Number of features in the dataset.
            num_classes {int} -- Number of classes in the dataset.
            list_hidden {list} -- List of integers representing the number of
                                  units per hidden layer in the network.
            activation {str, optional} -- Type of activation function. Choices
                                          include 'sigmoid', 'tanh', and 'relu'.
        """
        super(NeuralNetwork, self).__init__()

        self.input_size = input_size
        self.num_classes = num_classes
        self.list_hidden = list_hidden
        self.activation = activation

    def create_network(self):
        """Creates the layers of the neural network."""
        print(f"List Hidden: {self.list_hidden}")
        layers = []

        # First layer (Input layer → First Hidden Layer)
        layers.append(nn.Linear(self.input_size, self.list_hidden[0]))
        layers.append(self.get_activation(self.activation))  # Apply activation function

        # Hidden layers
        for i in range(len(self.list_hidden) - 1):
            layers.append(nn.Linear(self.list_hidden[i], self.list_hidden[i + 1]))
            layers.append(self.get_activation(self.activation))

        # Output layer (Last Hidden Layer → Output Layer)
        layers.append(nn.Linear(self.list_hidden[-1], self.num_classes))

        # Apply softmax activation to the output layer
        layers.append(nn.Softmax(dim=1))

        self.layers = nn.Sequential(*layers)

    def init_weights(self):
        """Initializes the weights of the network. Weights of a
        torch.nn.Linear layer should be initialized from a normal
        distribution with mean 0 and standard deviation 0.1. Bias terms of a
        torch.nn.Linear layer should be initialized with a constant value of 0.
        """
        torch.manual_seed(2)

        # For each layer in the network
        for module in self.modules():
            # If it is a torch.nn.Linear layer
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=0.1)
                nn.init.constant_(module.bias, 0)

    def get_activation(self, mode="sigmoid"):
        """Returns the torch.nn layer for the activation function.

        Arguments:
            mode {str, optional} -- Type of activation function. Choices
                                    include 'sigmoid', 'tanh', and 'relu'.

        Returns:
            torch.nn -- torch.nn layer representing the activation function.
        """
        activation = nn.Sigmoid()

        if mode == "tanh":
            activation = nn.Tanh()

        elif mode == "relu":
            activation = nn.ReLU(inplace=True)

        return activation

    def forward_manual(self, x, verbose=False):
        """Forward propagation of the model, implemented manually.

        Arguments:
            x {torch.Tensor} -- A Tensor of shape (N, D) representing input
                                features to the model.
            verbose {bool, optional} -- Indicates if the function prints the
                                        output or not.

        Returns:
            torch.Tensor, torch.Tensor -- A Tensor of shape (N, C) representing
                                          the output of the final linear layer in the network.
                                          A Tensor of shape (N, C) representing the probabilities
                                          of each class given by the softmax function.
        """
        for i in range(len(self.layers) - 1):
            layer = self.layers[i]

            if isinstance(layer, nn.Linear):
                x = torch.matmul(x, layer.weight.T) + layer.bias  # Linear transformation
            else:
                x = layer(x)  # Apply activation function

            if verbose:
                print(f"Output of layer {i}:")
                print(x, "\n")

        probabilities = self.layers[-1](x)  # Apply softmax

        if verbose:
            print(f"Output of layer {len(self.layers) - 1}:")
            print(probabilities, "\n")

        return x, probabilities

    def forward(self, x, verbose=False):
        """Forward propagation of the model, implemented using PyTorch.

        Arguments:
            x {torch.Tensor} -- A Tensor of shape (N, D) representing input
                                features to the model.
            verbose {bool, optional} -- Indicates if the function prints the
                                        output or not.

        Returns:
            torch.Tensor, torch.Tensor -- A Tensor of shape (N, C) representing
                                          the output of the final linear layer in the network.
                                          A Tensor of shape (N, C) representing the probabilities
                                          of each class given by the softmax function.
        """

        # For each layer in the network
        for i in range(len(self.layers) - 1):
            # Call the forward() function of the layer
            # and return the result to x.
            x = self.layers[i](x)

            if verbose:
                # Print the output of the layer
                print("Output of layer " + str(i))
                print(x, "\n")

        # Apply the softmax function
        probabilities = self.layers[-1](x)

        if verbose:
            print("Output of layer " + str(len(self.layers) - 1))
            print(probabilities, "\n")

        return x, probabilities

    def predict(self, probabilities):
        """Returns the index of the class with the highest probability.

        Arguments:
            probabilities {torch.Tensor} -- A Tensor of shape (N, C)
                                            representing the probabilities of N instances for C classes.

        Returns:
            torch.Tensor -- A Tensor of shape (N, ) containing the indices of
                            the class with the highest probability for N instances.
        """
        return torch.argmax(probabilities, dim=1)
