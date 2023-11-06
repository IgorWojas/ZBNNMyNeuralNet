// neural_network.h

#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include <cmath>
#include <fstream>
#include <random>
#include <iostream>
#include <ctime>

class NeuralNetwork {
private:
    int numLayers;  // Variable to store the number of layers in the network
    std::vector<int> numNeurons; // Vector to store the number of neurons in each layer
    std::vector<std::vector<std::vector<double>>> weights; // Vector to store weights
    std::string activationFunction; // Variable to store the chosen activation function

    double activate(double x); // Activation function
    double leakyReLU(double x, double alpha);
    double elu(double x, double alpha);
    double softplus(double x);
    double stepFunction(double x);
    double identity(double x);

public:
    NeuralNetwork(int layers, const std::vector<int>& neurons);  // Constructor to initialize layers and neurons
    void setNumLayers(int layers);  // Function to set the number of layers
    void setNumNeurons(const std::vector<int>& neurons); // Function to set the number of neurons in each layer
    void initializeWeights(double min, double max); // Function to populate weights with random values
    void setActivationFunction(std::string function); // Function to set activation function
    std::vector<double> feedForward(const std::vector<double>& input); // Function for forward pass
    void backpropagate(const std::vector<double>& input, const std::vector<double>& target, double learningRate); // Function for backpropagation
    void saveWeightsToFile(const std::string& filename); // Function to save weights to a binary file
    void loadWeightsFromFile(const std::string& filename); // Function to load weights from a binary file
};

inline NeuralNetwork::NeuralNetwork(int layers, const std::vector<int>& neurons) {
    numLayers = layers;
    numNeurons = neurons;
    activationFunction = "sigmoid"; // Default activation function is sigmoid
}

inline void NeuralNetwork::setNumLayers(int layers) {
    numLayers = layers;
}

inline void NeuralNetwork::setNumNeurons(const std::vector<int>& neurons) {
    numNeurons = neurons;
}

inline void NeuralNetwork::initializeWeights(double min, double max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(min, max);

    weights.clear(); // Clear any existing weights

    for (int i = 1; i < numLayers; ++i) {
        std::vector<std::vector<double>> layerWeights;
        for (int j = 0; j < numNeurons[i]; ++j) {
            std::vector<double> neuronWeights;
            for (int k = 0; k < numNeurons[i - 1]; ++k) {
                neuronWeights.push_back(dis(gen));
            }
            layerWeights.push_back(neuronWeights);
        }
        weights.push_back(layerWeights);
    }
}

inline void NeuralNetwork::setActivationFunction(std::string function) {
    activationFunction = function;
}

inline double NeuralNetwork::activate(double x) {
    if (activationFunction == "sigmoid") {
        return 1 / (1 + exp(-x));
    } else if (activationFunction == "relu") {
        return (x > 0) ? x : 0;
    } else if (activationFunction == "tanh") {
        return tanh(x);
    } else if (activationFunction == "leaky_relu") {
        return leakyReLU(x, 0.01); // You can adjust the alpha parameter as needed
    } else if (activationFunction == "elu") {
        return elu(x, 1.0); // You can adjust the alpha parameter as needed
    } else if (activationFunction == "softplus") {
        return softplus(x);
    } else if (activationFunction == "step") {
        return stepFunction(x);
    } else if (activationFunction == "identity") {
        return identity(x);
    } else {
        // Add more activation functions as needed
        return x; // Default to identity function
    }
}

inline double NeuralNetwork::leakyReLU(double x, double alpha) {
    return (x > 0) ? x : alpha * x;
}

inline double NeuralNetwork::elu(double x, double alpha) {
    return (x > 0) ? x : alpha * (exp(x) - 1);
}

inline double NeuralNetwork::softplus(double x) {
    return log(1 + exp(x));
}

inline double NeuralNetwork::stepFunction(double x) {
    return (x > 0) ? 1 : 0;
}

inline double NeuralNetwork::identity(double x) {
    return x;
}

inline std::vector<double> NeuralNetwork::feedForward(const std::vector<double>& input) {
    std::vector<std::vector<double>> activations; // Store activations for each layer
    std::vector<std::vector<double>> sums; // Store sums for each layer

    // Initialize input layer
    activations.push_back(input);
    sums.push_back(input);

    // Forward pass
    for (int i = 1; i < numLayers; ++i) {
        std::vector<double> layerActivations;
        std::vector<double> layerSums;

        for (int j = 0; j < numNeurons[i]; ++j) {
            double sum = 0.0;
            for (int k = 0; k < numNeurons[i - 1]; ++k) {
                sum += activations[i - 1][k] * weights[i - 1][j][k];
            }
            layerSums.push_back(sum);
            layerActivations.push_back(activate(sum));
        }

        activations.push_back(layerActivations);
        sums.push_back(layerSums);
    }

    return activations.back(); // Return output of the last layer
}

inline void NeuralNetwork::backpropagate(const std::vector<double>& input, const std::vector<double>& target, double learningRate) {
    std::vector<std::vector<double>> activations; // Store activations for each layer
    std::vector<std::vector<double>> sums; // Store sums for each layer

    // Initialize input layer
    activations.push_back(input);
    sums.push_back(input);

    // Forward pass
    for (int i = 1; i < numLayers; ++i) {
        std::vector<double> layerActivations;
        std::vector<double> layerSums;

        for (int j = 0; j < numNeurons[i]; ++j) {
            double sum = 0.0;
            for (int k = 0; k < numNeurons[i - 1]; ++k) {
                sum += activations[i - 1][k] * weights[i - 1][j][k];
            }
            layerSums.push_back(sum);
            layerActivations.push_back(activate(sum));
        }

        activations.push_back(layerActivations);
        sums.push_back(layerSums);
    }

    // Backpropagation
    std::vector<std::vector<double>> deltas; // Store deltas for each layer

    // Calculate output layer delta
    std::vector<double> outputDeltas;
    for (int i = 0; i < numNeurons.back(); ++i) {
        double output = activations.back()[i];
        outputDeltas.push_back(output * (1 - output) * (target[i] - output));
    }
    deltas.push_back(outputDeltas);

    // Calculate deltas for hidden layers
    for (int i = numLayers - 2; i > 0; --i) {
        std::vector<double> layerDeltas;
        for (int j = 0; j < numNeurons[i]; ++j) {
            double sum = 0.0;
            for (int k = 0; k < numNeurons[i + 1]; ++k) {
                sum += weights[i][k][j] * deltas[0][k];
            }
            double activation = activations[i][j];
            layerDeltas.push_back(activation * (1 - activation) * sum);
        }
        deltas.insert(deltas.begin(), layerDeltas);
    }

    // Update weights
    for (int i = 0; i < numLayers - 1; ++i) {
        for (int j = 0; j < numNeurons[i + 1]; ++j) {
            for (int k = 0; k < numNeurons[i]; ++k) {
                weights[i][j][k] += learningRate * deltas[i][j] * activations[i][k];
            }
        }
    }
}

inline void NeuralNetwork::saveWeightsToFile(const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (file.is_open()) {
        for (int i = 0; i < numLayers - 1; ++i) {
            for (int j = 0; j < numNeurons[i + 1]; ++j) {
                for (int k = 0; k < numNeurons[i]; ++k) {
                    file.write(reinterpret_cast<char*>(&weights[i][j][k]), sizeof(double));
                }
            }
        }
        file.close();
    } else {
        std::cerr << "Error: Unable to open file for writing." << std::endl;
    }
}

inline void NeuralNetwork::loadWeightsFromFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (file.is_open()) {
        // Clear any existing weights
        weights.clear();

        // Initialize weights with proper dimensions
        for (int i = 1; i < numLayers; ++i) {
            std::vector<std::vector<double>> layerWeights;
            for (int j = 0; j < numNeurons[i]; ++j) {
                std::vector<double> neuronWeights;
                for (int k = 0; k < numNeurons[i - 1]; ++k) {
                    neuronWeights.push_back(0.0); // Initialize to 0 (or any value you prefer)
                }
                layerWeights.push_back(neuronWeights);
            }
            weights.push_back(layerWeights);
        }

        for (int i = 0; i < numLayers - 1; ++i) {
            for (int j = 0; j < numNeurons[i + 1]; ++j) {
                for (int k = 0; k < numNeurons[i]; ++k) {
                    file.read(reinterpret_cast<char*>(&weights[i][j][k]), sizeof(double));
                }
            }
        }
        file.close();
    } else {
        std::cerr << "Error: Unable to open file for reading." << std::endl;
    }
}

int getRandomNumber(int min, int max) {                // dodane ze starszego headera
    // Seed the random number generator with the current time
    std::srand(std::time(0));

    // Generate a random number within the specified range
    return min + std::rand() % (max - min + 1);
}

double generateRandomDouble(double min, double max) {
    // Seed the random number generator
    std::srand(std::time(0));

    // Generate a random double between 0 and 1
    double randomDouble = (std::rand() / (RAND_MAX + 1.0));

    // Scale and shift the random double to fit the desired range
    return min + randomDouble * (max - min);
}


#endif
