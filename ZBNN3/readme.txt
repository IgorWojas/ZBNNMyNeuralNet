A simple header only "library" created utilising ChatGPT. Fully connected network. Some neuron activation functions to choose from. used along with openCV.
zbnn4.h is a core for the network.
zbmat3.h is a header file with openCV operations - cv:Mat to vector and opposite

example cpp below





//////////////////////////////// LAST WORKING CPP ////////////////////////////////////



#include "zbnn4.h"
#include "zbmat3.h"
#include <iostream>
#include <chrono>
#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;


int main() {
    //licz czas
    auto start = std::chrono::system_clock::now();

    int x = 100;
    int y = 100;
    int xy = x * y;
    Mat inputImg = imread("wp1.bmp", IMREAD_GRAYSCALE);  // ZAWSZE GRAYSCALE!!!!!!!!!!
    Mat targetImg = imread("up1.bmp", IMREAD_GRAYSCALE); // ZAWSZE GRAYSCALE!!!!!!!!!!
    Mat queryImg = imread("wp2.bmp", IMREAD_GRAYSCALE);  // ZAWSZE GRAYSCALE!!!!!!!!!!
    targetImg.convertTo(targetImg, CV_64F);
    inputImg.convertTo(inputImg, CV_64F);
    queryImg.convertTo(queryImg, CV_64F);
    targetImg /= 255;
    inputImg /= 255;
    queryImg /= 255;
    // vin.total()  vin.channels()

    // Define the neural network
    NeuralNetwork train(4, { xy, xy, xy, xy }); //train(4, { xy, xy, xy, xy });
    cout << "network initialized" << endl;

    // Set the activation function
    train.setActivationFunction("sigmoid"); // tanh sigmoid relu leaky_relu elu softplus step identity

    //load
    train.loadWeightsFromFile("dupa.dat");
    cout << "weights loaded" << endl;

    // Initialize weights with random values between -1 and 1
    //train.initializeWeights(-0.1, 0.1);
    //cout << "random weights distributed" << endl;

    // Define the training data 
    std::vector<std::vector<double>> trainingInput;
    std::vector<std::vector<double>> trainingTarget;
    std::vector<std::vector<double>> queryInput;
    trainingInput.push_back(MaToVe(inputImg));
    trainingTarget.push_back(MaToVe(targetImg));
    queryInput.push_back(MaToVe(queryImg));
    /*for (int i = 0; i < trainingInput[0].size(); i++) {
        cout << trainingTarget[0][i] << "|";
    }*/
    cout << "images translated to vectors" << endl;
    cout << "vector samples: input:" << trainingInput[0][650] << " target:" << trainingTarget[0][650] << " query:" << queryInput[0][650] << endl;

    // Train the neural network
    double learningRate = 0.01;
    int numEpochs = 1;
    for (int epoch = 0; epoch < numEpochs; ++epoch) {
        for (int i = 0; i < trainingInput.size(); i++) {
            std::vector<double> input = trainingInput[i];
            std::vector<double> target = trainingTarget[i];
            train.backpropagate(input, target, learningRate);
        }
        cout << "training progress " << epoch + 1 << "/" << numEpochs << "\r";
    }
    cout << "training complete " << endl;

    // save weights
    train.saveWeightsToFile("dupa.dat");
    cout << "weights saved" << endl;

    // build prediction network and query the prediction network with different activation functions
    //NeuralNetwork predict(4, { xy, xy, xy, xy });
    train.setActivationFunction("elu"); // tanh sigmoid relu leaky_relu elu softplus step identity
    //predict.loadWeightsFromFile("dupa.dat");
    //cout << "prediction weights loaded" << endl;

    // query 
    std::vector<double> output;
    for (int i = 0; i < queryInput.size(); i++) {
        std::vector<double> input = queryInput[i];
        output = train.feedForward(input); //std::vector<double> output = nn.feedForward(input);
        std::cout << "Input: " << input[660] << ", Target: " << trainingTarget[i][660] << ", Output: " << output[660] << std::endl;
    }
    cout << "prediction elu complete" << endl;

    Mat prediction(x, y, CV_64F);
    prediction = vectorToMat64(output, x, y);
    //cout << "debug: " << prediction.at<double>(0, 50) << endl;
    imwrite("predictionelu.bmp", prediction);
    
    //////////////////// relu /////////////////////
    train.setActivationFunction("relu"); // tanh sigmoid relu leaky_relu elu softplus step identity
    //std::vector<double> output;
    for (int i = 0; i < queryInput.size(); i++) {
        std::vector<double> input = queryInput[i];
        output = train.feedForward(input); //std::vector<double> output = nn.feedForward(input);
        std::cout << "Input: " << input[660] << ", Target: " << trainingTarget[i][660] << ", Output: " << output[660] << std::endl;
    }
    cout << "prediction relu complete" << endl;
    prediction = vectorToMat64(output, x, y);
    imwrite("predictionRelu.bmp", prediction);
    //////////////////// leaky_relu /////////////////////
    train.setActivationFunction("leaky_relu"); // tanh sigmoid relu leaky_relu elu softplus step identity
    //std::vector<double> output;
    for (int i = 0; i < queryInput.size(); i++) {
        std::vector<double> input = queryInput[i];
        output = train.feedForward(input); //std::vector<double> output = nn.feedForward(input);
        std::cout << "Input: " << input[660] << ", Target: " << trainingTarget[i][660] << ", Output: " << output[660] << std::endl;
    }
    cout << "prediction leaky_relu complete" << endl;
    prediction = vectorToMat64(output, x, y);
    imwrite("predictionLeakyRelu.bmp", prediction);
    //////////////////// identity /////////////////////
    train.setActivationFunction("identity"); // tanh sigmoid relu leaky_relu elu softplus step identity
    //std::vector<double> output;
    for (int i = 0; i < queryInput.size(); i++) {
        std::vector<double> input = queryInput[i];
        output = train.feedForward(input); //std::vector<double> output = nn.feedForward(input);
        std::cout << "Input: " << input[660] << ", Target: " << trainingTarget[i][660] << ", Output: " << output[660] << std::endl;
    }
    cout << "prediction identity complete" << endl;
    prediction = vectorToMat64(output, x, y);
    imwrite("predictionIdentity.bmp", prediction);




    auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << "s" << std::endl;
    return 0;
}
