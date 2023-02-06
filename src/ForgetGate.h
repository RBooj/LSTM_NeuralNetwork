#ifndef FORGETGATE_H
#define FORGETGATE_H

#include <iostream>
#include <cmath>
#include <vector>

using namespace std;
/*
    LSTM Neural Network
    Use the Sigmoid and tanh activation functions
    3 Stages -
    Calculate the percent to remmeber
    Update Long term memory
    Update short term memory

*/

// Each neuron will be made up of three stages which each accomplish a different goal
// Implement each stage in each neuron

// Rewriting with vectors and new architecture after studying equations of the architecture

class ForgetGate
{
private:
    // Input Stage data (Forget Gate)
    double _input;          // Input value to the neuron
    double _short_term_mem; // The value of the short term memory
    double _long_term_mem;  // The value of the long term memory (Cell State)

    double _input_weight; // The input is multiplied by a weight
    double _short_weight; // The short term memory value is multiplied by a weight
    double _sum_bias;     // The sum of short-term mem and input has a bias

    double sigmoid(int x); // Use a sigmoid activation function to calculate how much of the long therm memory to retain

    double _output; // retain the value f_t for use in backpropagation

public:
    // Constructor
    ForgetGate();

    // Return the output of feeding forward with forget gate
    double get_output();

    // Getters
    double get_uf(); // Input weight value
    double get_wf(); // Hidden state weight value
    double get_bf(); // Bias value

    // Set member variables
    void update_members(double input, double stm, double ltm);

    // Set weights and biases
    void set_bias(double x);
    void set_iw(double x); // Input weight
    void set_sw(double x); // Short weight

    // Output gate operations
    void feedforward(); // using the current member variables, generate an output
};

#endif