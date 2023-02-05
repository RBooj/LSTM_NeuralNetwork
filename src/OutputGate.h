#ifndef OUTPUTGATE_H
#define OUTPUTGATE_H

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

class OutputGate
{
private:
    // Third Stage (update STM stage) data
    /*
        Output gate has two parts:
        Creation of new short term memory
        Calculate how much of new memory is saved
    */
    double _input;          // Input value to the neuron
    double _short_term_mem; // The value of the short term memory
    double _long_term_mem;  // The value of the long term memory (Cell State)

    double _input_weight; // The input is multiplied by a weight
    double _short_weight; // The short term memory value is multiplied by a weight
    double _sum_bias;     // The sum of short-term mem and input has a bias

    // private functions
    double sigmoid(double x); // Use sigmoid to calculate how much of the short term mem should be remembered

    double _output; // Retain result of feedforward for backpropagation calculation

public:
    // Set member variables
    void update_members(double input, double ltm, double stm); // Encapsulate updating all 3 member variables
    void set_iw(double x);
    void set_sw(double x);
    void set_bias(double x);
    void update_weights(double w1, double w2, double b1);

    double get_output(); // Return result of feedfoward with output gate

    // Output gate operations
    void feedforward(); // using the current member variables, generate an output
};
#endif