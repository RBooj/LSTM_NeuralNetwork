#ifndef INPUTGATE_H
#define INPUTGATE_H

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

class InputGate
{
private:
    // Second stage (Input Gate) data
    /*
        Input gate has two parts:
        Creation of a new Long term memory
        Calculate how much of new memory is saved
    */
    double _input;          // Input value to the neuron
    double _short_term_mem; // The value of the short term memory
    double _long_term_mem;  // The value of the long term memory (Cell State)

    double _input_weight; // The input is multiplied by a weight (for calculating the percent to remember)
    double _short_weight; // The short term memory value is multiplied by a weight (for calculating the percent to remember)
    double _sum_bias;     // The sum of short-term mem and input has a bias (for calculating the percent to remember)

    // private functions
    double sigmoid(double x); // Use a sigmoid activation function for the percent of new memory to remember

    double _output; // Retain result from feeding forward with input gate for backpropagation later

public:
    InputGate();

    // Set member variables
    void update_members(double input, double stm, double ltm); // Encapsulate updating all 4 member variables
    void set_iw(double x);
    void set_sw(double x);
    void set_bias(double x);
    void update_weights(double w1, double w2, double b1);

    double get_output(); // RTeturn output value of input gate feedforward

    // Getters
    double get_ui();
    double get_wi();
    double get_bi();

    // Input gate operations
    void feedforward(); // using the current member variables, generate an output
};
#endif