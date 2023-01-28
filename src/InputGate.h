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

    double _input_weight_1; // The input is multiplied by a weight (for calculating the percent to remember)
    double _input_weight_2; // The input us multiplied by a weight (for creating a new memory)
    double _short_weight_1; // The short term memory value is multiplied by a weight (for calculating the percent to remember)
    double _short_weight_2; // The short term memory value is multiplied by a weight (for creating a new memory)
    double _sum_bias_1;     // The sum of short-term mem and input has a bias (for calculating the percent to remember)
    double _sum_bias_2;     // The sum of short-term mem and input has a bias (for creating a new memory)

    // private functions
    double remember_activation(double x);  // Use a sigmoid activation function for the percent of new memory to remember
    double potential_activation(double x); // Use a tanh activation function for the new memory

public:
    // Set member variables
    void update_members(double input, double stm, double ltm); // Encapsulate updating all 4 member variables
    void set_iw1(double x);                                    // Input wight 1
    void set_iw2(double x);                                    // Input weight 2
    void set_b1(double x);                                     // Sumb bias 1
    void set_sw1(double x);                                    // Short weight 1
    void set_sw2(double x);                                    // Short weight 2
    void set_b2(double x);                                     // sum bias 2
    void update_weights(double w1, double w2, double b1, double w3, double w4, double b2);

    // Input gate operations
    double feedforward(); // using the current member variables, generate an output
};
#endif