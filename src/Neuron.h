#ifndef NEURON_H
#define NEURON_H

// Packages
#include <iostream>
#include <cmath>
#include <vector>
#include <random>

// Loacal Includes
#include "ForgetGate.h"
#include "OutputGate.h"
#include "InputGate.h"

// Neuron class implements each of the gates and manages
// input/outputs to each of them.
class Neuron
{
private:
    // Each neuron will have three gates
    ForgetGate _forget_gate; // First gate
    InputGate _input_gate;   // Second gate
    OutputGate _output_gate; // Third gate

    // Each neuron has an input, a long term memory, and a short term memory
    double _input;          // The neuron takes in data from some outside source
    double _long_term_mem;  // The neuron gets a long term memory from previous neurons
    double _short_term_mem; // The neuron gets a short term memory from a previous neuron

public:
    // Constructor/destructor
    Neuron();

    // Get member variable values
    double get_input();
    double get_short();
    double get_long();

    // Set member variables
    void set_input(double new_input);
    void set_short(double new_STM);
    void set_long(double new_LTM);
    void update_members(double input, double ltm, double stm);

    // Neuron Operations
    double feedforward(); // Using the current member variable values, generate an output

    // Training
    // Calculate derivatives for the different functions used
    double diff_tanh(double x);

    double diff_sigmoid(double x);

    double ResidualSumOfSquares(double expected);

    double RSS_derivative(double expected);
};
#endif