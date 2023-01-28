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
    double _long_term_mem;  // The neuron saves a long term memory (cell state)
    double _short_term_mem; // The neuron saves a short term memory (hidden state)

    double _candidate_iw; // The neuron scales the values used to calculate the candidate state
    double _candidate_sw;
    double _candidate_bias;

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

    void set_cand_iw(double x);
    void set_cand_sw(double x);
    void set_cand_bias(double x);

    // Neuron Operations
    double feedforward(); // Using the current member variable values, generate an output and update internal memory
};
#endif