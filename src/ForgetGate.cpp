#include "ForgetGate.h"
using namespace std;

ForgetGate::ForgetGate()
{
    // Give member variables a default value
    _input = 0;
    _short_term_mem = 0;
    _long_term_mem = 0;

    _input_weight = 1;
    _short_weight = 1;
    _sum_bias = 0;
}

// Approximation of sigmoid function:
double ForgetGate::sigmoid(int x)
{
    return 1 / (1 + exp(-x));
}

// // Basic setters and getters for intercting with neuron class
// void ForgetGate::update_members(double input, double stm, double ltm)
// {
//     _input = input;
//     _short_term_mem = stm;
//     _long_term_mem = ltm;
// }

void ForgetGate::set_bias(double x)
{
    _sum_bias = x;
}

void ForgetGate::set_iw(double x)
{
    _input_weight = x;
}

void ForgetGate::set_sw(double x)
{
    _short_weight = x;
}
// ----------------------------------------------------------

// Generate the output value using the current member variables and return it
double ForgetGate::feedforward()
{
    // Collect values to use internally for determining what percentange of the ltm is kept
    // _input
    // _short_term_mem
    // _long_term_mem
    // and weights and bias for each

    // Calculate intermediate sum and add bias
    return sigmoid(_sum_bias + (_input * _input_weight) + (_short_term_mem * _short_weight));

    // Return the value of the sigmoid function with tmp as its input
}