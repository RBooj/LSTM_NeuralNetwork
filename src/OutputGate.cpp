#include "OutputGate.h"
using namespace std;
// Default constructor
OutputGate::OutputGate()
{
    _input = 0;          // Input value to the neuron
    _short_term_mem = 0; // The value of the short term memory
    _long_term_mem = 0;  // The value of the long term memory (Cell State)

    _input_weight = 1; // The input is multiplied by a weight
    _short_weight = 1; // The short term memory value is multiplied by a weight
    _sum_bias = 0;     // The sum of short-term mem and input has a bias
}

double OutputGate::potential_activation(double x)
{
    return tanh(x);
}

double OutputGate::remember_activation(double x)
{
    return 1 / (1 + exp(-x));
}

void OutputGate::update_members(double input, double ltm, double stm)
{
    _input = input;
    _long_term_mem = ltm;
    _short_term_mem = stm;
}

void OutputGate::set_iw(double x)
{
    _input_weight = x;
}
void OutputGate::set_sw(double x)
{
    _short_weight = x;
}
void OutputGate::set_bias(double x)
{
    _sum_bias = x;
}

void OutputGate::update_weights(double w1, double w2, double b1)
{
    _input_weight = w1;
    _short_weight = w2;
    _sum_bias = b1;
}

// Generate new short term memory - output of the neuron
double OutputGate::feedforward()
{
    // Calculate new memory
    double new_mem = (_short_term_mem * _short_weight) + (_input * _input_weight) + _sum_bias;
    new_mem = potential_activation(new_mem);

    // Calculate percent to remember
    double mem_percent = remember_activation(_long_term_mem);

    return new_mem * mem_percent;
}