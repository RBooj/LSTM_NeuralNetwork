#include "InputGate.h"
using namespace std;

InputGate::InputGate()
{
    _input = 0;          // Input value to the neuron
    _short_term_mem = 0; // The value of the short term memory
    _long_term_mem = 0;  // The value of the long term memory (Cell State)

    _input_weight_1 = 1; // The input is multiplied by a weight (for calculating the percent to remember)
    _input_weight_2 = 1; // The input us multiplied by a weight (for creating a new memory)
    _short_weight_1 = 1; // The short term memory value is multiplied by a weight (for calculating the percent to remember)
    _short_weight_2 = 1; // The short term memory value is multiplied by a weight (for creating a new memory)
    _sum_bias_1 = 0;     // The sum of short-term mem and input has a bias (for calculating the percent to remember)
    _sum_bias_2 = 0;     // The sum of short-term mem and input has a bias (for creating a new memory)
}

// Private functions
// Encapsulated for some reason if i decide to change them later
// Sigmoid for remember percentage
double InputGate::remember_activation(double x)
{
    return 1 / (1 + exp(-x));
}

// Tanh for generating a new memory
double InputGate::potential_activation(double x)
{
    return tanh(x);
}

void InputGate::update_members(double input, double stm, double ltm)
{
    _input = input;
    _short_term_mem = stm;
    _long_term_mem = ltm;
}

void InputGate::set_iw1(double x)
{
    _input_weight_1 = x;
}
void InputGate::set_iw2(double x)
{
    _input_weight_2 = x;
}
void InputGate::set_b1(double x)
{
    _sum_bias_1 = x;
}
void InputGate::set_sw1(double x)
{
    _short_weight_1 = x;
}
void InputGate::set_sw2(double x)
{
    _short_weight_2 = x;
}
void InputGate::set_b2(double x)
{
    _sum_bias_2 = x;
}

void InputGate::update_weights(double w1, double w2, double b1, double w3, double w4, double b2)
{
    _input_weight_1 = w1;
    _short_weight_1 = w2;
    _sum_bias_1 = b1;

    _input_weight_2 = w3;
    _short_weight_2 = w4;
    _sum_bias_2 = b2;
}

// Generate a new long term memory
double InputGate::feedforward()
{
    // Calculate new memory
    double new_mem = (_input * _input_weight_1) + (_short_term_mem * _short_weight_1) + _sum_bias_1;
    new_mem = potential_activation(new_mem);

    // Calculate percentage to remember
    double mem_percent = (_input * _input_weight_2) + (_short_term_mem * _short_weight_2) + _sum_bias_2;
    mem_percent = remember_activation(mem_percent);

    // Return new value of long term mem
    return (new_mem * mem_percent);
}