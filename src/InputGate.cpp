#include "InputGate.h"
using namespace std;

InputGate::InputGate()
{
    _input = 0;          // Input value to the neuron
    _short_term_mem = 0; // The value of the short term memory
    _long_term_mem = 0;  // The value of the long term memory (Cell State)
}

// Private functions
// Encapsulated for some reason if i decide to change them later
// Sigmoid for remember percentage
double InputGate::sigmoid(double x)
{
    return 1 / (1 + exp(-x));
}

void InputGate::update_members(double input, double stm, double ltm)
{
    _input = input;
    _short_term_mem = stm;
    _long_term_mem = ltm;
}

void InputGate::update_weights(double w1, double w2, double b1)
{
    _input_weight = w1;
    _short_weight = w2;
    _sum_bias = b1;
}

double InputGate::get_output()
{
    return _output;
}

double InputGate::get_ui()
{
    return _input_weight;
}

double InputGate::get_wi()
{
    return _short_weight;
}

double InputGate::get_bi()
{
    return _sum_bias;
}

void InputGate::set_iw(double x)
{
    _input_weight = x;
}
void InputGate::set_sw(double x)
{
    _short_weight = x;
}
void InputGate::set_bias(double x)
{
    _sum_bias = x;
}

// Input gate determines percentage of new memory
void InputGate::feedforward()
{
    _output = sigmoid(_input * _input_weight + _short_term_mem * _short_weight + _sum_bias);
}