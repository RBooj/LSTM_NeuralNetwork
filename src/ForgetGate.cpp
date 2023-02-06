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

double ForgetGate::get_output()
{
    return _output;
}

double ForgetGate::get_uf()
{
    return _input_weight;
}

double ForgetGate::get_wf()
{
    return _short_weight;
}

double ForgetGate::get_bf()
{
    return _sum_bias;
}

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
void ForgetGate::feedforward()
{
    // Collect values to use internally for determining what percentange of the ltm is kept
    // _input
    // _short_term_mem
    // _long_term_mem
    // and weights and bias for each

    // Calculate intermediate sum and add bias
    _output = sigmoid(_sum_bias + (_input * _input_weight) + (_short_term_mem * _short_weight));

    // Return the value of the sigmoid function with tmp as its input
}