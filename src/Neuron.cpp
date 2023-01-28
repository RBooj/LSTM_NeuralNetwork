#include "Neuron.h"
using namespace std;

// Neuron Constructor
Neuron::Neuron()
{
    srand(time(NULL));

    // Each Neuron will implement/manage 3 stages
    _forget_gate = ForgetGate();
    _input_gate = InputGate();
    _output_gate = OutputGate();

    // Each neuron has an input, a long term memory, and a short term memory
    // Initial Values are 0
    _input = 0;          // The neuron takes in data from some outside source
    _long_term_mem = 0;  // The neuron gets a long term memory from previous neurons
    _short_term_mem = 0; // The neuron gets a short term memory from a previous neuron

    // Determine the initial state of the weights/biases
    _forget_gate.set_iw(rand() / RAND_MAX);
    _forget_gate.set_sw(rand() / RAND_MAX);
    _forget_gate.set_bias(0);

    _input_gate.set_iw1(rand() / RAND_MAX);
    _input_gate.set_iw2(rand() / RAND_MAX);
    _input_gate.set_b1(0);
    _input_gate.set_sw1(rand() / RAND_MAX);
    _input_gate.set_sw2(rand() / RAND_MAX);
    _input_gate.set_b2(0);

    _output_gate.set_iw(rand() / RAND_MAX);
    _output_gate.set_sw(rand() / RAND_MAX);
    _output_gate.set_bias(0);
}

// Getting Member Variables
double Neuron::get_input()
{
    return _input;
}

double Neuron::get_short()
{
    return _short_term_mem;
}

double Neuron::get_long()
{
    return _long_term_mem;
}

// Setting Member Variables
void Neuron::set_input(double new_input)
{
    _input = new_input;
}

void Neuron::set_long(double new_LTM)
{
    _long_term_mem = new_LTM;
}

void Neuron::set_short(double new_STM)
{
    _short_term_mem = new_STM;
}

// Update members does all of the getting
void Neuron::update_members(double input, double stm, double ltm)
{
    _input = input;
    _short_term_mem = stm;
    _long_term_mem = ltm;
}

// Generate an output
double Neuron::feedforward()
{
    // Perform forget gate operations
    // update forget gate with internal values
    _forget_gate.update_members(_input, _short_term_mem, _long_term_mem); // Pass values to forget gate
    _long_term_mem *= _forget_gate.feedforward();                         // Perform forget gate's operations and update LTM

    // Perform input gate operations
    _input_gate.update_members(_input, _short_term_mem, _long_term_mem); // Pass values to input gate
    _long_term_mem += _input_gate.feedforward();                         // Perform input gate's operation and update LTM

    // Perform output gate operations
    _output_gate.update_members(_input, _long_term_mem, _short_term_mem); // Pass values to output gate
    _short_term_mem = _output_gate.feedforward();                         // Perform output gate's operations

    return _short_term_mem;
}

double diff_tanh(double x)
{
    return (1 - pow(tanh(x), 2));
}

double diff_sigmoid(double x)
{
    return (1 / (1 + exp(-x))) * (1 - 1 / (1 + exp(-x)));
}

double Neuron::ResidualSumOfSquares(double expected)
{
    // Squred error: expected - predicted
    return pow(expected - _short_term_mem, 2);
}

double Neuron::RSS_derivative(double expected)
{
    return -_short_term_mem * (expected - _short_term_mem);
}