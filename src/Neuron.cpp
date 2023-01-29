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

    _candidate_iw = 1;
    _candidate_sw = 1;
    _candidate_bias = 0;

    // Determine the initial state of the weights/biases
    _forget_gate.set_iw(1);
    _forget_gate.set_sw(1);
    _forget_gate.set_bias(0);

    _input_gate.set_iw(1);
    _input_gate.set_sw(1);
    _input_gate.set_bias(0);

    _output_gate.set_iw(1);
    _output_gate.set_sw(1);
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

void Neuron::set_cand_iw(double x)
{
    _candidate_iw = x;
}

void Neuron::set_cand_sw(double x)
{
    _candidate_sw = x;
}

void Neuron::set_cand_bias(double x)
{
    _candidate_bias = x;
}

// Update members does all of the getting
void Neuron::update_members(double input, double stm, double ltm)
{
    _input = input;
    _short_term_mem = stm;
    _long_term_mem = ltm;
}

// Return the value of the short term memory after propogating the current member variables
double Neuron::feedforward()
{
    // Perform feedforward with Forget gate first
    _forget_gate.update_members(_input, _short_term_mem, _long_term_mem);
    double f_t = _forget_gate.feedforward();

    // "Forget" part of the internal state
    _long_term_mem *= f_t;

    // Perform feedforward with input gate
    _input_gate.update_members(_input, _short_term_mem, _long_term_mem);
    double i_t = _input_gate.feedforward();

    // Calculate candidate state
    double g_t = tanh(_input * _candidate_iw + _short_term_mem * _candidate_sw + _candidate_bias);

    // Update long term memory (Cell state)
    double ltm_inc = i_t * g_t;
    _long_term_mem += ltm_inc;

    // Perform feedforward with output gate
    _output_gate.update_members(_input, _long_term_mem, _short_term_mem);
    double o_t = _output_gate.feedforward();

    // Update short term memory
    _short_term_mem = o_t * tanh(_long_term_mem);

    return _short_term_mem;
}

// TODO:: Write backpropogation algorithm
// TODO:: Write parent class to manage the network and handle many dimensions of data.