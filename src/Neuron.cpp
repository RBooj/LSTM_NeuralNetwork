#include "Neuron.h"
using namespace std;

double Neuron::sigmoid(double x)
{
    return 1 / (1 + exp(-x));
}

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
    // Save previous states
    _C_t_prev = _long_term_mem;
    _H_t_prev = _short_term_mem;

    // Perform feedforward with Forget gate first
    _forget_gate.update_members(_input, _short_term_mem, _long_term_mem);
    _forget_gate.feedforward();
    double f_t = _forget_gate.get_output();

    // "Forget" part of the internal state
    _long_term_mem *= f_t;

    // Perform feedforward with input gate
    _input_gate.update_members(_input, _short_term_mem, _long_term_mem);
    _input_gate.feedforward();
    double i_t = _input_gate.get_output();

    // Calculate candidate state
    _candidate_state = tanh(_input * _candidate_iw + _short_term_mem * _candidate_sw + _candidate_bias);
    double g_t = _candidate_state;

    // Update long term memory (Cell state)
    double ltm_inc = i_t * g_t;
    _long_term_mem += ltm_inc;

    // Perform feedforward with output gate
    _output_gate.update_members(_input, _long_term_mem, _short_term_mem);
    _output_gate.feedforward();
    double o_t = _output_gate.get_output();

    // Update short term memory
    _short_term_mem = o_t * tanh(_long_term_mem);

    return _short_term_mem;
}

void Neuron::backprop(double expected)
{
    // Calculate the new weights and biases according to the equation in the description document
    // W_new = W_old - (Loss derivative WRT weight/bias)*(learning rate)
    // Calculating intermediate values
    // Result from forget gate
    double f_t = _forget_gate.get_output();
    // Result from input gate
    double i_t = _input_gate.get_output();
    // Result from output gate
    double o_t = _output_gate.get_output();
    // Result from candidate state
    double g_t = _candidate_state;

    // weights/biases
    // TODO:: Clean up naming scheme for entire project
    double U_f = _forget_gate.get_uf();
    double W_f = _forget_gate.get_wf();
    double b_f = _forget_gate.get_bf();

    double U_i = _input_gate.get_ui();
    double W_i = _input_gate.get_wi();
    double b_i = _input_gate.get_bi();

    double U_o = _output_gate.get_uo();
    double W_o = _output_gate.get_wo();
    double b_o = _output_gate.get_bo();

    double U_g = _candidate_iw;
    double W_g = _candidate_sw;
    double b_g = _candidate_bias;

    // Calculate intermediate values
    double C_t_tanh_2 = pow(tanh(_long_term_mem), 2);
    double dLdCt = -2 * (expected - _short_term_mem) * (o_t - o_t * C_t_tanh_2);

    // Forget Gate
    // TODO:: Clean up repetitive calculations here
    double tmp_sig_f = U_f * _input + _H_t_prev * W_f + b_f;
    double dLdUf = dLdCt * _C_t_prev * sigmoid(tmp_sig_f) * (1 - sigmoid(tmp_sig_f)) * _input;
    double dLdWf = dLdCt * _C_t_prev * sigmoid(tmp_sig_f) * (1 - sigmoid(tmp_sig_f)) * _H_t_prev;
    double dLdbf = dLdCt * _C_t_prev * sigmoid(tmp_sig_f) * (1 - sigmoid(tmp_sig_f));

    // Input gate
    double tmp_sig_i = _input * U_i + _H_t_prev * W_i + b_i;
    double dLdUi = dLdCt * g_t * sigmoid(tmp_sig_i) * (1 - sigmoid(tmp_sig_i)) * _input;
    double dLdWi = dLdCt * g_t * sigmoid(tmp_sig_i) * (1 - sigmoid(tmp_sig_i)) * _H_t_prev;
    double dLdbi = dLdCt * g_t * sigmoid(tmp_sig_i) * (1 - sigmoid(tmp_sig_i));

    // Candidate State
    double tmp_tanh_g = pow(tanh(_input * U_g + _H_t_prev * W_g + b_g), 2);
    double dLdUg = dLdCt * i_t * _input * (1 - tmp_tanh_g);
    double dLdWg = dLdCt * i_t * _H_t_prev * (1 - tmp_tanh_g);
    double dLdbg = dLdCt * i_t * (1 - tmp_tanh_g);

    // Output Gate
    // C_t == long term mem
    // TODO::  Really need to fix naming convention for entire project
    double dLdHt = -2 * (expected - _short_term_mem);
    double tmp_sig_o = _input * U_o + _H_t_prev * W_o + b_o;
    double dLdUo = dLdHt * tanh(_long_term_mem) * sigmoid(tmp_sig_o) * (1 - sigmoid(tmp_sig_o)) * _input;
    double dLdWo = dLdHt * tanh(_long_term_mem) * sigmoid(tmp_sig_o) * (1 - sigmoid(tmp_sig_o)) * _H_t_prev;
    double dLdbo = dLdHt * tanh(_long_term_mem) * sigmoid(tmp_sig_o) * (1 - sigmoid(tmp_sig_o));

    // Use the derivatives WRT weights/biases in the formula for updating weights/biases
    double U_f_new = U_f - (dLdUf * _learning_rate);
    double U_i_new = U_i - (dLdUi * _learning_rate);
    double U_o_new = U_o - (dLdUo * _learning_rate);
    double U_g_new = U_g - (dLdUg * _learning_rate);

    double W_f_new = W_f - (dLdWf * _learning_rate);
    double W_i_new = W_i - (dLdWi * _learning_rate);
    double W_o_new = W_o - (dLdWo * _learning_rate);
    double W_g_new = W_g - (dLdWg * _learning_rate);

    double b_f_new = b_f - (dLdbf * _learning_rate);
    double b_i_new = b_i - (dLdbi * _learning_rate);
    double b_o_new = b_o - (dLdbo * _learning_rate);
    double b_g_new = b_g - (dLdbg * _learning_rate);

    // Update values
    _forget_gate.set_iw(U_f_new);
    _forget_gate.set_sw(W_f_new);
    _forget_gate.set_bias(b_f_new);

    _input_gate.set_iw(U_i_new);
    _input_gate.set_sw(W_i_new);
    _input_gate.set_bias(b_i_new);

    _output_gate.set_iw(U_o_new);
    _output_gate.set_sw(W_o_new);
    _output_gate.set_bias(b_o_new);

    _candidate_iw = U_g_new;
    _candidate_sw = W_g_new;
    _candidate_bias = b_g_new;
}
// TODO:: Write parent class to manage the network and handle many dimensions of data.