#include <string>
#include <fstream>
#include <vector>
#include <math.h>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <time.h>
#include <chrono>
#include <direct.h>
#include <stdio.h>
#include <random>
#include "ann.h"
#include "Misc.h"

#define zero_tol 0.00000001
#define ij_to_idx(i, j, num_cols) ((i*num_cols) + j) // convert (i,j) index to vector index for row-major ordering
#define my_sign(in_val) (in_val < zero_tol ? -1.0:1.0)

using namespace std;

void AnnClass::InitializeMatrices() {
	// Function to initialize the matrices and vectors responsible for things like weights and node activations
	// Compute the total number of nodes and weights in the ANN
	this->num_nodes = 0;
	this->num_weights = 0;
	for (int ii = 0; ii <= this->num_layers; ii++) {
		this->num_nodes += this->sum_nodes_per_layer[ii];
		if (ii < this->num_layers) {
			this->num_weights += this->sum_nodes_per_layer[ii] * this->sum_nodes_per_layer[ii + 1];
		}
	}

	this->node_f = new double[num_nodes];
	this->node_phi = new double[num_nodes];
	this->node_dphi = new double[num_nodes];
	this->node_dk = new double[num_nodes];
	this->temp_dk = new double[num_nodes]; // too big, but it's not going to make a difference (it's still relatively small)

	this->mean_node_phi = new double[num_nodes];
	this->mean_node_dphi = new double[num_nodes];
	this->output_error_mean = new double[this->num_output_nodes];

	if (this->do_dropout) {
		// Initialize the masks, but don't allocate space yet
		this->dropout_mask = NULL;
		this->weight_dropout_mask = NULL;
	}

	// Only allocate memory for the biases if biases are to be used
	if (this->use_biases) {
		this->biases = new double[num_nodes];
		this->bias_delta = new double[num_nodes];
		this->bias_deriv = new double[num_nodes];
		this->bias_prev_deriv = new double[num_nodes];
		this->bias_inc = new double[num_nodes];
	}
	else {
		this->biases = NULL;
		this->bias_delta = NULL;
		this->bias_deriv = NULL;
		this->bias_prev_deriv = NULL;
		this->bias_inc = NULL;
	}

	this->weights = new double[num_weights];
	this->deriv = new double[num_weights];
	this->prev_deriv = new double[num_weights];
	this->w_delta = new double[num_weights];
	this->w_inc = new double[num_weights];
	this->freeze_mask = NULL; // no need to allocate space for this quite yet

	// Initialize the weights by pulling values from a standard normal distribution
	// Also need to initialize the derivatives, increments, etc.
	// 2-5-18 Update
	// Switching to Xavier initialization for the weights
	default_random_engine rand_generator;
	normal_distribution<double> norm_dist(0.0, 1.0); // standard normal: 0-mean, unit variance
	uniform_real_distribution<double> uniform_dist(-1.0, 1.0);
	rand_generator.seed(std::chrono::system_clock::now().time_since_epoch().count());

	int layer_count = 0;
	int weight_count = 0;
	double init_r = 0.0;
	int num_lower, num_upper, weights_this_layer;

	// Starting in the first layer
	num_lower = this->sum_nodes_per_layer[0];
	num_upper = this->sum_nodes_per_layer[1];
	weights_this_layer = num_lower*num_upper;
	//init_r = 4.0 * sqrt(6.0 / (num_lower + num_upper));
	init_r = 0.2;
	for (int ww = 0; ww < num_weights; ww++) {
		//this->weights[ww] = norm_dist(rand_generator) * 0.2 - 0.1; // 

		// Because the sampling distribution is uniform over [-1,1], only need to multiply by the 'r' value
		// defined as r = 4*sqrt( 6 / (n_input + n_output))
		this->weights[ww] = uniform_dist(rand_generator)*init_r;
		weight_count++;

		/*
		// Change the layer count if necessary
		if (weight_count >= weights_this_layer) {
			layer_count++;
			weights_this_layer = num_lower*num_upper;
			weight_count = 0;

			num_lower = this->sum_nodes_per_layer[layer_count];
			num_upper = this->sum_nodes_per_layer[layer_count + 1];
			init_r = 4.0 * sqrt(6.0 / (num_lower + num_upper));
		}
		//*/

		this->deriv[ww] = 0.0;
		this->prev_deriv[ww] = 0.0;
		this->w_delta[ww] = this->delta0;
		this->w_inc[ww] = 0.0;
	}

	/*if (this->do_dropout) {
		for (int ww = 0; ww < num_weights; ww++) {
			this->dropout_mask[ww] = 1.0;
		}
	}*/

	for (int nn = 0; nn < num_nodes; nn++) {
		this->node_f[nn] = 0.0;
		this->node_phi[nn] = 0.0;
		this->node_dphi[nn] = 0.0;
		this->node_dk[nn] = 0.0;
	}

	if (this->use_biases) {
		for (int nn = 0; nn < num_nodes; nn++) {
			this->bias_delta[nn] = this->delta0;
			this->bias_deriv[nn] = 0.0;
			this->bias_inc[nn] = 0.0;
			this->bias_prev_deriv[nn] = 0.0;
			this->biases[nn] = uniform_dist(rand_generator); // biases need to be randomly initialized
		}
	}

	// If using #random, initialize the vector of random indices
	this->out_data << "1" << endl;
	if (this->random_pairs) {
		this->random_inds = new int[this->num_training_pairs];
		this->random_init = false;
	}
	this->out_data << "2" << endl;

}

void AnnClass::PropagateSignalForward(int input_idx) {
	// Function to complete a forward pass through the ANN, updating
	// node_f and node_phi along the way
	// input_idx is the row in the input data matrix to be used
	// (i.e., idx points to the input vector)

	// The activation for the input layer doesn't matter
	// However, the output function is found just by passing the input
	// vector into the activation function
	double tan_square;

	for (int nn = 0; nn < this->num_input_nodes; nn++) { // loop over the nodes in the input layer
		//this->node_phi[nn] = this->ComputeActivation(this->i_data[ij_to_idx(input_idx, nn, this->num_input_nodes)]);
		//this->node_dphi[nn] = this->ComputeActDeriv(this->i_data[ij_to_idx(input_idx, nn, this->num_input_nodes)]);

		tan_square = tanh(this->i_data[ij_to_idx(input_idx, nn, this->num_input_nodes)]);
		this->node_phi[nn] = tan_square;
		this->node_dphi[nn] = 1.0 - tan_square*tan_square;
	}

	// For the remaining layers, multiply the weights matrix by the output of the previous layer.
	// Because the number of slices of node_phi and node_f are +1 to the weight matrices, be
	// careful that the indexing is correct


	int start_weight_index = 0; // changes as the signal passes through the layers
	int node_index = this->num_input_nodes; // start at the first node in second layer
	int bias_index = node_index; // biases skip the first layer
	int prev_node = 0; // starting index for the nodes in layer k
	int nodes_prev_layer, nodes_next_layer;
	int temp_index = 0; // temp variable to hold index for desired weight
	this->last_nodes_added = 0;


	for (int l_count = 1; l_count <= this->num_layers; l_count++) {
		// Grab the number of nodes in layers k and k+1
		//nodes_prev_layer = this->sum_nodes_per_layer[l_count-1];
		//nodes_next_layer = this->sum_nodes_per_layer[l_count];

		// Because the weights are stored in row major format, each row of the weight matrix are all the connections to node J
		// in layer k+1 from every node in layer k. So, the outer loop should increment over the nodes in layer k+1
		for (int n_count = 0; n_count < this->sum_nodes_per_layer[l_count]; n_count++) { // loop over the next layer
			this->node_f[node_index] = 0.0;
			this->node_phi[node_index] = 0.0;

			for (int p_count = 0; p_count < this->sum_nodes_per_layer[l_count - 1]; p_count++) { // loop over the previous layer
				this->node_phi[node_index] += 
					this->weights[start_weight_index + p_count]*this->node_phi[prev_node + p_count];
			}

			// Add the bias, if necessary
			if (this->use_biases && l_count < this->num_layers) {
				this->node_phi[node_index] += this->biases[bias_index];
				bias_index++;
			}

			// 2-4-18
			// if syntax is l_count < this->num_layers:
			// All layers are tanh(), except the output which is logistic
			if (l_count <= this->num_layers) {
				// Compute the derivative first since node_phi gets overwritten
				//this->node_dphi[node_index] = this->ComputeActDeriv(this->node_phi[node_index]);
				// Compute the activation for this node
				//this->node_phi[node_index] = this->ComputeActivation(this->node_phi[node_index]);

				tan_square = tanh(this->node_phi[node_index]);
				this->node_phi[node_index] = tan_square;
				this->node_dphi[node_index] = 1.0 - tan_square*tan_square;
			}
			else {
				// Compute the derivative first since node_phi gets overwritten
				this->node_dphi[node_index] = this->ComputeLogDeriv(this->node_phi[node_index]);
				// Compute the activation for this node
				this->node_phi[node_index] = this->ComputeLogistic(this->node_phi[node_index]);
			}

			// Update the starting index for the weights (move to the next row)
			start_weight_index += this->sum_nodes_per_layer[l_count - 1];;

			// Move to the next node 
			node_index++;
		}

		// Update the index for the starting node in the previous layer
		prev_node += this->sum_nodes_per_layer[l_count - 1];;

		//if (this->do_dropout && l_count < this->num_layers - 1) {
		//	// Dropout is being used. Apply the mask to the node outputs to effectively drop nodes
		//	// Also, don't apply dropout to input/output layers
		//	this->node_phi[l_count + 1].array() *= this->dropout_mask[l_count].array();
		//	this->node_dphi[l_count + 1].array() *= this->dropout_mask[l_count].array();
		//}

		// If doing staggered training, zero the output of the nodes in module 1 if in the second stage
		/*if(this->stagger){
		if(this->stagger_stage == 2){
		int upper_bound = this->nodes_per_layer.array()(0,l_count+1);
		for(int nn=0; nn < upper_bound; nn++){
		this->node_phi[l_count].array()(nn) = 0.0;
		this->node_dphi[l_count].array()(nn) = 0.0;
		}
		}
		}*/
		//        cout << "Layer " << l_count+1 << ":" << endl;
		//        cout << "\tOutputs:" << endl << this->node_phi[l_count+1] << endl;

	}

	int a = 1;

	return;
}

void AnnClass::PropagateErrorBackward(int output_idx) {
	// Function to compute the deltas and dE/dw_ji for each node and weight

	double o_error = 0.0;

	int num_layers = this->num_layers; // Its going to be used a few times, pull it out now
	int num_output = this->num_output_nodes; // pull this out to make the code cleaner

									   // Compute the error at the output first
									   // E = (t_k - a_k), where t_k is the expected output,
									   // a_k is the actual output

	// Compute the output error
	for (int oo = 0; oo < this->num_output_nodes; oo++) {

		// Now compute the delta_k at the output
		o_error = this->o_data[ij_to_idx(output_idx, oo, this->num_output_nodes)] - 
			this->node_phi[this->num_nodes - this->num_output_nodes + oo];

		// Multiply the output error by the derivative of the output node's activation
		this->node_dk[this->num_nodes - this->num_output_nodes + oo] = 
			o_error * this->node_dphi[this->num_nodes - this->num_output_nodes + oo];
	}

	// Compute the delta_j for the hidden layers
	// delta_j = phi'(y_j)*sum(delta_k*w_kj), where k indexes the layer above
	// This is going to be a bit of a mess. Move backward through the network.
	int start_weight_index = this->num_weights-1; // changes as the signal passes through the layers
	int node_index = this->num_nodes-this->sum_nodes_per_layer[this->num_layers]-1; // start at the last node in the second to last layer
	int bias_index = node_index; // biases skip the last layer
	int prev_node = this->num_nodes-1; // starting index for the last node in the last layer
	int nodes_upper_layer, nodes_lower_layer;
	// nodes_upper_layers is "k+1"
	// nodes_lower_layer (nodes preceding layer) is "k"

	for (int ll = num_layers - 1; ll >= 0; ll--) { // move backward through the network, skipping the last layer
		nodes_upper_layer = this->sum_nodes_per_layer[ll + 1];
		nodes_lower_layer = this->sum_nodes_per_layer[ll];
		double *temp_dk = new double[nodes_lower_layer];
		// initialize temp_dk to zero
		for (int oo = 0; oo < nodes_lower_layer; oo++) {
			temp_dk[oo] = 0.0;
		}

		// Again, the rows of the weight matrix correspond to nodes in this_layer. That means moving along a row indexes the nodes
		// in prec_layer. We need sum(delta_k*w_kj). As such, temp_dk is going to hold the temporary values
		for (int t_count = 0; t_count < nodes_upper_layer; t_count++) {// index over layer k+1
			for (int p_count = 0; p_count < nodes_lower_layer; p_count++) { // index over layer k (preceding layer)
				temp_dk[p_count] += this->weights[start_weight_index - p_count] * this->node_dk[prev_node - t_count];
			}
			// Update the starting weight index
			start_weight_index -= nodes_lower_layer;
		}

		// Now compute the node_dk. BUT, temp_dk is indexed in reverse (i.e., [0] is actually the last node in the layer).
		// node_index starts at the last node in the lower layer.
		for (int p_count = 0; p_count < nodes_lower_layer; p_count++) {
			this->node_dk[node_index - p_count] = temp_dk[p_count] * this->node_dphi[node_index - p_count];
		}

		// Update the indices prev_node and node_index
		prev_node -= nodes_upper_layer;
		node_index -= nodes_lower_layer;

		// free memory before next loop 
		delete[] temp_dk;
	}

	// Now that we have all of the deltas, compute dE/dw_ji for each weight.
	// dE/dw_ji = delta_j*phi_i, where phi_i is the output of node 'i' in layer j-1.
	// If I had the node_dk and node_dphi vectors split out by layer, the gradient is easily computed
	// as an outer product. Unfortunately, I don't have the data structured that way.
	// I can still take the outer product, though. I just need to put the proper limits on the loops

	start_weight_index = 0;
	prev_node = 0; // starts indexing the first layer
	node_index = this->sum_nodes_per_layer[0]; // starts index the second layer
	for (int ll = 0; ll < this->num_layers; ll++) {
		int nodes_lower_layer = this->sum_nodes_per_layer[ll];
		int nodes_upper_layer = this->sum_nodes_per_layer[ll + 1];

		// Okay, w_ji connects node i in layer k to node j in layer k+1. If I want to be able to simply increment 
		// weight increment, I have to make sure the loops are ordered correctly.
		// The j are the rows in the weight matrix
		// The i are the columns in the weight matrix
		// The weight matrix is row-major, so 'i' should be nested inside 'j'
		for (int jj = 0; jj < nodes_upper_layer; jj++) {
			for (int ii = 0; ii < nodes_lower_layer; ii++) {
				this->deriv[start_weight_index] += this->node_dk[node_index + jj] * this->node_phi[prev_node + ii];
				start_weight_index++;
			}
		}

		// Update the starting node indices
		prev_node += sum_nodes_per_layer[ll];
		node_index += sum_nodes_per_layer[ll+1];
	}

	int a = 1;

	return;
}

void AnnClass::RunEpoch() {
	int pair_ind;
	double tan_square;
	int start_weight_index = 0; // changes as the signal passes through the layers
	int node_index = this->num_input_nodes; // start at the first node in second layer
	int bias_index = node_index; // biases skip the first layer
	int prev_node = 0; // starting index for the nodes in layer k
	int nodes_prev_layer, nodes_next_layer;
	int temp_index = 0; // temp variable to hold index for desired weight
	this->last_nodes_added = 0;

	double o_error = 0.0;
	double error_sum = 0.0;
	int num_output = this->num_output_nodes; // pull this out to make the code cleaner
	int num_layers = this->num_layers; // Its going to be used a few times, pull it out now
	int nodes_upper_layer, nodes_lower_layer;


	for (int pair_count = 0; pair_count < num_training_pairs; pair_count++) {

		if (this->random_pairs) {
			pair_ind = this->random_inds[pair_count];
		}
		else {
			pair_ind = pair_count;
		}
		// Perform back-propagation for all training pairs
		
		/*
		+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
						Forward Propagation					   						
		+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		*/
		
		for (int nn = 0; nn < this->num_input_nodes; nn++) { // loop over the nodes in the input layer
			//this->node_phi[nn] = this->ComputeActivation(this->i_data[ij_to_idx(input_idx, nn, this->num_input_nodes)]);
			//this->node_dphi[nn] = this->ComputeActDeriv(this->i_data[ij_to_idx(input_idx, nn, this->num_input_nodes)]);

			tan_square = tanh(this->i_data[ij_to_idx(pair_ind, nn, this->num_input_nodes)]);
			this->node_phi[nn] = tan_square;
			this->node_dphi[nn] = 1.0 - tan_square*tan_square;

			// Add to the mean vectors
			this->mean_node_phi[nn] += this->node_phi[nn] / num_training_pairs;
			this->mean_node_dphi[nn] += this->node_dphi[nn] / num_training_pairs;
		}

		// For the remaining layers, multiply the weights matrix by the output of the previous layer.
		// Because the number of slices of node_phi and node_f are +1 to the weight matrices, be
		// careful that the indexing is correct

		start_weight_index = 0; // changes as the signal passes through the layers
		node_index = this->num_input_nodes; // start at the first node in second layer
		bias_index = node_index; // biases skip the first layer
		prev_node = 0; // starting index for the nodes in layer k
		nodes_prev_layer, nodes_next_layer;
		temp_index = 0; // temp variable to hold index for desired weight


		for (int l_count = 1; l_count <= this->num_layers; l_count++) {
			// Grab the number of nodes in layers k and k+1
			//nodes_prev_layer = this->sum_nodes_per_layer[l_count-1];
			//nodes_next_layer = this->sum_nodes_per_layer[l_count];

			// Because the weights are stored in row major format, each row of the weight matrix are all the connections to node J
			// in layer k+1 from every node in layer k. So, the outer loop should increment over the nodes in layer k+1
			for (int n_count = 0; n_count < this->sum_nodes_per_layer[l_count]; n_count++) { // loop over the next layer
				this->node_f[node_index] = 0.0;
				this->node_phi[node_index] = 0.0;

				for (int p_count = 0; p_count < this->sum_nodes_per_layer[l_count - 1]; p_count++) { // loop over the previous layer
					this->node_phi[node_index] +=
						this->weights[start_weight_index + p_count] * this->node_phi[prev_node + p_count];
				}

				// Add the bias, if necessary
				if (this->use_biases && l_count < this->num_layers) {
					this->node_phi[node_index] += this->biases[bias_index];
					bias_index++;
				}

				// 2-4-18
				// if syntax is l_count < this->num_layers:
				// All layers are tanh(), except the output which is logistic
				if (l_count <= this->num_layers) {
					// Compute the derivative first since node_phi gets overwritten
					//this->node_dphi[node_index] = this->ComputeActDeriv(this->node_phi[node_index]);
					// Compute the activation for this node
					//this->node_phi[node_index] = this->ComputeActivation(this->node_phi[node_index]);

					tan_square = tanh(this->node_phi[node_index]);
					this->node_phi[node_index] = tan_square;
					this->node_dphi[node_index] = 1.0 - tan_square*tan_square;
				}
				else {
					// Compute the derivative first since node_phi gets overwritten
					this->node_dphi[node_index] = this->ComputeLogDeriv(this->node_phi[node_index]);
					// Compute the activation for this node
					this->node_phi[node_index] = this->ComputeLogistic(this->node_phi[node_index]);
				}

				this->mean_node_phi[node_index] += this->node_phi[node_index] / num_training_pairs;
				this->mean_node_dphi[node_index] += this->node_dphi[node_index] / num_training_pairs;

				// Update the starting index for the weights (move to the next row)
				start_weight_index += this->sum_nodes_per_layer[l_count - 1];;

				// Move to the next node 
				node_index++;
			}

			// Update the index for the starting node in the previous layer
			prev_node += this->sum_nodes_per_layer[l_count - 1];
		}

		/*
		+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
							Compute Errors					   						
		+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		*/
		// Compute the output error vector
		// Here, compute the square of the difference between the ANN output and the target output
		error_sum = 0.0;
		//double *ann_out = new double[this->num_output_nodes];
		//double *target_out = new double[this->num_output_nodes];
		//double *target_in = new double[this->num_input_nodes];
		//int *temp_inds = new int[this->num_output_nodes];
		for (int oo = 0; oo < this->num_output_nodes; oo++) {
			error_sum += pow(this->o_data[ij_to_idx(pair_ind, oo, this->num_output_nodes)] -
				this->node_phi[this->num_nodes - this->num_output_nodes + oo], 2);

			this->output_error_mean[oo] += (this->o_data[ij_to_idx(pair_ind, oo, this->num_output_nodes)] -
				this->node_phi[this->num_nodes - this->num_output_nodes + oo]) / num_training_pairs;

			//ann_out[oo] = this->node_phi[this->num_nodes - this->num_output_nodes + oo];
			//target_out[oo] = this->o_data[ij_to_idx(pair_id, oo, this->num_output_nodes)];
			//temp_inds[oo] = ij_to_idx(pair_id, oo, this->num_output_nodes);
		}

		// Update the mean error
		this->mean_error += error_sum / this->num_training_pairs;

		// If the current error is larger than the max error, update the max error and id_max
		if (error_sum > this->max_error) {
			this->max_error = error_sum;
			this->id_max = pair_ind;
		}

		// Update the percent correct
		if (error_sum < this->error_mean_limit) {
			this->percent_correct += 100.0 / this->num_training_pairs;
		}
	}

	/*
	+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	Propagate error backward
	+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	*/
	
	// In this version, compute the weight updates using the output error and node activations

	// Compute the error at the output first
	// E = (t_k - a_k), where t_k is the expected output,
	// a_k is the actual output

	// Compute the output error
	for (int oo = 0; oo < this->num_output_nodes; oo++) {

		// Multiply the output error by the derivative of the output node's activation
		this->node_dk[this->num_nodes - this->num_output_nodes + oo] =
			this->output_error_mean[oo] * this->mean_node_dphi[this->num_nodes - this->num_output_nodes + oo];
	}

	// Compute the delta_j for the hidden layers
	// delta_j = phi'(y_j)*sum(delta_k*w_kj), where k indexes the layer above
	// This is going to be a bit of a mess. Move backward through the network.
	// nodes_upper_layers is "k+1"
	// nodes_lower_layer (nodes preceding layer) is "k"
	start_weight_index = this->num_weights - 1; // changes as the signal passes through the layers
	node_index = this->num_nodes - this->sum_nodes_per_layer[this->num_layers] - 1; // start at the last node in the second to last layer
	bias_index = node_index; // biases skip the last layer
	prev_node = this->num_nodes - 1; // starting index for the last node in the last layer
	nodes_upper_layer, nodes_lower_layer;

	for (int ll = num_layers - 1; ll >= 0; ll--) { // move backward through the network, skipping the last layer
		nodes_upper_layer = this->sum_nodes_per_layer[ll + 1];
		nodes_lower_layer = this->sum_nodes_per_layer[ll];
		// initialize temp_dk to zero
		for (int oo = 0; oo < nodes_lower_layer; oo++) {
			temp_dk[oo] = 0.0;
		}

		// Again, the rows of the weight matrix correspond to nodes in this_layer. That means moving along a row indexes the nodes
		// in prec_layer. We need sum(delta_k*w_kj). As such, temp_dk is going to hold the temporary values
		for (int t_count = 0; t_count < nodes_upper_layer; t_count++) {// index over layer k+1
			for (int p_count = 0; p_count < nodes_lower_layer; p_count++) { // index over layer k (preceding layer)
				temp_dk[p_count] += this->weights[start_weight_index - p_count] * this->node_dk[prev_node - t_count];
			}
			// Update the starting weight index
			start_weight_index -= nodes_lower_layer;
		}

		// Now compute the node_dk. BUT, temp_dk is indexed in reverse (i.e., [0] is actually the last node in the layer).
		// node_index starts at the last node in the lower layer.
		for (int p_count = 0; p_count < nodes_lower_layer; p_count++) {
			this->node_dk[node_index - p_count] = temp_dk[p_count] * this->mean_node_dphi[node_index - p_count];
		}

		// Update the indices prev_node and node_index
		prev_node -= nodes_upper_layer;
		node_index -= nodes_lower_layer;

	}

	// Now that we have all of the deltas, compute dE/dw_ji for each weight.
	// dE/dw_ji = delta_j*phi_i, where phi_i is the output of node 'i' in layer j-1.
	// If I had the node_dk and node_dphi vectors split out by layer, the gradient is easily computed
	// as an outer product. Unfortunately, I don't have the data structured that way.
	// I can still take the outer product, though. I just need to put the proper limits on the loops

	start_weight_index = 0;
	prev_node = 0; // starts indexing the first layer
	node_index = this->sum_nodes_per_layer[0]; // starts index the second layer
	for (int ll = 0; ll < this->num_layers; ll++) {
		nodes_lower_layer = this->sum_nodes_per_layer[ll];
		nodes_upper_layer = this->sum_nodes_per_layer[ll + 1];

		// Okay, w_ji connects node i in layer k to node j in layer k+1. If I want to be able to simply increment 
		// weight increment, I have to make sure the loops are ordered correctly.
		// The j are the rows in the weight matrix
		// The i are the columns in the weight matrix
		// The weight matrix is row-major, so 'i' should be nested inside 'j'
		for (int jj = 0; jj < nodes_upper_layer; jj++) {
			for (int ii = 0; ii < nodes_lower_layer; ii++) {
				this->deriv[start_weight_index] += this->node_dk[node_index + jj] * this->mean_node_phi[prev_node + ii];
				start_weight_index++;
			}
		}

		// Update the starting node indices
		prev_node += sum_nodes_per_layer[ll];
		node_index += sum_nodes_per_layer[ll + 1];
	}
}

double AnnClass::ComputeActivation(double in_value) {
	// Computes the activation function tanh()
	return tanh(this->sigmoid_beta*in_value);
	//return this->ComputeLogistic(in_value);
	//return (in_value >= 0) ? in_value : 0.0;
}

double AnnClass::ComputeActDeriv(double in_value) {
	// Computes the derivative of the activation function
	double tan_square = tanh(this->sigmoid_beta*in_value)*tanh(this->sigmoid_beta*in_value);
	return this->sigmoid_beta*(1-tan_square);
	//return this->ComputeLogDeriv(in_value);
	//return (in_value >= 0) ? 1.0 : 0.01;
}

double AnnClass::ComputeLogistic(double in_value)
{
	return 1.0 / (1.0 + exp(-in_value));
}

double AnnClass::ComputeLogDeriv(double in_value) {
	// the derivative of the logistic function if f(x)*(1-f(x))
	return ComputeLogistic(in_value)*(1.0 - ComputeLogistic(in_value));
}

void AnnClass::ComputeWeightUpdates() {
	// Function to update the delta_ji values
	
	// Unlike the version using Eigen, weights and derivatives are stored as vectors
	// Just loop over the entire weight vector
	double deriv_sign;
	for (int ww = 0; ww < this->num_weights; ww++) {
		// Multiply the current and previous weight gradients
		deriv_sign = this->deriv[ww] * this->prev_deriv[ww];

		// The change in the update value is based off the sign of the above product
		// Note, this algorithm is based on iRPROP-
		if (deriv_sign > zero_tol) {
			// No change in derivative sign, increase the delta_ji
			this->w_delta[ww] *= this->eta_inc;

			// Make sure the weight update value is within upper bound
			if (this->w_delta[ww] > this->delta_max) {
				this->w_delta[ww] = this->delta_max;
			}
		}
		else if (deriv_sign < -zero_tol) {
			// There's been a sign change, decrease the delta_ji
			this->w_delta[ww] *= this->eta_dec;
			// Also need to zero out this weight gradient
			this->deriv[ww] = 0.0;

			// Make sure the weight update is within lower bound
			if (this->w_delta[ww] < this->delta_min) {
				this->w_delta[ww] = this->delta_min;
			}

		}
		else {
			// There's a zero, nothing needs to be done.
			int a = 1;
		}

		// Copy the current derivatives to prev_deriv
		this->prev_deriv[ww] = this->deriv[ww];
	}

	// If using biases, do the same as above
	//if (this->use_biases) {
	//	// When indexing the array for the biases, be sure to use -1 to start indexing at 0
	//	for (int ll = 1; ll < this->num_layers; ll++) { // loop through hidden layers.
	//													// Now, loop through the derivative matrix
	//		for (int rr = 0; rr < this->bias_deriv[ll - 1].rows(); rr++) {
	//			// Multiply the current and previous dE/dw_ji
	//			double deriv_sign = this->bias_deriv[ll - 1].array()(rr)*this->bias_prev_deriv[ll - 1].array()(rr);
	//			// The change in the update value is based off the sign of the above product
	//			// Note, this algorithm is based on iRPROP-
	//			if (deriv_sign > this->zero_tol) {
	//				// No change in derivative sign, increase the delta_ji
	//				this->bias_delta[ll - 1].array()(rr) *= this->eta_inc;
	//			}
	//			else if (deriv_sign < -this->zero_tol) {
	//				// There's been a sign change, decrease the delta_ji
	//				this->bias_delta[ll - 1].array()(rr) *= this->eta_dec;
	//				// Also need to zero out this weight gradient
	//				this->bias_deriv[ll - 1].array()(rr) = 0.0;
	//			}

	//			// Make sure the weight update value is within bounds
	//			if (this->bias_delta[ll - 1].array()(rr) > this->delta_max) {
	//				this->bias_delta[ll - 1].array()(rr) = this->delta_max;
	//			}
	//			else if (this->bias_delta[ll - 1].array()(rr) < this->delta_min) {
	//				this->bias_delta[ll - 1].array()(rr) = this->delta_min;
	//			}
	//		}
	//		// Copy the new derivative matrix to prev_deriv
	//		this->bias_prev_deriv[ll - 1].array() = this->bias_deriv[ll - 1].array();

	//	}
	//}
	
	int a = 1;
}

void AnnClass::ComputeWeightIncrements() {
	// Function to compute weight increments
	// Even though the increment computation and weight update can occur in the same step,
	// the use of nested networks and dynamic node creation would cause a lot of conditional statements.
	// Therefore, it's a little easier (and cleaner, arguably faster) to separate the computation of the increments 
	// before the weight update so masking can just be done on the increments.
	for (int ww = 0; ww < this->num_weights; ww++) {
		this->w_inc[ww] = my_sign(this->deriv[ww])*this->w_delta[ww];
	}
}

void AnnClass::UpdateWeights() {
	// Function to update the weights with the pre-computed weight increments
	for (int ww = 0; ww < this->num_weights; ww++) {
		this->weights[ww] += this->w_inc[ww];
	}
}

void AnnClass::ResetMatrices() {
	// Function to re-initialize the training matrices/vectors
	for (int ww = 0; ww < this->num_weights; ww++) {
		this->deriv[ww] = 0.0;
		this->prev_deriv[ww] = 0.0;
		this->w_delta[ww] = this->delta0;
		this->w_inc[ww] = 0.0;
	}
}

void AnnClass::ZeroMatrices() {
	// Function to zero out select training matrices/vectors at the start of each epoch
	for (int ww = 0; ww < this->num_weights; ww++) {
		this->deriv[ww] = 0.0;
		this->w_inc[ww] = 0.0;
	}

	for (int nn = 0; nn < num_nodes; nn++) {
		this->node_dk[nn] = 0.0;
		this->node_phi[nn] = 0.0;
		this->node_dphi[nn] = 0.0;

		this->mean_node_phi[nn] = 0.0;
		this->mean_node_dphi[nn] = 0.0;
	}
	
	for (int nn = 0; nn < this->num_output_nodes; nn++) {
		this->output_error_mean[nn] = 0.0;
	}

	if (this->use_biases) {
		for (int nn = 0; nn < this->num_nodes; nn++) {
			this->bias_deriv[nn] = 0.0;
		}
	}
}

void AnnClass::AddNodes() {
	// Function to add nodes to the hidden layers

	// First, check that nodes can be aded 
	if (this->nodes_added >= this->max_nodes) {
		// The maximum number of nodes have already been added. Do nothing.
		this->reached_node_limit = true;
		this->out_data << "Reached added node limit." << endl;
		return;
	} 

	this->epochs_between_add = 0;
	int num_add_nodes, node_diff;
	node_diff = this->max_nodes - this->nodes_added;
	if (node_diff > this->nodes_per_step) {
		// By adding the number specified in nodes_per_step, the total added nodes won't go over max_nodes. 
		// Therefore, add nodes_per_step nodes to each hidden layer
		num_add_nodes = this->nodes_per_step;
	}
	else {
		// In this case, adding nodes_per_step nodes would surpass max_nodes. So only add node_diff to each layer
		num_add_nodes = node_diff;
	}

	// Update how many nodes have been added 
	this->nodes_added += num_add_nodes;
	this->last_nodes_added = num_add_nodes;

	// Update prev_nodes_per_layer with the current number of nodes per layer
	this->prev_nodes_per_layer = this->nodes_per_layer;

	// Create the random number generator for initializing new weights (and biases)
	default_random_engine rand_generator;
	normal_distribution<double> norm_dist(0.0, 1.0); // standard normal: 0-mean, unit variance
	uniform_real_distribution<double> uniform_dist(-0.2,0.2);
	rand_generator.seed(std::chrono::system_clock::now().time_since_epoch().count());

	// Loop over the layers
	// For each layer, add the specified number of nodes and associated weights
	// But first, compute how many new nodes and weights will be added to create the temporary matrices/vectors
		// the first term accounts for the new nodes added to all hidden layers
	int new_num_nodes = (this->num_layers - 1)*num_add_nodes + this->num_nodes;
	// Computing the total number of new weights is a little more involved because it depends on the number of nodes
	// in each layer
	int new_num_weights = 0;
	int new_prev_layer, new_next_layer;
	for (int ll = 0; ll < this->num_layers; ll++) {
		if (ll == 0) {
			// No nodes are added to the input layer
			new_prev_layer = this->sum_nodes_per_layer[ll] ;
			new_next_layer = this->sum_nodes_per_layer[ll + 1] + num_add_nodes;
		}
		else if ((ll + 1) == this->num_layers) {
			// No nodes are added to the output layer
			new_prev_layer = this->sum_nodes_per_layer[ll] + num_add_nodes;
			new_next_layer = this->sum_nodes_per_layer[ll + 1];
		}
		else {
			// Both layers are hidden
			new_prev_layer = this->sum_nodes_per_layer[ll] + num_add_nodes;
			new_next_layer = this->sum_nodes_per_layer[ll + 1] + num_add_nodes;
		}

		new_num_weights += new_prev_layer*new_next_layer;
	}

	// Update the nodes_per_layer and sum_nodes_per_layer matrices
	int *new_nodes_per_layer = new int[this->num_modules*(this->num_layers + 1)]; // +1 to add input layer
	int *new_sum_nodes_per_layer = new int[(this->num_layers + 1)]; // +1 to include input layer

	for (int mm = 0; mm < this->num_modules; mm++) {
		for (int ll = 0; ll <= num_layers; ll++) {
			int temp_index = ij_to_idx(mm, ll, (num_layers + 1));
			new_nodes_per_layer[ij_to_idx(mm, ll, (num_layers + 1))] =
				this->nodes_per_layer[ij_to_idx(mm, ll, (num_layers + 1))];
			if (ll > 0 && ll < this->num_layers) {
				// Only add nodes to the hidden layers
				new_nodes_per_layer[ij_to_idx(mm, ll, (num_layers + 1))] += num_add_nodes;
			}
			if (mm == 0) {
				new_sum_nodes_per_layer[ll] = 0;
			}
		}
	}

	// Compute the new sum total nodes per layer
	for (int mm = 0; mm < this->num_modules; mm++) {
		for (int ll = 0; ll <= this->num_layers; ll++) {
			new_sum_nodes_per_layer[ll] += new_nodes_per_layer[ij_to_idx(mm, ll, (num_layers + 1))];
		}
	}

	// Create the temporary vectors
	// Go ahead and create the mask to avoid weight/bias updates for the existing nodes during the freeze period
	double *temp_weights = new double[new_num_weights];
	double *temp_prev_deriv = new double[new_num_weights];
	double *temp_w_inc = new double[new_num_weights];
	double *temp_w_delta = new double[new_num_weights];

	// the current gradient does not need to be carried over
	delete[] this->deriv;
	this->deriv = new double[new_num_weights];

	delete[] this->freeze_mask; // delete any existing mask
	this->freeze_mask = new double[new_num_weights]; // allocate space for the new mask

	// No data copying has to occur for node activations and derivatives. Just re-allocate memory for the new
	// vector sizes
	delete[] this->node_f;
	delete[] this->node_phi;
	delete[] this->node_dphi;
	delete[] this->node_dk;

	this->node_f = new double[new_num_nodes];
	this->node_phi = new double[new_num_nodes];
	this->node_dphi = new double[new_num_nodes];
	this->node_dk = new double[new_num_nodes];

	// Deal with the biases later 

	// Again, loop over the layers. For each layer, copy the existing weights, (previous) gradients, and 
	// weight increments. For the new connections, re-initialize as done in the constructor
	int old_weight_index = 0;
	int new_weight_index = 0; // starting index of weights
	int nodes_prev_layer, nodes_next_layer; // size of layers before node addition
	int new_npl, new_nnl; // size of layers after node addition
	for (int ll = 0; ll < this->num_layers; ll++) {

		// Loop over the old nodes in the next layer
		for (int nl = 0; nl < this->sum_nodes_per_layer[ll + 1]; nl++) {
			// Loop over the old nodes in the previous layer
			for (int pl = 0; pl < this->sum_nodes_per_layer[ll]; pl++) {
				temp_weights[new_weight_index] = this->weights[old_weight_index];
				temp_prev_deriv[new_weight_index] = this->prev_deriv[old_weight_index];
				temp_w_inc[new_weight_index] = this->w_inc[old_weight_index];
				temp_w_delta[new_weight_index] = this->w_delta[old_weight_index];
				this->freeze_mask[new_weight_index] = 0.0; // old weights are frozen
				new_weight_index++;
				old_weight_index++;
			}

			// Loop over the new nodes in the previous layer
			for (int pl = this->sum_nodes_per_layer[ll]; pl < new_sum_nodes_per_layer[ll]; pl++) {
				// these are new weights.
				temp_weights[new_weight_index] = uniform_dist(rand_generator);
				//temp_weights[new_weight_index] = norm_dist(rand_generator) * 0.2 - 0.1;
				temp_prev_deriv[new_weight_index] = 0.0;
				temp_w_inc[new_weight_index] = 0.0;
				temp_w_delta[new_weight_index] = this->delta0;
				this->freeze_mask[new_weight_index] = 1.0; // new weights are not frozen
				new_weight_index++;
			}
		}

		// Now, loop over the new nodes in the next layer
		for (int nl = this->sum_nodes_per_layer[ll + 1]; nl < new_sum_nodes_per_layer[ll+1]; nl++) {
			// These are all new weights, so loop over all nodes (old and new) in the previous layer
			// Loop over the new nodes in the previous layer
			for (int pl = 0; pl < new_sum_nodes_per_layer[ll]; pl++) {
				// these are new weights.
				temp_weights[new_weight_index] = uniform_dist(rand_generator);
				temp_prev_deriv[new_weight_index] = 0.0;
				temp_w_inc[new_weight_index] = 0.0;
				temp_w_delta[new_weight_index] = this->delta0;
				this->freeze_mask[new_weight_index] = 01.0; // new weights are not frozen
				new_weight_index++;
			}
		}

	}

	// Okay, now have to do a handoff between the new and old pointers.
	// I was trying to think of how to switch addresses using temporary pointer, but I think it would 
	// causes more issues when trying to de-allocate memory.
	// The more logically consistent (albeit slower) way is to delete the old pointers, copy over the new data,
	// and them delete the new pointer

	delete[] this->nodes_per_layer;
	delete[] this->sum_nodes_per_layer;

	this->nodes_per_layer = new_nodes_per_layer;
	this->sum_nodes_per_layer = new_sum_nodes_per_layer;

	//this->nodes_per_layer = new int[this->num_modules*(this->num_layers + 1)]; // +1 to add input layer
	//this->sum_nodes_per_layer = new int[(this->num_layers + 1)]; // +1 to include input layer

	// Copy over nodes_per_layer and sum_nodes_per_layer
	//for (int mm = 0; mm < this->num_modules; mm++) {
	//	for (int ll = 0; ll <= num_layers; ll++) {
	//		this->nodes_per_layer[ij_to_idx(mm, ll, (num_layers + 1))] =
	//			new_nodes_per_layer[ij_to_idx(mm, ll, (num_layers + 1))];

	//		if (mm == 0) {
	//			this->sum_nodes_per_layer[ll] = new_sum_nodes_per_layer[ll];
	//		}
	//	}
	//}

	//delete[] new_nodes_per_layer;
	//delete[] new_sum_nodes_per_layer;

	this->num_weights = new_num_weights;
	this->num_nodes = new_num_nodes;

	delete[] this->weights;
	delete[] this->prev_deriv;
	delete[] this->w_inc;
	delete[] this->w_delta;

	this->weights = temp_weights;
	this->prev_deriv = temp_prev_deriv;
	this->w_inc = temp_w_inc;
	this->w_delta = temp_w_delta;

	//this->weights = new double[this->num_weights];
	//this->prev_deriv = new double[this->num_weights];
	//this->w_inc = new double[this->num_weights];
	//this->w_delta = new double[this->num_weights];

	//for (int ww = 0; ww < this->num_weights; ww++) {
	//	this->weights[ww] = temp_weights[ww];
	//	this->prev_deriv[ww] = temp_prev_deriv[ww];
	//	this->w_inc[ww] = temp_w_inc[ww];
	//	this->w_delta[ww] = temp_w_delta[ww];
	//	this->deriv[ww] = 0.0; // initialize to avoid any issues
	//}

	//delete[] temp_weights;
	//delete[] temp_w_inc;
	//delete[] temp_w_delta;
	//delete[] temp_prev_deriv;

	this->freeze_count = 0;
	this->freeze_weights = true;
	this->out_data << "Freezing weights." << endl;

}

void AnnClass::MaskWeightsFrozen() {
	// Function to freeze old weights after dynamic node creation
	for (int ww = 0; ww < this->num_weights; ww++) {
		// Simply multiply by the freeze mask
		this->deriv[ww] *= this->freeze_mask[ww];
		this->w_inc[ww] *= this->freeze_mask[ww];
	}

	return;
}

void AnnClass::GetRandomIndices() {
	// Pulls out a subset of random indices.The array of indices has < num_training_pairs entries

	// To generate the list of random indices, the modified version of the Fisher-Yates shuffle is used

	// If not yet done, initialize the random shuffle array
	if (!this->random_init) {
		this->random_init = true;
		// Generate an array of numbers from 0 to (num_training_pairs-1) that will be repeatedly shuffled
		// for the random sequence generation
		// Memory for the array was allocated in InitializeMatrices()
		for (int ii = 0; ii < this->num_training_pairs; ii++) {
			this->random_inds[ii] = ii;
		}
	}

	int rand_ind;
	int temp_num;
	// Now do the shuffle
	for (int ii = this->num_training_pairs - 1; ii > 0; ii--) {
		rand_ind = rand() % ii; // generate and random number from 0 to ii-1
								// swap the entries at rand_ind and ii
		temp_num = this->random_inds[ii];
		this->random_inds[ii] = this->random_inds[rand_ind];
		this->random_inds[rand_ind] = temp_num;
	}

	return;

}

void AnnClass::MaskWeightsDropout() {
	// Function to mask weights that have been 'dropped'
	// At this time, I'm just going reset weights instead of dropping them
	default_random_engine rand_generator;
	uniform_real_distribution<double> uniform_dist(0.0, 1.0);
	uniform_real_distribution<double> weight_dist(-0.2, 0.2);
	rand_generator.seed(std::chrono::system_clock::now().time_since_epoch().count());
	for (int ww = 0; ww < this->num_weights; ww++) {
		// Generate a random number in the [0,1] range. If it falls below the dropout probability,
		// reset the weight.
		if (uniform_dist(rand_generator) < this->dropout_prob) {
			this->weights[ww] = weight_dist(rand_generator);
		}
	}
}