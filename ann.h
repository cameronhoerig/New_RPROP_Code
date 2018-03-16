#pragma once

#include <string>
#include <fstream>
#include <vector>

using namespace std;

class AnnClass {
	public:
		ofstream out_data;

		// List of matrices (stored as vectors in row-major form)
		double *weights; // A single vector holding all ANN weights
		double *prev_deriv; // "" "" "" previous dE/dw_ji for each weight
		double *deriv; // "" "" "" currenct dE/dw_ji for each weight
		double *w_delta; // Weight updates - used to determine weight increments
		double *w_inc; // Weight increments
		double *freeze_mask; // contains the masks used to freeze weights after dynamic node creation
		double *weight_dropout_mask; // Contains the masks for the ANN weights for dropout

		int *nodes_per_layer; // stores the number of nodes in each layer in each module
								// Each row of this matrix corresponds to a different module
		int *prev_nodes_per_layer;
		int *sum_nodes_per_layer; // The total nodes per layer over all modules

		// List of vectors
		double *node_f; // Activation for each node
		double *node_phi; // output of each node
		double *node_dphi; // Derivative at each node
		double *node_dk; // error delta at each node

		double *biases; // The biases for each node
		double *bias_deriv;
		double *bias_prev_deriv;
		double *bias_delta;
		double *bias_inc;
		double *freeze_vector_mask; // Contains the masks used to freeze biases after dynamic node creation
		double *dropout_mask; // Contains the masks for node dropout

		double *i_scale; // input scaling vector
		double *o_scale; // output scaling vector

		double *temp_dk; // vector used to hold temporary values
						 // Putting it as a class member so it doesn't have to be constantly created and destroyed

		double *mean_node_phi;
		double *mean_node_dphi;
		double *output_error_mean;

		int num_training_pairs;
		int num_layers;
		int num_modules;
		int nodes_per_step;
		int max_nodes;
		int num_epoch_freeze;
		int freeze_count;
		int min_epoch;
		int max_epoch;
		int error_freq;
		int weight_freq;
		int num_input_nodes; // The total number of input nodes over all modules
		int num_output_nodes; // The total number of output nodes over all modules
		int num_cycles; // The number of times to train over all epochs.
						// Used for staggered training
		int num_random_pairs; // specifies how many random pairs to use
		int random_epoch; // specifies the number of epochs before a new random subset of training pairs
						  // should be pulled out
		int *random_inds; // array of random indices specifying which training pairs to use
		int *shuffle_array;
		int mask_option; // changes the input masking option.
		int nodes_added; // keeps track of how many nodes have been dynamically added
		int dropout_epochs; // number of epochs between choosing new nodes for dropout
		int random_reset_epochs; // number of epochs between choosing new weights to reset
		int num_weights;
		int num_nodes;
		int epochs_between_add;
		int last_nodes_added;

		double delta0;
		double delta_min;
		double delta_max;
		double init_inc;
		double momentum_alpha;
		double sigmoid_beta;
		double zero_tol;
		double eta_inc;
		double eta_dec;
		double weight_init_factor;
		double error_max_limit;
		double error_mean_limit;
		double dropout_prob; // probability of dropping out a node
		double random_reset_prob; // probability of resetting a weight

		bool freeze_weights; // Indicates whether weights are frozen after nodes have been added
		bool stagger;
		bool separated;
		bool stagger_alt; // the alternative form of staggered training (for use with separated ANNs)
		bool change_stage;
		bool cartesian_module; // specifies if the last module is part of a Cartesian ANN
		bool do_dropout;
		bool random_reset;
		bool do_reset;
		bool retraining;
		bool random_pairs; // specifies if a random subset of training pairs should be used
		bool random_init;
		bool reached_node_limit; // flags whether or not the maximum number of nodes have been added
		bool prescaled;
		bool use_biases; // specifies if biases should be used for the hidden layers
		
		char *input_file;
		string analysis_type;
		string training_type;
		string data_file;
		string output_weight_fid;
		string prev_weight_fid;
		string output_data_fid;

		double *i_data;
		double *o_data;
		int *stagger_epochs;
		int *separated_params;
		
		// -------------------------------------
		// Variables for error calculation
		// -------------------------------------
		double mean_error;
		double max_error;
		double percent_correct;
		double prev_error; // holds the previous mean error 
		double slope_limit;

		int id_max;
		int epoch_check_slope;

		vector<double> error_vec;

		AnnClass(char *in_file = ".\\nann.inp"); // The constructor
		//~AnnClass(); // The destructor
		
		void ReadInputFile(); // Function that reads the training input file
		void ReadDataFile(); // Function that reads in the training data
		void ReadDataBinary(); // Function that reads in the training data from a binary file
		void InitializeMatrices(); // Function to initialize matrices and vectors for ANN
		//void GetRandomIndices(); // Function to pull out a subset of random indices of the training data
		void PrintDataHeader(); // Function to print the header to the output data file
		void PrintArchitecture(); // Function to print the ANN architecture to the output data file

		void ReadPreviousWeights();
		void MaskWeights(); // Function to mask weights when multiple modules are present
		//void MaskWeightsStagger(); // Function to mask weights for staggered training
		//void MaskWeightsSeparated(); // Function to mask weights for a separated architecture
		void MaskWeightsFrozen(); // Function to mask weights after nodes are dynamically added
		void MaskWeightsDropout(); // Function to mask weight increments during node dropout
		void UpdateWeights(); // Updates the value of the connection weights
		void PropagateSignalForward(int); // Perform a forward pass through the ANN
		void PropagateErrorBackward(int); // Compute error deltas and dE/dw_ji for each node and weight
		void RunEpoch(); // Loops over all training pairs to compute backpropagation without a bunch of function calls for each training pair
		void ComputeWeightUpdates(); // Function to compute the weight updates delta_ji
		void ComputeWeightIncrements(); // Function to compute the weight increments
		double ComputeActivation(double); // Computes the output of each node (the activation function)
		double ComputeActDeriv(double); // Computes the derivative of the activation function
		double ComputeLogistic(double); // the logistic function (alternative to hyperbolic tangent)
		double ComputeLogDeriv(double); // the derivative of the logistic function
		void ZeroMatrices(); // Function to zero out certain matrices at the start of a new epoch
		void ResetMatrices(); // Function to reset the matrices, like that done in the constructor.

		void GetRandomIndices(); // Function to pull out a subset of random training indices of the training data

		void PrintWeights(string, int); // Function to print the ANN weights to a file
		void AddNodes(); // Function to add nodes 
		//void CreateDropoutMask(); // Function to create the masks for node dropout
		//void ResetRandomWeights(); // Function to reset the weights connected to a particular node
		//

		// -------------------------------------
		// Methods for error calculation
		// -------------------------------------
		void CalculateErrors(int); // Function to calculate the errors
		void PrintErrors(int); // Function to print the errors to the output data file
		void ResetErrors(); // Function to reset the errors at the start of each epoch
		bool CheckSlope(); // Function to check the error slope to determine if nodes should be added


};