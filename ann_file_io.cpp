#include <string>
#include <fstream>
#include <vector>
#include <math.h>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <time.h>
#include <direct.h>
#include <stdio.h>
#include "ann.h"
#include "Misc.h"

#define GetCurrentDir _getcwd
#define ij_to_idx(i, j, num_cols) ((i*num_cols) + j) // convert (i,j) index to vector index for row-major ordering
// This next one takes some explaining. This function computes the index of the weight in the weight vector connecting
// node i in layer k to node j in layer k+1. Doing so requires that we know:
//	i,j,k (of course)
//  ss -> the total number of nodes before layer k
using namespace std;

// The only default is that input_file is ".\nann.exe"
AnnClass::AnnClass(char *in_file) {
	cout << "Start" << endl;
	this->input_file = in_file;
	this->retraining = false; // the default is the ANN is not being retrained
	this->stagger = false; // the default is no staggered training
	this->nodes_added = 0; // no nodes have been added at this point
	this->reached_node_limit = false;
	this->prescaled = false;
	this->cartesian_module = false;
	this->use_biases = false;

	this->freeze_weights = false;

	// set default values for eta_inc and eta_dec
	// These are the ones used in the paper.
	this->eta_inc = 1.2;
	this->eta_dec = 0.5;

	this->zero_tol = 0.00000001; // Set a default value for the zero tolerance

	// Set default values for members associated with the error calculations
	this->max_error = 0.0;
	this->mean_error = 0.0;
	this->percent_correct = 0.0;
	this->id_max = 0;
}

// The ReadInputFile function reads in and stores the training parameters
void AnnClass::ReadInputFile() {

	char buff[FILENAME_MAX];
	GetCurrentDir(buff, FILENAME_MAX);
	cout << "Current dir: " << buff << endl;

	ifstream in_file;
	in_file.open(this->input_file, ios::in);

	string this_line; // variable to hold each line of the input file as it is read
	size_t found; // variable for finding substrings within a string
	size_t sep_found; // like found, but specifically looking for the keyword #separated
	size_t random_found; // looking for the keyword #random
	size_t prescaled_found; // looking for the keyword #prescaled
	size_t cart_found; // looking for the keyword #cartesian
	size_t bias_found; // looking for the keyword #bias
	size_t drop_found; // looking for keyword #dropout
	size_t random_reset_found; // looking for keyword #reset

							   // The first line should contain the '*job' identifier. If it doesn't, there's an error
	this_line = GetNonemptyLine(in_file);
	found = this_line.find("*job");
	if (found == string::npos) {
		// The *job identifier not found. Close the file and throw an error
		cout << "*job identifier not found in " << this->input_file << endl;
		in_file.close();
		exit(EXIT_FAILURE);
	}

	// The next line specifies then number of layers (excluding input layer) and number of NANN modules
	this_line = GetNonemptyLine(in_file);
	vector<string> split_line;
	split_line = SplitString(this_line);
	int num_layers = atoi(split_line.at(0).c_str()); this->num_layers = num_layers;
	int num_modules = atoi(split_line.at(1).c_str()); this->num_modules = num_modules;

	// Initialize the nodes_per_layer member now that we know how many layers and modules
	this->nodes_per_layer = new int[num_modules*(num_layers + 1)]; // +1 to add input layer

	// The next num_modules lines specify the number of nodes in each layer
	// Also, initialize sum_nodes_per_layer to zero.
	this->sum_nodes_per_layer = new int[num_layers + 1]; // +1 to include input layer
	for (int mm = 0; mm < num_modules; mm++) {
		this_line = GetNonemptyLine(in_file);
		split_line = SplitString(this_line);

		for (int ll = 0; ll <= num_layers; ll++) {
			int temp_index = ij_to_idx(mm, ll, (num_layers + 1));
			this->nodes_per_layer[ij_to_idx(mm, ll, (num_layers + 1))] = atoi(split_line.at(ll).c_str());
			if (mm == 0) {
				this->sum_nodes_per_layer[ll] = 0;
			}
		}

	}

	// Compute the sum total nodes per layer
	
	for (int mm = 0; mm < num_modules; mm++) {
		for (int ll = 0; ll <= num_layers; ll++) {
			this->sum_nodes_per_layer[ll] += this->nodes_per_layer[ij_to_idx(mm, ll, (num_layers+1))];
		}
	}
	// Count the total number of input and output nodes
	this->num_input_nodes = this->sum_nodes_per_layer[0];
	this->num_output_nodes = this->sum_nodes_per_layer[num_layers];

	// Go ahead define the i_scale and o_scale members
	this->i_scale = new double[this->num_input_nodes];
	this->o_scale = new double[this->num_output_nodes];

	// The next line should start with "adaptive" and specifies some training parameters
	this_line = GetNonemptyLine(in_file);
	found = this_line.find("adaptive");
	if (found == string::npos) {
		// The *job identifier not found. Close the file and throw an error
		cout << "adaptive keyword not found in " << this->input_file << endl;
		in_file.close();
		exit(EXIT_FAILURE);
	}
	split_line = SplitString(this_line);
	this->nodes_per_step = atoi(split_line.at(1).c_str());
	this->max_nodes = atoi(split_line.at(2).c_str());
	this->epoch_check_slope = atoi(split_line.at(3).c_str());
	this->slope_limit = atof(split_line.at(4).c_str());
	this->num_epoch_freeze = atoi(split_line.at(5).c_str());
	this->freeze_count = 0;

	if (this->max_nodes < 1) {
		// No nodes are to be added during training
		this->reached_node_limit = true;
	}

	// Make sure the next line has the '*train' identifier
	this_line = GetNonemptyLine(in_file);
	found = this_line.find("*train");
	if (found == string::npos) {
		// The *job identifier not found. Close the file and throw an error
		cout << "*train keyword not found in " << this->input_file << endl;
		in_file.close();
		exit(EXIT_FAILURE);
	}
	this->analysis_type = "train";

	// The next line specifies the training algorithm. For now (5-14-16), only RPROP can be done
	this_line = GetNonemptyLine(in_file);
	found = this_line.find("rprop");
	if (found == string::npos) {
		// The *job identifier not found. Close the file and throw an error
		cout << "rprop keyword not found in " << this->input_file << endl;
		in_file.close();
		exit(EXIT_FAILURE);
	}
	this->training_type = "rprop";

	// The next line specifies RPROP parameters
	this_line = GetNonemptyLine(in_file);
	split_line = SplitString(this_line);
	this->delta0 = atof(split_line.at(0).c_str());
	this->delta_min = atof(split_line.at(1).c_str());
	this->delta_max = atof(split_line.at(2).c_str());
	this->momentum_alpha = atof(split_line.at(3).c_str());

	// The next line contains general training parameters
	this_line = GetNonemptyLine(in_file);
	split_line = SplitString(this_line);
	this->sigmoid_beta = atof(split_line.at(0).c_str());
	this->weight_init_factor = atof(split_line.at(1).c_str());
	this->error_max_limit = atof(split_line.at(2).c_str());
	this->error_mean_limit = atof(split_line.at(3).c_str());
	this->min_epoch = atoi(split_line.at(4).c_str());
	this->max_epoch = atoi(split_line.at(5).c_str());

	// The next line specifies the name of the file containing the training data
	this_line = GetNonemptyLine(in_file);
	this->data_file = this_line;

	// The next line states whether or not retraining is being done
	this_line = GetNonemptyLine(in_file);
	found = this_line.find("t");
	if (found == string::npos) {
		// Not retraining an ANN
		this->retraining = false;
	}
	else {
		this->retraining = true;
		// The next line gives the name of the file containing the existing connection weights
		this_line = GetNonemptyLine(in_file);
		this->prev_weight_fid = this_line;
	}

	// The next line specifies the file where the new weights are written
	this_line = GetNonemptyLine(in_file);
	this->output_weight_fid = this_line;

	// The next line specifies the name where training output data goes
	this_line = GetNonemptyLine(in_file);
	this->output_data_fid = this_line;

	// The next line specifies the number of epochs between which error data should
	// be written to the output data file and output weights
	// should be written tothe output weight file
	this_line = GetNonemptyLine(in_file);
	split_line = SplitString(this_line);
	this->error_freq = atoi(split_line.at(0).c_str());
	this->weight_freq = atoi(split_line.at(1).c_str());

	// Look for any remaining keywords
	// At this time (5-14-16), the only extra option is #stagger
	// (5-17-16) added the options #separated and #random
	// (11-7-16) added the option #prescaled

	// set some defaults for the options
	this->stagger = false;
	this->stagger_alt = false;
	this->do_reset = true; // default is to reset the matrices after stage 2, before stage 3
	this->separated = false;
	this->num_cycles = 1;
	this->mask_option = 0;
	this->random_pairs = false;
	this->do_dropout = false; // default is to not use node drop out
	this->random_reset = false; // default is to not do random weight reset

	this_line = GetNonemptyLine(in_file);
	while (!this_line.empty() && !in_file.eof()) { // Read until EOF
		found = this_line.find("#stagger");
		sep_found = this_line.find("#separated");
		random_found = this_line.find("#random");
		prescaled_found = this_line.find("#prescaled");
		cart_found = this_line.find("#cartesian");
		bias_found = this_line.find("#bias");
		drop_found = this_line.find("#dropout");
		random_reset_found = this_line.find("#reset");

		if (found != string::npos) {
			// Staggered training is to be done
			this->stagger = true;
			// There may be another option on this line, -no_reset, that skips the
			// matrix reset after stage 2 and before stage 3
			size_t reset_find = this_line.find("-no_reset");
			if (reset_find != string::npos) {
				this->do_reset = false;
			}
			// The next line should contain at least 3 numbers:
			// 1) number of epochs to train main network only
			// 2) number of epochs to train nested network only
			// 3) number of epochs for refinement training
			// The next parameters are optional
			// 4) number of training cycles (i.e. performing all training epochs)
			// 5) input masking option
			//      0: mask (x,y) in stage 1, mask strain in stage 2 (default)
			//      1: mask (x,y) in stage 1, no masking in stage 2
			//      2: no masking
			this_line = GetNonemptyLine(in_file);
			split_line = SplitString(this_line);
			this->stagger_epochs = new int[3];
			this->stagger_epochs[0] = atoi(split_line.at(0).c_str());
			this->stagger_epochs[1] = this->stagger_epochs[0] + atoi(split_line.at(1).c_str()); // Compute the running sum for later use
			this->stagger_epochs[2] = this->stagger_epochs[1] + atoi(split_line.at(2).c_str());

			if (split_line.size() == 4) {
				// Specifies the number of cycles
				this->num_cycles = atoi(split_line.at(3).c_str());
				this->mask_option = 1;
			}
			else if (split_line.size() == 5) {
				this->num_cycles = atoi(split_line.at(3).c_str());
				this->mask_option = atoi(split_line.at(4).c_str());
			}
			else {
				// set the defaults for number of cycles and input masking
				this->num_cycles = 1;
				this->mask_option = 1;
			}

			// Update max_epochs to the total number of staggered training epochs
			this->max_epoch = this->stagger_epochs[2];
		}
		else if (sep_found != string::npos) {
			continue; // I don't want to deal with this option right now

			//// Found the #separated keyword.
			//// There may be another option on this line, -do_alt, that specifies a slightly different
			//// version of staggered training should be performed for separated ANNs
			//size_t alt_find = this_line.find("-do_alt");
			//if (alt_find != string::npos) {
			//	this->stagger_alt = true;
			//}

			////
			//// One line follows this keyword that contains:
			////  1) Layer where the input starts for module 1
			////  2) Layer where the input starts for module 2
			////  3) At what layer module two connects to module one
			////
			//// Conditions that have to be met to use this keyword:
			//// 1) an ANN with 2 modules
			//// 2) input has to start a layer 1 for at least 1 module
			//// 3) module two must have zero nodes per layer after connecting to module one
			//// 4) If the input for a module starts after layer 1, all layers up to that point
			////      must have the same number of nodes
			//// 5) the input for modules must start at or before module 2 connects
			////
			//// A more thorough explanation of this architecture is at the beginning of the function
			//// WeightClass::MaskWeightsSeparated()
			//this->separated = true;
			//this_line = GetNonemptyLine(in_file);
			//split_line = SplitString(this_line);

			//// Copy the parameters
			//ArrayXi temp_params(3, 1);
			//temp_params(0) = atoi(split_line.at(0).c_str());
			//temp_params(1) = atoi(split_line.at(1).c_str());
			//temp_params(2) = atoi(split_line.at(2).c_str());

			//// Do the error checking described above
			//// First, make sure there are two modules
			//if (this->num_modules != 2) {
			//	this->separated = false;
			//	cout << "Two modules must be defined to use a separated architecture" << endl;
			//}

			//// Next, check that the input layer starts at the first layer for one of the modules
			//if (temp_params(0) > 1 && temp_params(1) > 1) {
			//	this->separated = false;
			//	cout << "Input must start at layer 1 for one of the modules to use a separated architecture." << endl;
			//}

			//// Ensure that module two has zero nodes per layer after connecting to module 1
			//// In the input file, the connection layer is based on indexes starting at one.
			//// However, in this->nodes_per_layer, indexing starts at zero.
			//for (int cc = temp_params(2) - 1; cc <= this->num_layers; cc++) {
			//	if (this->nodes_per_layer.array()(1, cc) != 0) {
			//		cout << "For a separated architecture, all layers in module 2 after connecting to module 1" <<
			//			" must have zero nodes." << endl;
			//		cout << "Updating the architecture to reflect this." << endl;
			//		this->nodes_per_layer.array()(1, cc) = 0;
			//	}
			//}

			//// Check that all layers in each module up to where the input starts have the same number of nodes
			//for (int mm = 0; mm < 2; mm++) {
			//	int input_start = temp_params(mm);
			//	for (int ll = 0; ll < input_start - 1; ll++) { //-1 to account for indexing starting at zero
			//		if (this->nodes_per_layer.array()(mm, ll) != this->nodes_per_layer.array()(mm, ll + 1)) {
			//			cout << "Each layer must have the same number of nodes up to where the input starts." << endl;
			//			cout << "Changing the nodes per layer to account for this." << endl;
			//			this->nodes_per_layer.array()(mm, ll + 1) = this->nodes_per_layer.array()(mm, ll);
			//		}
			//	}
			//}

			//// Finally, make sure the start of the input for both modules comes at or before module 2 connects to module 1
			//if (temp_params(0) > temp_params(2) || temp_params(1) > temp_params(2)) {
			//	this->separated = false;
			//	cout << "Module 2 can only connect to module 1 after both have received inputs." << endl;
			//}
			//this->separated_params = temp_params;

		}
		else if (random_found != string::npos) {
			// A subset of random training pairs should be used
			this->random_pairs = true;

			// The next line specifies two things
			// 1) The number of random training pairs to use
			// 2) The number of epochs before
			this_line = GetNonemptyLine(in_file);
			split_line = SplitString(this_line);

			// Make sure both parameters are present
			if (split_line.size() != 2) {
				this->random_pairs = false;
				cout << "Two parameters required to use random training pairs." << endl;
				cout << "Disabling this option." << endl;
			}
			else {
				this->num_random_pairs = atoi(split_line.at(0).c_str());
				this->random_epoch = atoi(split_line.at(1).c_str());
				// Initialize the array of random indices
				this->random_inds = new int[this->num_random_pairs];
				this->random_init = false;
			}

		}
		else if (prescaled_found != string::npos) {
			// The training data has already been scaled 
			this->prescaled = true;

		}
		else if (cart_found != string::npos) {
			// The last nested module is part of a Cartesian ANN
			this->cartesian_module = true;
		}
		else if (bias_found != string::npos) {
			this->use_biases = true;
		}
		else if (drop_found != string::npos) {
			// Node drop-out should be used

			// The next line specifies two things
			// 1) The number number of epochs before choosing new nodes to drop out
			// 2) The dropout probability
			this_line = GetNonemptyLine(in_file);
			split_line = SplitString(this_line);

			this->do_dropout = true;
			this->dropout_epochs = atoi(split_line.at(0).c_str());
			this->dropout_prob = atof(split_line.at(1).c_str());

		}
		else if (random_reset_found != string::npos) {
			// Random weight reset should be used

			// The next line specifies two things
			// 1) The number number of epochs before choosing new weights to reset
			// 2) The reset probability
			this_line = GetNonemptyLine(in_file);
			split_line = SplitString(this_line);

			this->random_reset = true;
			this->random_reset_epochs = atoi(split_line.at(0).c_str());
			this->random_reset_prob = atof(split_line.at(1).c_str());
		}
		else {
			// There's nothing else to look for, so break out of the loop
			break;
		}

		this_line = GetNonemptyLine(in_file);
	}

	in_file.close();


	return;
}

void AnnClass::ReadDataFile() {
	// Function to read in the training data from data_file
	string this_line;
	vector<string> split_line;
	ifstream in_file;

	in_file.open(this->data_file.c_str(), ios::in); // open the file in read-only mode

													// The first line in the file specifies the number of training pairs
	this_line = GetNonemptyLine(in_file);
	int num_pairs = atoi(this_line.c_str());
	this->num_training_pairs = num_pairs;
	// The next two lines are the input scaling factors and the output scaling
	// factors, respectively
	this_line = GetNonemptyLine(in_file); // input scaling factors
	split_line = SplitString(this_line);
	for (int s = 0; s < this->num_input_nodes; s++) {
		this->i_scale[s] = atof(split_line.at(s).c_str());
	}
	this_line = GetNonemptyLine(in_file); // output scaling factors
	split_line = SplitString(this_line);
	for (int s = 0; s < this->num_output_nodes; s++) {
		this->o_scale[s] = atof(split_line.at(s).c_str());
	}

	// Create the matrices to hold the input and output data
	this->i_data = new double[num_pairs*this->num_input_nodes];
	this->o_data = new double[num_pairs*this->num_output_nodes];

	// The training data switches between input and output data at each line,
	// starting with input data
	int start_i_index = 0; // initialize the starting index for input data
	int start_o_index = 0; // initialize the starting index for output data
	for (int ii = 0; ii < num_pairs; ii++) {
		this_line = GetNonemptyLine(in_file); // An input vector...
		split_line = SplitString(this_line);
		// Copy the data do the i_data matrix, scaling by i_scale
		start_i_index = this->num_input_nodes*ii;
		for (int ss = 0; ss < this->num_input_nodes; ss++) {
			this->i_data[start_i_index + ss]= atof(split_line.at(ss).c_str());// / this->i_scale.array()(ss);
			if (!this->prescaled) {
				// Training data has not been prescaled. Divide by scaling factor
				this->i_data[start_i_index + ss] /= this->i_scale[ss];
			}
		}

		this_line = GetNonemptyLine(in_file); // ...and the corresponding output vector
		split_line = SplitString(this_line);
		// Copy the data to the o_data matrix, scaling by o_scale
		start_o_index = this->num_output_nodes*ii;
		for (int ss = 0; ss < this->num_output_nodes; ss++) {
			this->o_data[start_o_index + ss] = atof(split_line.at(ss).c_str());// / this->o_scale.array()(ss);
			if (!this->prescaled) {
				// Training data has not been prescaled. Divide by scaling factor
				this->o_data[start_o_index + ss] /= this->o_scale[ss];
			}
		}
	}


	in_file.close();

}

void AnnClass::PrintDataHeader() {
	// Function to print the header to the output data file. 
	// The FORTRAN version of the RPROP algorithm split the output between a .log and .dat file. 
	// I'm going to combine the two in order to limit the number of files created.

	//ofstream out_file;
	//out_file.open(this->output_data_fid.c_str(),ios::out);

	// Open the output data file for writing
	this->out_data.open(this->output_data_fid.c_str(), ios::out);

	this->out_data << "--------------------------------------------------------------------------------" << endl << endl;
	this->out_data << "\tNested Adaptive Neural Networks (GPU)" << endl;
	this->out_data << "\tiRPROP+ training algorithm" << endl;
	this->out_data << "\tVersion 1.0 (12-6-17) " << endl;
	this->out_data << endl << "\tWritten by Cameron Hoerig (cameronhoerig@gmail.com)" << endl;
	this->out_data << "--------------------------------------------------------------------------------" << endl << endl;

	// Print the ANN architecture 

	// compute the minimum and maximum of each component for the input data
	double *min_input = new double[this->num_input_nodes];
	double *max_input = new double[this->num_input_nodes];
	double *min_output = new double[this->num_output_nodes];
	double *max_output = new double[this->num_output_nodes];

	// Initialize the min/max values to the first input/output vectors
	for (int nn = 0; nn < this->num_input_nodes; nn++) {
		min_input[nn] = this->i_data[nn];
		max_input[nn] = this->i_data[nn];
	}
	for (int nn = 0; nn < this->num_output_nodes; nn++) {
		min_output[nn] = this->o_data[nn];
		max_output[nn] = this->o_data[nn];
	}

	int start_i_index = 0;
	int start_o_index = 0;

	for (int nn = 1; nn < this->num_training_pairs; nn++) {
		start_i_index = this->num_input_nodes*nn;
		start_o_index = this->num_output_nodes*nn;
		
		for (int ii = 0; ii < this->num_input_nodes; ii++) {
			if (this->i_data[start_i_index + ii] < min_input[ii]) {
				min_input[ii] = this->i_data[start_i_index + ii];
			}
			else if (this->i_data[start_i_index + ii] > max_input[ii]) {
				max_input[ii] = this->i_data[start_i_index + ii];
			}
		}

		for (int oo = 0; oo < this->num_output_nodes; oo++) {
			if (this->o_data[start_o_index + oo] < min_output[oo]) {
				min_output[oo] = this->o_data[start_o_index + oo];
			}
			else if (this->o_data[start_o_index + oo] > max_output[oo]) {
				max_output[oo] = this->o_data[start_o_index + oo];
			}
		}

	}

	bool over_one = false; // set to true if an input/output component is greater than 1.0

						   // Print the scaled input data range
	this->out_data << "Scaled input range:" << endl;
	this->out_data << "\t";
	for (int ii = 0; ii < this->num_input_nodes; ii++) {
		this->out_data << showpos << fixed << setw(12) << setprecision(7) << min_input[ii];
		if (min_input[ii] < -1.0 || min_input[ii] > 1.0) { // test if a component is outside the +/- 1.0 range
			over_one = true;
		}
	}
	this->out_data << endl;

	this->out_data << "\t";
	for (int ii = 0; ii < this->num_input_nodes; ii++) {
		this->out_data << showpos << fixed << setw(12) << setprecision(7) << max_input[ii];
		if (max_input[ii] < -1.0 || max_input[ii] > 1.0) { // test if a component is outside the +/- 1.0 range
			over_one = true;
		}
	}
	this->out_data << endl << endl;

	// Let the user know if a component is outside the +/- 1 range
	if (over_one) {
		this->out_data << "A component of the input data is outside the +/- 1 range. Adjust the scale factors." << endl << endl;
		over_one = false; // reset to test the output data
	}

	// Print the scaled output data range
	this->out_data << "Scaled output range:" << endl;
	this->out_data << "\t";
	for (int ii = 0; ii < this->num_output_nodes; ii++) {
		this->out_data << showpos << fixed << setw(12) << setprecision(7) << min_output[ii];
		if (min_output[ii] < -1.0 || min_output[ii] > 1.0) { // test if a component is outside the +/- 1.0 range
			over_one = true;
		}
	}
	this->out_data << endl;

	this->out_data << "\t";
	for (int ii = 0; ii < this->num_output_nodes; ii++) {
		this->out_data << showpos << fixed << setw(12) << setprecision(7) << max_output[ii];
		if (max_output[ii] < -1.0 || max_output[ii] > 1.0) { // test if a component is outside the +/- 1.0 range
			over_one = true;
		}
	}
	this->out_data << endl << endl;

	// Let the user know if a component is outside the +/- 1 range
	if (over_one) {
		this->out_data << "A component of the output data is outside the +/- 1 range. Adjust the scale factors." << endl << endl;
		over_one = false; // reset to test the output data
	}

	this->out_data.unsetf(ios::showpos);
	this->PrintArchitecture();

	// Print the column headers for the error data
	this->out_data << setw(1) << " " << setw(6) << "Epoch" << setw(15) << "Max Error" << setw(10) << "ID Max" << setw(15) << "Mean Error" << setw(20) << "Below Avg Limit" << endl;


	//out_file.close();

}

void AnnClass::PrintWeights(string w_fid, int last_epoch) {

	// Open the file for writing
	ofstream out_file; 
	out_file.open(w_fid.c_str(), ios::out);

	if (this->use_biases) {
		out_file << "* weight of epoch " << last_epoch << ", using biases" << endl;
	}
	else {
		out_file << "* weight of epoch " << last_epoch << endl;
	}


	// The next line is the sigmoid beta
	out_file << "\t" << scientific << setw(15) << setprecision(7) << this->sigmoid_beta << endl;

	// The next line is the number of layers (excluding input layer) and number of modules
	out_file << "\t" << this->num_layers << "\t" << this->num_modules << endl;

	// The next num_modules lines are the nodes per layer, including the input layer
	for (int mm = 0; mm < this->num_modules; mm++) {
		for (int ll = 0; ll <= this->num_layers; ll++) {
			out_file << "\t" << fixed << setprecision(0) << this->nodes_per_layer[ij_to_idx(mm,ll,(this->num_layers + 1))];
		}
		out_file << endl;
	}

	// Now have to print all of the weights
	// Before each set, must print "layer:   x"

	// Copy the current state of the output stream
	int weight_index = 0;
	for (int ll = 0; ll < this->num_layers; ll++) {
		out_file << "layer:\t" << ll + 1 << endl;

		// Grab the number of nodes that reside in the current and next layers
		int nodes_this_layer = this->sum_nodes_per_layer[ll];
		int nodes_next_layer = this->sum_nodes_per_layer[ll + 1];
	
		for (int rr = 0; rr < nodes_next_layer; rr++) {
			for (int cc = 0; cc < nodes_this_layer; cc++) {
				out_file << "  " << setprecision(7) << setw(15) << scientific << this->weights[weight_index];
				weight_index++;
			}
			out_file << endl;
		}
	}

	// 5-30-17 Update
	// Print the scale vectors
	// First, print the input scaling vector
	out_file << endl;
	out_file << "Input scaling vector:" << endl;
	for (int rr = 0; rr < this->num_input_nodes; rr++) {
		out_file << "\t" << setprecision(7) << setw(15) << scientific << this->i_scale[rr];
	}

	// Now print the output scaling vector
	out_file << endl;
	out_file << "Output scaling vector:" << endl;
	for (int rr = 0; rr < this->num_output_nodes; rr++) {
		out_file << "\t" << setprecision(7) << setw(15) << scientific << this->o_scale[rr];
	}

	// If biases are present, skip a couple of lines and then print the biases
	if (this->use_biases) {
		// Before each set, must print "biases for layer:   x"

		// Copy the current state of the output stream
		out_file << endl << endl;
		int bias_index = this->sum_nodes_per_layer[0]; // skip the input layer
		for (int ll = 1; ll < this->num_layers; ll++) { // skip the input and output layers
			out_file << "biases for layer:\t" << ll << endl;

			int rows = this->sum_nodes_per_layer[ll];

			for (int rr = 0; rr < rows; rr++) {

				out_file << "  " << setprecision(7) << setw(15) << scientific << this->biases[bias_index];
				bias_index++;
			}
			out_file << endl;
		}
	}

	out_file.close();
	return;
}

void AnnClass::ReadPreviousWeights() {
	// Function to read in the previous weights from a file
	string this_line;
	vector<string> split_line;
	size_t found;
	size_t bias_found; // look for "using biases" in first line
	bool read_biases = false;
	ifstream in_file;
	in_file.open(this->prev_weight_fid.c_str(), ios::in); // open file in read only

	if (!in_file.good()) {
		// error opening the file. Don't try to read in the previous weights
		cout << "Error opening " << this->prev_weight_fid << endl;
		this->out_data << "Training a new ANN instead." << endl;
		this->retraining = false;
		in_file.close();
		return;
	}

	// The first line should have "* weight of...". Make sure it's present.
	// If not, the file is invalid
	this_line = GetNonemptyLine(in_file);
	found = this_line.find("* weight");
	if (found == string::npos) {
		// The *job identifier not found. Close the file and throw an error
		this->out_data << "* weight identifier not found in " << this->prev_weight_fid << endl;
		in_file.close();
		this->retraining = false; // Since the previous weight file is invalid, cannot retrain
		return;
	}

	// Look for "using biases" in this first line. 
	bias_found = this_line.find("using biases");
	if (bias_found != string::npos) {
		// The previous weights file says biases are present. Make sure the nann.inp file also specifies
		// that biases should be used
		if (!this->use_biases) {
			// The main input file says biases are not to be used. Disable retraining because the previous weights
			// will be dependent on the biases
			this->out_data << "Previous ANN used biases, but this option was not specified " << endl;
			this->out_data << "for the current ANN. Retraining will NOT be performed." << endl;
			in_file.close();
			this->use_biases = false;
			this->use_biases = false;
			this->retraining = false;
			return;
		}
		else {
			read_biases = true;
		}
	}
	else if (bias_found == string::npos && this->use_biases) {
		// This is the case where the input file says to use biases but the previous weights file does not 
		// use biases. In this case, keep the use_biases option but do not attempt to read them from the previous file
		read_biases = false;
		this->out_data << "Biases are to be used, but the previous ANN did not utilize biases." << endl;
		this->out_data << "The biases will be initialized to zero." << endl;
	}
	else {
		read_biases = true;
	}

	// The next line is the sigmoid beta. Ignore it.
	this_line = GetNonemptyLine(in_file);

	// The next line specifies the number of ANN layers
	this_line = GetNonemptyLine(in_file);
	split_line = SplitString(this_line);
	int prev_num_layers = atoi(split_line.at(0).c_str());
	int prev_num_modules = atoi(split_line.at(1).c_str());

	// Make sure the number of layers for the previous and current ANN match
	if (prev_num_layers != this->num_layers) {
		this->out_data << "Number of layers for previous and current ANN do not match." << endl;
		this->out_data << "Cannot do retraining." << endl;
		this->retraining = false;
		in_file.close();
		exit(1); // exit as failure
	}

	// The next prev_num_modules lines contain the number of nodes per layer per module
	int *prev_nodes_per_layer = new int[prev_num_modules*(prev_num_layers + 1)]; // +1 to include input layer
	int *prev_sum_nodes = new int[prev_num_layers + 1];

	for (int mm = 0; mm < prev_num_modules; mm++) {
		this_line = GetNonemptyLine(in_file);
		split_line = SplitString(this_line);

		for (int ll = 0; ll <= num_layers; ll++) {
			prev_nodes_per_layer[ij_to_idx(mm,ll,(prev_num_layers+1))] = atoi(split_line.at(ll).c_str());
			if (mm == 0) {
				prev_sum_nodes[ll] = 0; // initialize to zero
			}
		}
	}

	// Compute the total number of nodes per hidden layer over all modules for the previous ANN
	for (int mm = 0; mm < prev_num_modules; mm++) {
		for (int ll = 0; ll < prev_num_layers + 1; ll++) { // +1 to include input layer
			prev_sum_nodes[ll] += prev_nodes_per_layer[ij_to_idx(mm, ll, prev_num_modules)];
		}
	}

	// Now we have to check that the total number of nodes per layer for the previous ANN is not any
	// larger than the total nodes per layer for the new ANN
	bool more_prev_nodes = false;
	bool fewer_prev_nodes = false;
	for (int ll = 0; ll < prev_num_layers + 1; ll++) {
		int node_diff = prev_sum_nodes[ll] - this->sum_nodes_per_layer[ll];
		if (node_diff > 0) {
			// There are more nodes in the previous ANN
			more_prev_nodes = true;
		}
		else if (node_diff < 0) {
			// There are fewer nodes in the previous ANN
			fewer_prev_nodes = true;
		}
	}

	if (more_prev_nodes) {
		this->out_data << "The previous ANN has more nodes per hidden layer." << endl;
		this->out_data << "Retraining cannot be done." << endl;
		this->retraining = false;
		in_file.close();
		return;
	}
	else if (fewer_prev_nodes) {
		this->out_data << "Warning: mismatch in the number of nodes between" << endl;
		this->out_data << "\t\tprevious and current ANN." << endl;
	}

	// If the number of modules don't match, give a warning, but proceed with copying the weights
	if (prev_num_modules != this->num_modules) {
		this->out_data << "Warning: mismatch in the number of ANN modules between" << endl;
		this->out_data << "\t\tprevious and current ANN." << endl;
	}

	// We're going to assume that only one set of weights is defined in the previous weights file.
	// The weights are split up into prev_num_layers groups. Before each group, there will be a line
	// "layer: x". These lines show up predictably, so ignore them.
	// In each set of weights, the columns correspond to nodes in layer L-1, whereas the rows correspond to
	// nodes in layer L.
	// Also, in the Matlab code, everything was read in and then copied to the current weights matrices.
	// Here, because error checking was done above, we can copy directly into the current weights matrices.
	int weight_index = 0; 
	for (int ll = 0; ll < prev_num_layers; ll++) {
		// Read in and discard the "layer: x" line
		this_line = GetNonemptyLine(in_file);
		// The next prev_sum_nodes(ll+1) lines are the weights for the current layer
		for (int nn = 0; nn < prev_sum_nodes[ll+1]; nn++) {
			this_line = GetNonemptyLine(in_file);
			split_line = SplitString(this_line);
			// There are prev_sum_nodes(ll) components in the split_line vector
			// Each one is a number of type double that are the weights
			for (int kk = 0; kk < prev_sum_nodes[ll]; kk++) {
				this->weights[weight_index] = atof(split_line.at(kk).c_str());
				weight_index++;
			}
		}
	}

	// If using biases, look for them to be defined at the end. 
	if (this->use_biases && read_biases) {
		// Skip the four lines that correspond to the scaling vectors
		for (int kk = 0; kk < 4; kk++) {
			this_line = GetNonemptyLine(in_file);
		}

		// The next num_layers-1 pairs of lines are the biases.
		// -1 to skip the output layer
		int bias_index = 0;
		for (int ll = 1; ll < this->num_layers; ll++) {
			// The first line in the pair is of the form "biases for layer: X"\
						// Skip it
			this_line = GetNonemptyLine(in_file);
			this_line = GetNonemptyLine(in_file);
			split_line = SplitString(this_line); // Vector of biases
												 // Now fill out the bias vector for this layer
			for (int nn = 0; nn < prev_sum_nodes[ll]; nn++) {
				this->biases[bias_index] = atof(split_line.at(nn).c_str());
				bias_index++;
			}
		}
	}
	//

	in_file.close();

	this->out_data << "\tRetraining" << endl;

	// Free memory
	delete[] prev_sum_nodes;
	delete[] prev_nodes_per_layer;

	return;
}

void AnnClass::PrintArchitecture() {

	int layers = this->num_layers; // +1 to include input layer
	int modules = this->num_modules;

	this->out_data << endl;
	this->out_data << "ANN Architecture:" << endl;
	this->out_data << "\tNumber of layers: " << layers << endl;
	this->out_data << "\tNumber of modules: " << modules << endl;

	for (int mm = 0; mm < modules; mm++) {
		for (int ll = 0; ll <= layers; ll++) {
			if (ll == 0) { this->out_data << "\t\t"; }
			int temp_ind = ij_to_idx(mm, ll, layers);
			this->out_data << this->nodes_per_layer[ij_to_idx(mm,ll,(layers+1))];
			if (ll < this->num_layers) { this->out_data << "-"; }
		}
		this->out_data << endl;
	}
	this->out_data << endl;

	return;
}