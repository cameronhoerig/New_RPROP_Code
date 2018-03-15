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

#define ij_to_idx(i, j, num_cols) ((i*num_cols) + j) // convert (i,j) index to vector index for row-major ordering

using namespace std;

void AnnClass::ResetErrors() {
	this->mean_error = 0.0;
	this->max_error = 0.0;
	this->percent_correct = 0.0;
	this->id_max = 0;
}

void AnnClass::CalculateErrors(int pair_id) {
	// Calculates the maximum and average errors, as well as the "percent correct" and ID of the training
	// pair producing the maximum error

	// Compute the output error vector
	// Here, compute the square of the difference between the ANN output and the target output
	double error_sum = 0.0;
	//double *ann_out = new double[this->num_output_nodes];
	//double *target_out = new double[this->num_output_nodes];
	//double *target_in = new double[this->num_input_nodes];
	//int *temp_inds = new int[this->num_output_nodes];
	for (int oo = 0; oo < this->num_output_nodes; oo++) {
		error_sum += pow(this->o_data[ij_to_idx(pair_id, oo, this->num_output_nodes)] -
			this->node_phi[this->num_nodes - this->num_output_nodes + oo], 2);

		//ann_out[oo] = this->node_phi[this->num_nodes - this->num_output_nodes + oo];
		//target_out[oo] = this->o_data[ij_to_idx(pair_id, oo, this->num_output_nodes)];
		//temp_inds[oo] = ij_to_idx(pair_id, oo, this->num_output_nodes);
	}

	//for (int oo = 0; oo < this->num_output_nodes; oo++) {
	//	cout << ann_out[oo] << "\t\t" << target_out[oo] << "\t\t" << endl; // temp_inds[oo] << endl;
	//}
	//cout << endl << endl;

	//for (int ii = 0; ii < this->num_input_nodes; ii++) {
	//	cout << this->i_data[ij_to_idx(pair_id, ii, this->num_input_nodes)] << "\t";
	//}
	//cout << endl;

	// Update the mean error
	this->mean_error += error_sum / this->num_training_pairs;

	// If the current error is larger than the max error, update the max error and id_max
	if (error_sum > this->max_error) {
		this->max_error = error_sum;
		this->id_max = pair_id;
	}

	// Update the percent correct
	if (error_sum < this->error_mean_limit) {
		this->percent_correct += 100.0 / this->num_training_pairs;
	}

	return;
}

void AnnClass::PrintErrors(int epoch) {
	// Function to print the errors to the output data file

	this->out_data << fixed << setw(1) << "*" << setw(6) << epoch + 1 << setw(15) << setprecision(9) << this->max_error << setw(10) << this->id_max
		<< setw(15) << setprecision(9) << this->mean_error << setw(20) << setprecision(6) << this->percent_correct << endl;
}

bool AnnClass::CheckSlope() {
	// Function to check the error slope after every epoch_check_slope epochs and determine if nodes should be added to the ANN

	// First, compute the mean of the error vector
	double this_error = 0.0;

	for (unsigned int ii = 0; ii < this->error_vec.size(); ii++) {
		this_error += error_vec.at(ii);
	}

	this_error = this_error / static_cast<double>(this->epoch_check_slope);

	// compute the error slope
	// While it's generally okay the error will always be decreasing, this may not be the case in staggered training.
	// As such, take the magnitude during the slope check
	double slope = abs((this->prev_error - this_error)) / this_error;
	this->prev_error = this_error; // update prev_error

								   // clear the error vector
	this->error_vec.clear();

	if (slope < this->slope_limit) {
		// Error slope has fallen below the limit. A new node can be added
		return true;
	}
	else {
		// Error slope is still above the limit. A new node should not be added
		return false;
	}
}