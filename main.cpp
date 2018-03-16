// main.cpp : Defines the entry point for the console application.
//
#include <iostream>
#include <fstream>
#include <string>
#include <time.h>
#include "ann.h"
#include "Misc.h"


int main()
{
	// Seed the random number generator now so that if a random matrix is generated,
	// the same numbers aren't spit out every time this program runs
	srand(time(NULL));

	bool add_node = false; // flags whether any nodes should be added to the hidden layers

	int epoch = 0;
	int break_point;
	int num_training_pairs; // Maybe that defined in the data file OR a random subset of training pairs
	int pair_ind;

	string buffer;

	AnnClass ann_obj;
	ann_obj.ReadInputFile();
	//ann_obj.ReadDataFile();
	ann_obj.ReadDataBinary();
	ann_obj.PrintDataHeader();
	ann_obj.InitializeMatrices();

	// ---------------------------------------------
	// To add: support for random pairs 
	// ---------------------------------------------

	if (ann_obj.retraining) {
		// Retraining an ANN. Read in the previous weights.
		ann_obj.ReadPreviousWeights();

	}

	// ---------------------------------------------
	// To add: weight masking for nested networks
	// ---------------------------------------------

	// ---------------------------------------------
	// To add: finish support for node biases
	// ---------------------------------------------

	ann_obj.PrintWeights("stage_0.nnw", 0);

	if (ann_obj.prescaled) {
		ann_obj.out_data << "\tUsing pre-scaled training data" << endl;
	}

	if (ann_obj.random_pairs) {
		if (ann_obj.num_random_pairs > ann_obj.num_training_pairs) {
			ann_obj.out_data << "The number of specified random training pairs is greater than "
				<< "or equal to the total number of training pairs" << endl;
			ann_obj.out_data << "The random pairs option will be disabled." << endl;
			ann_obj.random_pairs = false;
			num_training_pairs = ann_obj.num_training_pairs;
		}
		else {
			// Otherwise, get the first set of random indices
			ann_obj.GetRandomIndices();
			// And set the number of training pairs to the specified number of random pairs
			num_training_pairs = ann_obj.num_random_pairs;
			ann_obj.out_data << "Using a subset of random pairs:" << endl;
			ann_obj.out_data << "\tNumber of pairs: " << ann_obj.num_random_pairs << endl;
			ann_obj.out_data << "\tEpochs before reselection: " << ann_obj.random_epoch << endl << endl;
		}
	}
	else {
		num_training_pairs = ann_obj.num_training_pairs;
		ann_obj.random_epoch = ann_obj.max_epoch + 3; // to make sure no random subsets will be selected
	}

	for (int cycle = 0; cycle < ann_obj.num_cycles; cycle++) {
		ann_obj.out_data << "Cycle " << cycle + 1 << endl;

		ann_obj.ResetMatrices();

		for (epoch = 0; epoch < ann_obj.max_epoch; epoch++) {

			if (epoch > 0 && ann_obj.random_pairs) {
				if ((epoch + 1) % ann_obj.random_epoch == 0) { // +1 to match indexing
					ann_obj.GetRandomIndices();
				}
			}

			ann_obj.ZeroMatrices();
			ann_obj.ResetErrors();

			ann_obj.RunEpoch();

			/*
			for (int pair_count = 0; pair_count < num_training_pairs; pair_count++) {

				if (ann_obj.random_pairs) {
					pair_ind = ann_obj.random_inds[pair_count];
				}
				else {
					pair_ind = pair_count;
				}
				// Perform back-propagation for all training pairs
				ann_obj.PropagateSignalForward(pair_ind); 
				ann_obj.PropagateErrorBackward(pair_ind);
				ann_obj.CalculateErrors(pair_ind);
			}
			//*/

			ann_obj.epochs_between_add++;

			if ((epoch + 1) % ann_obj.error_freq == 0) {
				// Based on the error data writing frequency, the current errors should be printed to 
				// the output data file
				ann_obj.PrintErrors(epoch);
			}

			// Add the mean error to the error vector
			ann_obj.error_vec.push_back(ann_obj.mean_error);

			if (epoch == 0) {
				// After the first epoch, set the previous error to the current error
				ann_obj.prev_error = ann_obj.mean_error;
			}

			// If nodes have been added and nodes need to be frozen, do the necessary masking
			// At a later time, I may change this so that the masking function just zeros out the 
			// weight increments instead of the gradients
			if (ann_obj.freeze_weights) {
				ann_obj.MaskWeightsFrozen();
				ann_obj.freeze_count++;
			}
		
			// Compute the weight increments
			ann_obj.ComputeWeightUpdates();
			ann_obj.ComputeWeightIncrements();		

			// Update the weights
			ann_obj.UpdateWeights();

			// If the minimum number of epochs have been reached and the max/mean errors fall below the specified limits, exit training
			// BUT ONLY IF WEIGHTS ARE NOT CURRENTLY FROZEN!
			// Don't want to exit if training newly added nodes, or if fewer than num_epoch_freeze epochs have occurred since the
			// last dynamic node addition
			if (epoch + 1 >= ann_obj.min_epoch && !ann_obj.freeze_weights) {
				if (ann_obj.last_nodes_added > 0 && ann_obj.epochs_between_add < ann_obj.num_epoch_freeze) {
					// Nodes have been added. Need to wait until at least num_epoch_freeze epochs have passed before exiting
				}
				else if (ann_obj.max_error <= ann_obj.error_max_limit && ann_obj.mean_error <= ann_obj.error_mean_limit) {
					ann_obj.out_data << "Minimum number of epochs and error tolerances have been met." << endl;
					ann_obj.out_data << "Exiting training." << endl;
					break;
				}
			}

			// Check if any weights should be dropped
			if (ann_obj.do_dropout) {
				// Only drop weights if at least 2*dropout_epochs remain
				if ((epoch + 1) % ann_obj.dropout_epochs == 0) {
					int remaining_epochs = 2*(ann_obj.max_epoch - epoch + 1);
					if (remaining_epochs > ann_obj.dropout_epochs) {
						ann_obj.out_data << "\t\tWeight Reset" << endl;
						ann_obj.MaskWeightsDropout();
					}
				}
				
			}

			if ((epoch + 1) % ann_obj.epoch_check_slope == 0) {
				// Time to check the error slope to determine if any nodes should be added
				add_node = ann_obj.CheckSlope();
				// Only add a node if the node addition limit has not already been reached AND 
				// not still training dynamically added nodes (i.e. num_epoch_freeze epochs have passed since
				// last node addition)

				// With the staggered training, this is going to get a bit messy. Go ahead and reuse the old code but 
				// wrap it in a conditional statement
				if (add_node) {
					// Also, only add a node if there are at least 2*num_epoch_freeze epochs left for training
					if ((ann_obj.max_epoch - epoch) < 2 * ann_obj.num_epoch_freeze) {
						add_node = false;
					}

					// Furthermore, only add a node if 2*num_epoch_freeze epochs have occurred since
					// the last dynamic node addition
					if (ann_obj.epochs_between_add < 2 * ann_obj.num_epoch_freeze) {
						add_node = false;
					}
					if (add_node && !ann_obj.reached_node_limit && !ann_obj.freeze_weights) {
						ann_obj.AddNodes();
					}
				}
			}


			// Unfreeze the weights after node addition if num_epoch_freeze epochs have passed
			if (ann_obj.freeze_weights) {
				if (ann_obj.freeze_count > ann_obj.num_epoch_freeze) {
					ann_obj.freeze_weights = false;
					ann_obj.freeze_count = 0;
					ann_obj.out_data << "Unfreezing weights." << endl;
				}
			}


		}


	}

	// Close the log file
	ann_obj.out_data.close();

	// Print the weights
	ann_obj.PrintWeights(ann_obj.output_weight_fid,epoch);

	return 0;
}

