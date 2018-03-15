#include <string>
#include <fstream>
#include <vector>
#include "Misc.h"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <functional>
#include <cctype>
#include <locale>

using namespace std;

string GetNonemptyLine(ifstream &in_file){
// Function to return the next non-blank line from a file
    string this_string;
    getline(in_file,this_string);
    while(this_string.empty() && !in_file.eof()){
        getline(in_file,this_string);
    }
    // Remove left and right white space
    //trim(this_string);
	StringTrim(this_string);
    //to_lower(this_string);
	transform(this_string.begin(), this_string.end(), this_string.begin(), ::tolower);
    return this_string;
}

vector<string> SplitString(string in_string){
// Function to split a string into tokens based on white space
    vector<string> split_vector;
	istringstream iss(in_string);
	copy(istream_iterator<string>(iss),
		istream_iterator<string>(),
		back_inserter(split_vector));

    //split(split_vector,in_string,is_any_of(" \t"),token_compress_on);

    return split_vector;
}

void LeftTrim(string &in_string) {
	in_string.erase(in_string.begin(), find_if(in_string.begin(), in_string.end(),
		not1(std::ptr_fun<int, int>(std::isspace))));
}

void RightTrim(string &in_string) {
	in_string.erase(find_if(in_string.rbegin(), in_string.rend(),
		not1(std::ptr_fun<int, int>(std::isspace))).base(), in_string.end());
}

void StringTrim(string &in_string) {
	LeftTrim(in_string);
	RightTrim(in_string);
}