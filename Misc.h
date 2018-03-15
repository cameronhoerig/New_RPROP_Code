#pragma once

#include <string>
#include <fstream>
#include <vector>
#include <string>

using namespace std;

string GetNonemptyLine(ifstream&); // Function to return the next non-blank line from a file
vector<string> SplitString(string); // Function to split a string into tokens based on white space
void LeftTrim(string&); // Function to trim leading white space from string
void RightTrim(string&); // Function to trim trailing white space from string
void StringTrim(string&); // Function to trim leading and trailing white space off a string

