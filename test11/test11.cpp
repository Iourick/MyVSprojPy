// test11.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>

#include "npy.hpp"
#include <vector>
#include <string>

int main() {
	const std::vector<long unsigned> shape{ 2, 3 };
	const bool fortran_order{ false };
	const std::string path{ "out.npy" };

	const std::vector<double> data1{ 1, 2, 3, 4, 5, 6 };
	npy::SaveArrayAsNumpy(path, fortran_order, shape.size(), shape.data(), data1);
}