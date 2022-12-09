#ifndef MLP_H
#define MLP_H

#include <algorithm>
#include <vector>
#include <iostream>
#include <random>
#include <ctime>
#include <numeric>
#include <cmath>
#include <time.h>

#define BETA1 0.9
#define BETA2 0.999
#define EPS 1e-7


using namespace std;

class Perceptron {

	public: 
		//Perceptron() = default;
		Perceptron(int inputs, double bias=1.0);
		vector<double> weights, m, v;
		double bias;
        double run(vector<double> x);
		void set_weights(vector<double> w_init);
		double sigmoid(double x);
};

class MultiLayerPerceptron {

	public: 
		MultiLayerPerceptron(vector<int> layers, double bias=1.0, bool adam=true, double eta = 0.5);
		void set_weights(vector<vector<vector<double> > > w_init);
		void print_weights();
		vector<double> run(vector<double> x);
		vector<double> bp(vector<double> error);
		void gd();
		double Adam(double &m, double &v, double derivative );

		
		vector<int> layers;
		double bias;
		double eta;

		int back_iter;
		bool b_adam;	

		vector<vector<Perceptron> > network;
		vector<vector<double> > values;
		vector<vector<double> > d;
		vector<vector<double> > loss_gradient;
};

#endif