#ifndef FILTERS_H
#define FILTERS_H

#define ALPHA 0.001

#include <iostream>
#include <algorithm>
#include <vector>
#include <random>
#include <ctime>
#include <iterator>
#include <stdlib.h>
#include <math.h>

#include "Volumes.h"

using namespace std;


void ReLu(volume& input_volume);
void deLeReLu(volume& input_volume);

class Convolutional {

    int _image_dim[3] ={1,16,16};     // image specification
    int _specs[4]     ={2,3,3,1}; 		// filter specifications
    int _out_dim[3]   ={2,13,13};     // convoluted output dimensions 

    int _padding=1;
    int _stride=2;
    int _iteration = 0;		  // To update the gradient descent 
    double _eta = 0.1;
    
    vector<double> _bias;   // The list of bias, same for each kernel so one value for each of it (kernels[0])		
    
    volume _cache;
    volume _filter;

    void _pad(volume& original_img, volume& out_img);

    void _gd(volume& d_filter, vector<double>& d_bias);

    void _out_dimension();


  public: 

    //Store a copy of the vectors since they can cange outside
    Convolutional(int image_dim[3], int kernels[4], int padding=1, int stride=1, double bias=0.1, double eta=0.01); 

    void new_epoch(double eta);

    void fwd(volume image, volume& out);

    void bp(volume d_out_vol, volume& d_input);

};



class Pooling {

};


#endif