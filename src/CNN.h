#ifndef CNN_H
#define CNN_H

#include <vector>
#include <ctime>
#include <math.h>
#include <string.h> 

#include "Volumes.h"
#include "Filters.h"
#include "Datasets.h"
#include "MLP.h"

using namespace std;

class CNN{

        vector<char> _layers;
        vector<double> _result;
        vector<Convolutional> _convs;
        vector<Pooling> _pools;
        vector<MultiLayerPerceptron> _dense;

        int _conv_index=0, _pool_index=0, _tot_layers=0, _num_classes=0;
        int _dense_input_shape[3]={0,0,0};
        int _image_shape[3]={0,0,0};

        volume Train_DS,Test_DS, Valid_DS;

        vector<int> Train_L, Test_L, Valid_L; //labels

        vector<double> train_acc, valid_acc, test_acc;
		vector<double> train_loss, valid_loss, test_loss;

        void _forward(volume& image);
        void _backward(vector<double>& gradient);
        void _iterate(volume& dataset, vector<int>& labels, vector<double>& loss_list, vector<double>& acc_list,int preview_period ,bool b_training = true);
        void _get_image(volume& image, volume& dataset, int index);
        
    public:
        
        CNN() = default; //Not auto-generated if other constructors are present
        void add_conv(vector<int>& image_dim, vector<int>& kernels, int padding=1, int stride=1, double bias=0.1, double eta=0.01);
        void add_pooling(int image_dim[3], char mode='a', int size=2, int stride=2, int padding=0);
        void add_dense(int input, vector<int>& hidden, int num_classes=10, double bias=1.0, bool adam=true, double eta = 0.01);
        void load_dataset(string data_name );  
        void training( int epochs = 1, int preview_period = 1);
        void testing( int preview_period = 1);
        void sanity_check(int set_size=50 ,int epochs=200);
        void plot_results();
        ~CNN();

};


#endif