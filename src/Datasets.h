#ifndef DATASETS_H
#define DATASETS_H

#define MNIST_TRAIN_LEN 60000
#define MNIST_TEST_LEN 10000
#define MNIST_TRAIN_SHAPES MNIST_TRAIN_LEN, 1, 28, 28
#define MNIST_TEST_SHAPES MNIST_TEST_LEN, 1, 28 , 28
#define IMAGE_DATA 784      // 28 x 28

#include <iostream>
#include <vector>
#include <fstream>

#include "Volumes.h"

using namespace std;


int ReverseInt (int i);
void _normalize_set(volume &set, int len, int n_rows, int n_cols);

class MNIST{
        
        void _get_set(string path, int NumberOfImages, volume& set, int DataOfAnImage);
        void _get_label(string path, int NumberOfImages, vector<int>& label);

        void _init_mnist(volume& Train_DS, vector<int>& Train_L,
                        volume& Test_DS,  vector<int>& Test_L,
                        volume& Valid_DS,  vector<int>& Valid_L);

    public:
        
        void get_mnist(volume& Train_DS, vector<int>& Train_L,
                        volume& Test_DS,  vector<int>& Test_L,
                        volume& Valid_DS,  vector<int>& Valid_L);

};

//Other datasets to be implemented

#endif