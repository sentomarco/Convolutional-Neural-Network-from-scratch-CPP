#ifndef VOLUMES_H
#define VOLUMES_H

#include <iostream>
#include <vector>

using namespace std;

class volume{

        vector<double> _mtx;
        vector<int> _shape;
        int _dim = 0, _length = 0;

    public:     

        volume();   //Void volume object

        volume(int H, int W);
        volume(int H, int W, int Depth);
        volume(int Layers, int H, int W, int Depth);
        volume(int* shapes, int dimensions);

        void init(const int* shapes, int dimensions); //Usefull to postpone istantiation
        void rebuild(int* shapes, int dimensions);

        int get_shape(int dim_n);
        int get_length();
        double get_value(int* index, int dimensions);
        vector<double>& get_vector();
        
        void assign(double val, int* index, int dimensions);
        void sum(double val, int* index, int dimensions);
        //void adjust(double val);

        volume& operator=(const volume &start_vol);

        double& operator[](int index);

};

#endif