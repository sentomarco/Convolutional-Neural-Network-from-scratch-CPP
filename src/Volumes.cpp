#include "Volumes.h"

volume::volume(){

}

volume::volume(int H, int W){

    _mtx.resize(H *W);
    _length=H*W;
    _dim = 2;
    _shape.resize(_dim);
    _shape[0]=H;
    _shape[1]=W;
}

volume::volume(int H, int W, int Depth){

    _mtx.resize(H *W *Depth, 0);
    _length=H*W*Depth;
    _dim = 3;
    _shape.resize(_dim);
    _shape[0]=H;
    _shape[1]=W;
    _shape[2]=Depth;
}

volume::volume(int Layers, int H, int W, int Depth){

    _mtx.resize(Layers *H *W *Depth);
    _length=Layers*H*W*Depth;
    _dim = 4;
    _shape.resize(_dim);
    _shape[0]=Layers;
    _shape[1]=H;
    _shape[2]=W;
    _shape[3]=Depth;
}

volume::volume(int* shapes, int dimensions){

        _length=1;
        _dim=dimensions;
        _shape.assign(shapes,shapes+dimensions);

        for (int i=0; i<dimensions; i++) _length*=shapes[i];

        _mtx.resize(_length);
}



void volume::init(const int* shapes, int dimensions){

    if(_mtx.size()!=0)  cerr<<("Error: volume already allocated.")<<endl;

    else{
        
        _length=1;
        _dim=dimensions;
        _shape.assign(shapes,shapes+dimensions);

        for (int i=0; i<dimensions; i++) _length*=shapes[i];

        _mtx.resize(_length);
    }

}

//The contenent is lost
void volume::rebuild(int* shapes, int dimensions){

    _length=1;
    _dim=dimensions;
    _shape.assign(shapes,shapes+dimensions);

    for (int i=0; i<dimensions; i++) _length*=shapes[i];

    _mtx.assign(_length,0);


}


//The access at the index (l,h,w,d) = mtx[l][h][w][d] is in the form of (4D case):  mtx[ l + h *Layers + w *Layers*Heigth + d *Layers*Heigth*Width]
void volume::assign(double val, int* index, int dimensions){

    if(dimensions!=_dim) cerr<<("Error: dimensions must match the dimensions of the volume.")<<endl;

    else{
        int element=0, offset=1;

        for (int i=0; i<dimensions; i++){
            offset=1;
            for(int sh=0; sh<i;sh++) offset*=_shape[sh];
            
            element+= index[i]*offset;
        }

        _mtx[element] = val;
    }

}


double volume::get_value(int* index, int dimensions){

    double rt=-1;

    if(dimensions!=_dim) cerr<<("Error: dimensions must match the dimensions of the volume.")<<endl;

    else{
        int element=0, offset=1;

        for (int i=0; i<dimensions; i++){
            offset=1;
            for(int sh=0; sh<i;sh++) offset*=_shape[sh];
            
            element+= index[i]*offset;
        }

        rt=_mtx[element];
    }

    return rt;
}

void volume::sum(double val, int* index, int dimensions){

    if(dimensions!=_dim) cerr<<("Error: dimensions must match the dimensions of the volume.")<<endl;

    else{
        int element=0, offset=1;

        for (int i=0; i<dimensions; i++){
            offset=1;
            for(int sh=0; sh<i;sh++) offset*=_shape[sh];
            
            element+= index[i]*offset;
        }

        _mtx[element] += val;
    }
}


int volume::get_shape(int dim_n){
    return _shape[dim_n];
}

int volume::get_length(){
    return _length;
}

vector<double>& volume::get_vector(){
    return _mtx;
}


volume& volume::operator=(const volume &start_vol) 
{   
    // 1.  Deallocate any memory that _mtx is using internally
    // 2.  Allocate some memory to hold the contents of start_vol
    // 3.  Copy the values from start_vol into this instance
    // 4.  Return *this
    if (this == &start_vol) return *this;
    
    else{

        // release resource in *this
        _mtx.resize(0), _shape.resize(0);
        _dim = 0, _length=0; 
        
        // allocate resource in *this
        this->init(&(start_vol._shape[0]), start_vol._dim);   
 
        //copy the content (_mtx)
        this->_mtx=start_vol._mtx;
        
    }

    return *this;  // Return a reference to myself.
}



//Return the single value in the inserted position of the vector volume.
//Keep in mind that is accessed as a vector
double& volume::operator[](int index)
{
    if (index >= _length) {
        cerr << "Array index out of bound, returned last element."<<endl;
        return _mtx.back();
    }
    
    return _mtx[index];
}
