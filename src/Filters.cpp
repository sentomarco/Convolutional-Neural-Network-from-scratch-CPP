
#include "Filters.h"

using namespace std;



void ReLu(volume& input_volume){
    for(int a=0; a<input_volume.get_length(); a++)
        if( input_volume[a]<0) input_volume[a] *= ALPHA;
}



void deLeReLu(volume& input_volume){

    //Same of doing:
    for(int a=0; a<input_volume.get_length(); a++)
        if( input_volume[a]<0) input_volume[a] = ALPHA;
}


//Store a copy of the vectors since they can change outside
Convolutional::Convolutional(int image_dim[3], int kernels[4], int padding, int stride, double bias, double eta){
    srand(time(NULL));
    
    if (image_dim[0]!=kernels[3]) 
        cerr<<("Error: depth of the filter must match the depth of the image.")<<endl;
    
    //Warning, copy with array must be used like this (arr + #elements)
    copy(image_dim, image_dim+3, begin(_image_dim));
    copy(kernels, kernels+4, begin(_specs));
    
    _padding=padding;
    _stride=stride;
    _eta=eta;
    _iteration=0;
    _filter.init(_specs,4);

    int pad_dim[3]={_image_dim[0], _image_dim[1]  + 2*_padding, _image_dim[2] + 2*_padding};
    copy(pad_dim, pad_dim+3, begin(_image_dim));

    _cache.init( pad_dim, 3 );
    
    for (int k=0; k<kernels[0]; k++) _bias.push_back( bias );
    
    double inputs=(kernels[0]*kernels[1]*kernels[2]*kernels[3]);

    for (int i=0; i<_filter.get_length(); i++) _filter[i]= ((double) (rand() % 100))/1000 ;

}


//Enlarge the shape of the image and _image_dim is changed accordingly     
void Convolutional::_pad(volume& original_img, volume& out_pad){

    for(int D=0; D<_image_dim[0]; D++)
         for (int H=0; H<_image_dim[1] - 2*_padding; H++)
            for (int W=0; W<_image_dim[2] - 2*_padding; W++){
                

                int output[3] = {D,H+_padding,W+_padding};
                int input[3] = {D, H, W};
                out_pad.assign(original_img.get_value(input,3), output, 3);

            }

    //for(int i=0; i<3; i++) _image_dim[i]=out_pad.get_shape(i);

}



void Convolutional::_out_dimension(){
        
        int f_y = _specs[1], f_x = _specs[2];
        double y_doub, x_doub;

		y_doub = (double)(_image_dim[1] - f_y + 2*_padding)/_stride +1;
	    x_doub = (double)(_image_dim[2] - f_x + 2*_padding)/_stride +1;	

        int y_int=(int)(y_doub+0.5), x_int=(int)(x_doub+0.5);

		//if( y_doub!=y_int || x_doub!=x_int) cerr<<"\nWarning: padding and stride combination is not integer."<<endl;

        _out_dim[0]=_specs[0]; // The output depth is equal to the number of kernels of the filter
        _out_dim[1]=y_int;
        _out_dim[2]=x_int;

        //printf("Out dimensions as calculated: %d %d %d\n",_out_dim[0], _out_dim[1], _out_dim[2] );

}




void Convolutional::fwd(volume image, volume& out){

    /*Produces a volume of size D2xH2xW2 where:
			#W2=(W1−F+2P)/S+1
			#H2=(H1−F+2P)/S+1
			#D2= kernels number
    */

    int f_y= _specs[1], f_x= _specs[2], f_d= _specs[3];
    int n_kernel = _specs[0];
     
    _out_dimension();
    int depth = _out_dim[0], out_H = _out_dim[1], out_W = _out_dim[2];

    out.rebuild( _out_dim, 3 );

    if(_padding!=0) _pad(image, _cache);   
    else _cache=image;                     
    // Now image is saved and adjusted in _cache 

	int y_out=0, x_out = 0;

    for(int kernel=0; kernel < n_kernel; kernel++){

        for (int layer=0; layer<depth; layer++ ){//each kernel has n (3) layers, one for each of the n (3) layers of the image, the depth.
            
            y_out = 0, x_out = 0;

            for (int y=0; y<_image_dim[1] - f_y; y+=_stride){	// image = ( depth x H x W )
                x_out=0;

                for (int x=0; x<_image_dim[2] - f_x; x+=_stride ){

                    for (int f_y_it=0; f_y_it<f_y; f_y_it++){
                        for(int f_x_it=0; f_x_it<f_x; f_x_it++){

                            int arr_out[3] = {kernel,y_out,x_out};
                            int in_cache[3] = {layer, y + f_y_it, x + f_x_it};
                            int in_filt[4] = {kernel, f_y_it, f_x_it, layer};
                            double val = _cache.get_value(in_cache,3)*_filter.get_value(in_filt,4);
                            out.sum(val, arr_out, 3);

                        }
                    }        
                    x_out++;
                }
                y_out++;
            }
        out[kernel]+=_bias[kernel];
        }

    }

    ReLu(out);	
}



void Convolutional::bp(volume d_out_vol, volume& d_input){

    deLeReLu(d_out_vol);

    // image (input or convolution result) - ( depth, out_H, out_W ) 
    // _specs = (n_kern x H x W x depth) The filters are in _filters
    
    int  n_kernel= _specs[0], f_y= _specs[1], f_x= _specs[2], f_d= _specs[3];

    //d_input = np.zeros( (  self.padded_dim[0], self.padded_dim[1], self.padded_dim[2]) )
    d_input.rebuild(_image_dim,3);

    volume d_filters(_specs[0], _specs[1], _specs[2], _specs[3]);   // The list of lists of error terms (lowercase deltas) 
    vector<double> d_bias;

    int y_out=0, x_out = 0;

    for(int kernel=0; kernel < n_kernel; kernel++){
    
        y_out=0, x_out = 0;

        for (int y=0; y<_image_dim[1] - f_y - 2* _padding; y+=_stride){		// image = ( depth x H x W )
            
            for (int x=0; x<_image_dim[2] - f_x - 2* _padding; x+=_stride ){

                // loss gradient of the input passed in the convolution operation

                for (int layer=0; layer<f_d; layer++ ){
                    
                    for (int f_y_it=0; f_y_it<f_y; f_y_it++){
                        for(int f_x_it=0; f_x_it<f_x; f_x_it++){
                            
                            int filt[4]     = {kernel, f_y_it, f_x_it, layer};
                            int out_in[3]   = {layer, y + f_y_it, x + f_x_it};
                            int in_cache[3] = {layer, y + f_y_it, x + f_x_it};
                            int in_vol[3]   = {kernel, y_out, x_out};
                            
                            double val_d_filt = _cache.get_value(in_cache,3)*d_out_vol.get_value(in_vol,3);
                            double val_d_in = d_out_vol.get_value(in_vol,3)*_filter.get_value(filt,4);

                            d_filters.sum(val_d_filt, filt, 4);     //d_filters[kernel, f_y_it, f_x_it, layer] +=  _cache[layer, y + f_y_it, x + f_x_it ] * d_out_vol[kernel, y_out, x_out ]
                            d_input.sum(val_d_in, out_in, 3);      //d_input[layer, y + f_y_it, x + f_x_it ] += d_out_vol[kernel, y_out, x_out ] * self.filters[kernel, f_y_it, f_x_it, layer]

                        }
                    }
                }
                x_out+=1;
            }
            x_out=0;
            y_out+=1;
        }
    }

    // loss gradient of the bias
    double k_bias=0;

    for(int kernel=0; kernel < n_kernel; kernel++){
        
        k_bias=0;
        for (int y=0; y<d_out_vol.get_shape(1); y++){
            for (int x=0; x<d_out_vol.get_shape(2); x++ ){
                int i[3]={kernel, y, x};
                k_bias+=d_out_vol.get_value(i, 3);
            }
        }
        d_bias.push_back( k_bias );            
    }

    _gd(d_filters, d_bias);

}



void Convolutional::_gd(volume& d_filter, vector<double>& d_bias){

    //NB d_filter and _filter same dimension
    
    //eta=eta*(exp(-x/10000))
    _eta = _eta * exp(((double)-_iteration)/10000);
    
    int  n_kernel= _specs[0], f_y= _specs[1], f_x= _specs[2], f_d= _specs[3];

    for(int kernel=0; kernel < n_kernel; kernel++){
        
        for (int y=0; y<f_y; y++){
            for (int x=0; x<f_x; x++ ){

                for (int layer=0; layer<f_d; layer++){

                        int index[4]={kernel, y, x, layer};

                        double delta = -_eta*d_filter.get_value(index, 4);

                        _filter.sum( delta, index, 4); 
                }
            }
        }
    }
    
    for(int i=0; i<(int)_bias.size(); i++) _bias[i] -= _eta * d_bias[i];

    _iteration ++;

}



void Convolutional::new_epoch(double eta){

    _eta=eta;
    _iteration=0;

}