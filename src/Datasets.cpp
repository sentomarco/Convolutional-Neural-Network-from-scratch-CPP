#include "Datasets.h"


int ReverseInt (int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1=i&255;
    ch2=(i>>8)&255;
    ch3=(i>>16)&255;
    ch4=(i>>24)&255;
    return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
}


//Should require also the depth value
void _normalize_set(volume &set, int len, int n_rows, int n_cols){

    for(int img=0; img<len; img++){

        //First: loop over an image to obtain the lower and higher value

        double max=0, min=255, val;

        for(int r=0;r<n_rows;++r)
            {
                for(int c=0;c<n_cols;++c)
                {   
                    int index[4]  = {img,0,r,c};
                    val = set.get_value(index, 4);
                    
                    if(val>max) max=val;
                    if(val<min) min=val;
                }
            }

        //Second: loop again over the image to normalize every value

        for(int r=0;r<n_rows;++r)
            {
                for(int c=0;c<n_cols;++c)
                {   
                    int index[4]  = {img,0,r,c};
                    val = set.get_value(index, 4);
                    
                    val = (val - min) / (max - min);

                    set.assign(val,index,4);

                }
            }

    }
}



void MNIST::_get_label(string path, int NumberOfImages, vector<int>& label){

    ifstream file(path,ios::binary);

    if (file.is_open())
    {
        int magic_number=0;
        int number_of_images=0;
    
        file.read((char*)&magic_number,sizeof(magic_number));
        magic_number= ReverseInt(magic_number);
        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= ReverseInt(number_of_images);

        for(int i=0;i<number_of_images;++i)
        {
                    unsigned char temp=0;
                    file.read((char*)&temp,sizeof(temp));
                    label[i]= (int)temp;

        }
    }
}


void MNIST::_get_set(string path, int NumberOfImages, volume& set, int DataOfAnImage){

    ifstream file(path,ios::binary);

    if (file.is_open())
    {
        int magic_number=0;
        int number_of_images=0;
        int n_rows=0;
        int n_cols=0;
        file.read((char*)&magic_number,sizeof(magic_number));
        magic_number= ReverseInt(magic_number);
        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= ReverseInt(number_of_images);
        file.read((char*)&n_rows,sizeof(n_rows));
        n_rows= ReverseInt(n_rows);
        file.read((char*)&n_cols,sizeof(n_cols));
        n_cols= ReverseInt(n_cols);

        for(int i=0;i<number_of_images;++i)
        {   
            for(int r=0;r<n_rows;++r)
            {
                for(int c=0;c<n_cols;++c)
                {   
                    unsigned char temp=0;
                    file.read((char*)&temp,sizeof(temp));
                    int index[4]  = {i,0,r,c};
                    set.assign((double)temp,index, 4);
                }
            }
        }

        //PREVIEW
        /*
        cout<<"First image:"<<endl;
        for(int r=0;r<28;++r)
        {
            for(int c=0;c<28;++c)
            {   
                int index[4]  = {0,r,c,0};
                cout<<set.get_value(index,4)<<" ";
            }
            cout<<endl;
        }
        */

    }
}



void MNIST::_init_mnist(volume& Train_DS, vector<int>& Train_L,
                        volume& Test_DS,  vector<int>& Test_L,
                        volume& Valid_DS,  vector<int>& Valid_L){

    
    int train_shapes[4]={MNIST_TRAIN_SHAPES};
    int test_shapes[4] ={MNIST_TEST_SHAPES};

    Train_DS.init(train_shapes,4);
    Test_DS.init(test_shapes,4);
    //Valid_DS.init();

    Train_L.assign(MNIST_TRAIN_LEN,0);
    Test_L.assign(MNIST_TEST_LEN,0);
    //Valid_L->assign();

}


void MNIST::get_mnist(volume& Train_DS, vector<int>& Train_L,
                        volume& Test_DS,  vector<int>& Test_L,
                        volume& Valid_DS,  vector<int>& Valid_L) {
    
    _init_mnist(Train_DS, Train_L, Test_DS, Test_L, Valid_DS, Valid_L);

    cout<<"\no Getting MNIST datasets\n"<<endl;

    _get_set("MNIST_data/train-images.idx3-ubyte", MNIST_TRAIN_LEN, Train_DS, IMAGE_DATA);
    _get_label("MNIST_data/train-labels.idx1-ubyte", MNIST_TRAIN_LEN, Train_L);

    _get_set("MNIST_data/t10k-images.idx3-ubyte", MNIST_TEST_LEN, Test_DS, IMAGE_DATA);
    _get_label("MNIST_data/t10k-labels.idx1-ubyte", MNIST_TEST_LEN, Test_L);

    _normalize_set(Train_DS, MNIST_TRAIN_LEN, 28, 28);
    _normalize_set(Test_DS, MNIST_TEST_LEN, 28, 28);
    // I still have to brake the training set in the validation sub-set.

     
}