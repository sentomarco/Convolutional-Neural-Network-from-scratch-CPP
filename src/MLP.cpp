#include "MLP.h"

double frand(){
	return (double)(rand() % 100)/1000 ;
    //return (2.0*(double)rand() / RAND_MAX) - 1.0;
}


// Return a new Perceptron object with the specified number of inputs (+1 for the bias).
Perceptron::Perceptron(int inputs, double bias){
    
	this->bias = bias;
	weights.resize(inputs+1); 
    m.resize(inputs+1,0);
    v.resize(inputs+1,0);
    generate(weights.begin(),weights.end(),frand);
}


// Run the perceptron. x is a vector with the input values.
double Perceptron::run(vector<double> x){
	x.push_back(bias);
	double sum = inner_product(x.begin(),x.end(),weights.begin(),(double)0.0);
	return sigmoid(sum);
}


// Set the weights. w_init is a vector with the weights.
void Perceptron::set_weights(vector<double> w_init){
	weights = w_init;
}


// Evaluate the sigmoid function for the floating point input x.
double Perceptron::sigmoid(double x){
	return 1.0/(1.0 + exp(-x));
}


// Return a new MultiLayerPerceptron object with the specified parameters.
MultiLayerPerceptron::MultiLayerPerceptron(vector<int> layers, double bias, bool adam, double eta) {

    srand(time(NULL));

    this->layers = layers;
    this->bias = bias;
    this->eta = eta;
    back_iter=0;
    b_adam=adam;
    
    for (int i = 0; i < (int) layers.size(); i++){
        values.push_back(vector<double>(layers[i],0.0));
        d.push_back(vector<double>(layers[i],0.0));
        loss_gradient.push_back(vector<double>(layers[i],0.0)); 
        network.push_back(vector<Perceptron>());
        if (i > 0)   //network[0] is the input layer,so it has no neurons
            for (int j = 0; j < layers[i]; j++)
                network[i].push_back(Perceptron(layers[i-1], bias));
    }
}



// Set the weights. w_init is a vector of vectors of vectors with the weights for all but the input layer.
void MultiLayerPerceptron::set_weights(vector<vector<vector<double> > > w_init) {
    for (int i = 0; i< (int) w_init.size(); i++)
        for (int j = 0; j < (int) w_init[i].size(); j++)
            network[i+1][j].set_weights(w_init[i][j]);
}


void MultiLayerPerceptron::print_weights() {
    cout << endl;
    for (int i = 1; i < (int) network.size(); i++){
        for (int j = 0; j < layers[i]; j++) {
            cout << "Layer " << i+1 << " Neuron " << j << ": ";
            for (auto &it: network[i][j].weights)
                cout << it <<"   ";
            cout << endl;
        }
    }
    cout << endl;
}


// Feed a sample x into the MultiLayer Perceptron.
vector<double> MultiLayerPerceptron::run(vector<double> x) {
    
    values[0] = x;
    for (int i = 1; i < (int) network.size(); i++)
        for (int j = 0; j < layers[i]; j++)
            values[i][j] = network[i][j].run(values[i-1]);
    return values.back();
}




//NB m and v are updated inside the function
double MultiLayerPerceptron::Adam(double &m, double &v, double derivative ){

    int t=back_iter;
    double dx=derivative;

    m=BETA1*m+(1-BETA1)*dx;
    double mt=m/(1-(double)pow(BETA1,t));
    v=BETA2*v + (1-BETA2)*((double)pow(dx,2));
    double vt = v / (1-(double)pow(BETA2,t));
    double delta = eta * mt / (sqrt(vt) + EPS);

    return delta;
}

void MultiLayerPerceptron::gd(){


    // STEPS 5 & 6: Calculate the deltas and update the weights
    for (int i = 1; i < (int) network.size(); i++)
        for (int j = 0; j < layers[i]; j++)
            for (int k = 0; k < layers[i-1]+1; k++){
                double delta;
                if (k==layers[i-1]){
                    delta = eta * d[i][j];
                }else{

                double dw = d[i][j] * values[i-1][k];		// dw = dscore * X

                if (!b_adam) delta = eta * dw;			// learning rate * dw
                else delta = Adam( network[i][j].m[k], network[i][j].v[k], dw);

                }
                network[i][j].weights[k] += delta;
            }
}



vector<double> MultiLayerPerceptron::bp(vector<double> error){
    
    back_iter++;    //To update Adam

    vector<double> outputs= values.back();

    // STEP 3: Calculate the output error terms
    for (int i = 0; i < (int) outputs.size(); i++)
        d.back()[i] = outputs[i] * (1 - outputs[i]) * (error[i]);

    // STEP 4: Calculate the error term of each unit on each layer    
    for (int i = ((int) network.size())-2; i > 0; i--)

        for (int h = 0; h < (int) network[i].size(); h++){

            double fwd_error = 0.0;

            for (int k = 0; k < layers[i+1]; k++)
                fwd_error += network[i+1][k].weights[h] * d[i+1][k];
            
            loss_gradient[i][h]=fwd_error;
            d[i][h] = values[i][h] * (1-values[i][h]) * fwd_error;
        }
    
    gd();

    return loss_gradient[1]; //Result from the first layer (not the input one)
}   

