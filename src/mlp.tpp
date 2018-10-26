/* Multi-layer perceptron implemetation
 *
 * Only forward pass.
 */

#include <cmath>

template<typename T>
T f_sigmoid(T x) {
  return 1.0 / (1.0 + exp(-x));  
}

template<typename T>
T f_sigmoidprim(T b) {
  return b * (1 - b);
}

template<typename T>
T f_tanh(T x) {
  T ex = exp(x);
  T negex = exp(-x);
  return (ex - negex) / (ex + negex);
}

template<typename T>
T f_tanhprim(T g) {
  return (1.0 - pow(g, 2));
}

template<class T>
MLP<T>::MLP(unsigned num_in, EActivation activation)
  : num_in_(num_in),
    layers_(0)
{

  switch(activation) {
  case ACTIVATION_TANH:
    f_activation_ = &(f_tanh<T>);
    break;
  case ACTIVATION_SIGMOID:
  default:
    f_activation_ = &(f_sigmoid<T>);
  }

  a_.push_back(Matrix<T>(1, num_in+1));

  Matrix<T> &a = a_[0];
  for (unsigned i=0; i < num_in+1; ++i) {
    a(num_in) = -1;
  }
}

template<class T>
void MLP<T>::add_layer(unsigned layer_num_out)
{
  unsigned layer_num_in = num_in_;

  if (layers_ > 0) {
    layer_num_in = W_[layers_-1].N_;
  }
  
  W_.push_back(Matrix<T>(layer_num_in + 1, layer_num_out));
  a_.push_back(Matrix<T>(1, layer_num_out + 1));

  layers_ += 1;

  a_[layers_](0, layer_num_out) = -1;
}

template<typename T>
void MLP<T>::set_parameters(std::vector<T> &x)
{
  int i = 0;
  for (unsigned l=0; l < layers_; ++l) {
    Matrix<T> &w = W_[l];
    for (unsigned j=0; j<w.M_; ++j) {
      for (unsigned k=0; k<w.N_; ++k) {
	w(j,k) = x[i++];
      }
    }
  }
}

template<class T>
void MLP<T>::activation_(Matrix<T> &a)
{
  for (unsigned i=0; i<a.N_; ++i) {
    a(i) = f_activation_(a(i));
  }
}

template<class T>
void MLP<T>::recall_()
{
  unsigned layer = 0;
  Matrix<T> &w1 = W_[layer];

  a_[layer].matmul(w1, a_[layer+1]);

  while (layer < layers_-1) {
    activation_(a_[layer+1]);
    layer += 1;
    Matrix<T> &w2 = W_[layer];
    a_[layer].matmul(w2, a_[layer+1]);
  }

  activation_(a_[layer+1]);
}

template<typename T>
void MLP<T>::recall(std::vector<T> &x, std::vector<T> &y)
{
  Matrix<T> &a = a_[0];
  
  for (unsigned i=0; i< num_in_; ++i) {
    a(i) = x[i];
  }

  recall_();

  Matrix<T> &output = a_[layers_];
  unsigned num_outputs = y.size() < output.N_ ? y.size() : output.N_;
  
  for (unsigned i=0; i < num_outputs; ++i) {
    y[i] = output(i);
  }
}

template<typename T>
void MLP<T>::print_parameters()
{
  for (unsigned i=0; i<layers_; ++i) {
    W_[i].print();
  }
}
