/** Multi-layer perceptron
 *
 * Only forward pass.
 */

#include <iostream>
#include <vector>

template<typename T>
inline T max(T p1, T p2) {
  return (p1 < p2) ? p2 : p1;
}

template<typename T>
inline T min(T p1, T p2) {
  return (p1 > p2) ? p2 : p1;
}

enum EActivation {
  ACTIVATION_SIGMOID,
  ACTIVATION_TANH,
};

template<typename T>
class Matrix
{
private:
  T *m_ = NULL;

  void Build() {
    if (m_ != NULL) {
      delete[] m_;
    }
    if (M_ * N_ > 0)
      m_ = new T[M_ * N_];
    set_all_(0);
  }

  void Init(unsigned _M, unsigned _N)
  {
    M_ = _M;   // Rows
    N_ = _N;   // Columns
    Build();
  };

  void set_all_(T v) {
    for (unsigned i=0; i<M_; ++i)
      for (unsigned j=0; j<N_; ++j)
	(*this)(i,j) = v;
  }
  
public:
  unsigned M_ = 0;
  unsigned N_ = 0;

  Matrix()
    : m_(NULL),
      M_(0), N_(0)
  {
  }

  Matrix(unsigned _M, unsigned _N)
    : m_(NULL),
      M_(_M), N_(_N)
  {
    Build();
  };

  Matrix(const Matrix &that)
    : m_(NULL),
      M_(that.M_), N_(that.N_)
  {
    Build();

    for (unsigned i=0; i<M_; ++i) {
      for (unsigned j=0; j<N_; ++j)
	(*this)(i,j) = that(i,j);
    }
  }
  
  Matrix& operator= (const Matrix &that)
  {
    if (M_ != that.M_ && N_ != that.N_) {
      Init(that.M_, that.N_);
    }
    for (unsigned i=0; i<M_; ++i) {
      for (unsigned j=0; j<N_; ++j)
	(*this)(i,j) = that(i,j);
    }
    return *this;
  }
    
  ~Matrix() {
    if (m_ != NULL) {
      delete[] m_;
      m_ = NULL;
    }
  }

  void zeros() {
    set_all_(0);
  }

  void ones() {
    set_all_(1);
  }

  inline void matmul (Matrix<T> &B, Matrix<T> &C) {
    if (B.M_ != N_ || C.M_ < M_ || C.N_ < B.N_)
      throw std::runtime_error("matmul: dimensions don't match");

    C.zeros();

    for (unsigned i=0; i<M_; ++i) {
      for (unsigned j=0; j<B.N_; ++j) {
	for (unsigned k=0; k<N_; k++) {
	  C(i,j) += (*this)(i,k) * B(k, j);
	}
      }
    }
  }

  // Element-wise operations
  inline Matrix& operator*= (Matrix &B) {
    if (B.M_ != M_ || B.N_ != N_)
      throw std::runtime_error("Dimensions don't match");
    for (unsigned i=0; i<M_; ++i)
      for (unsigned j=0; j<N_; ++j)
	m_[i*N_ + j] *= B(i,j);
    return *this;
  }

  inline Matrix& operator+= (Matrix &B) {
    if (B.M_ != M_ || B.N_ != N_)
      throw std::runtime_error("Dimensions don't match");
    for (unsigned i=0; i<M_; ++i)
      for (unsigned j=0; j<N_; ++j)
	m_[i*N_ + j] += B(i,j);
    return *this;
  }

  inline Matrix& operator-= (Matrix &B) {
    if (B.M_ != M_ || B.N_ != N_)
      throw std::runtime_error("Dimensions don't match");
    for (unsigned i=0; i<M_; ++i)
      for (unsigned j=0; j<N_; ++j)
	m_[i*N_ + j] -= B(i,j);
    return *this;
  }

  inline Matrix& operator/= (T b) {
    for (unsigned i=0; i<M_; ++i)
      for (unsigned j=0; j<N_; ++j)
	m_[i*N_ + j] = m_[i*N_ + j] / b;
    return *this;
  }

  inline Matrix& operator*= (T b) {
    for (unsigned i=0; i<M_; ++i)
      for (unsigned j=0; j<N_; ++j)
	m_[i*N_ + j] = m_[i*N_ + j] * b;
    return *this;
  }

  inline Matrix operator+ (Matrix &B) const {
    if (B.M_ != M_ || B.N_ != N_)
      throw std::runtime_error("Dimensions don't match");
    Matrix C(M_, N_);
    for (unsigned i=0; i<M_; ++i)
      for (unsigned j=0; j<N_; ++j)
	C(i,j) = m_[i*N_ + j] + B(i,j);
    return C;
  }

  inline Matrix operator- (Matrix &B) const {
    if (B.M_ != M_ || B.N_ != N_)
      throw std::runtime_error("Dimensions don't match");
    Matrix C(M_, N_);
    for (unsigned i=0; i<M_; ++i)
      for (unsigned j=0; j<N_; ++j)
	C(i,j) = m_[i*N_ + j] - B(i,j);
    return C;
  }

  inline Matrix operator/ (T d) const {
    Matrix C(M_, N_);
    for (unsigned i=0; i<M_; ++i)
      for (unsigned j=0; j<N_; ++j)
	C(i,j) = m_[i*N_ + j] / d;
    return C;
  }

  inline T& operator() (unsigned i)           { return m_[i]; }
  inline T  operator() (unsigned i) const     { return m_[i]; }
  inline T& operator() (unsigned i, unsigned j)       { return m_[i*N_ + j]; }
  inline T  operator() (unsigned i, unsigned j) const { return m_[i*N_ + j]; }

  void print() {
    unsigned i, j;
    for (i=0; i<M_; ++i) {
      for (j=0; j<N_; ++j) {
	std::cerr << std::fixed <<  m_[i*N_ + j] << " ";
      }
      std::cerr << "\n";
    }
  }
};

template<class T>
class MLP {
public:

  MLP(unsigned num_in, EActivation activation);
  
  void add_layer(unsigned num_out);
  void set_parameters(std::vector<T> &x);
  void recall(std::vector<T> &x, std::vector<T> &y);

  void print_parameters();
  
private:
  unsigned num_in_;
  unsigned layers_;

  std::vector<Matrix<T>> W_;  // Weights for layers
  std::vector<Matrix<T>> a_;  // Activation outputs
  
  T (*f_activation_)(T x);
  T (*f_activationprim_)(T x);

  void activation_(Matrix<T> &a);
  void recall_();
  
};

#include "mlp.tpp"
