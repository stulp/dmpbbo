#include <iostream>

#include "realtimechecks.hpp" // Must be included before including Eigen
#include <eigen3/Eigen/Core>

using namespace std;
using namespace Eigen;

void init(MatrixXd& a, MatrixXd& b, MatrixXd& c, int size)
{
  a = MatrixXd::Ones(size,size);
  b = MatrixXd::Ones(size,size);
  c = 0.00001*MatrixXd::Ones(size,size);
}

void update(MatrixXd& a, MatrixXd& b, MatrixXd& c, MatrixXd& tmp)
{
  ENTERING_REAL_TIME_CRITICAL_CODE  
  b += c;
  a.noalias() += b*c;
  // http://eigen.tuxfamily.org/dox/TopicLazyEvaluation.html
  // http://eigen.tuxfamily.org/dox/TopicWritingEfficientProductExpression.html  
  //a.noalias() = b*a; // Compiles, runs, but yields the wrong result!
  tmp.noalias() = b*c;
  c += tmp;
  EXITING_REAL_TIME_CRITICAL_CODE
}

int main(int n_args, char** args)
{
  int size = 2;
  if (n_args>1)
    size = atoi(args[1]);
  
  MatrixXd a,b,c;
  init(a,b,c,size);
  
  MatrixXd prealloc = MatrixXd(size,size);
  cout << a << endl; 
  
  ENTERING_REAL_TIME_CRITICAL_CODE
  for (float t=0.0; t<0.1; t+=0.01)
    update(a,b,c,prealloc);
  EXITING_REAL_TIME_CRITICAL_CODE 
}

