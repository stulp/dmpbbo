#include <iostream>
#include <eigen3/Eigen/Core>

using namespace std;
using namespace Eigen;

void init(MatrixXd& a, MatrixXd& b, MatrixXd& c, int size)
{
  a = MatrixXd::Ones(size,size);
  b = MatrixXd::Ones(size,size);
  c = 0.00001*MatrixXd::Ones(size,size);
}

void update(MatrixXd& a, MatrixXd& b, MatrixXd& c)
{
  // Pretty random equations to illustrate dynamic memory allocation
  b += c;
  a += b*c;
  c += b*c;
}

int main(int n_args, char** args)
{
  int size = 3;
  if (n_args>1)
    size = atoi(args[1]);
  
  // Initialization (not real-time)
  MatrixXd a,b,c;
  init(a,b,c,size);
  
  // Real-time loop
  for (float t=0.0; t<0.1; t+=0.01)
    update(a,b,c);
  
  return 0;
}
