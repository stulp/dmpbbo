# Using Eigen for real-time applications

Having C++ code running on robots means the code should be written to meet real-time requirements. In particular, dynamic memory allocations should be avoided in real-time critical parts. 

[Eigen](http://eigen.tuxfamily.org) is a highly optimized matrix library. For many operations Eigen must (dynamically) allocate temporary memory. So using Eigen in the context of real-time code means understanding when it is allocating memory "under the hood", and avoiding those situations.

## Finding and resolving dynamic memory allocations in Eigen

### When does Eigen allocate memory?
  
Take the following code, which does nothing useful, but provides a generic template for real-time code, especially in the context of robots. First some non-real-time initialization is done, and afterwards some function is called in a real-time loop. Question: Is Eigen doing dynamic allocations in the update() function? If so, where?

```c++
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
```

The benign line
```c++
a += b*c
```
causes a dynamic memory allocation. Because Eigen expands this line, as documented [here](http://eigen.tuxfamily.org/dox/TopicWritingEfficientProductExpression.html), to:
```c++
tmp = b*c
a += tmp
```
These expression are then turned into a bunch of for loops that loop over the individual entries of the matrices. 
    
Eigen has a way to determine whether it is necessary, or more efficient to use intermediate temporary matrices. For instance, it will use a tmp, and thus dynamically allocated memory, ["when its cost model shows that the total cost of an operation is reduced if a sub-expression gets evaluated into a temporary"](http://eigen.tuxfamily.org/dox/TopicLazyEvaluation.html). So knowing exactly when Eigen will allocated memory requires you to understand the cost function (good luck!), or determine it empirically with EIGEN_RUNTIME_NO_MALLOC (see below).   
  
### How can I find where exactly Eigen allocates memory during run-time?

To check where Eigen is making dynamic allocations explicitely, you can add 
```c++
#define EIGEN_RUNTIME_NO_MALLOC
```
*before* including the Eigen headers, as documented [here](http://eigen.tuxfamily.org/index.php?title=FAQ#Where_in_my_program_are_temporary_objects_created.3F). Then the function
    Eigen::internal::set_is_malloc_allowed(boolean)
allows you to specify in which parts of the code Eigen is not allowed to allocate dynamic memory. Since we don't want this to happen in the update() function, we add set_is_malloc_allowed() the top and the bottom.

```c++
void update(MatrixXd& a, MatrixXd& b, MatrixXd& c)
{
  Eigen::internal::set_is_malloc_allowed(false)
  b += c;
  a += b*c;
  c += b*c;
  Eigen::internal::set_is_malloc_allowed(true)
}
```


If you now run
```shell
g++ ggdb realtime.cpp -o realtime
./realtime
```
you will see that the code crashes. Eigen is dynamically allocating memory in the line 
```c++
a += b*c;
```
### How can I avoid dynamic memory allocation by Eigen?
  
As mentioned above, the line
```c++
a += b*c
```
is [expanded to](http://eigen.tuxfamily.org/dox/TopicWritingEfficientProductExpression.html):
```c++
tmp = b*c
a += tmp
```

To avoid this expansion, you can use the [noalias()](http://eigen.tuxfamily.org/dox/classEigen_1_1MatrixBase.html#ae77f3c3ccfb21694555dafc92c2da340) function. 
```c++
a.noalias() += b*c;
```
    
This will avoid the dynamic memory allocation, because you tell Eigen explicitely not to use the temporary matrix. See [here](http://eigen.tuxfamily.org/dox/TopicWritingEfficientProductExpression.html) and [here](http://eigen.tuxfamily.org/dox/TopicLazyEvaluation.html) for further documentation.

### When noalias() doesn't cut it 

But be careful with noalias()! There is another line where memory is also allocated: 
```c++
c += b*c
```
You may be tempted to write 
```c++
c.noalias() += b*c
```
While this will compile fine and run without allocating memory, it will give you the wrong result! Why? Because you are overwriting elements in c that are needed later to compute other elements in c. So here, you need the temporary expansion
```c++
tmp = b*c;
c += tmp
```
but without the memory allocation it implies...

So what do we do? We need to pre-allocate a matrix of the right size before entering the real-time loop, and use that pre-allocated matrix inside the real-time critical code. There are many ways to do this. In the code, I've added a variable prealloc, which is passed to the update function (this may not always be the best solution, but it is in our particular use case on the robot). In the function, we then explicitely write what Eigen would do under the hood
```c++
prealloc.noalias() = b*c;
c += prealloc;
```
In the context of the function, that looks like this:
```c++
void update(MatrixXd& a, MatrixXd& b, MatrixXd& c, MatrixXd& prealloc)
{
  Eigen::internal::set_is_malloc_allowed(false)
  b += c;
  a.noalias() += b*c;
  prealloc.noalias() = b*c 
  c += prealloc;
  Eigen::internal::set_is_malloc_allowed(true)
}
```

### When noalias() isn't necessary

Note that Eigen was apparently not allocating memory in the line:
```c++
b += c;
```
That is because this expression can simply be expanded to 
```c++
for i, for j, b(i,j) = b(i,j) + c(i,j) 
```
without requiring extra memory. No need to change it.

## Code Obfuscation

There is one big issue with all this. Pretty, compact code starts looking ugly. For instance, here's a multi-variate Gaussian probability density function with Eigen matrices:
(TODO: using inverse() here is not the right way...)
```c++
double normalPDF(const VectorXd& mean, const MatrixXd& covar, const VectorXd& input) {
  // -0.5 * (x-mu)^T * Sigma^-1 * (x-mu)
  double inside_exp = -0.5*(mean-input).dot(covar.inverse()*(mean-input));
  // (1/sqrt( (2*PI)^k * |Sigma| )) * exp(-0.5 * (x-mu)^T * Sigma^-1 * (x-mu)) 
  return pow(pow(2*M_PI,input.size()*covar.determinant()),-0.5)*exp(inside_exp);
}
```
There are so many possible dynamic allocations in there that it's not funny. Here's a version with no allocation:
(TODO: using inverse() here is not the right way...)
```c++
double normalPDFRealTime(const VectorXd& mean, const MatrixXd& covar, const VectorXd& input, VectorXd& diff_prealloc, VectorXd& covar_times_diff_prealloc, const MatrixXd& covar_inverse, double covar_determinant) {

  Eigen::internal::set_is_malloc_allowed(false);

  // (x-mu)
  diff_prealloc = input - mean;
  // Sigma^-1 * (x-mu)
  covar_times_diff_prealloc.noalias() = covar_inverse*diff_prealloc;
  // -0.5 * (x-mu)^T * Sigma^-1 * (x-mu)
  double inside_exp = -0.5*(diff_prealloc).dot(covar_times_diff_prealloc);
  
  // 1/sqrt( (2*PI)^k * |Sigma| ) 
  double normalization = pow(pow(2*M_PI,input.size()*covar_determinant),-0.5);

  Eigen::internal::set_is_malloc_allowed(true);
  
  // (1/sqrt( (2*PI)^k * |Sigma| )) * exp(-0.5 * (x-mu)^T * Sigma^-1 * (x-mu)) 
  return normalization*exp(inside_exp);
}
```
This is a bit annoying, because we have to preallocate/precompute several matrices, and our compact code has become ugly... 

## Eigen::Ref

If you call real-time functions (with Eigen matrices as arguments) from other real-time function, you'll also have to be careful in checking that Eigen won't make copies of the arguments. In many cases, you'll need to use [Eigen::Ref](http://eigen.tuxfamily.org/dox/TopicFunctionTakingEigenTypes.html#TopicUsingRefClass) "to avoid stupid copies of the arguments." (quote from the link).

## Integration into larger code bases

To automate things and avoid having to write Eigen::internal::set_is_malloc_allowed(...) all the time, I have the piece of code below in a header file.

```c++
#ifdef REALTIME_CHECKS

// If REALTIME_CHECKS is defined, we want to check for dynamic memory allocation.
// Make Eigen check for dynamic memory allocation
#define EIGEN_RUNTIME_NO_MALLOC
// We define ENTERING_REAL_TIME_CRITICAL_CODE and EXITING_REAL_TIME_CRITICAL_CODE to start/stop
// checking dynamic memory allocation
#define ENTERING_REAL_TIME_CRITICAL_CODE Eigen::internal::set_is_malloc_allowed(false);
#define EXITING_REAL_TIME_CRITICAL_CODE Eigen::internal::set_is_malloc_allowed(true);

#else // REALTIME_CHECKS

// REALTIME_CHECKS is not defined, not need to do any checks on real-time code. Simply set
// ENTERING_REAL_TIME_CRITICAL_CODE and EXITING_REAL_TIME_CRITICAL_CODE to empty strings.
#define ENTERING_REAL_TIME_CRITICAL_CODE
#define EXITING_REAL_TIME_CRITICAL_CODE
```    
    
So that allows you to simply write 
```c++
void update(MatrixXd& a, MatrixXd& b, MatrixXd& c, MatrixXd& prealloc)
{
  ENTERING_REAL_TIME_CRITICAL_CODE
  b += c;
  a.noalias() += b*c;
  prealloc.noalias() = b*c 
  c = prealloc;
  EXITING_REAL_TIME_CRITICAL_CODE
}
```
If I want to check Eigen's dynamic memory allocations, I compile as follows:
```shell
g++ -DREALTIME_CHECKS -ggdb realtime.cpp -o realtime.cpp
```

## Bottom line

Eigen is a great library, which make C++ code for linear algebra look almost as compact as Matlab/octave. But...  when using Eigen in a real-time context, explicitly avoiding dynamic memory allocation requires great care, and can make the code real ugly and obfuscated. 

## Downloading cpp/hpp files

You can download the demo cpp/hpp files, as well as a Makefile here: https://github.com/stulp/dmpbbo/tree/master/docs/eigenrealtime


TODO: Of course, simpler solution, if possible: use fixed size matrices.

