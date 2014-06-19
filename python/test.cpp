#include "pydmpbbo.h"
#include <eigen3/Eigen/Core>

using namespace std;
using namespace Eigen;

int main(void) {
    Dmp dmp(2, 3);
    set<string> selected;
    selected.insert("weights");
    dmp.getDmp().setSelectedParameters(selected);
    VectorXd values, values2;
    dmp.getDmp().getParameterVectorSelected(values);
    values(0) = 1;
    values(1) = 2;
    dmp.getDmp().setParameterVectorSelected(values);
    dmp.getDmp().getParameterVectorSelected(values2);
    //dmp.getDmp().getSelectableParameters(selected);
    for (set<string>::iterator it=selected.begin(); it!=selected.end(); ++it)
        cout << "__" << *it << endl;
    //cout << dmp.getDmp().getParameterVectorSelected() <<endl;
    cout << values2 << endl;
}
