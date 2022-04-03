#include "functionapproximators/eigen_conversion_mine.hpp"

using namespace std;
using namespace Eigen;

namespace DmpBbo {
  
void from_json(const nlohmann::json& j, VectorXi& vector)
{
    //using Scalar = typename MatrixXi::Scalar;
    using Index = typename MatrixXi::Index;

    if (j.is_array())
    {
        vector.resize(j.size());
        for (std::size_t ii = 0; ii < j.size(); ++ii)
        {
            const auto& value = j.at(ii);
            vector(static_cast<Index>(ii)) = value.get<int>();
        }
    }
    else
    {
        vector.resize(1);
        vector(0) = j.get<int>();
    }
}
  
void from_json(const nlohmann::json& j, VectorXd& vector)
{
    using Scalar = typename MatrixXd::Scalar;
    using Index = typename MatrixXd::Index;

    if (j.is_array())
    {
        vector.resize(j.size());
        for (std::size_t ii = 0; ii < j.size(); ++ii)
        {
            const auto& value = j.at(ii);
            vector(static_cast<Index>(ii)) = value.get<Scalar>();
        }
    }
    else
    {
        vector.resize(1);
        vector(0) = j.get<Scalar>();
    }
}
  
void from_json(const nlohmann::json& j, MatrixXd& matrix)
{
    using Scalar = typename MatrixXd::Scalar;
    using Index = typename MatrixXd::Index;

    bool resized = false;
    for (std::size_t row = 0; row < j.size(); ++row)
    {
        const auto& jrow = j.at(row);
        if (jrow.is_array())
        {
            if (!resized) 
            {
              matrix.resize(j.size(),jrow.size());
              resized = true;
            }
            for (std::size_t col = 0; col < jrow.size(); ++col)
            {
                const auto& value = jrow.at(col);
                matrix(static_cast<Index>(row), static_cast<Index>(col)) = value.get<Scalar>();
            }
        }
        else
        {
            if (!resized) 
            {
              matrix.resize(j.size(),1);
              resized = true;
            }
            matrix(static_cast<Index>(row), 0) = jrow.get<Scalar>();
        }
    }
}

}
