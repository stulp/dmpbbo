/**
 * @file eigen_json.tpp
 * @brief Source file for input/output of Eigen matrices to json.
 *
 * The Below code was adapted from
 * https://gitlab.com/Simox/simox/-/blob/master/SimoxUtility/json/eigen_conversion.h
 *
 * I added functionality for the appropriate resizing of matrices.
 */

template <typename Derived>
void from_json(const nlohmann::json& j, Eigen::MatrixBase<Derived>& matrix)
{
  using Scalar = typename Eigen::MatrixBase<Derived>::Scalar;
  using Index = typename Eigen::MatrixBase<Derived>::Index;

  bool resized = false;
  for (std::size_t row = 0; row < j.size(); ++row) {
    const auto& jrow = j.at(row);
    if (jrow.is_array()) {
      if (!resized) {
        matrix.derived().resize(j.size(), jrow.size());
        resized = true;
      }

      for (std::size_t col = 0; col < jrow.size(); ++col) {
        const auto& value = jrow.at(col);
        matrix(static_cast<Index>(row), static_cast<Index>(col)) =
            value.get<Scalar>();
      }
    } else {
      if (!resized) {
        // https://eigen.tuxfamily.org/dox/TopicFunctionTakingEigenTypes.html#title4
        matrix.derived().resize(j.size(), 1);
        resized = true;
      }
      matrix(static_cast<Index>(row), 0) = jrow.get<Scalar>();
    }
  }
}

/**
 * Read int from json. can be "4.0", "[4.0]", "[[4.0]]"
 */
inline double from_json_to_double(const nlohmann::json& j)
{
  if (j.contains("value")) {
    double dv = j.at("value");
    return dv;
  }

  // values
  if (j.contains("values")) {
    Eigen::MatrixXd matrix;
    from_json(j.at("values"), matrix);

    if (matrix.rows() != 1) {
      std::cerr << __FILE__ << ":" << __LINE__ << ":";
      std::cerr << "'values' should have 1 row, but has " << matrix.rows()
                << std::endl;
    }
    if (matrix.cols() != 1) {
      std::cerr << __FILE__ << ":" << __LINE__ << ":";
      std::cerr << "'values' should have 1 cols, but has " << matrix.rows()
                << std::endl;
    }

    return matrix(0, 0);
  }

  // double
  double d = j;
  return d;
}
