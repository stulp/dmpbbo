/**
 * @file   FunctionApproximator.hpp
 * @brief  FunctionApproximator class header file.
 * @author Freek Stulp
 *
 * This file is part of DmpBbo, a set of libraries and programs for the 
 * black-box optimization of dynamical movement primitives.
 * Copyright (C) 2014 Freek Stulp, ENSTA-ParisTech
 * 
 * DmpBbo is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
 * (at your option) any later version.
 * 
 * DmpBbo is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with DmpBbo.  If not, see <http://www.gnu.org/licenses/>.
 */

/** \defgroup FunctionApproximators Function Approximators
 */

#ifndef _FUNCTIONAPPROXIMATOR_H_
#define _FUNCTIONAPPROXIMATOR_H_

#include "Parameterizable.hpp"

#include <boost/serialization/nvp.hpp>

#include <string>
#include <vector>
#include <eigen3/Eigen/Core>

namespace DmpBbo {
  
// Forward declarations
class MetaParameters;
class ModelParameters;
class UnifiedModel;

/** \brief Base class for all function approximators.
 *  \ingroup FunctionApproximators
 */
class FunctionApproximator : public Parameterizable
{

public:
  
  /** Initialize a function approximator with meta- and optionally model-parameters
   *  \param[in] meta_parameters  The training algorithm meta-parameters
   *  \param[in] model_parameters The parameters of the trained model. If this parameter is not
   *                              passed, the function approximator is initialized as untrained. 
   *                              In this case, you must call FunctionApproximator::train() before
   *                              being able to call FunctionApproximator::predict().
   * Either meta_parameters XOR model-parameters can passed as NULL, but not both.
   */
  FunctionApproximator(const MetaParameters *const meta_parameters, const ModelParameters *const model_parameters=NULL);
  
  /** Initialize a function approximator with model-parameters
   *  \param[in] model_parameters The parameters of the trained model.
   */
  FunctionApproximator(const ModelParameters *const model_parameters);

  virtual ~FunctionApproximator(void);
  
  /** Return a pointer to a deep copy of the FunctionApproximator object.
   *  \return Pointer to a deep copy
   */
  virtual FunctionApproximator* clone(void) const = 0;
  
  /** Train the function approximator with corresponding input and target examples.
   *  \param[in] inputs  Input values of the training examples
   *  \param[in] targets Target values of the training examples
   */
  virtual void train(const Eigen::MatrixXd& inputs, const Eigen::MatrixXd& targets) = 0;
  
  /** Train the function approximator with corresponding input and target examples (and write results to file).
   *  \param[in] inputs  Input values of the training examples
   *  \param[in] targets Target values of the training examples
   *  \param[in] save_directory Directory to which to write results.
   * \param[in] overwrite Whether to overwrite existing files. true=do overwrite, false=don't overwrite and give a warning.
   */
  void train(const Eigen::MatrixXd& inputs, const Eigen::MatrixXd& targets, std::string save_directory, bool overwrite=false);

  /** Re-train the function approximator with corresponding input and target examples.
   *  \param[in] inputs  Input values of the training examples
   *  \param[in] targets Target values of the training examples
   *  Re-training could in principle have been enabled through FunctionApproximator::train, but we
   *  wanted to keep a clear disctinction between training (which must be done at least once before 
   *  FunctionApproximator::predict) can be called and re-training.
   */
  void reTrain(const Eigen::MatrixXd& inputs, const Eigen::MatrixXd& targets);
  
  /** Re-train the function approximator with corresponding input and target examples (and write results to file).
   *  \param[in] inputs  Input values of the training examples
   *  \param[in] targets Target values of the training examples
   *  \param[in] save_directory Directory to which to write results.
   * \param[in] overwrite Overwrite existing files in the directory above (default: false)
   *  Re-training could in principle have been enabled through FunctionApproximator::train, but we
   *  wanted to keep a clear disctinction between training (which must be done at least once before 
   *  FunctionApproximator::predict) can be called and re-training.
   */
  void reTrain(const Eigen::MatrixXd& inputs, const Eigen::MatrixXd& targets, std::string save_directory, bool overwrite=false);
  
  /** Query the function approximator to make a prediction
   *  \param[in]  inputs   Input values of the query
   *  \param[out] outputs  Predicted output values
   *
   * \remark This method should be const. But third party functions which is called in this function
   * have not always been implemented as const (Examples: LWPRObject::predict or IRFRLS::predict ).
   * Therefore, this function cannot be const.
   */
  virtual void predict(const Eigen::MatrixXd& inputs, Eigen::MatrixXd& outputs) = 0;

  /** Query the function approximator to make a prediction, and also to predict its variance
   *  \param[in]  inputs   Input values of the query (n_samples X n_dims_in)
   *  \param[out] outputs  Predicted output values (n_samples X n_dims_out)
   *  \param[out] variances Predicted variances for the output values  (n_samples X n_dims_out). Note that if the output has a dimensionality>1, these variances should actuall be covariance matrices (use function predict(const Eigen::MatrixXd& inputs, Eigen::MatrixXd& outputs, std::vector<Eigen::MatrixXd>& variances) to get the full covariance matrices). So for an output dimensionality of 1 this function works fine. For dimensionality>1 we return only the diagional of the covariance matrix, which may not always be what you want.
   *
   * \remark This method should be const. But third party functions which is called in this function
   * have not always been implemented as const (Examples: LWPRObject::predict or IRFRLS::predict ).
   * Therefore, this function cannot be const.
   */
  virtual void predict(const Eigen::MatrixXd& inputs, Eigen::MatrixXd& outputs, Eigen::MatrixXd& variances)
  {
    predict(inputs, outputs);
    variances.fill(0);
  }

  /** Query the function approximator to make a prediction, and also to predict its variance
   *  \param[in]  inputs   Input values of the query (n_samples X n_dims_in)
   *  \param[out] outputs  Predicted output values (n_samples X n_dims_out)
   *  \param[out] variances Predicted covariance matrices for the output values. It is is of size  (n_samples X n_dims_out X n_dims_out), which has been implemented as a std::vector of Eigen::MatrixXd.
   *
   * \remark This method should be const. But third party functions which is called in this function
   * have not always been implemented as const (Examples: LWPRObject::predict or IRFRLS::predict ).
   * Therefore, this function cannot be const.
   */
  virtual void predict(const Eigen::MatrixXd& inputs, Eigen::MatrixXd& outputs, std::vector<Eigen::MatrixXd>& variances)
  {
    predict(inputs, outputs);
    for (unsigned int i=0; i<variances.size(); i++)
      variances[i].fill(0);
  }

  
  /** Query the function approximator to get the variance of a prediction
   * This function is not implemented by all function approximators. Therefore, the default
   * implementation fills outputs with 0s.
   *  \param[in]  inputs   Input values of the query (n_samples X n_dims_in)
   *  \param[out] variances Predicted variances for the output values  (n_samples X n_dims_out). Note that if the output has a dimensionality>1, these variances should actuall be covariance matrices (use function predict(const Eigen::MatrixXd& inputs, Eigen::MatrixXd& outputs, std::vector<Eigen::MatrixXd>& variances) to get the full covariance matrices). So for an output dimensionality of 1 this function works fine. For dimensionality>1 we return only the diagional of the covariance matrix, which may not always be what you want.
   *
   * \remark This method should be const. But third party functions which is called in this function
   * have not always been implemented as const (Examples: LWPRObject::predict or IRFRLS::predict ).
   * Therefore, this function cannot be const.
   */
  virtual void predictVariance(const Eigen::MatrixXd& inputs, Eigen::MatrixXd& variances)
  {
    variances.fill(0);
  }
  
  /** Determine whether the function approximator has already been trained with data or not.
   *  \return true if the function approximator has already been trained, false otherwise.
   */
  bool isTrained(void) const
  {
    return (model_parameters_!=NULL);
  }
  
  /** The expected dimensionality of the input data.
   * \return Expected dimensionality of the input data
   */
  int getExpectedInputDim(void) const;
  
  /** The expected dimensionality of the output data.
   * \return Expected dimensionality of the output data
   */
  int getExpectedOutputDim(void) const;
  
  /** Get the name of this function approximator
   *  \return Name of this function approximator
   */
  virtual std::string getName(void) const = 0;
  
  void getSelectableParameters(std::set<std::string>& selected_values_labels) const;
  void setSelectedParameters(const std::set<std::string>& selected_values_labels);
  /**
   * Get the minimum and maximum of the selected parameters in one vector.
   * \param[out] min The minimum of the selected parameters concatenated in one vector
   * \param[out] max The minimum of the selected parameters concatenated in one vector
   */
  void getParameterVectorSelectedMinMax(Eigen::VectorXd& min, Eigen::VectorXd& max) const;
  int getParameterVectorSelectedSize(void) const;
  void setParameterVectorSelected(const Eigen::VectorXd& values, bool normalized=false);
  void getParameterVectorSelected(Eigen::VectorXd& values, bool normalized=false) const;

  void getParameterVectorMask(const std::set<std::string> selected_values_labels, Eigen::VectorXi& selected_mask) const;
  int getParameterVectorAllSize(void) const;
  void getParameterVectorAll(Eigen::VectorXd& values) const;
  void setParameterVectorAll(const Eigen::VectorXd& values);

  /** Return a representation of this function approximator's model as a unified model. 
   * See also the page on \ref page_unified_model 
   * \return Unified model representation of this function approximator's model.
   */
  UnifiedModel* getUnifiedModel(void) const;
  
  /** Print to output stream. 
   *
   *  \param[in] output  Output stream to which to write to
   *  \param[in] function_approximator Function approximator to write
   *  \return    Output stream
   *
   *  \remark Calls virtual function FunctionApproximator::toString, which must be implemented by
   * subclasses: http://stackoverflow.com/questions/4571611/virtual-operator
   */ 
  friend std::ostream& operator<<(std::ostream& output, const FunctionApproximator& function_approximator) {
    output << function_approximator.toString();
    return output;
  }
  
  /** Returns a string representation of the object.
   * \return A string representation of the object.
   */
  std::string toString(void) const;
  
  
  /** Accessor for FunctionApproximator::meta_parameters_
   *  \return The meta-parameters of the algorithm
   */
  const MetaParameters* getMetaParameters(void) const;
  
  /** Accessor for FunctionApproximator::model_parameters_
   *  \return The model parameters of the trained model
   */
  const ModelParameters* getModelParameters(void) const;
  
  void setParameterVectorModifierPrivate(std::string modifier, bool new_value);
  
  /** Generate a input samples that lie on a grid (much like Matlab's meshgrid)
   * For instance, if min = [2 6], and max = [3 8], and n_samples_per_dim = [3 5]
   * then this function first makes linearly spaces samples along each dimension, e.g.
   * samples_dim1 = [2 2.5 3] and samples_dim2 = [6 6.5 7 7.5 8]
   * and then combine the combination of samples along all dimensions: 
   * inputs_grid = [2 6; 2 6.5; 2 7; 2 7.5 2 8; 2.5 6; 2.5 6.5; 2.5 7; 2.5 7.5 2.5 8; 3 6; 3 6.5; 3 7; 3 7.5 3 8;
   * \param[in] min Minimum values in the grid along each dimension 
   * \param[in] max Maximum values in the grid along each dimension
   * \param[in] n_samples_per_dim Number of samples along each dimension in the grid
   * \param[out] inputs_grid A grid of samples
   */
  static void generateInputsGrid(const Eigen::VectorXd& min, const Eigen::VectorXd& max, const Eigen::VectorXi& n_samples_per_dim, Eigen::MatrixXd& inputs_grid);
  
  /** Generate a grid of inputs, and output the response of the basis functions and line segments
   * for these inputs.
   * This function is not pure virtual, because this might not make sense for every model parameters
   * class.
   *
   * \param[in] min Minimum values for the grid (one for each dimension)
   * \param[in] max Maximum values for the grid (one for each dimension)
   * \param[in] n_samples_per_dim Number of samples in the grid along each dimension
   * \param[in] directory Directory to which to save the results to.
   * \param[in] overwrite Whether to overwrite existing files. true=do overwrite, false=don't overwrite and give a warning.
   * \return Whether saving the data was successful.
   */
	virtual bool saveGridData(const Eigen::VectorXd& min, const Eigen::VectorXd& max, const Eigen::VectorXi& n_samples_per_dim, std::string directory, bool overwrite=false) const;
	
protected:

  /** Accessor for FunctionApproximator::model_parameters_
   *  \param[in] model_parameters The model parameters of a trained model
   */
  void setModelParameters(ModelParameters* model_parameters);
  
  
  /**
   * Default constructor.
   * \remarks This default constuctor is required for boost::serialization to work. Since this
   * constructor should not be called by other classes, it is private (boost::serialization is a
   * friend)
   */
  FunctionApproximator(void) {};
  
	
private:
  
  /** The meta-parameters of the function approximator.
   *
   *  These are all the algorithmic parameters that are used when training the function 
   *  approximator. These are thus only used in the FunctionApproximator::train function.
   */
  MetaParameters*  meta_parameters_;
  
  /** The model parameters of the function approximator.
   *
   *  These are all the parameters of a trained function approximator. These are determined when 
   *  training the function approximator in FunctionApproximator::train, and used to make
   *  predictions in FunctionApproximator::predict.
   */
  ModelParameters* model_parameters_;
  
  bool checkModelParametersInitialized(void) const;
 
  /** Give boost serialization access to private members. */  
  friend class boost::serialization::access;
  
  /** Serialize class data members to boost archive. 
   * \param[in] ar Boost archive
   * \param[in] version Version of the class
   * See http://www.boost.org/doc/libs/1_55_0/libs/serialization/doc/tutorial.html#simplecase
   */
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version)
  {
    // serialize base class information
    ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(Parameterizable);
    
    ar & BOOST_SERIALIZATION_NVP(meta_parameters_);
    ar & BOOST_SERIALIZATION_NVP(model_parameters_);
  }

};

}


/** Tell boost serialization that this class has pure virtual functions. */
#include <boost/serialization/assume_abstract.hpp>
BOOST_SERIALIZATION_ASSUME_ABSTRACT(DmpBbo::FunctionApproximator);
 
/** Don't add version information to archives. */
#include <boost/serialization/export.hpp>
BOOST_CLASS_IMPLEMENTATION(DmpBbo::FunctionApproximator,boost::serialization::object_serializable);

#endif // _FUNCTIONAPPROXIMATOR_H_

/** \page page_func_approx Function Approximation Module

\section sec_fa Function Approximation

This module implements a set of function approximators, i.e. supervised learning algorithms that are trained with demonstration pairs input/target, after which they make predictions for new inputs. For simplicity, this module implements only batch learning (not incremental), and does not allow trained function approximators to be retrained.

The two main functions are FunctionApproximator::train, which takes a set of inputs and corresponding targets, and FunctionApproximator::predict, which makes predictions for novel inputs. 

\subsection sec_fa_metaparameters MetaParameters and ModelParameters

In this module, algorithmic parameters are called MetaParameters, and the parameters of the model when the function approximator has been trained are called ModelParameters. The rationale for this is that an untrained function approximator can be entirely reconstructed if its MetaParameters are known; this is useful for saving to file and making copies. A trained function approximator can be compeletely reconstructed given only its ModelParameters.

The life-cycle of a function approximator is as follows:

\b 1. \b Initialization: The function approximator is initialized by calling the constructor with the MetaParameters. Its ModelParameters are set to NULL, indicating that the model is untrained.

\b 2. \b Training: FunctionApproximator::train is called, which performs the conversion: \f$ \mbox{train}: \mbox{MetaParameters} \times \mbox{Inputs} \times \mbox{Targets} \mapsto \mbox{ModelParameters} \f$

\b 3. \b Prediction: FunctionApproximator::predict is called, which performs the conversion: \f$ \mbox{predict}: \mbox{ModelParameters} \times \mbox{Input} \mapsto \mbox{Output}\f$

\em Remark. FunctionApproximator::train in Step \b 2. may only be called once. If you explicitly want to retrain the function approximator with novel input/target data call FunctionApproximator::reTrain() instead.

\em Remark. During the initialization, ModelParameters may also be passed to the constructor. This means that an already trained function approximator is initialized. Step \b 2. above is thus skipped.

\subsection sec_fa_changing_modelparameters Changing the ModelParameters of a FunctionApproximator

The user should not be allowed to set the ModelParameters of a trained function approximator directly. Hence, FunctionApproximator::setModelParameters is protected. However, in order to change the values inside the model parameters (for instance when optimizing them), the user may call ModelParameters::getParameterVectorSelected and ModelParameters::setParameterVectorSelected it inherits these functions from Parameterizable). These take a vector of doubles, check if the vector has the right size, and get/set the ModelParameters accordingly.

Function approximators often have different types of model parameters. For instance, the model parameters of Locally Weighted Regression (FunctionApproximatorLWR) represent the centers and widths of the basis functions, as well as the slopes of the line segments. If you only want to get/set the slopes when calling ModelParameters::getParameterVectorSelected and ModelParameters::setParameterVectorSelected, you must use ModelParameters::setSelectedParameters(const std::set<std::string>& selected_values_labels), for instance as follows:

\code
std::set<std::string> selected;
selected.insert("slopes");
model_parameters.setSelectedParameters(selected);
Eigen::VectorXd values;
model_parameters.getParameterVectorSelected(values);
// "values" now only contains the slopes of the line segments

selected.clear();
selected.insert("centers");
selected.insert("slopes");
model_parameters.setSelectedParameters(selected);
model_parameters.getParameterVectorSelected(values);
// "values" now contains the centers of the basis functions AND slopes of the line segments

\endcode

The rationale behind this implementation is that optimizers (such as evolution strategies) should not have to care about whether a particular set of model parameters contains centers, widths or slopes. Therefore, these different types of parameters are provided in one vector without semantics, and the generic interface is provided by the Parameterizable class.

Classes that inherit from Parameterizable (such as all ModelParameters and FunctionApproximator subclasses, must implement the pure virtual methods Parameterizable::getParameterVectorAll Parameterizable::setParameterVectorAll and Parameterizable::getParameterVectorMask. Which gets/sets all the possible parameters in one vector, and a mask specifying the semantics of each value in the vector. The work of setting/getting the selected parameters (and normalizing them) is done in the Parameterizable class itself. This approach is a slightly longer run-time than doing the work in the subclasses, but it leads to more legible and robust code (less code duplication).

\subsection sec_caching_basisfunctions Caching of basis functions

If the parameters of the basis functions (centers and widths of the kernels) do not change often, you can cache the basis function activations by calling ModelParameters::set_caching(true). This can lead to speed improvements because the activations are not computed over and over again. This function only makes senses if the inputs remain the same, i.e. this is not the case when running on a real robot. 

The reason why caching is implemented in ModelParameters, and not in FunctionApproximator is because ModelParameters knows which parts of the ModelParameters change the basis function activations, and which do not (for instance in RBFN, the widths and centers change the basis function activations, but the weights do not).


 */
