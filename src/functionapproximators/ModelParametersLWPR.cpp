/**
 * @file   ModelParametersLWPR.cpp
 * @brief  ModelParametersLWPR class source file.
 * @author Freek Stulp, Thibaut Munzer
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
 
#include <boost/serialization/export.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include "functionapproximators/ModelParametersLWPR.hpp"

/** For boost::serialization. See http://www.boost.org/doc/libs/1_55_0/libs/serialization/doc/special.html#export */
BOOST_CLASS_EXPORT_IMPLEMENT(DmpBbo::ModelParametersLWPR);

#include "dmpbbo_io/BoostSerializationToString.hpp"
#include "dmpbbo_io/EigenBoostSerialization.hpp"

#include "functionapproximators/UnifiedModel.hpp"


#include "lwpr.hh"

#include <iostream>
#include <fstream>

using namespace Eigen;
using namespace std;

namespace DmpBbo {
  
ModelParametersLWPR::ModelParametersLWPR(LWPR_Object* lwpr_object)
:
  lwpr_object_(lwpr_object)
{
  countLengths();
}

void ModelParametersLWPR::countLengths(void)
{
 
  // Determine the lengths of different vectors
  n_centers_ = 0;
  n_slopes_ = 0;
  n_offsets_ = 0;
	for (int iDim = 0; iDim < lwpr_object_->model.nOut; iDim++)
	{
		for (int iRF = 0; iRF < lwpr_object_->model.sub[iDim].numRFS; iRF++)
		{
      n_centers_ += lwpr_object_->nIn(); 
      n_slopes_  += lwpr_object_->model.sub[iDim].rf[iRF]->nReg;
      n_offsets_ += 1;
		}
	}
  n_widths_ = n_centers_;
  
}
  
ModelParametersLWPR::~ModelParametersLWPR(void)
{
  delete lwpr_object_;
}

ModelParameters* ModelParametersLWPR::clone(void) const 
{
  LWPR_Object* lwpr_object_clone = new LWPR_Object(*lwpr_object_);
  return new ModelParametersLWPR(lwpr_object_clone);
}

string ModelParametersLWPR::toString(void) const 
{
  RETURN_STRING_FROM_BOOST_SERIALIZATION_XML("ModelParametersLWPR");
  
  /*
  output << "{ \"ModelParametersLWPR\": { \"lwpr_object_xml_string\": \"";

  // Write file
  string filename("/tmp/lpwrfile_serialize.xml");
  lwpr_object_->writeXML(filename.c_str());
  
  // Read file into string stream (through file stream)
  ifstream t(filename);
  stringstream strstream;
  strstream << t.rdbuf(); 

  // Get the string and remove newlines  
  string str = strstream.str();
  str.erase(std::remove(str.begin(), str.end(), '\n'), str.end());
  
  // Output the XML string to the current output stream
  output << str;
  
  output << " \" } }";
  */
};



int ModelParametersLWPR::getExpectedInputDim(void) const
{
  return lwpr_object_->nIn();
}

void ModelParametersLWPR::getSelectableParameters(set<string>& selected_values_labels) const 
{
  selected_values_labels = set<string>();
  selected_values_labels.insert("centers");
  selected_values_labels.insert("widths");
  selected_values_labels.insert("slopes");
  selected_values_labels.insert("offsets");
}




void ModelParametersLWPR::getParameterVectorMask(const std::set<std::string> selected_values_labels, VectorXi& selected_mask) const
{
  selected_mask.resize(getParameterVectorAllSize());
  selected_mask.fill(0);
  
  int offset = 0;
  int size;
  
  // Centers
  size = n_centers_;
  if (selected_values_labels.find("centers")!=selected_values_labels.end())
    selected_mask.segment(offset,size).fill(1);
  offset += size;
  
  // Widths
  size = n_widths_;
  if (selected_values_labels.find("widths")!=selected_values_labels.end())
    selected_mask.segment(offset,size).fill(2);
  offset += size;
  
  // Offsets
  size = n_offsets_;
  if (selected_values_labels.find("offsets")!=selected_values_labels.end())
    selected_mask.segment(offset,size).fill(3);
  offset += size;

  // Slopes
  size = n_slopes_;
  if (selected_values_labels.find("slopes")!=selected_values_labels.end())
    selected_mask.segment(offset,size).fill(4);
  offset += size;

  assert(offset == getParameterVectorAllSize());   
}

void ModelParametersLWPR::getParameterVectorAll(VectorXd& values) const
{
  values.resize(getParameterVectorAllSize());
  int ii=0;
  
  for (int iDim = 0; iDim < lwpr_object_->model.nOut; iDim++)
    for (int iRF = 0; iRF < lwpr_object_->model.sub[iDim].numRFS; iRF++)
      for (int j = 0; j < lwpr_object_->nIn(); j++)
       values[ii++] = lwpr_object_->model.sub[iDim].rf[iRF]->c[j];

  for (int iDim = 0; iDim < lwpr_object_->model.nOut; iDim++)
    for (int iRF = 0; iRF < lwpr_object_->model.sub[iDim].numRFS; iRF++)
      for (int j = 0; j < lwpr_object_->nIn(); j++)
        values[ii++] = lwpr_object_->model.sub[iDim].rf[iRF]->D[j*lwpr_object_->model.nInStore+j];

  for (int iDim = 0; iDim < lwpr_object_->model.nOut; iDim++)
    for (int iRF = 0; iRF < lwpr_object_->model.sub[iDim].numRFS; iRF++)
      for (int j = 0; j < lwpr_object_->model.sub[iDim].rf[iRF]->nReg; j++)
        values[ii++] = lwpr_object_->model.sub[iDim].rf[iRF]->beta[j];
      
  for (int iDim = 0; iDim < lwpr_object_->model.nOut; iDim++)
    for (int iRF = 0; iRF < lwpr_object_->model.sub[iDim].numRFS; iRF++)
      values[ii++] = lwpr_object_->model.sub[iDim].rf[iRF]->beta0;
  
  assert(ii == getParameterVectorAllSize()); 
  
};

void ModelParametersLWPR::setParameterVectorAll(const VectorXd& values) {
  if (getParameterVectorAllSize() != values.size())
  {
    cerr << __FILE__ << ":" << __LINE__ << ": values is of wrong size." << endl;
    return;
  }
  
  int ii=0;
  for (int iDim = 0; iDim < lwpr_object_->model.nOut; iDim++)
    for (int iRF = 0; iRF < lwpr_object_->model.sub[iDim].numRFS; iRF++)
      for (int j = 0; j < lwpr_object_->nIn(); j++)
      lwpr_object_->model.sub[iDim].rf[iRF]->c[j] = values[ii++];

  for (int iDim = 0; iDim < lwpr_object_->model.nOut; iDim++)
    for (int iRF = 0; iRF < lwpr_object_->model.sub[iDim].numRFS; iRF++)
      for (int j = 0; j < lwpr_object_->nIn(); j++)
      lwpr_object_->model.sub[iDim].rf[iRF]->D[j*lwpr_object_->model.nInStore+j] = values[ii++];

  for (int iDim = 0; iDim < lwpr_object_->model.nOut; iDim++)
    for (int iRF = 0; iRF < lwpr_object_->model.sub[iDim].numRFS; iRF++)
      for (int j = 0; j < lwpr_object_->model.sub[iDim].rf[iRF]->nReg; j++)
        lwpr_object_->model.sub[iDim].rf[iRF]->beta[j] = values[ii++];

  for (int iDim = 0; iDim < lwpr_object_->model.nOut; iDim++)
    for (int iRF = 0; iRF < lwpr_object_->model.sub[iDim].numRFS; iRF++)
      lwpr_object_->model.sub[iDim].rf[iRF]->beta0 = values[ii++];

  assert(ii == getParameterVectorAllSize());   
};


UnifiedModel* ModelParametersLWPR::toUnifiedModel(void) const
{
  if (lwpr_object_->nIn()!=1)
  {
    //cout << "Warning: Can only call toUnifiedModel() when input dim of LWPR is 1" << endl;
    return NULL;
  }
  if (lwpr_object_->model.nOut!=1)
  {
    //cout << "Warning: Can only call toUnifiedModel() when output dim of LWPR is 1" << endl;
    return NULL;
  }
  
  int i_in = 0;
  int i_out = 0;

  int n_basis_functions = lwpr_object_->model.sub[i_out].numRFS;
  VectorXd centers(n_basis_functions);
  VectorXd widths(n_basis_functions);
  VectorXd offsets(n_basis_functions);
  VectorXd slopes(n_basis_functions);
  
  vector<double> tmp;
  
  for (int iRF = 0; iRF < lwpr_object_->model.sub[i_out].numRFS; iRF++)
  {
    LWPR_ReceptiveFieldObject rf = lwpr_object_->getRF(i_out,iRF);
    
    tmp = rf.center();
    centers[iRF] = tmp[i_in];
    
    offsets[iRF] = rf.beta0();
    
    tmp = rf.slope();
    slopes[iRF] = tmp[i_in];
    tmp = rf.beta();

    vector<vector<double> > D = rf.D();
    widths[iRF] = sqrt(1.0/D[i_in][i_in]);
    
  }

  
  vector<double> norm_out = lwpr_object_->normOut();
  vector<double> norm_in = lwpr_object_->normIn();
  //cout << "  centers=" << centers.transpose() << endl;
  //cout << "  widths=" << widths.transpose() << endl;
  //cout << "  offsets=" << offsets.transpose() << endl;
  //cout << "  slopes=" << slopes.transpose() << endl;
  //cout << "  norm_out[i_out]=" << norm_out[i_out] << endl;
  //cout << "  norm_in[i_in]=" << norm_out[i_in] << endl;

  offsets /= norm_out[i_out];
  slopes /= norm_out[i_out];
  centers /= norm_in[i_in];
  widths /= norm_in[i_in];
  
  //cout << "  centers=" << centers.transpose() << endl;
  //cout << "  widths=" << widths.transpose() << endl;
  //cout << "  offsets=" << offsets.transpose() << endl;
  //cout << "  slopes=" << slopes.transpose() << endl;

  //bool asymmetric_kernels=false;
  bool normalized_basis_functions=true;
  bool lines_pivot_at_max_activation=true;

  return new UnifiedModel(centers,widths,slopes,offsets,
                                    normalized_basis_functions, lines_pivot_at_max_activation);
}

template<class Archive>
void ModelParametersLWPR::serialize(Archive & ar, const unsigned int version)
{
  // serialize base class information
  ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(ModelParameters);
  
  std::cerr << "ERROR: Don't know how to serialize ModelParametersLWPR yet..." << std::endl;

  /*
  ar & BOOST_SERIALIZATION_NVP(centers_);
  ar & BOOST_SERIALIZATION_NVP(widths_);
  ar & BOOST_SERIALIZATION_NVP(slopes_);
  ar & BOOST_SERIALIZATION_NVP(offsets_);
  ar & BOOST_SERIALIZATION_NVP(asymmetric_kernels_);
  ar & BOOST_SERIALIZATION_NVP(lines_pivot_at_max_activation_);
  ar & BOOST_SERIALIZATION_NVP(slopes_as_angles_);
  ar & BOOST_SERIALIZATION_NVP(all_values_vector_size_);
  ar & BOOST_SERIALIZATION_NVP(caching_);
  */
}

/*
// void FunctionApproximatorLWPR::save(const char* file)
// {
// 	lwpr_object_->writeBinary(file);
// }
*/

}
