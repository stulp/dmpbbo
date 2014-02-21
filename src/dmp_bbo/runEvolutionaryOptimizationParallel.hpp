/**
 * @file runEvolutionaryOptimizationParallel.hpp
 * @brief  Header file for function to run multiple evolutionary optimization processes in parallel.
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
 
#ifndef RUNEVOLUTIONARYOPTIMIZATIONPARALLEL_H
#define RUNEVOLUTIONARYOPTIMIZATIONPARALLEL_H

#include <string>
#include <vector>

namespace DmpBbo {
  
// Forward declarations
class Task;
class TaskSolverParallel;
class DistributionGaussian;
class Updater;

/** Run several parallel evolutionary optimization processes.
 * \param[in] task The task to optimize
 * \param[in] task_solver The solver of the task
 * \param[in] distributions The initial parameter distribution (one for each parallel optimization)
 * \param[in] updater The Updater used to update the parameters
 * \param[in] n_updates The number of updates to perform
 * \param[in] n_samples_per_update The number of samples per update
 * \param[in] save_directory Optional directory to save to (default: don't save)
 * \param[in] overwrite Overwrite existing files in the directory above (default: false)
 */
void runEvolutionaryOptimizationParallel(Task* task, TaskSolverParallel* task_solver, std::vector<DistributionGaussian*> distributions, Updater* updater, int n_updates, int n_samples_per_update, std::string save_directory=std::string(""),bool overwrite=false);

}

#endif

/** \defgroup DMP_BBO Black Box Optimization of Dynamical Movement Primitives Module
 */

/** \page page_dmp_bbo Black Box Optimization of Dynamical Movement Primitives

 */
 
 
