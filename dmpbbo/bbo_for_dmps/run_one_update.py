# This file is part of DmpBbo, a set of libraries and programs for the
# black-box optimization of dynamical movement primitives.
# Copyright (C) 2022 Freek Stulp, ENSTA-ParisTech
#
# DmpBbo is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# DmpBbo is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DmpBbo.  If not, see <http://www.gnu.org/licenses/>.


from dmpbbo.bbo_for_dmps.LearningSessionTask import *


def runOptimizationTaskPrepare(
    directory,
    task,
    task_solver,
    distribution_initial,
    n_samples_per_update,
    updater,
    dmp_initial=None,
):

    args = {"task": task, "task_solver": task_solver}
    args.update({"distribution_initial": distribution_initial})
    args.update({"updater": updater})
    if dmp_initial:
        args.update({"dmp_initial": dmp_initial})
    session = LearningSessionTask(n_samples_per_update, directory, **args)

    # Generate first batch of samples
    i_update = 0
    _runOptimizationTaskGenerateSamples(
        session, distribution_initial, n_samples_per_update, i_update
    )

    return session


def _runOptimizationTaskGenerateSamples(session, distribution, n_samples, i_update):

    samples = distribution.generateSamples(n_samples)

    # Save some things to file
    session.tell(distribution, "distribution", i_update)
    session.tell(samples, "samples", i_update)

    # Load the initial DMP, and then set its perturbed parameters
    dmp = session.ask("dmp_initial")

    sample_labels = list(range(n_samples))
    sample_labels.append("eval")
    filenames = []
    for i_sample in sample_labels:

        if i_sample == "eval":
            # Evaluation rollout: no perturbation
            dmp.setParamVector(distribution.mean)
        else:
            dmp.setParamVector(samples[i_sample, :])

        f = session.tell(dmp, "dmp", i_update, i_sample)
        filenames.append(f)

    print("ROLLOUTS NOW REQUIRED ON FOLLOWING FILES:")
    print("  " + "\n  ".join(filenames))


def runOptimizationTaskOneUpdate(session, i_update):

    print("======================================================")
    print(f"i_update = {i_update}")

    print("EVALUATING ROLLOUTS")
    costs_per_sample = []

    n_samples = session.ask("n_samples_per_update")
    task = session.ask("task")

    sample_labels = list(range(n_samples))
    sample_labels.append("eval")
    for i_sample in sample_labels:

        cost_vars = session.ask("cost_vars", i_update, i_sample)
        dmp = session.ask("dmp", i_update, i_sample)
        sample = dmp.getParamVector()

        costs = task.evaluateRollout(cost_vars, sample)

        session.tell(costs, "costs", i_update, i_sample)

        if i_sample != "eval":  # Do not include evaluation rollout
            costs_per_sample.append(costs[0])  # Sum over cost components only

    # 3. Update parameters
    print("UPDATING DISTRIBUTION")
    distribution_prev = session.ask("distribution", i_update)
    samples = session.ask("samples", i_update)
    updater = session.ask("updater")

    distribution_new, weights = updater.updateDistribution(
        distribution_prev, samples, costs_per_sample
    )

    session.tell(weights, "weights", i_update)
    session.tell(distribution_new, "distribution_new", i_update)

    # Update done: generate new samples
    print("GENERATE NEW SAMPLES")
    i_update += 1  # Next batch of samples are for the next update.
    _runOptimizationTaskGenerateSamples(session, distribution_new, n_samples, i_update)
