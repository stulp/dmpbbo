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
""" Module with functions to run the optimization of a task in multiple updates. """

from dmpbbo.bbo_of_dmps.LearningSessionTask import LearningSessionTask


def run_optimization_task_prepare(
    directory,
    task,
    task_solver,
    distribution_initial,
    n_samples_per_update,
    updater,
    dmp_initial=None,
    save_trajectory=False,
):
    """ Run the optimization of a task with a task solver

    @param directory:  The directory to save results to
    @param task:  The task to solve
    @param task_solver: The task solver which solves the task
    @param distribution_initial:  The initial distribution from which to sample.
    @param updater:  The updater to update the distribution.
    @param n_samples_per_update:  The number of samples for one update
    @param dmp_initial: The initial DMP.
    @param save_trajectory: Whether to save trajectories also, or only DMPs
    @return: The learning session (see LearningSessionTask)
    """

    args = {"task": task, "task_solver": task_solver}
    args.update({"distribution_initial": distribution_initial})
    args.update({"updater": updater})
    if dmp_initial is not None:
        args.update({"dmp_initial": dmp_initial})
    session = LearningSessionTask(n_samples_per_update, directory, **args)

    # Generate first batch of samples
    i_update = 0
    _run_optimization_task_generate_samples(
        session, distribution_initial, n_samples_per_update, i_update, save_trajectory
    )

    return session


def _run_optimization_task_generate_samples(
    session, distribution, n_samples, i_update, save_trajectory=False
):

    samples = distribution.generate_samples(n_samples)

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
            dmp.set_param_vector(distribution.mean)
        else:
            dmp.set_param_vector(samples[i_sample, :])

        f = session.tell(dmp, "dmp", i_update, i_sample)
        if save_trajectory:
            ts = dmp.ts_train
            xs, xds, _, _ = dmp.analytical_solution(ts)
            traj = dmp.states_as_trajectory(ts, xs, xds)
            f = session.tell(traj.as_matrix(), "traj", i_update, i_sample)
        filenames.append(f)

    print("ROLLOUTS NOW REQUIRED ON FOLLOWING FILES:")
    print("  " + "\n  ".join(filenames))


def run_optimization_task_one_update(session, i_update, save_trajectory=False):
    """ Do one update for the optimization of a task

    @param session:  The learning session (LearningSessionTask)
    @param i_update: The update number (how many updates so far?)
    @param save_trajectory: Whether to save trajectories also, or only DMPs
    """
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
        sample = dmp.get_param_vector()

        costs = task.evaluate_rollout(cost_vars, sample)

        session.tell(costs, "costs", i_update, i_sample)

        if i_sample != "eval":  # Do not include evaluation rollout
            costs_per_sample.append(costs[0])  # Sum over cost components only

    # 3. Update parameters
    print("UPDATING DISTRIBUTION")
    distribution_prev = session.ask("distribution", i_update)
    samples = session.ask("samples", i_update)
    updater = session.ask("updater")

    distribution_new, weights = updater.update_distribution(
        distribution_prev, samples, costs_per_sample
    )

    session.tell(weights, "weights", i_update)
    session.tell(distribution_new, "distribution_new", i_update)

    # Update done: generate new samples
    print("GENERATE NEW SAMPLES")
    i_update += 1  # Next batch of samples are for the next update.
    _run_optimization_task_generate_samples(
        session, distribution_new, n_samples, i_update, save_trajectory
    )
