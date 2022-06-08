""" Module with the class TaskThrowBall """

import numpy as np
from matplotlib import pyplot as plt

from dmpbbo.bbo_of_dmps.Task import Task


class TaskThrowBall(Task):
    """ Task in which a ball is thrown towards a certain positions. """

    def __init__(self, x_goal, x_margin, y_floor, acceleration_weight=0.0001):

        self.x_goal = x_goal
        self.x_margin = x_margin
        self.y_floor = y_floor
        self.acceleration_weight = acceleration_weight

    def get_cost_labels(self):
        """Labels for the different cost components.

        The cost function evaluateRollout may return an array of costs. The first one cost[0] is
        always the sum of the other ones, i.e. costs[0] = sum(costs[1:]). This function returns
        labels for the individual cost components.
        """
        return ["landing site", "acceleration"]

    def evaluate_rollout(self, cost_vars, sample):
        """The cost function which defines the task.

        @param cost_vars: All the variables relevant to computing the cost. These are determined by
            TaskSolver.perform_rollout(). For further information see the tutorial on "bbo_of_dmps".
        @param sample: The sample from which the rollout was generated. Passing this to the cost
            function is useful when performing regularization on the sample.
        @return: costs The scalar cost components for the sample. The first item costs[0] should
            contain the total cost.
        """
        n_dims = 2
        n_time_steps = cost_vars.shape[0]

        # ts = cost_vars[:,0]
        # y = cost_vars[:,1:1+n_dims]
        ydd = cost_vars[:, 1 + n_dims * 2 : 1 + n_dims * 3]
        ball = cost_vars[:, 7:9]
        ball_final_x = ball[-1, 0]

        dist_to_landing_site = abs(ball_final_x - self.x_goal)
        dist_to_landing_site -= self.x_margin
        if dist_to_landing_site < 0.0:
            dist_to_landing_site = 0.0

        sum_ydd = 0.0
        if self.acceleration_weight > 0.0:
            sum_ydd = np.sum(np.square(ydd))

        costs = np.zeros(1 + 2)
        costs[1] = dist_to_landing_site
        costs[2] = self.acceleration_weight * sum_ydd / n_time_steps
        costs[0] = np.sum(costs[1:])
        return costs

    def plot_rollout(self, cost_vars, ax=None):
        """ Plot a rollout (the cost-relevant variables).

        @param cost_vars: Rollout to plot
        @param ax: Axis to plot on (default: None, then a new axis a created)
        @return: line handles and axis
        """

        if not ax:
            ax = plt.axes()
        # Indexing
        #   t = 0
        #   y, yd, ydd = 1, 3, 5
        #   y_ball, yd_ball, ydd_ball = 7, 9, 11
        # t = cost_vars[:, 0]
        y = cost_vars[:, 1:3]
        ball = cost_vars[:, 7:9]

        line_handles = ax.plot(y[:, 0], y[:, 1], linewidth=0.5)
        line_handles_ball_traj = ax.plot(ball[:, 0], ball[:, 1], "-")
        # line_handles_ball = ax.plot(ball[::5,0],ball[::5,1],'ok')
        # plt.setp(line_handles_ball,'MarkerFaceColor','none')

        line_handles.extend(line_handles_ball_traj)

        # Plot the floor
        xg = self.x_goal
        x_floor = [-1.0, xg - self.x_margin, xg, xg + self.x_margin, 0.4]
        yf = self.y_floor
        y_floor = [yf, yf, yf - 0.05, yf, yf]
        ax.plot(x_floor, y_floor, "-k", linewidth=1)
        ax.plot(self.x_goal, self.y_floor - 0.05, "og")
        ax.axis("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim([-0.9, 0.3])
        ax.set_ylim([-0.4, 0.3])

        return line_handles, ax


def run_python_simulation(dmp, y_floor=-0.3):
    """ Run a ball throwing simulation with a DMP.

    @param dmp: The DMP to integrate.
    @param y_floor:  The height of the floor
    @return: The cost-relevant variables in a matrix.
    """
    dt = 0.01
    ts = np.arange(0, 1.5 * dmp.tau, dt)
    n_time_steps = len(ts)

    x, xd = dmp.integrate_start()

    # ts = cost_vars[:,0]
    # y = cost_vars[:,1:1+n_dims]
    # ydd = cost_vars[:,1+n_dims*2:1+n_dims*3]
    # ball = cost_vars[:,-2:]
    n_dims_y = dmp.dim_dmp()
    ys = np.zeros([n_time_steps, n_dims_y])
    yds = np.zeros([n_time_steps, n_dims_y])
    ydds = np.zeros([n_time_steps, n_dims_y])
    ys_ball = np.zeros([n_time_steps, n_dims_y])
    yd_ball = np.zeros([1, n_dims_y])
    ydd_ball = np.zeros([1, n_dims_y])

    (ys[0, :], yds[0, :], ydds[0, :]) = dmp.states_as_pos_vel_acc(x, xd)
    ys_ball[0, :] = ys[0, :]

    ball_in_hand = True
    ball_in_air = False
    for ii in range(1, n_time_steps):
        (x, xd) = dmp.integrate_step(dt, x)
        (ys[ii, :], yds[ii, :], ydds[ii, :]) = dmp.states_as_pos_vel_acc(x, xd)

        if ball_in_hand:
            # If the ball is in your hand, it moves along with your hand
            ys_ball[ii, :] = ys[ii, :]
            yd_ball = yds[ii, :]
            ydd_ball = ydds[ii, :]  # noqa

            if ts[ii] > 0.6:
                # Release the ball to throw it!
                ball_in_hand = False
                ball_in_air = True

        elif ball_in_air:
            # Ball is flying through the air
            ydd_ball[0] = 0.0  # No friction
            ydd_ball[1] = -9.81  # Gravity

            # Euler integration
            yd_ball = yd_ball + dt * ydd_ball
            ys_ball[ii, :] = ys_ball[ii - 1, :] + dt * yd_ball

            if ys_ball[ii, 1] < y_floor:
                # Ball hits the floor (floor is at -0.3)
                ball_in_air = False
                ys_ball[ii, 1] = y_floor

        else:
            # Ball on the floor: does not move anymore
            ys_ball[ii, :] = ys_ball[ii - 1, :]

    ts = np.atleast_2d(ts).T
    cost_vars = np.concatenate((ts, ys, yds, ydds, ys_ball), axis=1)
    return cost_vars
