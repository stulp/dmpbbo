import numpy as np


def perform_rollout(dmp_sched, integrate_time, n_time_steps):
    ts = np.linspace(0.0, integrate_time, n_time_steps)
    dt = ts[1]

    ys_des = np.zeros([n_time_steps, dmp_sched.dim_y])
    yds_des = np.zeros([n_time_steps, dmp_sched.dim_y])
    ydds_des = np.zeros([n_time_steps, dmp_sched.dim_y])

    schedules = np.zeros([n_time_steps, dmp_sched.dim_y])

    ys_cur = np.zeros([n_time_steps, dmp_sched.dim_y])
    yds_cur = np.zeros([n_time_steps, dmp_sched.dim_y])
    ydds_cur = np.zeros([n_time_steps, dmp_sched.dim_y])

    x_des, xd_des, sch = dmp_sched.integrate_start_sched()
    tt = 0
    ys_des[tt, :], yds_des[tt, :], ydds_des[tt, :] = dmp_sched.states_as_pos_vel_acc(x_des,
                                                                                     xd_des)
    schedules[tt, :] = sch
    ys_cur[tt, :] = ys_des[tt, :]
    yds_cur[tt, :] = yds_des[tt, :]
    ydds_cur[tt, :] = ydds_des[tt, :]
    for tt in range(1, n_time_steps):
        x_des, xd_des, sch = dmp_sched.integrate_step_sched(dt, x_des)
        ys_des[tt, :], yds_des[tt, :], ydds_des[tt, :] = dmp_sched.states_as_pos_vel_acc(x_des,
                                                                                         xd_des)
        # Compute error terms
        y_err = ys_cur[tt - 1, :] - ys_des[tt - 1, :]
        yd_err = yds_cur[tt - 1, :] - yds_des[tt - 1, :]

        # Force due to PD-controller
        gain = 100.0
        ydds_cur[tt, :] = -gain * y_err - np.sqrt(gain) * yd_err
        # Euler integration
        yds_cur[tt, :] = yds_cur[tt - 1, :] + dt * ydds_cur[tt, :]
        ys_cur[tt, :] = ys_cur[tt - 1, :] + dt * yds_cur[tt, :]

    return ts, ys_cur, yds_cur, ydds_cur, schedules, ys_des, yds_des, ydds_des


def main():
    pass


if __name__ == "__main__":
    main()
