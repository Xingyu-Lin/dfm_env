# Created by Xingyu Lin, 2019/1/26                                                                                  
from dfm_env.rope_float_env import RopeFloatEnv
import numpy as np
import datetime
import dateutil.tz
from baselines.her.util import mpi_fork
from mpi4py import MPI
import sys
import os

if __name__ == '__main__':
    # generate N starting positions
    num_cpu = 50
    N = 2000
    goal_push_num = 2
    os.system('rm -rf ./dfm_env/cached/*_{}.npz'.format(goal_push_num))

    whoami = mpi_fork(num_cpu, nv_profile=False)
    if whoami == 'parent':
        sys.exit(0)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    env = RopeFloatEnv(n_substeps=20, use_image_goal=False, use_visual_observation=False, visualization_mode=False,
                       with_goal=False, action_type='endpoints', goal_push_num=goal_push_num)
    all_init_qpos = np.empty(shape=(N, len(env.qpos_rope_inds)), dtype=float)
    all_target_qpos = np.empty(shape=(N, len(env.qpos_target_rope_inds)), dtype=float)

    for i in range(N):
        if i % 100 == 0:
            now = datetime.datetime.now(dateutil.tz.tzlocal())
            timestamp = now.strftime('%Y-%m-%d %H:%M:%S.%f %Z')
            print('{}: {}/{} pairs finished'.format(timestamp, i, N))
        env.reset()
        init_rope_qpos = env.physics.data.qpos[env.qpos_rope_inds]
        target_rope_qpos = env.physics.data.qpos[env.qpos_target_rope_inds]
        all_init_qpos[i, :] = init_rope_qpos
        all_target_qpos[i, :] = target_rope_qpos
    # print('Generation finished at cpu {}. Saving to {}...'.format(rank, worker_file_name))
    # np.savez(worker_file_name, all_init_qpos=all_init_qpos, all_target_qpos=all_target_qpos)

    # Aggregate all files at worker 0
    if rank == 0:
        worker_all_init_qpos = np.empty(shape=(N, len(env.qpos_rope_inds)), dtype=float)
        worker_all_target_qpos = np.empty(shape=(N, len(env.qpos_target_rope_inds)), dtype=float)

        for i in range(num_cpu - 1):
            comm.Recv(worker_all_init_qpos, source=i + 1, tag=13)
            comm.Recv(worker_all_target_qpos, source=i + 1, tag=13)
            all_init_qpos = np.vstack([all_init_qpos, worker_all_init_qpos])
            all_target_qpos = np.vstack([all_target_qpos, worker_all_target_qpos])
        file_name = './dfm_env/cached/generated_rope_{}.npz'.format(goal_push_num)
        np.savez(file_name, all_init_qpos=all_init_qpos, all_target_qpos=all_target_qpos)
    else:
        comm.Send(all_init_qpos, dest=0, tag=13)
        comm.Send(all_target_qpos, dest=0, tag=13)
