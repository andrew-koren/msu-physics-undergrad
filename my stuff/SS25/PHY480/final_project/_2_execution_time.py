import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

from _1_molecular_dynamics import *

boundary = [boundary_wall, boundary_periodic]

def run_simulation_tracking(r, v, f, masses, Nsim, PBC, lbox, dt, rcut, skipframes=0, Ninner=1, forces_lj=forces_lj):

    r_storage = np.zeros([Nsim,*r.shape])
    f_storage = np.zeros([Nsim,*f.shape])

    segment_times = np.zeros((Nsim, 5))

    frame = 0
    for i in range(Nsim+skipframes):
        # inner loop
        for j in range(Ninner):
            t0 = perf_counter()
            # propagate velocities by dt/2
            update_v(dt/2, v, f, masses)
            t1 = perf_counter()
            # propagate positions r by dt
            update_r(dt, r, v)
            t2 = perf_counter()
            # apply boundary conditions
            boundary[PBC]( lbox, r, v )
            t3 = perf_counter()
            # compute forces
            forces_lj( lbox, r, f, rcut, PBC)
            t4 = perf_counter()

            # propagate velocities by dt/2
            update_v(dt/2, v, f, masses)
            t5 = perf_counter()

        if i >= skipframes:
            r_storage[frame] = r
            f_storage[frame] = f
            frame += 1

            segment_times[i-skipframes] = [t1-t0,t2-t1,t3-t2,t4-t3,t5-t4]

    return r_storage, f_storage, segment_times

def trailing_mean(array, trail_len=None):
    ar2 = np.zeros(array.shape)
    if trail_len:
        for i in range(len(array)):
            ar2[i] = array[max(i-trail_len,0):i+1].mean()
    else:
        for i in range(len(array)):
            ar2[i] = array[:i+1].mean()
    return ar2

def plot_frametimes(segment_times):
    names  = ['1. v+at/2', '2. x+vt/2', '3. apply boundary', '4. forces', '5. v+at/2']
    for i, name in enumerate(names):
        plt.plot(range(len(segment_times)), trailing_mean(segment_times[:,i], 5), label=name)
    plt.grid()
    plt.yscale('log')
    plt.legend()
    plt.xlabel('frames')
    plt.ylabel('seconds per frame')

def plot_scaling_chart(meta_segment_times, labels=None):
    force_times = []
    other_times = []
    for N, seg_times in meta_segment_times:
        x = N
        mean_seg_times = seg_times.mean(axis=0)
        y1 = mean_seg_times[3] * 1000
        y2 = mean_seg_times[[0,1,2,4]].sum() * 1000

        force_times.append((x, y1))
        other_times.append((x, y2))
    
    if labels == None:
        plt.scatter(*np.array(force_times).T, label='Force update')
        plt.scatter(*np.array(other_times).T, label='Everything else')
    else:
        plt.scatter(*np.array(force_times).T, label=labels[0])
        plt.scatter(*np.array(other_times).T, label=labels[1])


    plt.xlabel('N')
    plt.ylabel('Time per frame (ms)')


