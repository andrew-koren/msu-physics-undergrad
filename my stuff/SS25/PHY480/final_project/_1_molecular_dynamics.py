
import numpy as np
from numba import njit

from scipy.spatial.distance import cdist
from random import sample

@njit
def forces_lj( lbox, r, f, rcut, PBC ):

    # set the parameters for calculation
    N = r.shape[0]  # read off the number of particles
    nd = r.shape[1] # read off the dimensions
    rcut_sq = rcut*rcut # to speed up comparisons #whats this for?
    dr = np.zeros(nd) # vector between two particles
    df = np.zeros(nd) # force contribution from one pair
    lbox2 = lbox/2 # speed up calculations
    
    # set the force array to zero
    f *= 0
    
    # loop over the first particle
    for i in range(N):
        # loop over the second particle
        for j in range(i+1, N):
            Dx = r[j] - r[i]
            if PBC:
                Dx = (Dx + lbox2) % lbox - lbox2

            # if dist_sq is less than rcut_sq, there is a contribution to the force from this pair
            dist = np.sqrt(np.sum(Dx**2))
            if dist < rcut:
                continue
            
            df = 24/dist**8 * ( 1 - 2/dist) * Dx
            f[i] += df
            f[j] -= df

@njit
def measure_pot( lbox, r, rcut, PBC=True ):
    
    # read off parameters
    N = r.shape[0]
    nd = r.shape[1]
    rcut_sq = rcut*rcut
    dr = np.zeros(nd)
    lbox2 = lbox/2

    pot = 0
    
    # loop over the first particle
    for i in range(N):
        # loop over the second particle
        for j in range(i+1, N):
            Dx = r[j] - r[i]
            if PBC:
                Dx = (Dx + lbox2) % lbox - lbox2
            
            dist = np.sqrt(np.sum(Dx**2))
            if dist < rcut:
                continue    

            pot += 4 * (dist**(-12) - dist**(-6))


    return pot

def measure_kin( v, masses ):

    v2 = np.sum(v**2)
    return sum(v2*masses/2)

def initial_r( lbox, r ):
    N, dim = r.shape
    r[:] = np.random.rand(*r.shape) * lbox
    i = 1
    breaker = 0  

    while i < N:
        while (cdist(r[:N+1],r[:N+1]) + np.eye(len(r[:N+1]), len(r[:N+1]))*2 < 1).any():
            r[i] = np.random.rand(2) * lbox
            breaker += 1
            if breaker == 1000:
                r[:] = np.random.rand(*r.shape) * lbox
                breaker = 0
                i = 1
        breaker = 0
        i += 1
    
def easy_initial_r( lbox, r, min_dist = 1, jitter=0):
    grid_spacing = min_dist+jitter
    grids = np.meshgrid(*[np.arange(min_dist, l-min_dist, grid_spacing) for l in lbox])
    all_points = np.column_stack([grid.ravel() for grid in grids])

    apply_jitter = lambda x: x + (np.random.rand()-0.5)*jitter
    all_points = np.vectorize(apply_jitter)(all_points)
    r[:] = np.array(sample(list(all_points), k=len(r)))

def initial_v( lbox, v, T, masses ):
    masses = np.array([[m] for m in masses])
    sigma = np.sqrt(T/masses)

    v[:] = np.random.randn(*v.shape)  * sigma
    
    total_momentum = np.sum(masses * v, axis=1, keepdims=True)
    total_mass = np.sum(masses)
    
    v[:] -= total_momentum / total_mass

def update_r( dt, r, v ):

    r[:] = r + v*dt

def update_v( dt, v, f, masses ):

    v[:] = v + f/np.array([[m] for m in masses])*dt

def boundary_periodic( lbox, r, v ):

    r[:] = r % lbox

def boundary_wall( lbox, r, v ):

    # loop on particles and r components,
    # if r is outside, reflect back in,
    # reverse the velocity if needed
    
    #safety
    if (abs(r - lbox/2) > lbox*1.5).any():
        raise ValueError('full ring not implemented!')

    #hard wall
    v[r<0] *= -1
    r[r<0] *= -1

    v[r>lbox] *= -1
    r[:] = np.where(r>lbox, 2*lbox-r, r)