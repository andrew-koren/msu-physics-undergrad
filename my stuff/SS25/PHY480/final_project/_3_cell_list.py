import numpy as np
from numba import njit, prange

@njit #to adjust indexing
def compute_strides(n_cells):
    return np.array([1, n_cells[0], n_cells[0] * n_cells[1]], dtype=np.int32)

@njit
def compute_cell_sizes(lbox, rcut):
    dim = len(lbox)
    n_cells = np.empty_like(lbox, dtype=np.int32)
    cell_size = np.empty_like(lbox)
    for d in range(dim):
        n_cells[d] = int(lbox[d] / rcut)
        cell_size[d] = lbox[d] / n_cells[d]
    return n_cells, cell_size

@njit
def setup_linked_list(r, lbox, rcut, padding=0):
    N, dim = r.shape
    n_cells, cell_size = compute_cell_sizes(lbox, rcut+padding)
    strides = compute_strides(n_cells)
    cells = np.prod(n_cells)
    
    head = -np.ones(cells, dtype=np.int32)
    linked_list = -np.ones(N, dtype=np.int32) # works up to 2mil lol

    for i in range(N):
        r_cell = np.empty(dim, dtype=np.int32)
        for d in range(dim):
            r_cell[d] = int(r[i,d] / cell_size[d]) % n_cells[d]
        
        flat_index = np.sum(r_cell * strides)  # Faster with precomputed strides
        
        linked_list[i] = head[flat_index]
        head[flat_index] = i
    
    return head, linked_list, n_cells, cell_size, strides
    
# planned to dynamically update, but skipped due to time

# get_neighbors written by DeepSeek
@njit
def get_neighbors(i, r, head, linked_list, n_cells, cell_size, strides, PBC):
    """Faster neighbor search with precomputed strides"""
    neighbors = []
    ncx, ncy, ncz = n_cells
    cx = int(r[i,0] / cell_size[0]) % ncx
    cy = int(r[i,1] / cell_size[1]) % ncy
    cz = int(r[i,2] / cell_size[2]) % ncz
    
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if PBC:
                    nx = (cx + dx) % ncx
                    ny = (cy + dy) % ncy
                    nz = (cz + dz) % ncz
                else:
                    nx, ny, nz = cx + dx, cy + dy, cz + dz
                    if (nx < 0 or nx >= ncx or 
                        ny < 0 or ny >= ncy or 
                        nz < 0 or nz >= ncz):
                        continue
                
                # Faster flat index calculation with strides
                flat_index = nx*strides[0] + ny*strides[1] + nz*strides[2]
                
                j = head[flat_index]
                while j != -1:
                    if j > i:
                        neighbors.append(j)
                    j = linked_list[j]
    return neighbors

@njit(parallel=False)
def cl_forces_lj( lbox, r, f, rcut, PBC, padding=0):

    # set the parameters for calculation
    N, nd = r.shape #read off the dimensions
    rcut_sq = rcut*rcut # to speed up comparisons #whats this for?
    dr = np.zeros(nd) # vector between two particles
    df = np.zeros(nd) # force contribution from one pair
    lbox2 = lbox/2 # speed up calculations
    
    # set the force array to zero
    f[:,:] = 0

    head, linked_list, n_cells, cell_size, strides = setup_linked_list(r, lbox, rcut, padding=padding)

    # loop over the first particle
    for i in prange(N):
        # loop over the second particle
        neighbors = get_neighbors(i, r, head, linked_list, n_cells, cell_size, strides, PBC)
        for j in neighbors:
            Dx = r[j] - r[i]
            if PBC:
                Dx = (Dx + lbox2) % lbox - lbox2

            # if dist_sq is less than rcut_sq, there is a contribution to the force from this pair
            dist_sq = np.sum(Dx**2)
            if dist_sq < rcut_sq:
                continue
            
            #df = 24/dist**8 * ( 1 - 2/dist) * Dx
            # much faster to pre-compute inverse distance
            # saves 4x time!!!!!
            inv_dist_sq = 1.0/dist_sq
            inv_d6 = inv_dist_sq**3
            inv_d12 = inv_d6**2

            df = (48*inv_d12-24*inv_d6)*inv_dist_sq*Dx


            f[i] += df
            f[j] -= df

# _4_
@njit(parallel=True)
def cl_forces_lj_parallel( lbox, r, f, rcut, PBC, padding=0):

    # set the parameters for calculation
    N, nd = r.shape #read off the dimensions
    rcut_sq = rcut*rcut # to speed up comparisons #whats this for?
    dr = np.zeros(nd) # vector between two particles
    df = np.zeros(nd) # force contribution from one pair
    lbox2 = lbox/2 # speed up calculations
    
    # set the force array to zero
    f[:,:] = 0

    head, linked_list, n_cells, cell_size, strides = setup_linked_list(r, lbox, rcut, padding=padding)

    # loop over the first particle
    for i in prange(N):
        # loop over the second particle
        neighbors = get_neighbors(i, r, head, linked_list, n_cells, cell_size, strides, PBC)
        for j in neighbors:
            Dx = r[j] - r[i]
            if PBC:
                Dx = (Dx + lbox2) % lbox - lbox2

            # if dist_sq is less than rcut_sq, there is a contribution to the force from this pair
            dist_sq = np.sum(Dx**2)
            if dist_sq < rcut_sq:
                continue
            
            #df = 24/dist**8 * ( 1 - 2/dist) * Dx
            # much faster to pre-compute inverse distance
            # saves 4x time!!!!!
            inv_dist_sq = 1.0/dist_sq
            inv_d6 = inv_dist_sq**3
            inv_d12 = inv_d6**2

            df = (48*inv_d12-24*inv_d6)*inv_dist_sq*Dx


            f[i] += df
            f[j] -= df
