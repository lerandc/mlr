import numpy as np

def hilbert_sort_2D(points, eps=1e-4, return_hidx=False):

    order = [hilbert_index_2D(p[0], p[1], eps) for p in points]

    if return_hidx:
        return order
    else:
        order = np.argsort(order)

    if len(order) == len(points):
        return order
    else:
        return hilbert_sort_2D(points, eps=eps/2.0)

def hilbert_sort_2D_unrolled(points, eps_0=1e-4, return_hidx=False, return_points=True):

    x_arr = np.copy(points[:,0])
    y_arr = np.copy(points[:,1])
    h_arr = np.zeros(points.shape[0])
    dfact = 0.25 # initialize with 2^-d for d=dim=2
    eps = eps_0

    i = 0
    while(eps <= 1):
        hilbert_index_2D_state_loop(x_arr, y_arr, h_arr, dfact)
        dfact *= 0.25
        eps *= 4.0
        i += 1
    
    if return_points:
        points[:,0] = x_arr
        points[:,1] = y_arr
        
    if return_hidx:
        return h_arr
    else:
        return np.argsort(h_arr, kind="stable")

def hilbert_sort_3D_unrolled(points, eps_0=1e-4, return_hidx=False, return_points=False):

    x_arr = np.copy(points[:,0])
    y_arr = np.copy(points[:,1])
    z_arr = np.copy(points[:,2])
    h_arr = np.zeros(points.shape[0])
    dfact = 0.125 # initialize with 2^-d for d=dim=3
    eps = eps_0

    i = 0
    while(eps <= 1):
        hilbert_index_3D_state_loop(x_arr, y_arr, h_arr, dfact)
        dfact *= 0.125
        eps *= 8.0
        i += 1
    
    if return_points:
        points[:,0] = x_arr
        points[:,1] = y_arr
        points[:,2] = y_arr
        
    if return_hidx:
        return h_arr
    else:
        return np.argsort(h_arr, kind="stable")

def hilbert_sort_2D_unrolled_fixed_iter(points, n_iter, orig_order=None, return_hidx=False, return_points=True):

    x_arr = np.copy(points[:,0])
    y_arr = np.copy(points[:,1])
    h_arr = np.zeros(points.shape[0])


    if orig_order is None:
        o_arr = np.arange(points.shape[0], dtype=int)
    else:
        o_arr = np.copy(orig_order)

    oc_arr = np.copy(o_arr)
    dfact = 0.25 # initialize with 2^-d for d=dim=2

    for i in range(n_iter):
        hilbert_index_2D_state_loop(x_arr, y_arr, h_arr, dfact)
        dfact *= 0.25
        
    if return_hidx:
        return h_arr
    else:
        h_unique = np.unique(h_arr)
        h_unique_sorted = np.argsort(h_unique)
        h_map = {k:v for k, v in zip(h_unique, h_unique_sorted)}
        hilbert_order_iter(o_arr, oc_arr, h_arr, h_map)
        return o_arr

def hilbert_sort_3D_unrolled_fixed_iter(points, n_iter, orig_order=None, return_hidx=False, return_points=True):

    x_arr = np.copy(points[:,0])
    y_arr = np.copy(points[:,1])
    z_arr = np.copy(points[:,2])
    h_arr = np.zeros(points.shape[0])

    if orig_order is None:
        o_arr = np.arange(points.shape[0], dtype=int)
    else:
        o_arr = np.copy(orig_order)

    oc_arr = np.copy(o_arr)
    dfact = 0.125 # initialize with 2^-d for d=dim=3

    for i in range(n_iter):
        hilbert_index_3D_state_loop(x_arr, y_arr, z_arr, h_arr, dfact)
        dfact *= 0.125
        
    if return_hidx:
        return h_arr
    else:
        h_unique = np.unique(h_arr)
        h_unique_sorted = np.argsort(h_unique)
        h_map = {k:v for k, v in zip(h_unique, h_unique_sorted)}
        hilbert_order_iter(o_arr, oc_arr, h_arr, h_map)
        return o_arr

def hilbert_index_2D(x, y, eps):

    if eps > 1:
        return 0

    if x < 0.5:
        if y < 0.5:
            return (0 + hilbert_index_2D(2*y, 2*x, 4*eps))/4
        else:
            return (1 + hilbert_index_2D(2*x, 2*y - 1, 4*eps))/4
    else:
        if y >= 0.5:
            return (2 + hilbert_index_2D(2*x-1, 2*y - 1, 4*eps))/4
        else:
            return (3 + hilbert_index_2D(1-2*y,2-2*x, 4*eps))/4


# put within a wrapper function so that dfact is initialized correctly
def hilbert_index_2D_state_loop(x_arr, y_arr, h_arr, dfact):

    # unsafe not to assert that arr lengths aren't all equal
    for i in range(len(x_arr)):
        x_c = x_arr[i]
        y_c = y_arr[i]

        if x_c < 0.5:
            if y_c < 0.5:
                x_n = 2.0*y_c 
                y_n = 2.0*x_c
                h_arr[i] += 0*dfact
            else:
                x_n = 2.0*x_c 
                y_n = 2.0*y_c - 1.0
                h_arr[i] += 1*dfact
        else:
            if y_c >= 0.5:
                x_n = 2.0*x_c - 1.0
                y_n = 2.0*y_c - 1.0
                h_arr[i] += 2*dfact
            else:
                x_n = 1.0 - 2.0*y_c
                y_n = 2.0 - 2.0*x_c
                h_arr[i] += 3*dfact

        x_arr[i] = x_n
        y_arr[i] = y_n



def hilbert_index_3D_state_loop(x_arr, y_arr, z_arr, h_arr, dfact):

    # unsafe not to assert that arr lengths aren't all equal
    for i in range(len(x_arr)):
        x_c = x_arr[i]
        y_c = y_arr[i]
        z_c = z_arr[i]

        # k_i is correct with the new coords;
        # but should check ordering of if statements
        if z_c < 0.5:
            if x_c < 0.5:
                if y_c < 0.5:
                    x_n = 2.0*z_c 
                    y_n = 2.0*x_c
                    z_n = 2.0*y_c

                    h_arr[i] += 0*dfact
                else:
                    x_n = 2.0*y_c - 1.0
                    y_n = 2.0*z_c
                    z_n = 2.0*x_c

                    h_arr[i] += 1*dfact 
            else:
                if y_c >= 0.5:
                    x_n = 2.0*y_c - 1.0
                    y_n = 2.0*z_c
                    z_n = 2.0*x_c - 1.0

                    h_arr[i] += 2*dfact
                else:
                    x_n = 2.0 - 2.0*x_c
                    y_n = 1.0 - 2.0*y_c
                    z_n = 2.0*z_c # I think there might be a typo on pg. 113; -1/2 * z does not map to [0, 1)
                    h_arr[i] += 3*dfact 
        else:
            if x_c >= 0.5:
                if y_c < 0.5:
                    x_n = 2.0 - 2.0*x_c
                    y_n = 1.0 - 2.0*y_c
                    z_n = 2.0*z_c - 1.0

                    h_arr[i] += 4*dfact 
                else:
                    x_n = 2.0*y_c - 1.0
                    y_n = 2.0 - 2.0*z_c
                    z_n = 2.0 - 2.0*x_c

                    h_arr[i] += 5*dfact 
            else:
                if y_c >= 0.5:
                    x_n = 2.0*y_c - 1.0
                    y_n = 2.0 - 2.0*z_c
                    z_n = 1.0 - 2.0*x_c

                    h_arr[i] += 6*dfact
                else:
                    x_n = 2.0 - 2.0*z_c
                    y_n = 2.0*x_c
                    z_n = 1.0 - 2.0*y_c

                    h_arr[i] += 7*dfact

        x_arr[i] = x_n
        y_arr[i] = y_n
        z_arr[i] = z_n

def hilbert_order_iter(o, o_c, h_arr, h_map):

    h_l = len(h_map.keys())
    N = len(o)
    N_block = N // h_l
    block_idx = np.zeros(o.shape[0], dtype=int)
    hilbert_order_iter_block_cumsum(block_idx, h_arr, h_map)

    for i in range(N):
        j = N_block * h_map[h_arr[i]] + block_idx[i]  
        o[j] = o_c[i]

def hilbert_order_iter_block_cumsum(arr, h_arr, h_map):
    for k, v in h_map.items():
        cur_idx = 0
        for j in range(len(arr)):
            if h_map[h_arr[j]] == v:
                arr[j] = 0 + cur_idx
                cur_idx += 1
