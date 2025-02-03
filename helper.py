import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import gridspec
from scipy.interpolate import griddata
from matplotlib.colors import ListedColormap
from scipy import sparse
import copy

#NASA Logo

def png_to_matrix(file_path):
    """
    Convert a PNG image to a NumPy matrix.

    Args:
    file_path (str): Path to the PNG file.

    Returns:
    numpy.ndarray: 2D matrix representing the image.
    """
    # Open the image using PIL
    img = Image.open(file_path)
    
    # Convert the image to grayscale if it's not already
    img = img.convert('L')
    
    # Convert the image to a NumPy array
    matrix = np.array(img)
    
    return matrix

file_path = 'ImageProcessingDemo128.png'
NASA_map = png_to_matrix(file_path)[::-1,::]



### Diffusion-Advection-Forcing Model ###

def make_M_2d_diffusion_advection_forcing(nr: int, nc: int, dt: float,
                                KX: np.ndarray, KY: np.ndarray,
                                DX_C: np.ndarray, DY_C: np.ndarray,
                                DX_G: np.ndarray, DY_G: np.ndarray,
                                VX: np.ndarray, VY: np.ndarray,
                                RAC: np.ndarray,
                                F : float,
                                cyclic_east_west:   bool = True, 
                                cyclic_north_south: bool = False,
                                M_is_sparse=False):
    """
    Creates linear model M which can be used to forward-simulate a discrete approximation of a 
    2D diffusion-advection-forcing model.

    c(t+1) = Mc(t) + F f(t)

    Spatially-variant case

    nr:     the number of rows of discrete cells.
    nc:     the number of cols of discrete cells.
    dt:     duration of a timestep

    (i,j) = (0,  0) is the southwesternmost cell
    (i,j) = (-1,-1) is the northeasternmost cell

    For the below definitions:

    KX:     the diffusivity constant matrix along the x-axis (between columns)
                KX[i,j]   gives diffusivity at the boundary between cells [i,j-1] and [i,j] 

    KY:     the diffusivity constant matrix along the y-axis (between rows)
                KY[i,j]   gives diffusivity at the boundary between cells [i-1,j] and [i,j]

    DX_C:   the horizontal distance (x-axis) matrix between the centers cells in adjacent columns
                DX_C[i,j] gives the distance between the centers of cells [i,j-1] and [i,j]

    DY_C:   the vertical distance (y-axis) matrix between the centers of cells in adjacent rows
                DY_C[i,j] gives the distance between the centers of cells [i-1,j] and [i,j]

    DX_G:   the horizontal length (x-axis) matrix of a cell along one edge.
                DX_G[i,j] gives the length of the "south" side of cell [i,j]  

    DY_G:   the vertical length (y-axis) matrix of a cell along one edge.
                DY_G[i,j] gives the length of the "west" side of cell [i,j]  

    VX:     the velocity constant matrix along the x-axis (between columns)
                VX[i,j]   gives the velocity at the boundary between cells [i,j-1] and [i,j]

    VY:     the velocity constant matrix along the y-axis (between rows)
                VY[i,j]   gives the velocity at the boundary between cells [i-1,j] and [i,j]

    RAC:    the area of a cell.
                RAC[i,j]  gives the area of cell [i,j]
    
    F:     the forcing term constant. This is the same for all cells.

    cyclic_east_west:   if True, cell [i, 0] is east of cell [i,-1]

    cyclic_north_south: if True, cell [0, j] is north of cell [-1,j]

    M_is_sparse: if True, return a sparse matrix. If False, return a dense matrix.
    """

    if KX.shape !=   (nr, nc+1):
        raise ValueError("KX doesn't have the right shape for your dimensions!")
    if KY.shape !=   (nr+1, nc):
        raise ValueError("KY doesn't have the right shape for your dimensions!")
    if DX_C.shape != (nr, nc+1):
        raise ValueError("DX_C doesn't have the right shape for your dimensions!")
    if DY_C.shape != (nr+1, nc):
        raise ValueError("DY_C doesn't have the right shape for your dimensions!")
    if DX_G.shape != (nr+1, nc):
        raise ValueError("DX_G doesn't have the right shape for your dimensions!")
    if DY_G.shape != (nr, nc+1):
        raise ValueError("DX_G doesn't have the right shape for your dimensions!")
    if VX.shape !=   (nr, nc+1):
        raise ValueError("VX doesn't have the right shape for your dimensions!")
    if VY.shape !=   (nr+1, nc):
        raise ValueError("VY doesn't have the right shape for your dimensions!")

    size = nr * nc

    if M_is_sparse:
        M = sparse.lil_matrix((size, size))
    else:
        M = np.zeros((size, size))
    
    beta = dt / RAC
    


    # Shorthand variables

    S = KX*DY_G/DX_C
    T = KY*DX_G/DY_C

    S_IJ, T_IJ     = S[:,   :-1], T[:-1,  :]
    S_IJP1, T_IP1J = S[:,   1:],  T[1:,  :]

    R = VX*DY_G 
    Q = VY*DX_G 

    R_IJ, Q_IJ     = R[:,   :-1], Q[:-1,  :]
    R_IJP1, Q_IP1J = R[:,   1:],  Q[1:,  :]

    # Contributions from diffusion (d)

    d_IP1_J = beta * T_IP1J
    d_IM1_J = beta * T_IJ
    d_I_JP1 = beta * S_IJP1
    d_I_JM1 = beta * S_IJ

    d_IJ = - d_IP1_J - d_IM1_J - d_I_JP1 - d_I_JM1

    # Contributions from advection (a)

    a_IM1_J = beta * np.maximum(Q_IJ, 0)
    a_IP1_J = - beta * np.minimum(Q_IP1J, 0)
    a_I_JM1 = beta * np.maximum(R_IJ, 0)
    a_I_JP1 = - beta * np.minimum(R_IJP1, 0)

    a_IJ = beta * (\
        np.minimum(Q_IJ, 0) - np.maximum(Q_IP1J, 0) + \
        np.minimum(R_IJ, 0) - np.maximum(R_IJP1, 0))


    #Create array to store indices

    c = np.zeros([nr,nc])
    c_indices = np.arange(len(c.ravel()))
    c_indices = np.array(np.reshape(c_indices, [nr, nc]))

    for i in range(nr): #y-axis (north, south)
        for j in range(nc): #x-axis (east, west)
            
            #Get current position
            ind_here = c_indices[i,j]

            # Currently we have no adjacent cells, we need to populate them
            ind_N = np.nan
            ind_E = np.nan
            ind_S = np.nan
            ind_W = np.nan
            
            # Get indices for each direction
            # south
            if i > 0:
                ind_S = c_indices[i-1, j]
            elif cyclic_north_south:
                ind_S = c_indices[-1, j]

            # north
            if i < nr-1:
                ind_N = c_indices[i+1, j]
            elif cyclic_north_south:
                ind_N = c_indices[0, j]

            # west
            if j > 0:
                ind_W = c_indices[i, j-1]
            elif cyclic_east_west:
                ind_W = c_indices[i, -1]
            
            # east
            if j < nc-1:
                ind_E = c_indices[i, j+1]
            elif cyclic_east_west:
                ind_E = c_indices[i, 0]
    
            # Now that we have our indices, we can fill in our matrix
            
            M[ind_here, ind_here] = 1 + d_IJ[i,j] + a_IJ[i,j] \
                                    - F #Forcing term is the same for all cells
            
            if np.isfinite(ind_W):
                # cell to the west 
                M[ind_here, ind_W] = 0 + d_I_JM1[i,j] + a_I_JM1[i,j]
            if np.isfinite(ind_E):
                # cell to the east
                M[ind_here, ind_E] = 0 + d_I_JP1[i,j] + a_I_JP1[i,j]
            if np.isfinite(ind_N):
                # cell to the north
                M[ind_here, ind_N] = 0 + d_IP1_J[i,j] + a_IP1_J[i,j]
            if np.isfinite(ind_S):
                # cell to the south
                M[ind_here, ind_S] = 0 + d_IM1_J[i,j] + a_IM1_J[i,j]
    
    if M_is_sparse:
        M = M.tocsr()

    return M
    




### Simulate Model ###

def compute_linear_time_evolution(c0, M, saved_timesteps, duration,
        debug = False):
    """
    Compute linear time evolution of model: c(t+1) = Mc(t) 

    Args:
    c0 (array): Initial state vector (a,1)
    M (array): Linear model matrix (a,a)
    saved_timesteps (list): Timesteps to save state
    duration (int): Number of timesteps to simulate
    debug (bool): If True, print progress every 10 steps

    Returns:
    saved_timesteps (list): List of timesteps where state was saved
    saved (list): List of state vectors at each saved timestep
    """
    r = np.zeros_like(c0)
    return compute_affine_time_evolution(c0, M, r, saved_timesteps, duration, debug)

def compute_linear_time_evolution_simple(c0, M, num_saved_timesteps, duration_per_saved_timestep,
        debug = False):
    """
    Compute linear time evolution of model: c(t+1) = Mc(t) 

    Simplified with evenly spaced saved_timesteps.

    Args:
    c0 (array): Initial state vector (a,1)
    M (array): Linear model matrix (a,a)
    num_saved_timesteps (int): Number of saved_timesteps to save
    duration_per_saved_timestep (int): Number of timesteps between saved_timesteps
    debug (bool): If True, print progress every 10 steps

    Returns:
    list: 1D arrays representing saved_timesteps in time
    """
    saved_timesteps = [i*duration_per_saved_timestep for i in range(num_saved_timesteps)] # Evenly spaced saved_timesteps
    duration = num_saved_timesteps * duration_per_saved_timestep + 1 # Duration must be longer than the last saved timestep

    return compute_linear_time_evolution(c0, M, saved_timesteps, duration, debug)

def compute_affine_time_evolution(c0, M, r, saved_timesteps, duration,
        debug = False):
    """
    Compute affine time evolution model with affine term: c(t+1) = Mc(t) + r

    Args:
    c0 (array): Initial state vector (a,1)
    M (array): Linear model matrix (a,a)
    r (array): Affine term vector (a,1)
    saved_timesteps (list): Timesteps to save state
    duration (int): Number of timesteps to simulate
    debug (bool): If True, print progress every 10 steps

    Returns:
    saved_timesteps (list): List of timesteps where state was saved
    state_over_time (list): List of state vectors at each saved timestep
    """
    state_over_time=[]
    c = c0

    for i in range(duration): #Iterate over all timesteps
        if i%10 == 0 and debug:
            print(i)

        if i in saved_timesteps:
            state_over_time.append(c)
        c = M.dot(c) + r #Update the state

    return saved_timesteps, state_over_time

def compute_affine_time_evolution_simple(c0, M, r, num_saved_timesteps, 
                                         duration_per_saved_timestep=1, debug = False):
    """
    Compute affine time evolution model with affine term: c(t+1) = Mc(t) + r

    Simplified with evenly spaced saved_timesteps.

    Args:
    c0 (array): Initial state vector (a,1)
    M (array): Linear model matrix (a,a)
    r (array): Affine term vector (a,1)
    num_saved_timesteps (int): Number of saved_timesteps to save
    duration_per_saved_timestep (int): Number of timesteps between saved_timesteps
    debug (bool): If True, print progress every 10 steps

    Returns:
    list: 1D arrays representing saved_timesteps in time
    """
    saved_timesteps = [i*duration_per_saved_timestep for i in range(num_saved_timesteps)] # Evenly spaced saved_timesteps
    duration = num_saved_timesteps * duration_per_saved_timestep + 1 # Duration must be longer than the last saved timestep
    
    return compute_affine_time_evolution(c0, M, r, saved_timesteps, duration, debug)



### Plotting Heatmaps ###

def plot_multi_heatmap_time_evolution(saved_timesteps, many_states_over_time, 
                                 nr, nc, titles, big_title, 
                                 vmin=None, vmax=None, is_1d=False, transpose=False):
    """
    Display time evolution of multiple 1D or 2D arrays over time, side by side.

    Args:
    saved_timesteps (list or None): List of times that we save our state, or None if not shown
    many_states_over_time (list of lists): Each inner list contains 1D arrays of state at each timestep
    nr (int): Number of rows in each 2D array (or 1 for 1D state)
    nc (int): Number of columns in each 2D array (or length of 1D array)
    titles (list): Titles for each subplot
    big_title (str): Overall title for the entire plot
    vmin, vmax (float, optional): Min/max values for color scaling
    is_1d (bool): Whether the input state is 1D (True) or 2D (False)
    transpose (bool): If True, transpose the grid layout of plots
    """
    # Reconstruct 1D arrays into 2D arrays
    if is_1d:
        plottable_states = [[state.reshape(1, -1) for state in state_over_time] 
                            for state_over_time in many_states_over_time]
    else:
        plottable_states = [[state.reshape(nr, nc) for state in state_over_time] 
                            for state_over_time in many_states_over_time]
    
    # Calculate global min and max for consistent color scaling if not provided
    if vmin is None:
        vmin = min(np.min(state) for state_over_time in many_states_over_time for state in state_over_time)
    if vmax is None:
        vmax = max(np.max(state) for state_over_time in many_states_over_time for state in state_over_time)
    
    # Create the heatmaps
    num_states = len(many_states_over_time)
    num_timesteps = len(many_states_over_time[0])  # Assume all state runs have the same number of timesteps
    
    # Calculate figure size and height ratios based on transpose option
    title_height = 0.5  # inches
    subplot_height = 2 if is_1d else 5  # inches
    
    if transpose:
        # For transposed layout, swap width and height calculations
        total_height = title_height + (subplot_height * num_states)
        fig_width = 5 * num_timesteps + 1
    else:
        # Original layout
        total_height = title_height + (subplot_height * num_timesteps)
        fig_width = 5 * num_states + 1
    
    # Create figure with two subfigures
    fig = plt.figure(figsize=(fig_width, total_height))
    subfigs = fig.subfigures(2, 1, height_ratios=[title_height, subplot_height * (num_states if transpose else num_timesteps)])
    
    # Add the main title to the top subfigure
    subfigs[0].suptitle(big_title, fontsize=16)
    
    # Create gridspec for the bottom subfigure (plot grid)
    if transpose:
        gs = subfigs[1].add_gridspec(num_states, num_timesteps + 1, width_ratios=[1]*num_timesteps + [0.05])
    else:
        gs = subfigs[1].add_gridspec(num_timesteps, num_states + 1, width_ratios=[1]*num_states + [0.05])
    
    for i, state_over_time in enumerate(plottable_states):
        for j, state in enumerate(state_over_time):
            if transpose:
                ax = subfigs[1].add_subplot(gs[i, j])
            else:
                ax = subfigs[1].add_subplot(gs[j, i])
            
            im = ax.imshow(state, aspect='auto', cmap='coolwarm', vmin=vmin, vmax=vmax, origin='lower')
            
            # Set the title, including timestep if provided
            if saved_timesteps is None:
                ax.set_title(f'{titles[i]}')
            else:
                ax.set_title(f'{titles[i]}\nt={saved_timesteps[j]}')
            
            ax.set_xlabel('$x$')
            if not is_1d:
                ax.set_ylabel('$y$')
            else:
                ax.set_yticks([])
            
            # Add colorbar for each row
            if transpose:
                if j == num_timesteps - 1:  # Only for the last column
                    cbar_ax = subfigs[1].add_subplot(gs[i, -1])
                    plt.colorbar(im, cax=cbar_ax)
            else:
                if i == num_states - 1:  # Only for the last column
                    cbar_ax = subfigs[1].add_subplot(gs[j, -1])
                    plt.colorbar(im, cax=cbar_ax)
    
    plt.tight_layout()
    plt.show()

def plot_1d_heatmap_time_evolution(saved_timesteps, state_over_time, transpose=False):
    """
    Displays the time evolution of a 1D state as a series of heatmaps.

    Args:
    saved_timesteps (list): List of time values corresponding to each state array
    state_over_time (list): List of 1D numpy arrays, each representing the state at a time saved
    
    Returns:
    None: Displays the plot using matplotlib

    """
    plot_multi_heatmap_time_evolution(
        saved_timesteps=saved_timesteps,
        many_states_over_time=[state_over_time],
        nr=1,
        nc=len(state_over_time[0]),
        titles=["Evolution in time"],
        big_title="Evolution in time",
        transpose=transpose
    )



def plot_2d_heatmap_time_evolution(saved_timesteps, state_over_time, nr, nc, vmin = None, vmax = None,
                                   transpose=False):
    """
    Displays the time evolution of 2D state as a series of heatmaps.

    Args:
    saved_timesteps (list): List of time values corresponding to each state array in time
    state_over_time (list): List of 1D numpy arrays, each representing the state at a time saved
    nr (int): Number of rows in the 2D grid
    nc (int): Number of columns in the 2D grid
    vmin (float, optional): Minimum value for color scaling. If None, calculated from state.
    vmax (float, optional): Maximum value for color scaling. If None, calculated from state.

    Returns:
    None: Displays the plot using matplotlib
    """
    # We're passing a single state over time, so we wrap it in another list
    plot_multi_heatmap_time_evolution(
        saved_timesteps=saved_timesteps,
        many_states_over_time=[state_over_time],
        nr=nr,
        nc=nc,
        titles=["Evolution in time"],
        big_title="Evolution in time",
        vmin=vmin,
        vmax=vmax,
        transpose=transpose
    )


def plot_multi_heatmap(many_states, nr, nc, titles, big_title, vmin=None, vmax=None, transpose=False):
    """
    Display heatmaps of multiple 2D states side by side for comparison.

    Args:
    many_states (list): List of 1D arrays to be reshaped into 2D
    nr, nc (int): Number of rows and columns for reshaping
    titles (list): Titles for each heatmap
    big_title (str): Overall title for the plot
    vmin, vmax (float, optional): Min/max values for color scaling

    Displays heatmaps side by side for comparison
    """
    # We're comparing at a single timestep, so we wrap each data array in its own list
    many_states_over_time = [[state] for state in many_states]
    
    plot_multi_heatmap_time_evolution(
        saved_timesteps=None,  # Single timestep
        many_states_over_time=many_states_over_time,
        nr=nr,
        nc=nc,
        titles=titles,
        big_title=big_title,
        vmin=vmin,
        vmax=vmax,
        transpose=transpose
    )



### Covariance functions ###

def compute_covariance_gaussian_dropoff(a, b, std_dev = 1):
    """
    Compute Gaussian covariance matrix with exponential dropoff.

    Calculates covariance using e^{ -d / 2s }, where d is squared
    Euclidean distance between a and b, and s is the standard deviation std_dev.

    Args:
    a, b (array-like): Input vectors, represents spatial coordinates
    std_dev (float): Standard deviation of gaussian dropoff (default=1)

    Returns:
    numpy.ndarray: Covariance matrix
    """

    d = np.linalg.norm(a - b, axis=-1)**2      #Distance between a and b
    covariance = np.exp( -d / (2*std_dev))       #Covariance matrix

    return covariance


def vector_of_2d_indices(nr, nc):
    """
    Convert 2D array of indices to a 1D vector representation containing the same indices.

    Args:
    nr (int): Number of rows
    nc (int): Number of columns

    Returns:
    numpy.ndarray: Shape (nr*nc, 2), each row is [row, col] index
    """
    y, x = np.mgrid[:nr, :nc] #Get meshgrid of all of our indices
    vector_form = np.column_stack((y.ravel(), x.ravel())) #Stack
    return vector_form


def compute_covariance_matrix_gaussian_dropoff(nr, nc, std_dev=1):
    """
    Compute covariance matrix with Gaussian dropoff for 2D grid.

    Compares all pairs of 2D indices and computes covariance using
    exponential dropoff function e^{ -d / 2s }, where 
    - d is squared Euclidean distance between a and b
    - s is the standard deviation std_dev.

    Args:
    nr, nc (int): Number of rows and columns in 2D grid
    std_dev (float): Standard deviation for Gaussian dropoff (default=1)

    Returns:
    np.ndarray: Covariance matrix of size (nr*nc, nr*nc)
    """
    # Get all 2D indices as a 1D vector of (row, col) pairs
    index_vector = vector_of_2d_indices(nr, nc)
    
    # Prepare indices for broadcasting
    indices_i = index_vector[:, np.newaxis, :]  # Shape: (nr*nc, 1, 2)
    indices_j = index_vector[np.newaxis, :, :]  # Shape: (1, nr*nc, 2)
    
    # Compute covariance matrix
    covariance_matrix = compute_covariance_gaussian_dropoff(
        indices_i,
        indices_j,
        std_dev=std_dev
    )
    
    return covariance_matrix



### Generating Smooth Data ###

def add_random_circles(matrix, num_circles, radius, values):
    """
    Add random circles to a matrix for interesting initial conditions.

    Args:
    matrix (numpy.ndarray): 2D array to modify
    num_circles (int): Number of circles to add
    radius (int): Radius of circles
    values (list): Possible values for circles

    Returns:
    numpy.ndarray: Modified matrix with added circles
    """

    nr, nc = matrix.shape
    for _ in range(num_circles):
        center_r = np.random.randint(0, nr)
        center_c = np.random.randint(0, nc)
        r, c = np.ogrid[:nr, :nc]
        mask = ((r - center_r)**2 + (c - center_c)**2 <= radius**2)

        value = np.random.choice(values)
        matrix[mask] = value
    return matrix

def generate_random_vectors_mean_0_cov_C(nr, nc, C, num_vectors):
    """
    Generates random vectors from a normal distribution with mean 0 and covariance C.

    Args:
    nr, nc (int): Dimensions of the 2D grid
    C (np.ndarray): Covariance matrix
    num_vectors (int): Number of random vectors to generate

    Returns:
    tuple: (zs, Lzs)
        zs (list): Random normal vectors with mean 0 and covariance I    
            z ~ N(0, I)
        Lzs (list): Random vectors with mean 0 and covariance C
            Lz ~ N(0, C)
    """
    size = nr * nc #Size of the grid
    zs = [np.random.randn(size, 1) for _ in range(num_vectors)] #Random normal vectors with mean 0 and covariance I

    # If z ~ N(0, I), then Lz ~ N(0, C)
    L = np.linalg.cholesky(C) 
    Lzs = [L @ z for z in zs] #Random vectors with mean 0 and covariance C

    return zs, Lzs

def generate_random_vector_mean_0_cov_C(nr, nc, C):
    """
    Generates a random vector from a normal distribution with mean 0 and covariance C.

    Args:
    nr, nc (int): Dimensions of the 2D grid
    C (np.ndarray): Covariance matrix

    Returns:
    tuple: (z, Lz)
        z (np.ndarray): Random normal vector with mean 0 and covariance I    
            z ~ N(0, I)
        Lz (np.ndarray): Random vector with mean 0 and covariance C
            Lz ~ N(0, C)
    """
    zs, Lzs = generate_random_vectors_mean_0_cov_C(nr, nc, C, 1)
    return zs[0], Lzs[0]





def generate_true_and_first_guess_field_uniform_cov(C, nr, nc, gamma):
    """
    Generates two fields: a true field and a first-guess field, both with the same covariance C.

    Args:
    C (np.ndarray): Covariance matrix for both fields
    nr (int): Number of rows in the grid
    nc (int): Number of columns in the grid
    gamma (float): Fraction of the field shared between true and first-guess fields

    Returns:
    tuple: (true_field, first_guess_field)
        true_field (np.ndarray): Random field with covariance C
        first_guess_field (np.ndarray): Random field sharing a component with true_field, covariance C

    Notes:
    - f2 replaces f1 in the "first-guess" field to represent inaccuracies
    - The shared component (f0) represents the fraction of the field that is 
      common between the true and first-guess fields
    """
    f0,f1,f2 = generate_random_vectors_mean_0_cov_C(nr, nc, C, 3)[1]

    true_field        = f0 * gamma + f1 * (1-gamma)
    first_guess_field = f0 * gamma + f2 * (1-gamma)

    return true_field, first_guess_field

def generate_true_and_first_guess_field(C_known, C_error, nr, nc):
    """
    Generates two fields: a true field and a first-guess field, both with specified covariances.

    Args:
    C_known (np.ndarray): Covariance matrix for the known part of the field
    C_error (np.ndarray): Covariance matrix for the error part of the field
    nr, nc (int): Dimensions of the 2D grid

    Returns:
    tuple: (true_field, first_guess_field)
        true_field (np.ndarray): Random field with covariance C_known + C_error
        first_guess_field (np.ndarray): Random field sharing a component with true_field, covariance C_known + C_error

    Notes:
    - f2 replaces f1 in the "first-guess" field to represent inaccuracies
    - The shared component (f0) represents the fraction of the field that is 
      common between the true and first-guess fields
    """
    f0 = generate_random_vectors_mean_0_cov_C(nr, nc, C_known, 1)[1][0]
    f1, f2 = generate_random_vectors_mean_0_cov_C(nr, nc, C_error, 2)[1]

    true_field = f0 + f1
    first_guess_field = f0 + f2

    return true_field, first_guess_field


def generate_gaussian_field(n, nrv, ncv):
    """
    Randomly generates a 2D field composed of n Gaussian functions with distinct means and standard deviations.

    Args:
    n (int): Number of Gaussian functions to generate
    nrv (int): Number of rows in the field
    ncv (int): Number of columns in the field

    Returns:
    np.ndarray: A 2D array representing the generated Gaussian field

    Notes:
    - The field is made pseudo-periodic by creating three copies of each Gaussian function
    """
    mux = np.random.choice(ncv, n)
    muy = np.random.choice(range(2, nrv - 2), n)
    sigmax = np.random.uniform(1,ncv/4,n)
    sigmay = np.random.uniform(1,nrv/4,n)

    #Combine all the gaussian functions to get the field

    v = np.zeros((nrv,ncv))
    for i in range(n):
        for x in range(ncv):
            for y in range(nrv):
                #We create three copies of our gaussian so that we get a pseudo-periodic field

                # Original Gaussian
                gauss  = np.exp(-((x-mux[i])**2/(2*sigmax[i]**2) + (y-muy[i])**2/(2*sigmay[i]**2)))
                # Shifted left
                gauss += np.exp(-((x-(mux[i]-ncv))**2/(2*sigmax[i]**2) + (y-muy[i])**2/(2*sigmay[i]**2)))
                # Shifted right
                gauss += np.exp(-((x-(mux[i]+ncv))**2/(2*sigmax[i]**2) + (y-muy[i])**2/(2*sigmay[i]**2)))

                v[y,x] += gauss 

    return v


def generate_circular_field(v):
    """
    Generates a circular field by taking the gradient of the input field and rotating it by 90 degrees.

    Args:
    v (np.ndarray): Input 2D field

    Returns:
    tuple: (grad_v_x, grad_v_y)
        grad_v_x (np.ndarray): X-component of the circular field
        grad_v_y (np.ndarray): Y-component of the circular field
    """
    grad_v_y, grad_v_x = np.gradient(v)

    return -grad_v_y, grad_v_x


def create_random_model(nr, nc, dt, F, 
                        num_gauss = 16,
                        DX_C = None, DY_C = None, DX_G = None, DY_G = None, RAC = None,
                        cyclic_east_west=True, cyclic_north_south=False):
    """
    Creates a random model with a new velocity field and diffusivity field.

    - Velocity field is generated from a field of circular patterns, in order to create a field
    with low divergence.

    - Diffusivity field is generated randomly, with 0 diffusivity on the boundaries.

    Args:
    nr (int): Number of rows in the grid
    nc (int): Number of columns in the grid
    dt (float): Time step
    F (float): Forcing coefficient
    num_gauss (int, optional): Number of Gaussian functions for velocity field generation. Defaults to 16.
    DX_C, DY_C, DX_G, DY_G, RAC (np.ndarray, optional): Grid spacing and area parameters. If None, set to arrays of ones.
    cyclic_east_west (bool, optional): If True, applies cyclic conditions east-west. Defaults to True.
    cyclic_north_south (bool, optional): If True, applies cyclic conditions north-south. Defaults to False.

    Returns:
    tuple: (M, params)
        M (np.ndarray): Model matrix for 2D diffusion-advection-forcing
        params (dict): Dictionary of parameters used to create the model

    """
    # If none, just set everything to appropriately-sized array of 1's
    if DX_C is None:
        DX_C = np.ones((nr, nc+1))
    if DY_C is None:
        DY_C = np.ones((nr+1, nc))
    if DX_G is None:
        DX_G = np.ones((nr+1, nc))
    if DY_G is None:
        DY_G = np.ones((nr, nc+1))
    if RAC is None:
        RAC = np.ones((nr, nc))
    


    # Randomly generate diffusivities: must be positive
    KX = np.random.rand(nr, nc+1)
    KY = np.random.rand(nr+1, nc)
    KX = np.abs(KX)
    KY = np.abs(KY)

    # Randomly generate velocities as above
    gauss = generate_gaussian_field(num_gauss,nr+1,nc+1)
    VX, VY = generate_circular_field(gauss)

    # Create the model matrix
    params = {
        'nr': nr,
        'nc': nc,
        'dt': dt,
        'KX': KX,
        'KY': KY,
        'DX_C': DX_C,
        'DY_C': DY_C,
        'DX_G': DX_G,
        'DY_G': DY_G,
        'VX': 100*VX[:-1,:],
        'VY': 100*VY[:,:-1],
        'RAC': RAC,
        'F': F,
        'cyclic_east_west':  cyclic_east_west,
        'cyclic_north_south': cyclic_north_south
    }

    M = make_M_2d_diffusion_advection_forcing(**params)

    return M, params

def create_random_initial_ocean_state(nr, nc, C, num_circles, radius, values):
    """
    Creates a random initial ocean state with specified covariance matrix.

    Args:
    nr (int): Number of rows in the grid
    nc (int): Number of columns in the grid
    C (np.ndarray): Covariance matrix for the initial state
    num_circles (int): Number of random circles to add
    radius (int): Radius of circles
    values (list): Possible values for circles

    Returns:
    tuple: (z, Lz)
        z (np.ndarray): Random initial state with covariance C
        Lz (np.ndarray): Random initial state with covariance
    """

    z = np.random.rand(nr,nc)
    z = add_random_circles(z, num_circles, radius, values)
    z = z.reshape((nr*nc,1))

    L = np.linalg.cholesky(C)
    Lz = L @ z

    return z, Lz

def generate_world(nr, nc, dt, F, num_gauss=16, num_circles=20, radius=5, values=[2,-2], std_dev=2):
    """
    Generates a world with an ocean state, atmosphere, and model matrix.

    Args:
    nr (int): Number of rows in the grid
    nc (int): Number of columns in the grid
    dt (float): Time step
    F (float): Forcing parameter
    num_gauss (int, optional): Number of Gaussian functions for velocity field generation. Defaults to 16.
    num_circles (int, optional): Number of circles for ocean state generation. Defaults to 20.
    radius (int, optional): Radius of circles for ocean state generation. Defaults to 5.
    values (list, optional): Values of circles for ocean state generation. Defaults to [2,-2].
    std_dev (int, optional): Standard deviation for Gaussian dropoff. Defaults to 2.

    Returns:
    tuple: (C, c0, f, M)
        C (np.ndarray): Covariance matrix
        c0 (np.ndarray): Initial ocean state
        f (np.ndarray): Atmosphere
        M (np.ndarray): Model matrix
    """
    # Generate covariance matrix
    C = compute_covariance_matrix_gaussian_dropoff(nr, nc, std_dev)

    # Generate model matrix
    M, params = create_random_model(nr, nc, dt, F, num_gauss=num_gauss)

    # Generate initial ocean state
    _, c0 = create_random_initial_ocean_state(nr, nc, C, num_circles=num_circles, radius=radius, values=values)

    # Generate atmosphere
    _, f = generate_random_vector_mean_0_cov_C(nr, nc, C)

    return C, c0, f, M

#C, c0, f, M = generate_world(50, 50, 0.1, 1, num_gauss=16, num_circles=20, radius=5, values=[2,-2], std_dev=2)

# Get magnitude of f
#f = f/3
#f_mag = np.linalg.norm(f)
#print(f'Magnitude of f: {f_mag:.2f}')


### Observe field ###

def observe(real, sigma, num_observations):
    """
    Generates noisy observations of a true state at randomly selected indices.

    Args:
    real (np.ndarray): The true state of the system, as a 1D array
    sigma (float): The standard deviation of the observation noise
    num_observations (int): The number of observations to make

    Returns:
    tuple: (indices, observations)
        indices (np.ndarray): Array of randomly selected indices for observation
        observations (np.ndarray): Noisy observations of the true state at the selected indices

    Notes:
    - Observations are made by adding Gaussian noise to the true state values
    - The noise is generated as a 2D column vector
    - Indices are selected without replacement, ensuring unique observation points
    """
    # We randomly select which indices to observe
    indices = np.random.choice(len(real), num_observations, replace=False)

    # We observe the true state plus Gaussian noise
    noise = np.random.normal(0, sigma, (num_observations, 1))
    
    observations = real[indices] + noise

    return indices, observations

def fill_nan_map_with_observations(indices, observations, nr, nc):
    """
    Maps observations to their corresponding positions in a 2D grid, filling the rest with NaNs.

    Args:
    indices (np.ndarray): Indices of the observations in the flattened grid
    observations (np.ndarray): Observed values
    nr (int): Number of rows in the grid
    nc (int): Number of columns in the grid

    Returns:
    np.ndarray: 2D array with observations at their corresponding positions and NaNs elsewhere
    """
    observed_state_2d = np.full((nr, nc), np.nan)
    observed_state_2d.flat[indices] = observations.flatten()
    return observed_state_2d

def interpolate_observation_map(observed_state_2d, extend=False):
    """
    Interpolates a 2D grid of observed values, to fill in NaN values (representing unobserved points).

    Args:
    observed_state_2d (np.ndarray): 2D array with observed values and NaNs
    extend (bool, optional): If True, extends the observed state by three copies horizontally. Defaults to False.

    Returns:
    np.ndarray: 2D array of interpolated values

    Notes:
    - If extend is True:
        - The observed state is extended by three copies horizontally
        - Interpolation is performed on the extended grid
        - The middle third of the interpolated result is returned
    - If extend is False:
        - Interpolation is performed on the original grid
    - Linear interpolation is used, with NaN values for points outside the convex hull of observations
    """
    nr, nc = observed_state_2d.shape

    if extend:
        # Extend state by three copies
        observed_state_2d = np.hstack((observed_state_2d, observed_state_2d, observed_state_2d))
    
    extended_nr, extended_nc = observed_state_2d.shape
    
    # Create grid coordinates
    x, y = np.meshgrid(np.arange(extended_nc), np.arange(extended_nr))
    
    # Find non-NaN indices and values
    observed_indices = np.where(~np.isnan(observed_state_2d.flatten()))[0]
    observed_values = observed_state_2d.flatten()[observed_indices]
    
    # Create points and grid for interpolation
    points = np.column_stack((x.flat[observed_indices], y.flat[observed_indices]))
    grid_x, grid_y = np.meshgrid(np.arange(extended_nc), np.arange(extended_nr))
    
    # Perform interpolation
    interpolated_2d = griddata(points, observed_values, (grid_x, grid_y), method='linear', fill_value=np.nan)
    
    if extend:
        # Extract the middle third
        return interpolated_2d[:, nc:2*nc]
    else:
        return interpolated_2d


def observe_over_time(ocean_states, sigma, num_obs_per_timestep, nr, nc):
    """
    Observes the ocean state at each timestep, and places those observations in
    a 2d array with NaNs for unobserved points.

    Args:
    ocean_states (list): List of ocean states at each timestep
    sigma (float): Standard deviation of observation noise
    num_obs_per_timestep (int): Number of observations per timestep
    nr (int): Number of rows in the grid
    nc (int): Number of columns in the grid

    Returns:
    list: List of observed ocean states at each timestep, each as a 2D array with NaNs for unobserved points

    """
    # Observe ocean state at each timestep
    indices_and_observations_over_time = [observe(ocean_state, sigma, num_obs_per_timestep) 
                                            for ocean_state in ocean_states]
    
    # Place on a map: unobserved points are filled with NaN
    observed_state_over_time_2d = [fill_nan_map_with_observations(indices, observations_t, nr, nc) 
                                    for indices, observations_t in indices_and_observations_over_time]
    
    return observed_state_over_time_2d




### Compute Adjoints ###

def compute_Jt(xt_true, xt_guess): 
    """
    Computes squared loss between two vectors at time t.
    
    Args:
    xt_true (np.ndarray): True state vector at time t
    xt_guess (np.ndarray): Guessed state vector at time t
    
    Returns:
    float: Squared loss, or 0 if no valid terms
    """
    
    # Sum over all valid terms, using numpy to treat nans as zeros
    result = np.nansum((xt_true - xt_guess)**2)
    if np.isnan(result): 
        return 0
    else: 
        return result

def compute_J(x_true, x_guess): 
    """
    Computes total squared loss between two vectors across all timesteps.
    
    Args:
    x_true (list): List of true state vectors at each timestep
    x_guess (list): List of guessed state vectors at each timestep
    
    Returns:
    float: Total squared loss across all timesteps
    """
    return np.sum([
                compute_Jt(x_true[i], x_guess[i]) for i in range(len(x_true))]
                )

def compute_DJ_Dxt(xt_true, xt_guess):
    """
    Computes partial derivative of squared loss w.r.t. guessed state at time t.
    
    Args:
    xt_true (np.ndarray): True state vector at time t
    xt_guess (np.ndarray): Guessed state vector at time t
    
    Returns:
    np.ndarray: Partial derivative of loss, with NaNs treated as 0
    """
    return np.nan_to_num( 2*(xt_guess - xt_true), nan = 0 )


def compute_adjoints(DJ_Dx, dxtp1_dxt):
    """
    Computes adjoint variables for optimization using backwards-time recursion.

    Args:
    DJ_Dx (list): List of partial derivatives of loss w.r.t. state at each timestep
    dxtp1_dxt (list): List of total derivatives of next state w.r.t. current state at each timestep

    Returns:
    list: Adjoint variables for each timestep, in forward time order
    """

    tau = len(DJ_Dx)
    adjoints = [0] * tau # Initialize list of adjoints

    adjoints[tau-1] = DJ_Dx[tau-1]
    
    for t in range(tau-2, -1, -1):  # Backwards in time
        adjoint = DJ_Dx[t] + dxtp1_dxt[t].dot( adjoints[t+1] )

        adjoints[t] = adjoint

    return adjoints

def compute_dJ_df(M, F, observed_state_over_time, simulated_state_over_time):
    """
    Computes the gradient of the loss with respect to the forcing field f for the linear model:
    x(t+1) = Mx(t) + Ff

    Args:
    M (np.ndarray): Model matrix
    F (float): Forcing coefficient
    observed_state_over_time (list): List of observed states at each timestep
    simulated_state_over_time (list): List of simulated states at each timestep

    Returns:
    np.ndarray: Gradient of the loss with respect to the forcing field f
    """
    num_timesteps = len(observed_state_over_time)
    vec_length = len(observed_state_over_time[0])

    #Compute adjoints
    DJ_Dx = [compute_DJ_Dxt(observed_state_over_time[i], simulated_state_over_time[i])
             for i in range(num_timesteps)] # partial J / partial x(t)
    
    dxtp1_dxt = [M.T for i in range(num_timesteps-1)] #dx(t+1)/dx(t)

    adjoints = compute_adjoints(DJ_Dx, dxtp1_dxt) # dJ/dx(t) = lambda(t)

    # Compute gradient for each timestep: how f being applied at time t affects J
    dJ_dft = [ F * adjoint for adjoint in adjoints[1:] ] # dJ/df(t) = dx(t+1)/df(t) dJ/dx(t+1)
    dJ_dft.append(np.zeros((vec_length,1))) # dJ/df(tau) = 0 

    #f is applied the same at all timesteps
    dJ_df = np.sum(dJ_dft, axis=0) # dJ/df = sum_t dJ/df(t)
    return dJ_df

### Gradient Descent ###

losses_template = { #Losses at each iteration
        "ocean_misfit": [],
        "atmosphere_misfit": [],
        "mahalanobis(covariance similarity)": [],
    }


def update_losses(losses, ocean_states_observed, ocean_states_simulated, f_guess, f_adjust, f_true, C_error):
    """
    Updates the losses dictionary with new loss values for ocean, atmosphere, and control adjustment.

    Args:
    losses (dict): Dictionary containing lists of loss values
    ocean_states_observed (list): List of observed ocean states
    ocean_states_simulated (list): List of simulated ocean states
    f_guess (np.ndarray): Initial guess for the atmospheric forcing field
    f_adjust (np.ndarray): Adjustment to the atmospheric forcing field
    f_true (np.ndarray): True atmospheric forcing field
    C_error (np.ndarray): Covariance matrix for the control error

    Returns:
    dict: Updated losses dictionary with new loss values appended
    """
    f_i = f_guess + f_adjust

    # Compute losses
    ocean_loss_i = compute_J(ocean_states_observed, ocean_states_simulated) # J_{ocean} = J
    atmos_loss_i = compute_Jt(f_true, f_i)                                  # J_{atm} = misfit of atm
    
    mahal = f_adjust.T @ np.linalg.inv(C_error) @ f_adjust # C_error should be the covariance of our adjustment
    mahal_loss_i = np.linalg.norm(mahal)

    #Store losses
    losses["ocean_misfit"].append(ocean_loss_i)
    losses["atmosphere_misfit"].append(atmos_loss_i)
    losses["mahalanobis(covariance similarity)"].append(mahal_loss_i)

    return losses

possible_debug_vars = { #Debug variables to compute at each iteration
    "Norm of s_i": [],
    "Expected Delta J w simple gd": [],
    "Expected Delta J w update rule": [],
    "Norm of simple gd ui": [],
    "Norm of update rule ui": [],
    "Normalized dot product $a_i$ and $u_i$": [],
    "$s_i^T Cs_i$": [],
    "$a_i$": [],
    "Normalized $a_i^T s_i$": [],
    "Normalized $-Cs_i \cdot a_i$": [],
    "Norm of $a_i$": [],
    "Actual Delta J": []
}

def update_debug_vars(debug_vars, x0, M, F, f_guess, f_adjust, 
                      C_known, C_error, 
                      s, step_size, ui, 
                      ocean_states_simulated, ocean_states_observed):
    """
    Updates the debug variables dictionary with various metrics for gradient descent analysis.

    Args:
    debug_vars (dict): Dictionary containing lists of debug variable values
    x0 (np.ndarray): Initial ocean state
    M (np.ndarray): Model matrix
    F (float): Forcing coefficient
    f_guess (np.ndarray): Initial guess for the atmospheric forcing field
    f_adjust (np.ndarray): Adjustment to the atmospheric forcing field
    C_known (np.ndarray): Covariance matrix for the known portion of the control
    C_error (np.ndarray): Covariance matrix for the control error
    s (np.ndarray): Gradient of the loss with respect to the forcing field
    step_size (float): Step size for gradient descent
    ui (np.ndarray): Update vector for the current iteration
    ocean_states_simulated (list): List of simulated ocean states
    ocean_states_observed (list): List of observed ocean states

    Returns:
    dict: Updated debug_vars dictionary with new values appended to each metric
    """
    #Initialize useful variables
    num_timesteps = len(ocean_states_observed)
    ui_simple_gd = -step_size * s
    delta = s.T @(ui_simple_gd)

    #Compute debug vars
    norm_s = np.linalg.norm(s)
    exp_delta_J_simple_gd = (s.T @ ui_simple_gd)[0,0]
    exp_delta_J_update_rule = (s.T @ ui)[0,0]
    norm_simple_ui = np.linalg.norm(ui_simple_gd)
    norm_ui = np.linalg.norm(ui)
    norm_dot_product = (f_adjust.T @ ui)[0,0] / (np.linalg.norm(f_adjust) * np.linalg.norm(ui))
    sTCs = (s.T @ C_error @ s)[0,0]
    ai = f_adjust
    normalized_aiTs = (f_adjust.T @ s)[0,0] / (np.linalg.norm(f_adjust) * np.linalg.norm(s))
    normalized_Csdotai = (f_adjust.T @ (C_error @ s))[0,0] / (np.linalg.norm(f_adjust) * np.linalg.norm(C_error @ s))
    norm_ai = np.linalg.norm(f_adjust)

    #Store debug vars
    debug_vars["Norm of s_i"].append(norm_s)
    debug_vars["Expected Delta J w simple gd"].append(exp_delta_J_simple_gd)
    debug_vars["Expected Delta J w update rule"].append(exp_delta_J_update_rule)
    debug_vars["Norm of simple gd ui"].append(norm_simple_ui)
    debug_vars["Norm of update rule ui"].append(norm_ui)
    debug_vars["Normalized dot product $a_i$ and $u_i$"].append(norm_dot_product)
    debug_vars["$s_i^T Cs_i$"].append(sTCs)
    debug_vars["$a_i$"].append(ai)
    debug_vars["Normalized $a_i^T s_i$"].append(normalized_aiTs)
    debug_vars["Normalized $-Cs_i \cdot a_i$"].append(-normalized_Csdotai)
    debug_vars["Norm of $a_i$"].append(norm_ai)

    #Handle last debug var: Actual Delta J
    f_new = f_guess + f_adjust + ui
    _, new_ocean_states_simulated = compute_affine_time_evolution_simple(x0, M, F*f_new, num_timesteps)

    new_J = compute_J(ocean_states_observed, new_ocean_states_simulated)
    old_J = compute_J(ocean_states_observed, ocean_states_simulated)

    actual_delta_J = new_J - old_J
    
    debug_vars["Actual Delta J"].append(actual_delta_J)
    
    return debug_vars


def gradient_descent_template(M, F, f_true, f_guess, C_known, C_error,         # World parameters
                              x0, num_timesteps,                                    # Simulation parameters
                              ocean_states_observed, num_iters, step_size,      # Optimization parameters
                              update_rule, update_params, disp=False):
    """
    Perform gradient descent to optimize the atmospheric forcing field. 
    
    The step we take at each iteration is computed using a modified update rule, and extra parameters as necessary.

    Args:
        M: Model matrix
        F: Scalar constant for forcing
        f_true: True atmospheric forcing field
        f_guess: Initial guess for the atmospheric forcing field
        x0: Initial ocean state
        num_timesteps: Number of timesteps
        ocean_states_observed: Observed ocean states
        num_iters: Number of iterations
        step_size: Step size for gradient descent
        C_known: Covariance matrix for the known portion of the control
        C_error: Covariance matrix for the control error
        update_params: Function to compute the update rule to take at each iteration
        extra_params: Extra parameters to pass to the update rule
        disp: Flag to print information

    Returns:
        f: Optimized atmospheric forcing field
        losses: Dictionary of losses
            ocean_misfit: Ocean loss at each iteration
            atmosphere: Atmospheric loss at each iteration
            $a_iC^{-1}a_i$: Mahalanobis distance for the control adjustment at each iteration
    """
    size = f_guess.shape[0]
    f_adjust = np.zeros((size,1))

    losses = copy.deepcopy(losses_template)
    debug_vars = copy.deepcopy(possible_debug_vars)

    for i in range(num_iters):
        if i%10==0 and disp:
            print("Iteration", i)
        
        #Compute results of previous update rule
        f_i = f_guess + f_adjust #f_i = f_0 + a_i
        _, ocean_states_simulated = compute_affine_time_evolution_simple(x0, M, F*f_i, num_timesteps)
        

        # Compute and store losses
        losses = update_losses(losses, ocean_states_observed, ocean_states_simulated, f_guess, f_adjust, f_true, C_error)
        
        # Compute and store debug variables
        
        s = compute_dJ_df(M, F, ocean_states_observed, ocean_states_simulated)
        ui = update_rule(i, s, step_size, f_adjust, *update_params) #Update rule

        debug_vars = update_debug_vars(debug_vars, x0, M, F, f_guess, f_adjust, 
                                       C_known, C_error, 
                                       s, step_size, ui, 
                                       ocean_states_simulated, ocean_states_observed)

        # Apply update rule to f_adjust

        f_adjust = f_adjust + ui

        

    return f_adjust, losses, debug_vars


def simple_gradient_update_rule(curr_iter, s, step_size, f_adjust):
    """
    Computes the update step for simple gradient descent.

    Args:
    curr_iter (int): Current iteration number (unused in this function)
    s (np.ndarray): Gradient of the loss with respect to the forcing field
    step_size (float): Step size for gradient descent
    f_adjust (np.ndarray): Current adjustment to the forcing field (unused in this function)

    Returns:
    np.ndarray: Update step for the forcing field adjustment
    """
    return -step_size * s  # Just use the gradient of the loss

def simple_gradient_descent(M, F, f_true, f_guess, C_known, C_error,       # World parameters
                            x0, timesteps,                                   # Simulation parameters
                            ocean_states_observed, num_iters, step_size,      # Optimization parameters
                            disp=False):                                      # Optimization method
    """
    Implements simple gradient descent for optimizing the atmospheric forcing field.

    Args:
    M (np.ndarray): Model matrix
    F (float): Forcing coefficient
    f_true (np.ndarray): True atmospheric forcing field
    f_guess (np.ndarray): Initial guess for the atmospheric forcing field
    C_known (np.ndarray): Covariance matrix for the known portion of the control
    C_error (np.ndarray): Covariance matrix for the control error
    x0 (np.ndarray): Initial ocean state
    timesteps (int): Number of timesteps for simulation
    ocean_states_observed (list): List of observed ocean states
    num_iters (int): Number of iterations for gradient descent
    step_size (float): Step size for gradient descent
    disp (bool, optional): If True, display progress. Defaults to False.

    Returns:
    tuple: (f_adjust, losses, debug_vars)
        f_adjust (np.ndarray): Final adjustment to the atmospheric forcing field
        losses (dict): Dictionary of loss values over iterations
        debug_vars (dict): Dictionary of debug variables over iterations
    """
    return gradient_descent_template(M, F, f_true, f_guess, C_known, C_error, 
                                     x0, timesteps, 
                                     ocean_states_observed, num_iters, step_size,
                                     simple_gradient_update_rule, [], disp)



def cholesky_update_rule(curr_iter, s, step_size, f_adjust, C_error):
    """
    Computes the update step using Cholesky decomposition of the error covariance matrix.

    Args:
    curr_iter (int): Current iteration number (unused in this function)
    s (np.ndarray): Gradient of the loss with respect to the forcing field
    step_size (float): Step size for gradient descent
    f_adjust (np.ndarray): Current adjustment to the forcing field (unused in this function)
    C_error (np.ndarray): Covariance matrix for the control error

    Returns:
    np.ndarray: Update step for the forcing field adjustment

    Notes:
    Applies the Cholesky decomposition L of C_error to s, then rescales the result 
    to match the original gradient's magnitude.
    """
    # Apply cholesky decomposition L : C_error = L @ L.T
    L = np.linalg.cholesky(C_error)
    cholesky_s = L @ s

    # Rescale so magnitude is the same
    rescaled_cholesky_s = cholesky_s * (np.linalg.norm(s) / np.linalg.norm(cholesky_s))
    step = - step_size * rescaled_cholesky_s

    return step

def cholesky_gradient_descent(M, F, f_true, f_guess, C_known, C_error,       # World parameters
                              x0, timesteps,                                 # Simulation parameters
                              ocean_states_observed, num_iters, step_size,   # Optimization parameters
                              disp=False):                                   # Optimization method
    """
    Implements gradient descent using Cholesky decomposition for optimizing the atmospheric forcing field.

    Args:
    M (np.ndarray): Model matrix
    F (float): Forcing coefficient
    f_true (np.ndarray): True atmospheric forcing field
    f_guess (np.ndarray): Initial guess for the atmospheric forcing field
    C_known (np.ndarray): Covariance matrix for the known portion of the control
    C_error (np.ndarray): Covariance matrix for the control error
    x0 (np.ndarray): Initial ocean state
    timesteps (int): Number of timesteps for simulation
    ocean_states_observed (list): List of observed ocean states
    num_iters (int): Number of iterations for gradient descent
    step_size (float): Step size for gradient descent
    disp (bool, optional): If True, display progress. Defaults to False.

    Returns:
    tuple: (f_adjust, losses, debug_vars)
        f_adjust (np.ndarray): Final adjustment to the atmospheric forcing field
        losses (dict): Dictionary of loss values over iterations
        debug_vars (dict): Dictionary of debug variables over iterations
    """
    return gradient_descent_template(M, F, f_true, f_guess, C_known, C_error,
                                     x0, timesteps, 
                                     ocean_states_observed, num_iters, step_size,
                                     cholesky_update_rule, [C_error], disp)


def cov_constraint_J_update_rule(curr_iter, s, step_size, f_adjust, C_error, weight_cov_term):
    """
    Computes the update step using a covariance constraint on the loss function.

    Args:
    curr_iter (int): Current iteration number (unused in this function)
    s (np.ndarray): Gradient of the loss with respect to the forcing field
    step_size (float): Step size for gradient descent
    f_adjust (np.ndarray): Current adjustment to the forcing field
    C_error (np.ndarray): Covariance matrix for the control error
    weight_cov_term (float): Weight for the covariance constraint term

    Returns:
    np.ndarray: Update step for the forcing field adjustment

    Notes:
    Adds a weighted covariance constraint term to the original gradient,
    then rescales the result to match the original gradient's magnitude.
    """
    cov_term_grad = 2 * np.linalg.inv(C_error) @ f_adjust  # Covariance term

    s_prime = s + weight_cov_term * cov_term_grad  # Gradient of J' with respect to the forcing field

    norm_s_prime = s_prime * (np.linalg.norm(s) / np.linalg.norm(s_prime))  # Rescale magnitude to match original
    
    return -step_size * norm_s_prime

def cov_constraint_J_gradient_descent(M, F, f_true, f_guess, C_known, C_error,       # World parameters
                                      x0, timesteps,                                 # Simulation parameters
                                      ocean_states_observed, num_iters, step_size,   # Optimization parameters
                                      weight_cov_term, disp=False):                  # Optimization method
    """
    Implements gradient descent with a covariance constraint for optimizing the atmospheric forcing field.

    Args:
    M (np.ndarray): Model matrix
    F (float): Forcing coefficient
    f_true (np.ndarray): True atmospheric forcing field
    f_guess (np.ndarray): Initial guess for the atmospheric forcing field
    C_known (np.ndarray): Covariance matrix for the known portion of the control
    C_error (np.ndarray): Covariance matrix for the control error
    x0 (np.ndarray): Initial ocean state
    timesteps (int): Number of timesteps for simulation
    ocean_states_observed (list): List of observed ocean states
    num_iters (int): Number of iterations for gradient descent
    step_size (float): Step size for gradient descent
    weight_cov_term (float): Weight for the covariance constraint term
    disp (bool, optional): If True, display progress. Defaults to False.

    Returns:
    tuple: (f_adjust, losses, debug_vars)
        f_adjust (np.ndarray): Final adjustment to the atmospheric forcing field
        losses (dict): Dictionary of loss values over iterations
        debug_vars (dict): Dictionary of debug variables over iterations
    """
    return gradient_descent_template(M, F, f_true, f_guess, C_known, C_error,
                                     x0, timesteps, 
                                     ocean_states_observed, num_iters, step_size,
                                     cov_constraint_J_update_rule, [C_error, weight_cov_term], disp)

def dan_update_rule(curr_iter, s, step_size, f_adjust, C_error):
    """
    Computes the update step using Dan's method for improving the Mahalanobis distance.

    Args:
    curr_iter (int): Current iteration number (unused in this function)
    s (np.ndarray): Gradient of the loss with respect to the forcing field
    step_size (float): Step size for gradient descent
    f_adjust (np.ndarray): Current adjustment to the forcing field (unused in this function)
    C_error (np.ndarray): Covariance matrix for the control error

    Returns:
    np.ndarray: Update step for the forcing field adjustment

    Notes:
    Modifies the gradient direction to improve the Mahalanobis distance of u_i (update rule) while
    maintaining the desired improvement in the loss function J.
    """
    ui_simple_gd = -step_size * s  # Pre-dan step
    delta = s.T @ (ui_simple_gd)   # Compute desired improvement of J
    
    new_vec = (C_error @ s) / (s.T @ C_error @ s)  # Direction modified to improve Mahalanobis distance

    return delta * new_vec

def dan_gradient_descent(M, F, f_true, f_guess, C_known, C_error,       # World parameters
                         x0, timesteps,                                 # Simulation parameters
                         ocean_states_observed, num_iters, step_size,   # Optimization parameters
                         disp=False):                                   # Optimization method
    """
    Implements gradient descent using Dan's method for optimizing the atmospheric forcing field.

    Args:
    M (np.ndarray): Model matrix
    F (float): Forcing coefficient
    f_true (np.ndarray): True atmospheric forcing field
    f_guess (np.ndarray): Initial guess for the atmospheric forcing field
    C_known (np.ndarray): Covariance matrix for the known portion of the control
    C_error (np.ndarray): Covariance matrix for the control error
    x0 (np.ndarray): Initial ocean state
    timesteps (int): Number of timesteps for simulation
    ocean_states_observed (list): List of observed ocean states
    num_iters (int): Number of iterations for gradient descent
    step_size (float): Step size for gradient descent
    disp (bool, optional): If True, display progress. Defaults to False.

    Returns:
    tuple: (f_adjust, losses, debug_vars)
        f_adjust (np.ndarray): Final adjustment to the atmospheric forcing field
        losses (dict): Dictionary of loss values over iterations
        debug_vars (dict): Dictionary of debug variables over iterations
    """
    return gradient_descent_template(M, F, f_true, f_guess, C_known, C_error,
                                     x0, timesteps, 
                                     ocean_states_observed, num_iters, step_size,
                                     dan_update_rule, [C_error], disp)


def dan_modified_update_rule(curr_iter, s, step_size, f_adjust, C_error):
    """
    Computes the update step using a modified version of Dan's method for improving the Mahalanobis distance.

    Args:
    curr_iter (int): Current iteration number (unused in this function)
    s (np.ndarray): Gradient of the loss with respect to the forcing field
    step_size (float): Step size for gradient descent
    f_adjust (np.ndarray): Current adjustment to the forcing field
    C_error (np.ndarray): Covariance matrix for the control error

    Returns:
    np.ndarray: Update step for the forcing field adjustment

    Notes:
    Modifies the gradient direction to improve the Mahalanobis distance while
    maintaining the desired improvement in the loss function J. This version
    intends to improve Mahalanobis distance of a_i+u_i (f_adjust+gradient update), instead of just u_i.
    """
    ui_simple_gd = -step_size * s  # Pre-dan step
    delta = s.T @ (ui_simple_gd)   # Compute desired improvement of J
    
    new_vec = (C_error @ s) / (s.T @ C_error @ s)  # Direction modified to improve Mahalanobis distance
    vec_scale = delta + s.T @ f_adjust

    return -f_adjust + vec_scale * new_vec

def dan_modified_gradient_descent(M, F, f_true, f_guess, C_known, C_error,       # World parameters
                                  x0, timesteps,                                 # Simulation parameters
                                  ocean_states_observed, num_iters, step_size,   # Optimization parameters
                                  disp=False):                                   # Optimization method
    """
    Implements gradient descent using a modified version of Dan's method for optimizing the atmospheric forcing field.

    Args:
    M (np.ndarray): Model matrix
    F (float): Forcing coefficient
    f_true (np.ndarray): True atmospheric forcing field
    f_guess (np.ndarray): Initial guess for the atmospheric forcing field
    C_known (np.ndarray): Covariance matrix for the known portion of the control
    C_error (np.ndarray): Covariance matrix for the control error
    x0 (np.ndarray): Initial ocean state
    timesteps (int): Number of timesteps for simulation
    ocean_states_observed (list): List of observed ocean states
    num_iters (int): Number of iterations for gradient descent
    step_size (float): Step size for gradient descent
    disp (bool, optional): If True, display progress. Defaults to False.

    Returns:
    tuple: (f_adjust, losses, debug_vars)
        f_adjust (np.ndarray): Final adjustment to the atmospheric forcing field
        losses (dict): Dictionary of loss values over iterations
        debug_vars (dict): Dictionary of debug variables over iterations
    """
    return gradient_descent_template(M, F, f_true, f_guess, C_known, C_error,
                                     x0, timesteps, 
                                     ocean_states_observed, num_iters, step_size,
                                     dan_modified_update_rule, [C_error], disp)


### Gradient Descent Testing ###

def compare_gd_methods_once(M, F, f_true, f_guess, C_known, C_error, 
                            x0, timesteps, 
                            ocean_states_observed, num_iters, step_size, 
                            methods, disp=False):
    """
    Runs multiple methods of gradient descent on the same dataset and compares their performance.

    Args:
    M (np.ndarray): Model matrix
    F (float): Forcing coefficient
    f_true (np.ndarray): True atmospheric forcing field
    f_guess (np.ndarray): Initial guess for the atmospheric forcing field
    C_known (np.ndarray): Covariance matrix for the known portion of the control
    C_error (np.ndarray): Covariance matrix for the control error
    x0 (np.ndarray): Initial ocean state
    timesteps (int): Number of timesteps for simulation
    ocean_states_observed (list): List of observed ocean states
    num_iters (int): Number of iterations for gradient descent
    step_size (float): Step size for gradient descent
    methods (list): List of gradient descent methods to compare. Each method is a list of the form
                    ["Method Name", method_func, extra_params]
    disp (bool, optional): If True, display progress. Defaults to False.

    Returns:
    dict: A dictionary where keys are method names and values are tuples containing:
          - f_adjust (np.ndarray): Final adjustment to the atmospheric forcing field
          - losses (dict): Dictionary of loss values over iterations
          - debug_vars (dict): Dictionary of debug variables over iterations

    Notes:
    This function applies each specified gradient descent method to the same initial conditions
    and dataset, allowing for direct comparison of their performance.
    """
    results = {}

    for method_name, method_func, extra_params in methods:
        if disp:
            print(f"Running method {method_name}")

        f_adjust, losses, debug_vars = gradient_descent_template(M, F, f_true, f_guess, C_known, C_error, 
                                                     x0, timesteps, 
                                                     ocean_states_observed, num_iters, step_size,
                                                     method_func, extra_params, disp)
        results[method_name] = (f_adjust, losses, debug_vars)

    return results

def compare_gd_methods_many_times(nr, nc, dt, F, gamma, sigma, num_obs_per_timestep, 
                                  num_timesteps, num_iters, step_size, 
                                  C_known, C_error, methods, num_runs, disp=False):
    """
    Create many different sets of data. 
    For each one, we will run each of our gradient descent methods.
    Once we finish, we average losses across all runs.

    Args:
    nr (int): Number of rows in the grid
    nc (int): Number of columns in the grid
    dt (float): Time step
    F (float): Forcing parameter
    gamma (float): Proportion of the control vector that is correct
    sigma (float): Standard deviation of observation noise
    num_obs_per_timestep (int): Number of observations per timestep
    num_timesteps (int): Number of timesteps
    num_iters (int): Number of iterations of gradient descent
    step_size (float): Step size for gradient descent
    methods (list): List of gradient descent methods to compare
    num_runs (int): Number of times to run the whole optimization process
    disp (bool): If True, print progress
    
    Returns:
    dict: Dictionary containing averaged losses and debug variables for each method
    """

    # Initialize results dictionary
    losses =     {method[0]: copy.deepcopy(losses_template) 
                for method in methods}
    debug_vars = {method[0]: copy.deepcopy(possible_debug_vars) 
                  for method in methods}

    for run in range(num_runs):
        if disp:
            print(f"Run {run + 1}/{num_runs}")

        # Generate world
        C_control, x0, _, M = generate_world(nr, nc, dt, F)
        C_known, C_error = C_control * gamma, C_control * (1-gamma)
        
        f_true, f_guess = generate_true_and_first_guess_field(C_known, C_error, nr, nc)

        # Run the simulation with the true and guessed control vector
        _, real_state_over_time  = compute_affine_time_evolution_simple(x0, M, F*f_true,  num_timesteps)

        observed_state_over_time_2d = observe_over_time(real_state_over_time, sigma, 
                                                               num_obs_per_timestep, nr, nc)

        observed_state_over_time = [np.reshape(observed_state_2d, (nr*nc, 1)) 
                                    for observed_state_2d in observed_state_over_time_2d]

        # Run each method
        for method_name, method_func, extra_params in methods:
            if disp:
                print(f"  Method: {method_name}")

            results = compare_gd_methods_once(M, F, f_true, f_guess, C_known, C_error,
                                                x0, num_timesteps, 
                                                observed_state_over_time, num_iters, step_size, 
                                                [[method_name, method_func, extra_params]], disp)
            
            # Include new losses
            for method_name, (_, method_losses, method_debug_vars) in results.items():
                for loss_name, loss_list in losses[method_name].items():
                    loss_list.append(method_losses[loss_name])

                for debug_name, debug_list in debug_vars[method_name].items():
                    debug_list.append(method_debug_vars[debug_name])

    # Average losses

    for method_name, method_losses in losses.items():
        for loss_name, loss_list in method_losses.items():
            losses[method_name][loss_name] = np.mean(loss_list, axis=0)

    for method_name, method_debug_vars in debug_vars.items():
        for debug_name, debug_list in method_debug_vars.items():
            debug_vars[method_name][debug_name] = np.mean(debug_list, axis=0)

    return losses, debug_vars

### Gradient Descent Visualization ###

def plot_losses(losses_many, num_obs_per_timestep, step_size, num_timesteps, num_iters, min_iter=None, max_iter=None):
    """
    Plots the losses for multiple gradient descent methods over iterations.

    Args:
    losses_many (dict): Dictionary of losses for each method. 
                        Keys are method names, values are dictionaries containing losses.
    num_obs_per_timestep (int): Number of observations per timestep
    step_size (float): Step size used in gradient descent
    num_timesteps (int): Number of timesteps in the simulation
    num_iters (int): Number of iterations of gradient descent
    min_iter (int): Lowest plotted iter (default: None, plots from the beginning)
    max_iter (int): Highest plotted iter (default: None, plots until the end)

    Returns:
    None: This function displays the plot using matplotlib.pyplot.show()

    Notes:
    Creates a 2x2 grid of plots:
    1. Ocean misfit
    2. Atmosphere loss
    3. Control adjust Mahalanobis distance
    4. J' (combined loss for covariance constraint method, ocean loss for others)

    Each plot shows the evolution of the respective loss over iterations for all methods.
    """
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    loss_funcs = ["$\sum_t (Ex(t)-y(t))^{T} (Ex(t)-y(t))$", 
                  "$\sum_t (f_i(t)-f_{true}(t) )^{T} ( f_i(t)-f_{true}(t) )$", 
                  "$a_i^T C^{-1} a_i$"]

    # Determine the range of iterations to plot
    min_iter = 0 if min_iter is None else max(0, min_iter)
    max_iter = num_iters if max_iter is None else min(num_iters, max_iter)
    plot_range = slice(min_iter, max_iter)

    for i, (loss_name, ax, func) in enumerate(zip(["ocean_misfit", "atmosphere_misfit", "mahalanobis(covariance similarity)"], axs.flatten(), loss_funcs)):
        for method_name, losses_dict in losses_many.items():
            ax.plot(range(min_iter, max_iter), losses_dict[loss_name][plot_range], label=method_name)
        ax.set_xlabel("Iteration $i$")
        ax.set_ylabel(loss_name+" loss:    "+func)
        ax.legend()
        ax.set_title(f"{loss_name}: "+func)
        
        # Set integer ticks on x-axis
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Fourth plot: ocean_misfit + mahalanobis if using covariance control adjust, just ocean otherwise
    ax = axs[1, 1]
    for method_name, losses_dict in losses_many.items():
        if method_name == r"Covariance Constraint J Gradient Descent":
            combined_loss = [o + c for o, c in zip(losses_dict["ocean_misfit"], losses_dict["mahalanobis(covariance similarity)"])]
            ax.plot(range(min_iter, max_iter), combined_loss[plot_range], label=method_name)
        else:
            ax.plot(range(min_iter, max_iter), losses_dict["ocean_misfit"][plot_range], label=method_name)

    ax.set_xlabel("Iteration $i$")
    ax.set_ylabel("J'")
    ax.legend()
    ax.set_title("J'")
    
    # Set integer ticks on x-axis for the fourth plot
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    fig.suptitle(f"Gradient Descent Variants: num_obs={num_obs_per_timestep}, step_size={step_size}, num_timesteps={num_timesteps}, num_iters={num_iters}")

    plt.tight_layout()
    plt.show()

def plot_debug(debug_vars, min_iter=None, max_iter=None, tickwidth=1, vlines = []):
    """
    Plot 9 chosen debug variables in a 3x3 grid for each method.

    Args:
    debug_vars (dict): Dictionary containing debug variables for each method
    min_iter (int, optional): Start index for plotting. If None, starts from the beginning.
    max_iter (int, optional): End index for plotting. If None, plots until the end.
    tickwidth (int): Width between ticks on x-axis
    vlines (list): List of vertical lines to add to the plot

    Returns:
    None: Displays the plot
    """
    # Determine the actual range of iterations
    all_iters = next(iter(debug_vars.values()))["Norm of s_i"]
    total_iters = len(all_iters)

    # Set min_iter and max_iter if they are None
    min_iter = 0 if min_iter is None else max(0, min_iter)
    max_iter = total_iters if max_iter is None else min(total_iters, max_iter)

    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(f"Debug Variables: Iterations {min_iter} to {max_iter}")

    # List of debug variables to plot
    plot_vars = [
        "Norm of s_i",
        "Expected Delta J w simple gd",
        "Expected Delta J w update rule",
        "Actual Delta J",
        "Norm of simple gd ui",
        "Norm of update rule ui",
        "Normalized dot product $a_i$ and $u_i$",
        "Normalized $-Cs_i \cdot a_i$",
        "Norm of $a_i$",
    ]

    for i, (var_name, ax) in enumerate(zip(plot_vars, axs.flatten())):
        for method_name, method_debug_vars in debug_vars.items():
            if var_name in method_debug_vars:
                plot_data = method_debug_vars[var_name][min_iter:max_iter]
                ax.plot(range(min_iter, max_iter), plot_data, ".-", label=method_name)
        
        ax.set_xlabel("Iteration $i$")
        ax.set_ylabel(var_name)
        ax.legend()
        ax.set_title(var_name)
        ax.grid(True)
        
        # Set x-axis ticks to reflect the actual iteration numbers
        ticks = range(min_iter, max_iter, tickwidth)
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks)

        # Add vertical lines if specified
        for vline in vlines:
            if min_iter <= vline < max_iter:
                ax.axvline(x=vline, color='r', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()