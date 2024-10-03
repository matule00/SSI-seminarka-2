
import pandas as pd
import numpy as np
import math as math
import random as rn
from matplotlib import pyplot as plt
import matplotlib as mtl
from fitter import Fitter
from scipy import stats
from operator import le
import multiprocessing as mp

pd.options.mode.chained_assignment = None  # default='warn' # to supress "A value is trying to be set on a copy of a slice from a DataFrame"

def init_ped_data(t,x,y):
    # initialize ped data container
    # INPUT: vector of init time, init positions

    # init dataframe with the first ped
    ped_data = pd.DataFrame({'ped_id': 0,
                             't': [[t[0]]],                        # to be list of recorded time
                             'x': [[x[0]]],                        # to be list of recorded x position
                             'y': [[y[0]]],                         # to be list of recorded y position
                             'dec_x': np.nan,
                             'dec_y': np.nan
                             }, index = [0])

    # add peds one by one
    rep = range(len(x)-1)
    for i in rep:
        ped_data_n = pd.DataFrame({'ped_id': i+1,
                                't': [[t[i+1]]],
                                'x': [[x[i+1]]],
                                'y': [[y[i+1]]],
                                }, index = [i+1])
        ped_data = pd.concat([ped_data, ped_data_n], ignore_index=True)

    return ped_data


def one_ped_decision(ped_data, ped_idx, distance_grid, const):

    # If ped is in the leaving tourniquet, leave the system with desired probability
    if ped_data.x[ped_idx][-1] == 1 and ped_data.y[ped_idx][-1] == 13:
        r = np.random.rand()
        if r < const['p']:
            ped_data.dec_x[ped_idx] = -1
            ped_data.dec_y[ped_idx] = -1

            # print('   Ped ' + str(ped_idx) + ' choosed to leave')
        else:
            ped_data.dec_x[ped_idx] = np.nan
            ped_data.dec_y[ped_idx] = np.nan

    # If ped already left, skip it
    elif ped_data.x[ped_idx][-1] == -1 and ped_data.y[ped_idx][-1] == -1:
        ped_data.dec_x[ped_idx] = np.nan
        ped_data.dec_y[ped_idx] = np.nan

        # print('   Ped ' + str(ped_idx) + ' already left, therefore skip')

    else:
        # get profitability
        dist_u = distance_grid[ped_data.x[ped_idx][-1]-1][ped_data.y[ped_idx][-1]]
        dist_d = distance_grid[ped_data.x[ped_idx][-1]+1][ped_data.y[ped_idx][-1]]
        dist_l = distance_grid[ped_data.x[ped_idx][-1]][ped_data.y[ped_idx][-1]-1]
        dist_r = distance_grid[ped_data.x[ped_idx][-1]][ped_data.y[ped_idx][-1]+1]

        # calculate probability
        p_norm = np.exp(-1*dist_u) + np.exp(-1*dist_d) + np.exp(-1*dist_l) + np.exp(-1*dist_r)
        p_u = np.exp(-1*dist_u)/p_norm
        p_d = np.exp(-1*dist_d)/p_norm
        p_l = np.exp(-1*dist_l)/p_norm
        p_r = np.exp(-1*dist_r)/p_norm

        # distribution function
        cum_p_u = p_u
        cum_p_d = p_d + cum_p_u
        cum_p_l = p_l + cum_p_d
        cum_p_r = p_r + cum_p_l

        r = rn.random()

        if r < cum_p_u:
            ped_data.dec_x[ped_idx] = ped_data.x[ped_idx][-1] -1
            ped_data.dec_y[ped_idx] = ped_data.y[ped_idx][-1]
        elif r < cum_p_d:
            ped_data.dec_x[ped_idx] = ped_data.x[ped_idx][-1] +1
            ped_data.dec_y[ped_idx] = ped_data.y[ped_idx][-1]
        elif r < cum_p_l:
            ped_data.dec_x[ped_idx] = ped_data.x[ped_idx][-1]
            ped_data.dec_y[ped_idx] = ped_data.y[ped_idx][-1] -1
        else:
            ped_data.dec_x[ped_idx] = ped_data.x[ped_idx][-1]
            ped_data.dec_y[ped_idx] = ped_data.y[ped_idx][-1] +1

    return ped_data


def save_step(ped_data, ped_id, new_x, new_y, new_t):

    ped_data.dec_x[ped_id] = np.nan
    ped_data.dec_y[ped_id] = np.nan
    ped_data.x[ped_id] = ped_data.x[ped_id] + [int(new_x)]
    ped_data.y[ped_id] = ped_data.y[ped_id] + [int(new_y)]
    ped_data.t[ped_id] = ped_data.t[ped_id] + [new_t]

    return ped_data


def resolve_conflicts(ped_data, const, act_t):

    # print('   Conflict resolution started')

    rep_x = range(const['grid_size_x'])                                         # For all cells
    for i in rep_x:
        rep_y = range(const['grid_size_y'])
        for j in rep_y:

            ped_conf = []                                                       # Initiate empty "waiting room"

            rep_k = range(const['N_ped']-1)                                     # For all peds
            for k in rep_k:

                if  (ped_data.dec_x[k] == i) & (ped_data.dec_y[k] == j):        # Check whether they want to enther this cell
                    ped_conf = ped_conf + [k]                                   # If so, they are written to waiting list

            if len(ped_conf) > 1:                                               # If waiting room is occupied by more than 2 peds
                r = rn.randint(0,len(ped_conf)-1)                               # Pick one randomly to keep his decision

                rep_id = range(len(ped_conf))
                for p in rep_id:                                                # Others will change they mind and stay at their positions
                    if p != r:
                        ped_data = save_step(ped_data, ped_conf[p], ped_data.x[ped_conf[p]][-1], ped_data.y[ped_conf[p]][-1], act_t)  # Make "stay" step

    return ped_data


def cell_guest(ped_data, const, x, y):

    ped_id = np.nan

    rep_k = range(const['N_ped']-1)
    for k in rep_k:

        if (ped_data.x[k][-1] == x) & (ped_data.y[k][-1] == y):
            ped_id = ped_data.ped_id[k]

    return ped_id


def execute_all_steps(ped_data, const, act_t, left_peds):
# Move pedestrians to cell they picked, if it is empty
# Kind of smart logic to resolve the situation when the selected cell is occupied but the blocker ped would move
# I.e. logic here enables the decision algorithm to pick occupied cell

    # print('   Movement started')

    peds_to_move = ped_data.ped_id[~ped_data.dec_x.isna()]                  # Initialy, chance to move is defined as True if at least one ped has decision
    chance_to_move = len(peds_to_move) > 0

    while chance_to_move:                                                   # We may need more loops in case of complex blocking situation
                                                                            # The loop will repeated if there was at least one move in previous one
        chance_to_move = False
        peds_to_move = ped_data.ped_id[~ped_data.dec_x.isna()]
        peds_to_move.reset_index(inplace=True, drop=True)

        rep_k = range(len(peds_to_move))                                    # For all peds that may move
        for k in rep_k:

            # If ped choosed to leave, leave
            if ped_data.dec_x[peds_to_move[k]] == -1 and ped_data.dec_y[peds_to_move[k]] == -1:
                ped_data = save_step(ped_data, peds_to_move[k], ped_data.dec_x[peds_to_move[k]], ped_data.dec_y[peds_to_move[k]], act_t)

                #print('     Ped ' + str(peds_to_move[k]) + ' left')

                # Add the left ped into the set to control the number of them
                left_peds.add(peds_to_move[k])

                chance_to_move = True



            else:

                blocking_ped = cell_guest(ped_data, const, ped_data.dec_x[peds_to_move[k]], ped_data.dec_y[peds_to_move[k]])   # Who is in his desired cell

                if pd.isna(blocking_ped):                                                     # Noone is blocking
                    ped_data = save_step(ped_data, peds_to_move[k], ped_data.dec_x[peds_to_move[k]], ped_data.dec_y[peds_to_move[k]], act_t)  # Make step

                    chance_to_move = True
                    # print('     Ped ' + str(peds_to_move[k]) + ' moved')

                elif pd.isna(ped_data.dec_x[blocking_ped]):                # Blocking ped will not move this time step -> No chance to move this time step
                    ped_data = save_step(ped_data, peds_to_move[k], ped_data.dec_x[peds_to_move[k]], ped_data.dec_y[peds_to_move[k]], act_t)   # Make "stay" step

                    # print('     Ped ' + str(peds_to_move[k]) + ' blocked')

                # else:
                    # print('     Ped ' + str(peds_to_move[k]) + ' blocker may move')


    return ped_data, left_peds

"""#Generate map and peds"""

def make_storey(num_classes):
  if num_classes % 2 != 0:
    raise ValueError("Number of classes must be even")
  else:
    # Define the one class with escape distances
    one_class = np.array([[10,      9,       8,       7,       6,       5,       4,       3,       2,        1,       0],
                          [11,     10,       9,       8,       7,       6,       5,       4,       3,        2,  np.inf],
                          [12,     11,      10,       9,       8,       7,       6,       5,       4,        3,  np.inf],
                          [13,     12,      11,      10,       9,       8,       7,       6,       5,        4,  np.inf],
                          [14,     13,      12,      11,      10,       9,       8,       7,       6,        5,  np.inf],
                          [15,     14,      13,      12,      11,      10,       9,       8,       7,        6,  np.inf],
                          [16,     15,      14,      13,      12,      11,      10,       9,       8,        7,  np.inf],
                          [17,     16,      15,      14,      13,      12,      11,      10,       9,        8,  np.inf],
                          [18,     17,      16,      15,      14,      13,      12,      11,      10,        9,  np.inf],
                          [19,     18,      17,      16,      15,      14,      13,      12,      11,       10,  np.inf]])

    # Define an array with all classes, which will be merged
    all_classes = np.zeros((num_classes, one_class.shape[0], one_class.shape[1]))
    for line in range(int(num_classes/2)):
        all_classes[2*line] = one_class + 3 + line * 11             # Control the escape distances
        all_classes[2*line + 1] = one_class[:, ::-1] + 3 + line * 11    # Invert the class on the right hand-side


    # Create the corridor between classes
    corridor_left = np.arange(1, num_classes/2 * 11 + 1).reshape(-1, 1)
    corridor_middle = np.arange(0, num_classes/2 * 11).reshape(-1, 1)
    corridor_right = np.arange(1, num_classes/2 * 11 + 1).reshape(-1, 1)

    left = np.concatenate((np.inf * np.ones((1, one_class.shape[1])), all_classes[0]), axis=0)
    right = np.concatenate((np.inf * np.ones((1, one_class.shape[1])), all_classes[1]), axis=0)

    # Number of parallel classes
    lines = int(num_classes/2)
    i=1

    # First, merge the classes below
    while i < lines:
      left = np.concatenate((left, np.inf * np.ones((1, one_class.shape[1])), all_classes[2*i]), axis=0)
      right = np.concatenate((right, np.inf * np.ones((1, one_class.shape[1])), all_classes[2*i+1]), axis=0)
      i += 1

    # Concatenate classes with the corridor in between
    storey = np.concatenate((left, corridor_left, corridor_middle, corridor_right, right), axis=1)

    # Add an outer boundary of np.inf
    storey = np.pad(storey, pad_width=1, mode='constant', constant_values=np.inf)


    print('number of rows:', len(storey))
    print('number of columns:', len(storey[0]))
    print('middle:',int(len(storey[0])/2))
    print('storey shape:',storey.shape)

    return storey



def generate_ped(num_ped_in_class, storey, seed_value=None):
    if seed_value is not None:
        np.random.seed(seed_value)  # Set the seed for reproducibility

    num_of_lines = int((storey.shape[0] - 1) / 11)

    x, y = [], []

    for i in range(num_of_lines):
        # Define the bounds of the sub-matrix within the matrix
        row_start, row_end = 2 + i * 11, 12 + i * 11  # Row indices for the sub-matrix
        col_start_left, col_end_left = 1, 11  # Column indices for the sub-matrix
        col_start_right, col_end_right = 16, 25  # Column indices for the sub-matrix

        # Extract the sub-matrix
        sub_matrix_left = storey[row_start:row_end, col_start_left:col_end_left]
        sub_matrix_right = storey[row_start:row_end, col_start_right:col_end_right]

        # Get available positions in the sub-matrix
        left_positions = [(row_start + r, col_start_left + c)
                          for r in range(sub_matrix_left.shape[0])
                          for c in range(sub_matrix_left.shape[1])]
        right_positions = [(row_start + r, col_start_right + c)
                           for r in range(sub_matrix_right.shape[0])
                           for c in range(sub_matrix_right.shape[1])]

        # Randomly select unique positions
        if num_ped_in_class <= len(left_positions):
            left_selected = np.random.choice(len(left_positions), num_ped_in_class, replace=False)
        else:
            left_selected = np.random.choice(len(left_positions), len(left_positions), replace=False)

        if num_ped_in_class <= len(right_positions):
            right_selected = np.random.choice(len(right_positions), num_ped_in_class, replace=False)
        else:
            right_selected = np.random.choice(len(right_positions), len(right_positions), replace=False)

        # Store the positions based on the selected indices
        for idx in left_selected:
            row, col = left_positions[idx]
            x.append(col)
            y.append(row)

        for idx in right_selected:
            row, col = right_positions[idx]
            x.append(col)
            y.append(row)

    return np.array(x), np.array(y)

def show_storey(storey, pedestrian_x, pedestrian_y):
    # Replace np.inf with np.nan for proper handling in the plot
    storey = np.where(np.isinf(storey), np.nan, storey)

    # Set up the colormap and normalization based on values in the storey matrix
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=np.nanmin(storey), vmax=np.nanmax(storey))

    # Plot the storey matrix as a grid of colorful cells
    plt.figure(figsize=(10, 10))
    plt.imshow(storey, cmap=cmap, norm=norm)

    # Add colorbar
    plt.colorbar(label='Distance to the exit', shrink=0.65)
    # Plot pedestrian positions on top of the storey with white markers and black edges
    plt.scatter(pedestrian_x, pedestrian_y, facecolor='white', edgecolor='black', s=50, marker='o', label='Pedestrians')

    # Show the plot
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title='Legend')  # Adjust the location as needed
    plt.title('Storey Distance Heatmap')

    # Save the plot to a file
    plt.savefig('heatmap_with_positions.png', bbox_inches='tight')  # Use bbox_inches='tight' to adjust the bounding box

    plt.show()




def show_fundamental_diagram(ped_data, const):
    # Define the number of pedestrians
    num_peds = len(ped_data.t)

    # Create a colormap
    cmap = mtl.colormaps['tab20']  # You can choose other colormaps if needed

    ratio_of_storey = const['grid_size_x'] / const['grid_size_y']
    plt.figure(figsize=(8, 8 * ratio_of_storey))

    # Plot each pedestrian's data with a different color
    for i in range(num_peds):
        plt.plot(ped_data.t[i], ped_data.x[i], 'o', color=cmap(i % 20), label=f'ped {i+1}' if i < 20 else "")

    plt.title('Timespace Fundamental Diagram')
    plt.xlabel(r'$t \,\,\mathrm{[s]}$')
    plt.ylabel(r'$x \,\,\, \mathrm{[m]}$')

    # Avoid cluttering the legend by only showing the first 20 labels
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[:20], labels[:20], loc='best')

    plt.show()


def show_aerial_plot(ped_data, const):
    # Define the number of pedestrians
    num_peds = len(ped_data.t)

    # Create a colormap
    cmap = mtl.colormaps['tab20']  # You can choose other colormaps if needed

    ratio_of_storey = const['grid_size_x'] / const['grid_size_y']
    plt.figure(figsize=(8, 8 * ratio_of_storey))

    # Plot each pedestrian's data with a different color
    for i in range(num_peds):
        plt.plot(ped_data.y[i][:-1], ped_data.x[i][:-1], 'o-', color=cmap(i % 20), label=f'ped {i+1}' if i < 20 else "")

    # Plot special markers (e.g., attractors)
    plt.plot(const['attractor_y'], const['attractor_x'], 'r*', label='Attractor', markersize=10)

    plt.title('Aerial Plot')
    plt.xlabel(r'$x \,\,\mathrm{[m]}$')
    plt.ylabel(r'$y \,\,\, \mathrm{[m]}$')
    plt.xlim(-1, const['grid_size_y'])
    plt.ylim(-1, const['grid_size_x'])

    # Avoid cluttering the legend by only showing the first 20 labels
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[:20] + [handles[-1]], labels[:20] + [labels[-1]], loc='center left',  # Position the legend to the left center
           bbox_to_anchor=(1, 0.5))

    plt.gca().invert_yaxis()  # Reverse the direction of the y-axis

    # Save the plot to a file
    plt.savefig('aerial_plot.png', bbox_inches='tight')  # Use bbox_inches='tight' to adjust the bounding box

    plt.show()

"""#Script"""

def check_waiting_pedestrians(ped_data, const):
    # Store visited cells to avoid counting the same pedestrian multiple times
    visited = set()

    exit_x, exit_y = const['attractor_x'], const['attractor_y']
    grid_size_x, grid_size_y = const['grid_size_x'], const['grid_size_y']

    # Define a function to recursively check neighbors
    def check_neighbors(x, y):
        # Base case: if the cell is out of bounds or already visited, return 0
        if (x < 0 or x >= grid_size_x or y < 0 or y >= grid_size_y) or (x, y) in visited:
            return 0

        # Mark this cell as visited
        visited.add((x, y))

        # Check if there is a pedestrian in the current cell
        ped_in_cell = any((ped_data['x'].apply(lambda positions: positions[-1] == x)) &
                          (ped_data['y'].apply(lambda positions: positions[-1] == y)))

        # If no pedestrian is here, return 0 (no waiting pedestrian in this cell)
        if not ped_in_cell:
            return 0

        # Otherwise, there is a pedestrian here, check neighbors recursively
        return 1 + (
            check_neighbors(x + 1, y) +  # Down
            check_neighbors(x - 1, y) +  # Up
            check_neighbors(x, y + 1) +  # Right
            check_neighbors(x, y - 1)    # Left
        )

    # Start checking from the exit cell
    return check_neighbors(exit_x, exit_y)

def single_simulation(k, x, y, const, storey, num_of_sim = 5):

    t = np.zeros(len(x))
    # Init data containers
    ped_data = init_ped_data(t,y,x)

    # Set of left pedestrians
    left_peds = set()
    waiting_peds = []

    i=0
    act_t = 0
    # Iterate while all pedestrians wont leave the storey
    print('\nSimulation', k, 'started...')
    while len(left_peds) < const['N_ped']:

        act_t = (i+1)*const['dt']                                       # i+1 is current itteration as i = 0  was defined in init step

        #print('  Step ', act_t, 'of simulation', k, 'started')

        # Calculate the number of waiting pedestrians
        waiting_pedestrians_count = check_waiting_pedestrians(ped_data, const)
        waiting_peds.append(waiting_pedestrians_count)

        # model desision loop over all peds
        #print('   Decision started')
        rep2 = range(const['N_ped'])
        for j in rep2:
            ped_data = one_ped_decision(ped_data, j, storey, const)

        # conflict resolution
        ped_data = resolve_conflicts(ped_data, const, act_t)

        # model movement loop over all peds
        ped_data, left_peds = execute_all_steps(ped_data, const, act_t, left_peds)
        i += 1

    print('\n Simulation', k, 'finished')

    return ped_data, waiting_peds, act_t


def simulation_model(x, y, const, storey, num_of_sim=5):
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.starmap(single_simulation, [(k, x, y, const, storey) for k in range(1, num_of_sim + 1)])

    # Unpack results from multiprocessing
    ped_data_all = [result[0] for result in results]
    waiting_peds_all = [result[1] for result in results]
    simulations_time = [result[2] for result in results]

    '''
    # Save waiting_peds_all as a CSV
    waiting_peds_df = pd.DataFrame(waiting_peds_all)
    waiting_peds_df.to_csv('waiting_peds_all.csv', index=False)

    simulations_time_df = pd.DataFrame(simulations_time)
    simulations_time_df.to_csv('simulations_time.csv', index=False)
    '''

    return ped_data_all, waiting_peds_all, simulations_time








#============================================#
#              SCRIPT STARTS HERE            #
#============================================#
if __name__ == "__main__":  # Add this line

    #======================#
    #     PRELIMINARIES    #
    #======================#

    num_of_classes = 8
    num_peds_in_class = 10

    # Generate storey with given number of classes
    storey = make_storey(num_of_classes)

    # Constants - dictionary
    const = {'N_ped': 0,                                # numer of peds in the system
             'N_step': 30,                              # number of steps
             'grid_size_x': 2 + num_of_classes/2*11,    # number of rows
             'grid_size_y': 26,                         # number of columns
             'dt': 1,                                   # time step length [s]
             'attractor_x': 1,                          # x position of attractor [cell]
             'attractor_y': 13,                         # y position of attractor [cell]
             'p': 0.9                                   # probability of leaving the system
            }


    #======================#
    #         MODEL        #
    #======================#

    '''
    # Convert to DataFrame
    const_df = pd.DataFrame(list(const.items()), columns=['Parameter', 'Value'])
    const_df.to_csv('const_parameters.csv', index=False)
    '''




    # Randomly generate given number of pedestrians in each class of the storey
    # Use seed to generate still the same positions
    x, y = generate_ped(num_peds_in_class, storey, 42)
    print('X-coordinates:', x)
    print('Y-coordinates:', y)
    show_storey(storey, x, y)


    # Set number of pedestrians
    const['N_ped'] = len(x)

    const['grid_size_x'] = storey.shape[0]
    const['grid_size_y'] = storey.shape[1]


    # Run simulation
    ped_data_all, waiting_peds_all, simulations_time = simulation_model(x, y, const, storey, num_of_sim = 5)


    #======================#
    #     POSTPROCESSING   #
    #======================#



    # Timespace fundamental diagram
    show_fundamental_diagram(ped_data_all[0], const)

    # Aerial plot
    show_aerial_plot(ped_data_all[0], const)


    '''
    # Number of subplots
    num_plots = len(waiting_peds_all)
    
    # Create subplots
    fig, axes = plt.subplots(nrows=1, ncols=num_plots, figsize=(15, 5))
    
    plt.suptitle('Histogram of Waiting peds', fontsize=16)
    
    # Loop through each array and create a histogram in each subplot
    for i, waiting_peds in enumerate(waiting_peds_all):
        axes[i].hist(waiting_peds, bins=10, edgecolor='black', density=True)
    
        params = stats.expon.fit(waiting_peds)
    
        # Generate x values for the fitted distribution
        x = np.linspace(0, max(waiting_peds), 100)
    
        # Calculate the PDF of the fitted distribution
        p = stats.expon.pdf(x, *params)
    
        # Plot the fitted exponential distribution
        axes[i].plot(x, p, 'r-', linewidth=2, label='Fitted Exponential')  # Fitted line
    
        axes[i].set_title(f'Histogram {i + 1}')
        axes[i].set_xlabel('Number of waiting peds')
        axes[i].legend()
        axes[i].set_ylabel('Frequency')
    
        print('Params of exponential distribution in run', i,':', params)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Show the plot
    plt.show()
    '''

    print('\nEvacuation times:',simulations_time)
    plt.hist(simulations_time, bins=10, edgecolor='black', density=True, color='green')

    # Fit a normal distribution to the data
    mu, std = stats.norm.fit(simulations_time)  # Estimate parameters

    # Create an array of x values for the fitted distribution
    x = np.linspace(min(simulations_time), max(simulations_time), 100)
    p = stats.norm.pdf(x, mu, std)  # Calculate the PDF

    # Plot the fitted distribution
    #plt.plot(x, p, 'r-', linewidth=2, label='Fitted Normal Distribution')  # Fitted line
    plt.xlabel('Evacuation time')
    plt.ylabel('Frequency')
    plt.title('Histogram of evacuation times')
    # Save the plot to a file
    plt.savefig('sim_time.png', bbox_inches='tight')  # Use bbox_inches='tight' to adjust the bounding box

    plt.show()


    num_of_runs = len(waiting_peds_all)
    # Step 1: Combine all sub-arrays into one
    combined_data = np.concatenate(waiting_peds_all)

    # Step 2: Create a histogram for the combined data
    plt.figure(figsize=(8, 6))
    plt.hist(combined_data, bins=10, edgecolor='black', label = 'Histogram', alpha=0.7, density=True)  # Use density=True for normalized histogram
    params = stats.expon.fit(combined_data)

    # Generate x values for the fitted distribution
    x = np.linspace(0, max(combined_data), 100)

    # Calculate the PDF of the fitted distribution
    p = stats.expon.pdf(x, *params)


    # Plot the fitted exponential distribution
    plt.plot(x, p, 'r-', linewidth=2, label='Fitted Exponential')  # Fitted line
    plt.title(f'Histogram of all waiting peds during all {num_of_runs} runs')
    plt.xlabel('Number of Waiting Peds')
    plt.ylabel('Density')
    plt.legend()

    # Save the plot to a file
    plt.savefig('wait_peds_hist_fitted.png', bbox_inches='tight')  # Use bbox_inches='tight' to adjust the bounding box

    plt.show()