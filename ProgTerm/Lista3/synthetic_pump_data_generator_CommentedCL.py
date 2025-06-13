from pathlib import Path

#Plot and numpy libraries are imported
import matplotlib.pyplot as plt
import numpy as np
#Plot font is set
plt.rcParams.update({'font.size': 14})

#Variable for converting h to s
HOUR_TO_SECOND = 3600

#function for calculating H using the maximum head, Q and A
def H_equation(Q, H_shut_off, A):
    return H_shut_off - A * np.power(Q, 2.0)


H_shut_off = 50.0  # m
A = 80000

#Experiments have and associated uncertainty which is contained in H_unc
FS_uncertainty = 0.05
H_unc = FS_uncertainty * H_shut_off

Q_max = 80  # m3/h
#Flow rate values are set by dividing the Qmax range in 10 values, and converted in m3/s
flowrate_values = np.linspace(0.0, Q_max, 10)
flowrate_values = np.linspace(0.0, Q_max, 10) / HOUR_TO_SECOND

#Output folder is set (where output data will be stored)
output_folder = Path('synthetic_data')
output_folder.mkdir(parents=True, exist_ok=True)

N_experiments = 20

#Head values loop repeating N_experiments times
for synthetic_experiment in range(N_experiments):
    #Randomly Head values are calculated using the flowrate array and H_unc with a normal distribution and standar deviation H_unc
    H_unc_values = np.random.normal(flowrate_values, H_unc)
    #Head values are calculated with the H_equation function and the A parameter
    H_values = H_equation(flowrate_values, H_shut_off, A)
    #Calculated values are summed with the uncertain head values 
    H_values += H_unc_values

    #Dispersion plot is generated with Head and Flowrate values
    plt.scatter(flowrate_values, H_values, marker='o', edgecolor='k')

    #Experiment values are organized in columns
    data = np.column_stack((flowrate_values, H_values))
    #output file path is built and set
    out_file = Path(output_folder, f'synthetic_experiment_{synthetic_experiment:02d}.txt')
    #Data is saved with set headers and delimited by ,
    np.savetxt(out_file,
               data,
               delimiter=',',
               header='Q[m3/s], H[m]',
               comments='',
               fmt='%.6e')