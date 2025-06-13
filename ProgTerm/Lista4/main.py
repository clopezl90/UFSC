from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os

#Setting data importing path and manually experiments and rows quantities
folder_path = "synthetic_data"
n_exp = len(os.listdir(folder_path))
n_lineas = 10

#Initializing flow rate and head arrays
Q_point = np.zeros((n_lineas, n_exp))
H_point = np.full((n_lineas, n_exp), np.nan)

def fillingPumpData():
    '''
    This function reads the information of measured head for every flow rate in experiments.
    :return: flow rate and head arrays
    '''
    # Initializing second column counter
    j = 0
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            with open(file_path, "r") as file:
                linhas = file.readlines()[1:]

                for i, linea in enumerate(linhas):
                    columnas = linea.strip().split(',')
                    Q_point[i, j] = float(columnas[0])
                    try:
                        H_point[i, j] = float(columnas[1])
                    except ValueError:
                        H_point[i, j] = np.nan

            j += 1
    return Q_point, H_point
#Executing of function
fillingPumpData()

#Arrays of data for plot are set
Qplot = []
for flow in Q_point:
    Qplot.append(flow[0])
Hplot = []
Hplotstd=[]
for i in range(n_lineas):
    Hplot.append(np.nanmean(H_point[i]))
    Hplotstd.append(np.nanstd(H_point[i]))

#Arrays are written in terms of the numpy library for vectorial manipulating
Qplot = np.array(Qplot)
Hplot = np.array(Hplot)
Hplotstd = np.array(Hplotstd)

#Similariry scaling
Qplot_esc = Qplot * 2
Hplot_esc = Hplot * 4
Hplotstd_esc = Hplotstd * 4

#Setting variables for plot
x = Qplot
y = Hplot
xesc = Qplot_esc
yesc = Hplot_esc

plt.figure()
#Setting a 2 plots figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

#Adding measured points in graph with alpha parameter, error bar and labels in plot1
for j in range(n_exp):
    ax1.scatter(Q_point[:, j], H_point[:, j], color='gray', alpha=0.2, marker='o', edgecolors='none')
ax1.errorbar(x, y, yerr=Hplotstd, fmt='o-', capsize=4, ecolor='red', label='1200 rpm')
ax1.set_xlabel('Flow rate Q [m³/s]')
ax1.set_ylabel('Head H [m]')
ax1.set_title('Pump performance @ 1200 rpm')
ax1.grid(True)
ax1.legend()

#Adding measured points in graph with alpha parameter, error bar and labels in plot2
ax2.plot(x, y, 'o-', color='blue', label='1200 rpm')
ax2.errorbar(xesc, yesc, yerr=Hplotstd_esc, fmt='o-', capsize=4, ecolor='red', label='2400 rpm')
ax2.set_xlabel('Flow Rate Q [m³/s]')
ax2.set_ylabel('Head H [m]')
ax2.set_title('Performance by Similarity @1200rpm, 2400rpm')
ax2.grid(True)
ax2.legend()
fig.tight_layout()
plt.show()

#Writing data in txt file, by looping the flowrate and head arrays, and writing coordinates in two columns. Values are rounded when necessary.
with open("OutputData.txt", "w") as f:
    f.write(f"Data for 1200 rpm \n")
    f.write(f"Flow rate [kg/s] // Head [m] \n")
    for i in range (len(Qplot)):
        f.write(f"{np.round(Qplot[i], decimals=4)} , {np.round(Hplot[i])}\n")


    f.write(f"Data for 2400 rpm \n")
    f.write(f"Flow rate [kg/s] // Head [m] \n")
    for i in range (len(Qplot_esc)):
        f.write(f"{np.round(Qplot_esc[i], decimals=4)} , {np.round(Hplot_esc[i])}\n")





