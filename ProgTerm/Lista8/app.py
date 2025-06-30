import pandas as pd
import matplotlib.pyplot as plt
from src.measurements import Measurement

#Data read with panda
df_tsv = pd.read_csv('experiments.tsv', sep='\t')

#with open("experiments.tsv", "r") as file:
 #   lines = file.readlines()
#variable_names = lines[0].strip().split("\t")

#Needed variables are located with its key
nom_usg = df_tsv["nom_usg"]
wc = df_tsv["WC"]
dpdx= df_tsv["DPDX_FRIC"]
usl = df_tsv["USL"]

#print(nom_usg,wc,dpdx,usl)

#List for storing measurement with the Measurements clss
measurements = []

for _, row in df_tsv.iterrows():
    measurements.append(Measurement(row["nom_usg"], row["WC"], row["DPDX_FRIC"], row["USL"]))
#Dictionary for store data for each superficial velocity as a key
measurements_usg = {}

for exp in measurements:
    key = exp.nom_usg
    if key not in measurements_usg:
        measurements_usg[key] = []
    measurements_usg[key].append(exp)

#Loop for ploting each superficial velocity case
for usg_value, group in measurements_usg.items():
    x = [m.wc for m in group]
    y = [m.dpdx for m in group]
    c = [m.usl for m in group]

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(x, y, c=c, cmap='viridis', edgecolor='k', s=80)

    plt.xlabel("Water Cut [%]")
    plt.ylabel(r"$\frac{dP}{dX}_{\mathrm{fric}}$ [Pa/m]")
    plt.title(rf"$j_g = {usg_value:.2f}\ \mathrm{{m/s}}$")

    cbar = plt.colorbar(sc)
    cbar.set_label(r"$j_l$ [m/s]")
    plt.grid(True)
    plt.tight_layout()
    #plt.show()
    plt.savefig(f"grafico_jg_{usg_value:.2f}.png")
