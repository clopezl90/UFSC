
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class FlowDataAnalyzer:
    def __init__(self, filepath):
        """Inicializa y carga el archivo TSV"""
        self.df = pd.read_csv(filepath, sep="\t")
        self.df["USG_group"] = self.df["USG"].round(1)

    def agrupar_usg(self):
        """Devuelve resumen de grupos por USG redondeado"""
        grupos = self.df["USG_group"].value_counts().sort_index()
        print("Grupos por USG redondeado (m/s):")
        print(grupos)
        return grupos

    def plot_wc_vs_dpdx_for_usg(self, usg_value, save_path=None):
        """Gera gráfico de WC vs dP/dX_fric para um valor específico de USG com colorbar de USL"""
        df_usg = self.df[self.df["USG_group"] == round(usg_value, 1)]
        if df_usg.empty:
            print(f"Nenhum dado encontrado para USG ≈ {usg_value}")
            return

        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(df_usg["WC"], df_usg["DPDX_FRIC"], c=df_usg["USL"],
                        cmap='jet', edgecolor='k', s=80)

        ax.set_xlabel("Water Cut [%]")
        ax.set_ylabel(r"$\\frac{dP}{dX}_{\\text{fric}}$ [Pa/m]")
        ax.set_title(rf"$j_g = {usg_value:.2f}\ \mathrm{{m/s}}$", fontsize=14)

        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label(r"$j_l$ [m/s]")

        ax.grid(True)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Gráfico salvo como {save_path}")
        else:
            plt.show()


if __name__ == "__main__":
    filepath = "experiments.tsv"  # Cambia si usas otra ruta
    analizador = FlowDataAnalyzer(filepath)
    analizador.agrupar_usg()
    analizador.plot_wc_vs_dpdx_for_usg(1.75)  # Ejemplo: para j_g = 1.75 m/s
