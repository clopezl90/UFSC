class Measurement():
    def __init__(self, nom_usg, wc, dpdx, usl):
        self.nom_usg = round(float(nom_usg), 2)
        self.wc = round(float(wc),2)
        self.dpdx = round(float(dpdx),2)
        self.usl = round(float(usl),2)