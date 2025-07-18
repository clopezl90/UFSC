from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt

'''
Problem relevant information
As Re < 1000, velocity limit is 0,044 m/s, assuming that max bubble diameter would be 
0,02 that is the channel lenght h.
'''
#Problem data
r=0.1 #[m]
d=2*r #[m]
h=0.02 #[m]
v_angular= 209.4 #[rad/s]
rho_l = 997 #[kg/m3]
rho_g = 1.184 #[kg/m3]
viscosity = 0.00089 #[Pa s]
x0=0.004 #[m]

#Calculation of non linear forces balance
def force_balance(db,u_l):
    re = (rho_l*u_l*db)/viscosity
    if re <= 0:
        return np.inf, 0
    cd = (24/re)*(1+(0.15*re**0.687))
    rhs= (cd*rho_l*(u_l**2)*(db**2))/8
    lhs=((db**3)*(rho_l-rho_g)*r*(v_angular**4))/4
    return rhs,lhs

#Function for calculation with db, u_l

def f(db,u_l):
    rhs,lhs = force_balance(db,u_l)
    return rhs-lhs


#raiz = fsolve(f, x0)
#print("db =", raiz[0])


flow_rates=[]
db_group=[]

#Velocity values for looping
u_l_values = np.linspace(0.005, 0.044, 20)

#loop for ql and critical db
for u_L in u_l_values:
    q = 2*np.pi*r*h*u_L
    flow_rates.append(q)

    def f_local(db):
        return f(db, u_L)

    raiz = fsolve(f_local, x0)
    db_group.append(raiz[0])

print(flow_rates,db_group)

#Plotting
plt.plot(flow_rates, db_group, marker='o')
plt.xlabel("Flow rate Q_L [m3/s]")
plt.ylabel("Critical diameter Db [m]")
plt.title("Db vs Ql")
plt.grid(True)
plt.tight_layout()
plt.show()