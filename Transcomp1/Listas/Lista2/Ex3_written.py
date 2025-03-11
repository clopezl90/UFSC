import numpy as np
import matplotlib.pyplot as plt

L=1
max_iter = 100
tolerance = 1e-6


def temp_initialization(N):
    x= np.linspace(0,L,N)
    T= np.sin(np.pi*x) + (0.1*(np.sin(N*np.pi*x)))
    T[0]=0
    T[N-1]=100
    return T,x

def calculation_GS(T, N):
    for iteration in range (0,5):    
        for i in range(1,N-1):
            T[i] = (T[i+1] + T[i-1])/2                            

    return T



def solution(N):
    dx= L/(N-1)
    T, x = temp_initialization(N)
    plt.figure()
    plt.plot(x, T, label="Distribución inicial")
    
    T_solved = calculation_GS(T,N)
    plt.plot(x, T_solved, label="Distribución final")
    plt.show()
    


solution(5)
    
