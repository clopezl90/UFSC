import numpy as np
import matplotlib.pyplot as plt

NodesX =20
NodesY =20 
maxIter = 10000
convergence = 1e-6

def inicialization(a,b):
    matrix = np.zeros((NodesX,NodesY))
    x = np.linspace(0,a,NodesX)
    y = np.linspace(0,b,NodesY)

    #BC top border
    for i in range (NodesX):
        matrix[i,-1] = np.sin((np.pi*x[i])/a)
    
    return matrix, x, y

def GSsolution(matrix,deltax, deltay):
    maxTotalResidual = []
    

    for iteration in range (maxIter):
        prevMatrix = matrix.copy()

        for i in range (1,NodesX-1):
            for j in range (1,NodesY-1):
                matrix[i,j] = ((((matrix[i+1,j] + matrix[i-1,j])/deltax**2) + ((matrix[i,j+1] + matrix[i,j-1])/deltay**2))/(2 / deltax**2 + 2 / deltay**2))  
        
        residual = np.abs(matrix-prevMatrix)
        maxResidual = np.max(residual)
        maxTotalResidual.append(maxResidual)

        if maxResidual<convergence:
            print(f'Convergence reached in iteration {iteration}')
            break
    else:
        print(f'Convergence not reached')
    
    return matrix, maxTotalResidual


#Solucion para relacion 1
a = 1
b = 1
deltax = a/NodesX
deltay = b/NodesY
matrix, x, y = inicialization(a,b)
SolvedMatrix, maxTotalResidual = GSsolution(matrix, deltax, deltay)

#Solucion para relacion 10

a = 1
b = 1
deltax = a/NodesX
deltay = deltax/10
matrix10, x, y = inicialization(a,b)
SolvedMatrix10, maxTotalResidual10 = GSsolution(matrix, deltax, deltay)


plt.figure(figsize=[10,6])
plt.plot(maxTotalResidual,label="Δx / Δy = 1")
plt.plot(maxTotalResidual10,label="Δx / Δy = 10")
plt.yscale("log")
plt.xlabel("Iteration")
plt.ylabel("Máx residual")
plt.title("Residuals Evolution")
plt.legend()
plt.grid()
plt.show()