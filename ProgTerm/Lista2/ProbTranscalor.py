import numpy as np

def ler_propriedades(arquivo):
    """
    Le os valores de propiedades en el arquivo, y com um for identifica os valores e as chaves para logo
    colocarlos en um array propiedades
    Retorna o array propiedades

    """
    propriedades = {}
    with open(arquivo, "r") as f:
        for linha in f:
            if '=' in linha:
                chave, valor = linha.strip().split('=')
                propriedades[chave.strip()] = float(valor.strip())
    return propriedades

props = ler_propriedades("props.txt")

#As variaveis sao lidas do array propiedades e asociadas a novas variaveis

m_ponto = props['m_ponto'] # [kg/s]
q_area = props['q_area'] # [W/m2]
rho = props['rho'] # [kg/m3]
k = props['k'] # [w/mk]
cp = props['cp']  # [J/kg.K]
miu_visc = props['miu_visc'] #[N.s/m2]
d = props['d'] #[m]
t_in = props['t_in'] #[K]
t_out = props['t_out'] #[K]
deltaT=t_out-t_in

#Cria listas para ler dinamicamente diferentes valores das propiedades de interes

valores_fluxo=[2000,20000,50000]
valores_vazao=[0.5,2.5,10]
valores_diametro=[0.01, 0.07, 0.2]



def calculoFluxoTermico():
    """
    Calcula comprimento e a taxa de calor con los valores de fluxo por area de la lista valores_fluxo
    Escreve logo os valores em arquivo txt
    """
    with open('DadosSaida.txt', 'a') as f:
        f.write(f"Fluxo [W/m2] // comprimento [m] // taxa de calor transferida [kW]\n")
        for fluxo in valores_fluxo:
            comprimento=(m_ponto*cp*deltaT)/(fluxo*(np.pi*d))
            calor=(fluxo*(np.pi*d*comprimento))/1000
            f.write(f"{fluxo} // {comprimento:.2f} // {calor:.2f}\n")

calculoFluxoTermico()

def calculoVazao():
    """
    Calcula comprimento e a taxa de calor con los valores de vazao de la lista valores vazao
    Escreve logo os valores em arquivo txt
    """
    with open('DadosSaida.txt', 'a') as f:

        f.write(f"Vacao [kg/s] // comprimento [m] // taxa de calor transferida [kW]\n")
        for vacoes in valores_vazao:
            comprimento=(vacoes*cp*deltaT)/(q_area*(np.pi*d))
            calor=(vacoes*cp*deltaT)/1000
            f.write(f"{vacoes} // {comprimento:.2f} // {calor:.2f}\n")
calculoVazao()

def calculoDiametro():
    """
    Calcula comprimento e a taxa de calor con los valores de diametro de la lista valores diametro
    Escreve logo os valores em arquivo txt
    """
    with open('DadosSaida.txt', 'a') as f:
        f.write(f"Diametro [m] // comprimento [m] // taxa de calor transferida [W]\n")
        for diametro in valores_diametro:
            comprimento=(m_ponto*cp*deltaT)/(q_area*(np.pi*diametro))
            calor=m_ponto*(np.pi*diametro*comprimento)
            f.write(f"{diametro} // {comprimento:.2f} // {calor:.2f}\n")
calculoDiametro()








