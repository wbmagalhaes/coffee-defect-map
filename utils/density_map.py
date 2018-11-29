import numpy as np
from scipy import integrate as intg

def gaussian_kernel(center,
                    map_size,
                    A=1,
                    sx=1,
                    sy=1,
                    theta=0,
                    ):
    """Calcula os pontos de um kernel Gaussiano com integral igual a 1.
    Args:
        center: ponto (x,y) no centro do kernel.
        map_size: tamanho do kernel.
        A: amplitude do kernel. Valor padrão é 1.
        sx: sigma_x, espalhamento em x. Valor padrão é 1.
        sy: sigma_y, espalhamento em y. Valor padrão é 1.
        theta: angulo de rotacao do kernel. Valor padrão é 0.
        
    Returns:
        2D numpy array com forma (map_size, map_size) com os valores calculados.
    """
    x, y = np.meshgrid(np.arange(map_size), np.arange(map_size))
    
    x0 = center[0]
    y0 = center[1]
    
    a = np.cos(theta)**2/(2*sx**2) + np.sin(theta)**2/(2*sy**2)
    b = -np.sin(2*theta)/(4*sx**2) + np.sin(2*theta)/(4*sy**2)
    c = np.sin(theta)**2/(2*sx**2) + np.cos(theta)**2/(2*sy**2)

    z = A * np.exp(-(a*(x-x0)**2 + 2*b*(x-x0)*(y-y0) + c*(y-y0)**2))
    return z / (2*np.pi*sx*sy)

def integrate(dmap):
    """Calcula a integral do mapa de densidade aplicando duas vezes o método dos trapézios.
    Args:
        dmap: mapa de densidade a ser integrado.
    Returns:
        float com o resultado da integral dupla.
    """
    size = len(dmap)
    x = np.arange(size)
    y = np.arange(size)
    z = dmap[x][y]
    
    return intg.trapz(intg.trapz(z, y), x)

def sum(dmap):
    """Calcula a soma dos pontos do mapa de densidade.
    Args:
        dmap: mapa de densidade a ser somado.

    Returns:
        float com o resultado da somatória.
    """
    return np.sum(dmap)
