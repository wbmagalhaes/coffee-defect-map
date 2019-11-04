import numpy as np
from scipy import integrate as intg


def gaussian_kernel(center,
                    map_size,
                    A=1,
                    sx=1,
                    sy=1):
    """Calcula os pontos de um kernel Gaussiano com integral igual a 1.

    Args:
        center: ponto (x,y) no centro do kernel.
        map_size: tamanho do mapa (w, h).
        A: amplitude do kernel. Valor padrão é 1.
        sx: sigma_x, espalhamento em x. Valor padrão é 1.
        sy: sigma_y, espalhamento em y. Valor padrão é 1.

    Returns:
        2D numpy array com forma (map_size, map_size) com os valores calculados.
    """
    x, y = np.meshgrid(np.arange(map_size[1]), np.arange(map_size[0]))

    x0 = center[0]
    y0 = center[1]

    a = 1 / (2 * sx * sx)
    c = 1 / (2 * sy * sy)

    x = x - x0
    y = y - y0

    e = a*x*x + c*y*y

    z = A * np.exp(-e) / (2 * np.pi * sx * sy)
    return z


def gaussian_kernel_theta(center,
                          map_size,
                          A=1,
                          sx=1,
                          sy=1,
                          theta=0,
                          ):
    """Calcula os pontos de um kernel Gaussiano com integral igual a 1.

    Args:
        center: ponto (x,y) no centro do kernel.
        map_size: tamanho do mapa (w, h).
        A: amplitude do kernel. Valor padrão é 1.
        sx: sigma_x, espalhamento em x. Valor padrão é 1.
        sy: sigma_y, espalhamento em y. Valor padrão é 1.
        theta: angulo de rotacao do kernel. Valor padrão é 0.

    Returns:
        2D numpy array com forma (map_size, map_size) com os valores calculados.
    """
    x, y = np.meshgrid(np.arange(map_size[1]), np.arange(map_size[0]))

    x0 = center[0]
    y0 = center[1]

    sx_2 = sx * sx
    sy_2 = sy * sy

    cos_theta = np.cos(theta)
    cos_theta_2 = cos_theta * cos_theta

    sin_theta = np.sin(theta)
    sin_theta_2 = sin_theta * sin_theta

    sin_2_theta = np.sin(2 * theta)

    a = +cos_theta_2 / (2 * sx_2) + sin_theta_2 / (2 * sy_2)
    b = -sin_2_theta / (4 * sx_2) + sin_2_theta / (4 * sy_2)
    c = +sin_theta_2 / (2 * sx_2) + cos_theta_2 / (2 * sy_2)

    x = x-x0
    y = y-y0

    e = a*x*x + 2*b*x*y + c*y*y

    z = A * np.exp(-e) / (2 * np.pi * sx * sy)
    return z


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
