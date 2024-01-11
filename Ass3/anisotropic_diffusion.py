import sys

import imageio.v3 as imageio
import math_tools
import numpy as np
import scipy.sparse as sp
from scipy.ndimage import gaussian_filter
from scipy.sparse.linalg import spsolve


def diffusion_tensor(
    u: np.ndarray,
    sigma_g: float,
    sigma_u: float,
    alpha: float,
    gamma: float,
    nabla: sp.csr_matrix,
    mode: str,
):
    # Implement the diffusion tensor (9)
    # Keep in mind which operations require a flattened image, and which don't

    U_x, U_y = (nabla @ gaussian_filter(u, sigma_u,).ravel()).reshape(2, *u.shape)


    Sxx = gaussian_filter(np.multiply(U_x, U_x), sigma_g)
    Syx = gaussian_filter(np.multiply(U_y, U_x), sigma_g)
    Sxy = gaussian_filter(np.multiply(U_x, U_y), sigma_g)
    Syy = gaussian_filter(np.multiply(U_y, U_y), sigma_g)

    S = np.array([[Sxx, Sxy], [Syx, Syy]])

    d = np.zeros(S.shape)

    for i in range(0, u.shape[0]):
        for j in range(0, u.shape[1]):
            [mu_1,mu_2],v = np.linalg.eig(np.array([[S[0,0][i,j],S[0,1][i,j]],[S[1,0][i,j],S[1,1][i,j]]]))

            if(mu_1 < mu_2):
                mu_1,mu_2 = mu_2,mu_1
                v = np.array([[v[1,0],v[1,1]],[v[0,0],v[0,1]]])

            # Creating the diagonal matrix with the eigenvalues
            vt = np.transpose(v)

            #Creating lambda_1 and lambda_2 for either CED or EED
            if mode == 'ced':
                lambda_1 = alpha
                x = mu_1 - mu_2
                g = np.exp(-(x**2 / (2 * gamma**2)))
                lambda_2 = alpha + (1 - alpha) * (1 - g)
            elif mode == 'eed':
                #Warning gamma is in this case delta
                lambda_1 = (1 + (mu_1 / gamma**2))**(-1/2)
                lambda_2 = 1
            else:
                raise ValueError("Invalid mode. Supported modes are 'ced' and 'eed'.")

            d[:,:,i,j] = v * np.array([[lambda_1, 0],[0, lambda_2]]) * vt
    d = d.reshape(2,2, u.size)

    d = sp.bmat([[sp.diags(d[0,0]),sp.diags(d[0,1])],[sp.diags(d[1,0]),sp.diags(d[1,1])]])
    return d


def nonlinear_anisotropic_diffusion(
    image: np.ndarray,
    sigma_g: float,
    sigma_u: float,
    alpha: float,
    gamma: float,
    tau: float,
    T: float,
    mode: str,
):
    t = 0.
    U_t = image.ravel()
    nabla = math_tools.spnabla_hp(*image.shape)
    id = sp.eye(U_t.shape[0], format="csc")
    while t < T:
        print(f'{t=}')
        D = diffusion_tensor(
            U_t.reshape(image.shape), sigma_g, sigma_u, alpha, gamma, nabla,
            mode
        )
        U_t = spsolve(id + tau * nabla.T @ D @ nabla, U_t)
        t += tau
    return U_t.reshape(image.shape)


params = {
    'ced': {
        'sigma_g': 1.5,
        'sigma_u': 0.7,
        'alpha': 0.0005,
        'gamma': 1e-4,
        'tau': 5.,
        'T': 100.,
    },
    'eed': {
        'sigma_g': 0.,
        'sigma_u': 10,
        'alpha': 0.,
        # This is delta in the assignment sheet, for the sake of an easy
        # implementation we use the same name as in CED
        'gamma': 1e-4,
        'tau': 1.,
        'T': 10.,
    },
}

inputs = {
    'ced': 'starry_night.png',
    'eed': 'fir.png',
}

if __name__ == "__main__":
    mode = sys.argv[1]
    input = imageio.imread(inputs[mode]) / 255.
    output = nonlinear_anisotropic_diffusion(input, **params[mode], mode=mode)
    imageio.imwrite(
        f'./{mode}_out.png', (output.clip(0., 1.) * 255.).astype(np.uint8)
    )
