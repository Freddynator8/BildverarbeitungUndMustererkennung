import os

import imageio
import matplotlib.pyplot as plt
import numpy as np
import numpy.lib.stride_tricks as ns
import torch as th


def image_to_patches(im: np.ndarray, w: int = 2):
    return ns.sliding_window_view(im, (w, w)).reshape(-1, w**2)


def train_data(w: int = 5):
    imnames = [os.path.join('./train', name) for name in os.listdir('./train')]
    images = [imageio.imread(imname) / 255. for imname in imnames]
    return np.concatenate([image_to_patches(im, w) for im in images])


def imsave(uri: str, im: np.ndarray):
    imageio.imsave(uri, (np.clip(im, 0, 1) * 255.).astype(np.uint8))


def patches_to_image(P: np.ndarray, shape: tuple[int, ...], w: int):
    '''See slide 9 in the density estimation lecture material'''
    tmp = th.from_numpy(P)[None].permute(0, 2, 1)
    # This essentially undoes sliding_window_view
    p_th = th.nn.functional.fold(tmp, shape, (w, w), 1, 0, (1, 1))
    # Get the denominator (= number of overlapping patches for any pixel)
    # by folding the one-image
    denom = th.nn.functional.fold(
        th.ones_like(tmp), shape, (w, w), 1, 0, (1, 1)
    )
    return (p_th / denom).numpy().squeeze()


def psnr(x: np.ndarray, y: np.ndarray):
    return (10 * np.log10(1.0 / np.mean((x - y)**2)))


_param_names = ['alphas', 'mus', 'sigmas']


def _get_model_path(K: int, w: int):
    return os.path.join('models', f'K_{K:02d}_w_{w:02d}')


def save_gmm(
    K: int, w: int, alphas: np.ndarray, mus: np.ndarray, sigmas: np.ndarray
):
    out_path = _get_model_path(K, w)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    for name, arr in zip(_param_names, [alphas, mus, sigmas]):
        np.save(os.path.join(out_path, name + '.npy'), arr)


def load_gmm(K: int, w: int):
    out_path = _get_model_path(K, w)
    return [
        np.load(os.path.join(out_path, which + '.npy'))
        for which in _param_names
    ]


_mus = np.array([[-2., 2.], [3., 0.]])
_sigmas = np.array([[[2., 0.], [0., 1.]], [[6., 2], [2, 1.3]]])


def generate_toy():
    # alpha_1 = 0.2, alpha_2 = 0.8
    n1 = 200
    n2 = 800
    rng = np.random.default_rng()
    samples_1 = rng.multivariate_normal(_mus[0], _sigmas[0], size=(n1, ))
    print(_sigmas[0])
    samples_2 = rng.multivariate_normal(_mus[1], _sigmas[1], size=(n2, ))
    samples = np.concatenate((samples_1, samples_2))
    np.save('toy.npy', samples)


def plot_gmm(
    data: np.ndarray, alphas: np.ndarray, mus: np.ndarray, covars: np.ndarray
):
    '''
    Visualize the GMM by their means (size proportional to their weights)
    and the eigenvectors of the covariance matrix.
    '''
    w, v = np.linalg.eigh(covars)
    w_ref, v_ref = np.linalg.eigh(_sigmas)
    _, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim([-7, 8])
    ax.set_ylim([-4, 6])
    ax.set_aspect('equal')
    ax.scatter(data[:, 0], data[:, 1])
    ax.scatter(mus[:, 0], mus[:, 1], marker='*', s=400 * alphas)
    ax.scatter(
        _mus[:, 0], _mus[:, 1], marker='*', s=400 * np.array([0.2, 0.8])
    )
    for comp in range(alphas.shape[0]):
        for eig in range(2):
            ax.arrow(
                mus[comp, 0],
                mus[comp, 1],
                w[comp, eig] * v[comp, 0, eig],
                w[comp, eig] * v[comp, 1, eig],
                width=0.05,
                color='k'
            )
            ax.arrow(
                _mus[comp, 0],
                _mus[comp, 1],
                w_ref[comp, eig] * v_ref[comp, 0, eig],
                w_ref[comp, eig] * v_ref[comp, 1, eig],
                width=0.05,
                color='c'
            )
    ax.legend(['Data', 'GMM Means', 'True Means', 'GMM Cov', 'True Cov'])
    plt.show()
