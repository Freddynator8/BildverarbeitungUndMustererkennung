import imageio.v3 as imageio
import numpy as np
import utils


def expectaion_maximization(
    X: np.ndarray,
    K: int,
    max_iter: int = 50,
    plot: bool = False,
    show_each: int = 5,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Number of data points, features
    m = X.shape[1]
    # Init: Uniform weights, first K points as means, identity covariances
    alphas = np.full((K, ), 1. / K)
    mus = X[:K]
    sigmas = np.tile(np.eye(m)[None], (K, 1, 1))

    for it in range(max_iter):
        # TODO: Implement
        gammas = np.empty((X.shape[0], K))
        alphas = alphas
        mus = mus
        sigmas = sigmas

        if it % show_each == 0 and plot:
            utils.plot_gmm(X, alphas, mus, sigmas)

    return alphas, mus, sigmas


def denoise(
    alphas: np.ndarray,
    mus: np.ndarray,
    sigmas: np.ndarray,
    y: np.ndarray,
    lamda: float,
    alpha: float = 0.5,
    max_iter: int = 30,
):
    x_est = y.copy()

    # TODO: Precompute A, b and implement the loop
    for it in range(max_iter):
        x_est = x_est

    return x_est


def train(K: int = 2, w: int = 5):
    data = utils.train_data(w)
    alphas, mus, sigmas = expectaion_maximization(data, K=K)
    utils.save_gmm(K, w, alphas, mus, sigmas)


if __name__ == "__main__":
    # Set to True if you want to debug your EM implementation
    # with the toy data. The asserts should not throw.
    debug_em = False
    if debug_em:
        data = np.load('./toy.npy')
        alphas, mus, sigmas = expectaion_maximization(data, K=2, plot=True)
        for n, arr in zip(['alphas', 'mus', 'sigmas'], [alphas, mus, sigmas]):
            assert np.allclose(arr, np.load(f'./models_ref/toy/{n}.npy'))

    # Fix seed if you want reproducibility
    seed = 42
    rng = np.random.default_rng(seed)

    # Parameters for the GMM: Components `K` and window size `w`, m = w ** 2
    K = 5
    w = 5

    # Set to true if you want to train the GMM
    do_training = False
    if do_training:
        train(K, w)

    alphas, mus, sigmas = utils.load_gmm(K, w)

    sigma = 0.1
    # Tunable, this was the choice for the PSNR in the assignment sheet
    lamda = 1 / sigma**2
    for i in range(5):
        x = imageio.imread(f'./validation/img{i}.png') / 255.
        y = x + sigma * rng.normal(size=x.shape)
        y_patched = utils.image_to_patches(y, w)
        x_map_patched = denoise(alphas, mus, sigmas, y_patched, lamda)
        x_map = utils.patches_to_image(x_map_patched, x.shape, w)
        print(utils.psnr(x_map, x))
        utils.imsave(f'./validation/img{i}_map.png', x_map)
