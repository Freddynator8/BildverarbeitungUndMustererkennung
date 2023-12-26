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
        beta = np.zeros((N, K))
        # perform cholesky decomposition
        L = np.linalg.cholesky(np.linalg.inv(sigmas))
        # iterate over all k for computing betas
        for k in range(K):
            # calculate beta_k(x_i); used transpose of L_k in norm!!!
            beta[:, k] = -0.5 * (np.square(np.linalg.norm(np.matmul(X - mus[k], L[k]), axis=1)) + m * np.log(2 * np.pi)) + np.linalg.slogdet(L[k])[1] + np.log(alphas[k])
        # calculate all logsumexp(beta(x_i))
        logsumexp = np.max(beta, axis=1) + np.log(np.sum(np.exp(beta - np.max(beta, axis=1)[:, None]), axis=1))
        # calculate all gammas
        gammas = np.exp(beta - logsumexp[:, None])
        # precalc sum of all gamma_k
        gamma_sum = np.sum(gammas, axis=0)
        # calculate all alphas
        alphas = gamma_sum / N
        # calculate all mus
        mus = np.divide(np.matmul(gammas.T, X), gamma_sum[:, None])
        # iterate over all k for computing all sigma_k
        for k in range(K):
            diff = X - mus[k]
            sum_mat = np.matmul(np.multiply(gammas[:, k], diff.T), diff)
            sigmas[k] = sum_mat / gamma_sum[k] + eps * np.eye(m)

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
    N = y.shape[0]
    K = mus.shape[0]
    m = mus.shape[1]
    # precomputation
    A = np.zeros((K, m, m))
    b = np.zeros((K, m))
    for k in range(K):
        A[k] = np.linalg.inv(np.add(lamda * np.identity(m), np.linalg.inv(sigmas[k])))
        b[k] = np.matmul(np.linalg.inv(sigmas[k]), mus[k])

    L = np.linalg.cholesky(np.linalg.inv(sigmas))

    for it in range(max_iter):
        # create buffer to determine argmax
        # resp = np.zeros((N, K))
        beta = np.zeros((N, K))
        # calculate responsibilities
        for k in range(K):
            beta[:, k] = -0.5 * (
                        np.square(np.linalg.norm(np.matmul(x_est - mus[k], L[k]), axis=1)) + m * np.log(2 * np.pi)) + \
                         np.linalg.slogdet(L[k])[1] + np.log(alphas[k])
            # diff = x_est - mus[k]
            # exponential = np.exp(-np.sum(np.multiply(np.matmul(diff, np.linalg.inv(sigmas[k]).T), diff), axis=1) / 2)
            # resp[:, k] = np.log(alphas[k] * np.power((np.power(2 * np.pi, m) * np.linalg.det(sigmas[k])), -0.5) * exponential)
        # find responsible component for each frame
        # resp_comp = np.argmax(resp, axis=1)
        resp_comp = np.argmax(beta, axis=1)
        # update x_est
        x_bar = np.sum(np.multiply(A[resp_comp], (lamda * y + b[resp_comp])[:, :, None]), axis=1)
        x_est = alpha * x_est + (1 - alpha) * x_bar

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
