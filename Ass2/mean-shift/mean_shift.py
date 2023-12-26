import imageio.v3 as imageio
import torch as th

# It is highly recommended to set up pytorch to take advantage of CUDA GPUs!
device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')

# Choose the size of the image here. Prototyping is easier with 128.
M = 128
# The reference implementation works in chunks. Set this as high as possible
# while fitting the intermediary calculations in memory.
simul = M**2

im = th.from_numpy(imageio.imread(f'./{M}.png') / 255.).to(device)

for zeta in [1.8]:
    y, x = th.meshgrid(
        th.linspace(-zeta, zeta, M, device=device),
        th.linspace(-zeta, zeta, M, device=device),
        indexing='xy'
    )
    features = th.cat((im, y[..., None], x[..., None]), dim=-1).reshape(-1, 5)
    for h in [0.8]:
        # The `shifted` array contains the iteration variables
        shifted = features.clone()
        # The `to_do` array contains the indices of the pixels for which the
        # stopping criterion is _not_ yet met.
        to_do = th.arange(M**2, device=device)
        while len(to_do):
            # We walk through the points in `shifted` in chunks of `simum`
            # points. Note that for each point, you should compute the distance
            # to _all_ other points, not only the points in the current chunk.
            chunk = shifted[to_do[:simul]].clone()
            # TODO: Mean shift iterations, writing back the result into shifted

            distances = th.cdist(chunk,features, 2)

            for d in range(distances.shape[0]):
                shifted[to_do[d]] = th.sum(features * (distances[d]**2 <= h**2)[:, None], dim = 0) / th.sum((distances[d]**2 <= h**2), dim = 0)


            cond = (th.norm(shifted[to_do[:simul]] - chunk, dim = 1) >= 1e-6)

            # We only keep the points for which the stopping criterion is not
            # met. `cond` should be a boolean array of length `simul` that
            # indicates which points should be kept.
            to_do = to_do[th.cat(
                (cond, cond.new_ones(to_do.shape[0] - cond.shape[0]))
            )]
            print(len(to_do))
        # Reference images were saved using this code.
        imageio.imwrite(
            f'./reference/{M}/8C_zeta_{zeta:1.1f}_h_{h:.2f}.png',
            (shifted * 255.).to(th.uint8).reshape(M, M, 5)[..., :3].cpu().numpy()
        )
