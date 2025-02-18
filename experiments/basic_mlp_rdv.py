import rendervous as rdv
import vulky.datasets as datasets
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

L = 8
W = 64

# load the example image
# im = rdv.load_image('reconstructed_cloud.png')[..., 0:3].to(rdv.device())
im = datasets.Images.environment_example.cuda()
im = torch.clamp(im ** (1.0/2.2), 0.0, 1.0)


def dense(input_dim, output_dim):
    k = 1 / input_dim
    A_0 = torch.nn.Parameter((torch.rand(output_dim, input_dim, device=rdv.device()) * 2 - 1) * np.sqrt(k))
    B_0 = torch.nn.Parameter((2 * torch.rand(output_dim, device=rdv.device()) - 1) * np.sqrt(k))
    return A_0 @ rdv.X + rdv.const[B_0]

x = torch.cartesian_prod(
    torch.arange(-1.0, 1.0, 2.0 / im.shape[0], device=rdv.device()),
    torch.arange(-1.0, 1.0, 2.0 / im.shape[1], device=rdv.device())
).view(im.shape[0], im.shape[1], -1)


maps = []
input_dim = 2
output_dim = 3
for i in range(L):
    current_dim = output_dim if i == L-1 else W
    maps.append(dense(input_dim, current_dim))
    if i < L - 1:
        maps.append(rdv.relu)
    input_dim = current_dim


mlp = None
for map in maps:
    if mlp is None:
        mlp = map
    else:
        mlp = mlp.then(map)


optimizer = torch.optim.NAdam(mlp.parameters(), lr=0.002)

mlp(x)  # prewarm forward
mlp(x).sum().backward()  # prewarm backward

# Evaluation time
with torch.no_grad():
    for test in tqdm(range(1000)):
        y = mlp(x)
        torch.cuda.empty_cache()


import time
backward_duration = 0
steps_iterator = tqdm(range(500))
for s in steps_iterator:
    optimizer.zero_grad()
    inf_y = mlp(x)
    loss = torch.nn.functional.mse_loss(inf_y, im)
    backward_duration_start = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize()
    backward_duration += time.perf_counter() - backward_duration_start
    optimizer.step()
    steps_iterator.set_description_str(f"Loss: {loss.item()} BT: {backward_duration / (s + 1)}")
    torch.cuda.empty_cache()

with torch.no_grad():
    plt.imshow(mlp(x).cpu())
    plt.show()