import rendervous as rdv
import vulky.datasets as datasets
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


# load the example image
# im = rdv.load_image('reconstructed_cloud.png')[..., 0:3].to(rdv.device())
im = datasets.Images.environment_example.cuda()
im = torch.clamp(im ** (1.0/2.2), 0.0, 1.0)

L = 8
W = 64

maps = []
indices = []
input_dim = 2
output_dim = 3
for i in range(L):
    current_dim = output_dim if i == L-1 else W
    if i < L - 1:  # no create checkpoint for last linear
        indices.append(len(maps))
    maps.append(rdv.LinearNode(input_dim, current_dim))
    if i < L - 1:
        maps.append(rdv.MapNode(rdv.relu))
    input_dim = current_dim


mlp = rdv.Sequential(*maps,
    # distance_between_checkpoints=0,
    checkpoint_indices=indices, # [1::2],
    forward_technique=rdv.ForwardTechnique.technique_1,
    backward_technique=rdv.BackwardTechnique.technique_3
).build()


optimizer = torch.optim.NAdam(mlp.parameters(), lr=0.002)

x = torch.cartesian_prod(
    torch.arange(-1.0, 1.0, 2.0 / im.shape[0], device=rdv.device()),
    torch.arange(-1.0, 1.0, 2.0 / im.shape[1], device=rdv.device())
).view(im.shape[0], im.shape[1], -1)

mlp(x).sum().backward()  # prewarm


# x = torch.rand(10000000, 2)

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