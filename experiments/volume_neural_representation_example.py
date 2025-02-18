import torch
import rendervous as rdv
import matplotlib.pyplot as plt
import vulky.datasets as datasets
import numpy as np
from tqdm import tqdm


# load the disney cloud as a tensor
cloud = datasets.Volumes.disney_cloud
environment_tensor = datasets.Images.environment_example

camera = rdv.PerspectiveCameraSensor(512, 512, rdv.look_at_poses((2.8, 0.2, -2.4)))

# create a grid map as reference
bmin, bmax = rdv.normalized_box(cloud)
grid = rdv.Grid3D(cloud, bmin, bmax)

boundary = rdv.ray_box_intersection[bmin, bmax]
field_T = rdv.GridDDATransmittance(grid, boundary)

T = camera.capture(field_T)

plt.imshow(T[0].cpu(), cmap='Blues_r')
plt.gca().invert_yaxis()
plt.gca().axis('off')
plt.show()

# create a latent grid to represent the compact feature
latent = torch.nn.Parameter(torch.randn(16, 16, 16, 16, device=rdv.device()))
latent_grid = rdv.Grid3D(latent, bmin, bmax)
# create a MLP to represent the scene
def dense(input_dim, output_dim):
    k = 1 / input_dim
    A_0 = torch.nn.Parameter((torch.rand(output_dim, input_dim, device=rdv.device())*2 - 1)*np.sqrt(k))
    B_0 = torch.nn.Parameter((2 * torch.rand(output_dim, device=rdv.device())-1)*np.sqrt(k))
    return A_0 @ rdv.X + rdv.const[B_0]

maps = [dense(16, 32), rdv.relu, dense(32, 32), rdv.relu, dense(32, 32), rdv.relu, dense(32, 1), rdv.relu]
mlp = None
for m in maps: mlp = m if mlp is None else mlp.then(m)

rep_map = latent_grid.then(mlp)

# train the representation
bmin, bmax = bmin.to(rdv.device()), bmax.to(rdv.device())
opt = torch.optim.NAdam(list(mlp.parameters())+[latent], lr=0.02)
sch = torch.optim.lr_scheduler.ExponentialLR(opt, 0.999)
steps_iterator = tqdm(range(1000))
view_steps = [0, 10, 20, 40, 100, 250, 500, 999]
for s in steps_iterator:
    with torch.no_grad():
        x = torch.rand(128*1024, 3, device=rdv.device()) * (bmax - bmin) + bmin
        ref_values = grid(x)

    opt.zero_grad()
    inf_values = rep_map(x)
    loss = torch.nn.functional.mse_loss(ref_values, inf_values, reduction='sum')
    loss.backward()
    opt.step()
    sch.step()
    steps_iterator.set_description_str(f"Loss: {loss.item()}")

    if s in view_steps:
        with torch.no_grad():

            ds = rdv.DependencySet()
            ds.add_parameters(
                sigma = grid * 50,
                majorant = rdv.const[50.0, 10000],
                scattering_albedo = rdv.const[0.99, 0.98, 0.97],
                environment_tensor = environment_tensor
            )
            ds.requires(rdv.medium_box_AABB, bmin=bmin, bmax=bmax)
            ds.requires(rdv.medium_radiance_path_integrator_NEE_DT)

            # field_T = rdv.RaymarchingTransmittance(rep_map, boundary)

            import time

            t = time.perf_counter()
            R = camera.capture(ds.radiance, fw_samples=8)
            t = time.perf_counter() - t
            print(f"[INFO] Rendered in {t} secs")

            plt.imshow(R[0].cpu() ** (1.0/2.2))
            plt.gca().invert_yaxis()
            plt.gca().axis('off')
            plt.show()
