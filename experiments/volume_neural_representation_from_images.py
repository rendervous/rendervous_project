import torch
import rendervous as rdv
import matplotlib.pyplot as plt
import vulky.datasets as datasets
from tqdm import tqdm
import numpy as np


density_scale = 5.0
cloud = datasets.Volumes.disney_cloud.to(rdv.device()) * density_scale
bmin, bmax = rdv.normalized_box(cloud)

boundary = rdv.ray_box_intersection[bmin, bmax]

N = 8
training_poses = rdv.oct_camera_poses(N, radius=3.0)
training_camera = rdv.PerspectiveCameraSensor(256, 256, training_poses)
training_transmittance_field = rdv.GridDDATransmittance(rdv.Grid3D(cloud, bmin, bmax), rdv.ray_box_intersection[bmin, bmax])
reference_images = training_camera.capture(training_transmittance_field)

fig, axes = plt.subplots(N, N, figsize=(N, N), dpi=512)
for i in range(N):
    for j in range(N):
        axes[i,j].imshow(reference_images[i*8+j].cpu(), cmap='Blues_r', vmin=0.0, vmax=1.0)
        axes[i,j].invert_yaxis()
        axes[i,j].axis('off')
plt.tight_layout(pad=0.0)
plt.show()


rec_camera = rdv.PerspectiveCameraSensor(256, 256, training_poses, jittered=True)

# rec_tensor = torch.nn.Parameter(torch.zeros_like(cloud))
#
# rec_grid = rdv.Grid3D(rec_tensor, bmin, bmax)
#
# # rec_field = rdv.RatiotrackingTransmittance(rec_grid, rdv.ray_box_intersection[bmin, bmax], majorant=rdv.const[density_scale, 100000.0])
#
# rec_field = rdv.GridDDATransmittance(rec_grid, rdv.ray_box_intersection[bmin, bmax])

# create a latent grid to represent the compact feature
latent = torch.nn.Parameter(0.1*torch.randn(16, 16, 16, 16, device=rdv.device()))
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

rec_field = rdv.RatiotrackingTransmittance(
    rep_map,
    boundary=boundary,
    majorant=rdv.const[density_scale, 10000]
)



STEPS = 200

opt = torch.optim.NAdam(rep_map.parameters(), lr=0.002, betas=(0.9, 0.999))
sch = torch.optim.lr_scheduler.OneCycleLR(opt, 0.005, 200)
#sch = torch.optim.lr_scheduler.ExponentialLR(opt, 0.99)
reconstruction_steps = tqdm(range(STEPS))
for step in reconstruction_steps:
    opt.zero_grad()
    inferred_images = rec_camera.capture(rec_field, fw_samples=16)
    loss = torch.nn.functional.huber_loss(
        reference_images,
        inferred_images,
        reduction='mean'
    )
    loss.backward()
    reconstruction_steps.set_description_str(f"Loss: {loss.item()} LR: {opt.param_groups[0]['lr']}")
    opt.step()
    sch.step()


testing_poses = rdv.look_at_poses((-2.8, 0.2, -.4))
testing_camera = rdv.PerspectiveCameraSensor(512, 512, testing_poses)
test_field = rdv.RaymarchingTransmittance(rep_map, boundary)

with torch.no_grad():
    T = testing_camera.capture(test_field)
    plt.imshow(T[0].cpu(), cmap="Blues_r", vmin=0., vmax=1.0)
    plt.gca().invert_yaxis()
    plt.gca().axis('off')
    plt.show()

    T = testing_camera.capture(training_transmittance_field)
    plt.imshow(T[0].cpu(), cmap="Blues_r", vmin=0., vmax=1.0)
    plt.gca().invert_yaxis()
    plt.gca().axis('off')
    plt.show()

