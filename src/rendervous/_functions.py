from . import _internal
import typing as _typing
import torch as _torch
import vulky as _vk
import numpy as _np
import os as _os
import random as _random


__FUNCTIONS_FOLDER__ = _os.path.dirname(__file__).replace('\\', '/') + "/include/functions"


class _oct_inv_projection(_internal.FunctionBase):
    __extension_info__ = dict(
        path=__FUNCTIONS_FOLDER__ + '/tools/oct_inv_projection.comp.glsl',
        parameters = dict(
            in_tensor = _torch.int64,
            out_tensor = _torch.int64,
        )
    )

    def bind(self, coordinates: _torch.Tensor, out: _typing.Optional[_torch.Tensor] = None) -> _typing.Tuple:
        import math
        if out is None:
            out = _vk.tensor(*coordinates.shape[:-1], 3, dtype=_torch.float)
            # out = _torch.zeros(N, len(shape), dtype=_torch.long)
        self.out_tensor = _vk.wrap_gpu(out, 'out')
        self.in_tensor = _vk.wrap_gpu(coordinates, 'in')
        return (math.prod(coordinates.shape[:-1]), 1, 1)

    def result(self) -> _torch.Tensor:
        self.out_tensor.mark_as_dirty()
        self.out_tensor.invalidate()
        return self.out_tensor.obj


def oct_inv_projection(coordinates: _torch.Tensor, out: _typing.Optional[_torch.Tensor] = None):
    return _oct_inv_projection.eval(coordinates, out=out)


def random_sphere_points(N, *, seed: int = 13, radius: float = 1.0):
    _torch.manual_seed(seed)
    samples = _torch.randn(N, 3)
    samples /= _torch.sqrt((samples ** 2).sum(-1, keepdim=True))
    return _vk.vec3(samples * radius)


def random_equidistant_sphere_points(N, *, seed: int = 13, radius: float = 1.0):
    _torch.manual_seed(103)
    initial_samples = _torch.randn(N * 100, 3)
    initial_samples /= _torch.sqrt((initial_samples ** 2).sum(-1, keepdim=True))
    initial_samples = initial_samples.numpy()
    from sklearn.cluster import KMeans
    import numpy as np
    kmeans = KMeans(n_clusters=N, random_state=seed).fit(initial_samples)
    CAMERAS = kmeans.cluster_centers_
    CAMERAS /= _np.sqrt((CAMERAS ** 2).sum(-1, keepdims=True))
    CAMERAS *= radius
    return _torch.as_tensor(CAMERAS, device=_internal.device())


def random_equidistant_camera_poses(N, *, seed: int = 13, radius: float = 1.0):
    origins = random_equidistant_sphere_points(N, seed=seed, radius=radius)
    camera_poses_tensor = _torch.zeros(N, 9, device=_internal.device())
    camera_poses_tensor[:, 0:3] = origins
    camera_poses_tensor[:, 3:6] = _vk.vec3.normalize(-1*origins)
    camera_poses_tensor[:, 7] = 1.0
    return camera_poses_tensor


def oct_camera_poses(N, *, seed: int = 13, radius: float = 1.0):
    with _torch.no_grad():
        u = _torch.arange(-1.0 + 1.0/N, 1.0, 2.0/N, device=_internal.device())
        c = _torch.cartesian_prod(u, u)
        origins =  oct_inv_projection(c) * radius
        # origins = random_equidistant_sphere_points(N, seed=seed, radius=radius)
        camera_poses_tensor = _torch.zeros(N*N, 9, device=_internal.device())
        camera_poses_tensor[:, 0:3] = origins
        camera_poses_tensor[:, 3:6] = _vk.vec3.normalize(-1*origins)
        camera_poses_tensor[:, 7] = 1.0
        return camera_poses_tensor



class _dummy_function(_internal.FunctionBase):
    __extension_info__ = dict(
        path=__FUNCTIONS_FOLDER__ + '/dummy_example/forward.comp.glsl',
        parameters = dict(
            a = _torch.int64,
            b = _torch.int64,
            out=_torch.int64,
            alpha=float,
        )
    )

    def bind(self, a: _torch.Tensor, b: _torch.Tensor, alpha: float = 1.0, out: _typing.Optional[_torch.Tensor] = None) -> _typing.Tuple:
        self.alpha = alpha
        if out is None:
            out = _vk.tensor_like(a)
        self.a = _vk.wrap_gpu(a)
        self.b = _vk.wrap_gpu(b)
        self.out = _vk.wrap_gpu(out, 'out')
        return (a.numel(), 1, 1)

    def result(self) -> _torch.Tensor:
        self.out.invalidate()
        return self.out.obj


def dummy_function(a: _torch.Tensor, b: _torch.Tensor, alpha: float = 1.0, out: _typing.Optional[_torch.Tensor] = None):
    return _dummy_function.eval(a, b, alpha = alpha, out = out)


class _random_ids(_internal.FunctionBase):
    __extension_info__ = dict(
        path=__FUNCTIONS_FOLDER__ + '/tools/random_ids.comp.glsl',
        parameters = dict(
            out_tensor = _torch.int64,
            shape = [4, int],
            dim = int,
            output_dim = int,
        )
    )

    def bind(self, N: int, shape: _typing.Tuple[int], out: _typing.Optional[_torch.Tensor] = None) -> _typing.Tuple:
        if out is None:
            out = _vk.tensor(N, len(shape), dtype=_torch.long)
            # out = _torch.zeros(N, len(shape), dtype=_torch.long)
        self.out_tensor = _vk.wrap_gpu(out, 'out')
        self.dim = len(shape)
        for i in range(self.dim):
            self.shape[i] = shape[i]
        return (N, 1, 1)

    def result(self) -> _torch.Tensor:
        self.out_tensor.mark_as_dirty()
        self.out_tensor.invalidate()
        return self.out_tensor.obj


def random_ids(N: int, shape: _typing.Tuple[int], out: _typing.Optional[_torch.Tensor] = None):
    return _random_ids.eval(N, shape, out = out)


class _gridtoimg(_internal.FunctionBase):
    __extension_info__ = dict(
        path=__FUNCTIONS_FOLDER__ + "/tools/gridtoimg.comp.glsl",
        parameters = dict(
            in_tensor = _torch.int64,
            out_tensor = _torch.int64,
            shape = [4, int],
            dim = int,
            output_dim = int
        )
    )

    def bind(self, in_tensor: _torch.Tensor, out: _typing.Optional[_torch.Tensor] = None) -> _typing.Tuple:
        out_shape = tuple(d - 1 for d in in_tensor.shape[:-1]) + (in_tensor.shape[-1],)
        if out is None:
            out = _vk.tensor(*out_shape, dtype=_torch.float)
        else:
            assert out.shape == out_shape
        self.in_tensor = _vk.wrap_gpu(in_tensor, 'in')
        self.out_tensor = _vk.wrap_gpu(out, 'out')
        self.dim = len(out_shape) - 1
        self.output_dim = out_shape[-1]
        elements = 1
        for i in range(self.dim):
            self.shape[i] = in_tensor.shape[i]
            elements *= out_shape[i]
        return (elements, 1, 1)

    def result(self) -> _torch.Tensor:
        self.out_tensor.mark_as_dirty()
        self.out_tensor.invalidate()
        return self.out_tensor.obj


class _imgtogrid(_internal.FunctionBase):
    __extension_info__ = dict(
        path=__FUNCTIONS_FOLDER__ + "/tools/imgtogrid.comp.glsl",
        parameters = dict(
            in_tensor = _torch.int64,
            out_tensor = _torch.int64,
            shape = [4, int],
            dim = int,
            output_dim = int
        )
    )

    def bind(self, in_tensor: _torch.Tensor, out: _typing.Optional[_torch.Tensor] = None) -> _typing.Tuple:
        out_shape = tuple(d + 1 for d in in_tensor.shape[:-1]) + (in_tensor.shape[-1],)
        if out is None:
            out = _vk.tensor(*out_shape, dtype=_torch.float)
        else:
            assert out.shape == out_shape
        self.in_tensor = _vk.wrap_gpu(in_tensor, 'in')
        self.out_tensor = _vk.wrap_gpu(out, 'out')
        self.dim = len(out_shape) - 1
        self.output_dim = out_shape[-1]
        elements = 1
        for i in range(self.dim):
            self.shape[i] = in_tensor.shape[i]
            elements *= out_shape[i]
        return (elements, 1, 1)

    def result(self) -> _torch.Tensor:
        self.out_tensor.mark_as_dirty()
        self.out_tensor.invalidate()
        return self.out_tensor.obj


class _Grid2ImageFunction(_torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args):
        in_tensor, = args
        return _gridtoimg.eval(in_tensor)

    @staticmethod
    def backward(ctx, *args):
        out_grad, = args
        return _imgtogrid.eval(out_grad)


class _Image2GridFunction(_torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args):
        in_tensor, = args
        return _imgtogrid.eval(in_tensor)

    @staticmethod
    def backward(ctx, *args):
        out_grad, = args
        return _gridtoimg.eval(out_grad)


def gridtoimg(in_tensor: _torch.Tensor, out: _typing.Optional[_torch.Tensor] = None):
    if out is not None:
        assert not in_tensor.requires_grad
        return _gridtoimg.eval(in_tensor, out = out)
    return _Grid2ImageFunction.apply(in_tensor)


def imgtogrid(in_tensor: _torch.Tensor, out: _typing.Optional[_torch.Tensor] = None):
    if out is not None:
        return _imgtogrid.eval(in_tensor, out = out)
    return _Image2GridFunction.apply(in_tensor)


class _resample_grid(_internal.FunctionBase):
    __extension_info__ = dict(
        path=__FUNCTIONS_FOLDER__ + "/tools/resample_grid.comp.glsl",
        parameters=dict(
            in_tensor=_torch.int64,
            out_tensor=_torch.int64,
            in_shape=[4, int],
            out_shape=[4, int],
            dim=int,
            output_dim=int
        )
    )

    def bind(self, in_tensor: _torch.Tensor, dst_shape: _typing.Tuple[int], out: _typing.Optional[_torch.Tensor] = None) -> _typing.Tuple:
        assert len(in_tensor.shape) == len(dst_shape) + 1
        out_shape = dst_shape + (in_tensor.shape[-1],)
        if out is None:
            out = _vk.tensor(*out_shape, dtype=_torch.float)
        else:
            assert out.shape == out_shape
        self.in_tensor = _vk.wrap_gpu(in_tensor, 'in')
        self.out_tensor = _vk.wrap_gpu(out, 'out')
        self.dim = len(out_shape) - 1
        self.output_dim = out_shape[-1]
        elements = 1
        for i in range(self.dim):
            self.in_shape[i] = in_tensor.shape[i]
            self.out_shape[i] = out_shape[i]
            elements *= out_shape[i]
        return (elements, 1, 1)

    def result(self) -> _torch.Tensor:
        self.out_tensor.mark_as_dirty()
        self.out_tensor.invalidate()
        return self.out_tensor.obj


def resample_grid(in_tensor: _torch.Tensor, dst_shape: _typing.Tuple[int,...], out: _typing.Optional[_torch.Tensor] = None):
    dst_shape = tuple(dst_shape)
    min_shape = tuple((d + 1)//2 for d in in_tensor.shape[:-1])
    max_shape = tuple(d*2 for d in in_tensor.shape[:-1])
    clamp_shape = tuple(max(min(dst_shape[i], max_shape[i]), min_shape[i]) for i in range(len(dst_shape)))
    if clamp_shape == dst_shape:
        return _resample_grid.eval(in_tensor, dst_shape, out=out)
    g = _resample_grid.eval(in_tensor, clamp_shape, out=out)
    return resample_grid(g, dst_shape, out=out)


class _resample_img(_internal.FunctionBase):
    __extension_info__ = dict(
        path=__FUNCTIONS_FOLDER__ + "/tools/resample_img.comp.glsl",
        parameters=dict(
            in_tensor=_torch.int64,
            out_tensor=_torch.int64,
            in_shape=[4, int],
            out_shape=[4, int],
            dim=int,
            output_dim=int
        )
    )

    def bind(self, in_tensor: _torch.Tensor, dst_shape: _typing.Tuple[int], out: _typing.Optional[_torch.Tensor] = None) -> _typing.Tuple:
        assert len(in_tensor.shape) == len(dst_shape) + 1
        out_shape = dst_shape + (in_tensor.shape[-1],)
        if out is None:
            out = _vk.tensor(*out_shape, dtype=_torch.float)
        else:
            assert out.shape == out_shape
        self.in_tensor = _vk.wrap_gpu(in_tensor, 'in')
        self.out_tensor = _vk.wrap_gpu(out, 'out')
        self.dim = len(out_shape) - 1
        self.output_dim = out_shape[-1]
        elements = 1
        for i in range(self.dim):
            self.in_shape[i] = in_tensor.shape[i]
            self.out_shape[i] = out_shape[i]
            elements *= out_shape[i]
        return (elements, 1, 1)

    def result(self) -> _torch.Tensor:
        self.out_tensor.mark_as_dirty()
        self.out_tensor.invalidate()
        return self.out_tensor.obj


def resample_img(in_tensor: _torch.Tensor, dst_shape: _typing.Tuple[int,...], out: _typing.Optional[_torch.Tensor] = None):
    dst_shape = tuple(dst_shape)
    min_shape = tuple((d + 1)//2 for d in in_tensor.shape[:-1])
    max_shape = tuple(d*2 for d in in_tensor.shape[:-1])
    clamp_shape = tuple(max(min(dst_shape[i], max_shape[i]), min_shape[i]) for i in range(len(dst_shape)))
    if clamp_shape == dst_shape:
        return _resample_img.eval(in_tensor, dst_shape, out=out)
    g = _resample_img.eval(in_tensor, clamp_shape, out=out)
    return resample_img(g, dst_shape, out=out)


def _power_of_two(x):
    return (x and (not (x & (x - 1))))


class _total_variation(_internal.FunctionBase):
    __extension_info__ = dict(
        path=__FUNCTIONS_FOLDER__ + '/regularizers/total_variation.comp.glsl',
        parameters= dict(
            in_tensor=_torch.int64,
            out_tensor=_torch.int64,
            shape=[4, int],
            dim=int
        )
    )

    def bind(self, in_tensor: _torch.Tensor, out: _typing.Optional[_torch.Tensor] = None) -> _typing.Tuple:
        import math
        assert len(in_tensor.shape) <= 4
        out_shape = in_tensor.shape[:-1] + (1,)
        if out is None:
            out = _vk.tensor(*out_shape, dtype=_torch.float)
        else:
            assert out.shape == out_shape
        self.in_tensor = _vk.wrap_gpu(in_tensor, 'in')
        self.out_tensor = _vk.wrap_gpu(out, 'out')
        self.dim = len(out_shape)-1
        for i in range(self.dim + 1):
            self.shape[i] = in_tensor.shape[i]
        return (math.prod(in_tensor.shape[:-1]), 1, 1)

    def result(self) -> _torch.Tensor:
        self.out_tensor.mark_as_dirty()
        self.out_tensor.invalidate()
        return self.out_tensor.obj


class _total_variation_backward(_internal.FunctionBase):
    __extension_info__ = dict(
        path=__FUNCTIONS_FOLDER__ + '/regularizers/total_variation_backward.comp.glsl',
        parameters=dict(
            in_tensor=_torch.int64,
            out_grad_tensor=_torch.int64,
            in_grad_tensor=_torch.int64,
            shape=[4, int],
            dim=int
        )
    )

    def bind(self, in_tensor: _torch.Tensor, out_grad_tensor: _torch.Tensor, out: _typing.Optional[_torch.Tensor] = None) -> _typing.Tuple:
        import math
        assert len(in_tensor.shape) <= 4
        out_shape = in_tensor.shape
        if out is None:
            out = _torch.zeros(*out_shape, dtype=_torch.float, device=_internal.device())
        else:
            assert out.shape == out_shape
        self.in_tensor = _vk.wrap_gpu(in_tensor, 'in')
        self.out_grad_tensor = _vk.wrap_gpu(out_grad_tensor, 'in')
        self.in_grad_tensor = _vk.wrap_gpu(out, 'out')
        self.dim = len(out_shape)-1
        for i in range(self.dim + 1):
            self.shape[i] = in_tensor.shape[i]
        return (math.prod(in_tensor.shape[:-1]), 1, 1)

    def result(self) -> _torch.Tensor:
        self.in_grad_tensor.mark_as_dirty()
        self.in_grad_tensor.invalidate()
        return self.in_grad_tensor.obj


class _total_variation_diff(_torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        in_tensor, = args
        ctx.save_for_backward(in_tensor)
        return _total_variation.eval(in_tensor)

    @staticmethod
    def backward(ctx, *args):
        out_grad, = args
        in_tensor, = ctx.saved_tensors
        return _total_variation_backward.eval(in_tensor, out_grad)


def total_variation(in_tensor: _torch.Tensor) -> _torch.Tensor:
    return _total_variation_diff.apply(in_tensor)


class _copy_img_to_morton(_internal.FunctionBase):
    __extension_info__ = dict(
        path=__FUNCTIONS_FOLDER__ + "/tools/copy_img_to_morton.comp.glsl",
        parameters = dict(
            in_tensor = _torch.int64,
            out_tensor = _torch.int64,
            resolution = int,
            output_dim = int
        )
    )

    def bind(self, in_tensor: _torch.Tensor, out: _typing.Optional[_torch.Tensor] = None) -> _typing.Tuple:
        assert len(in_tensor.shape) == 3, 'in_tensor should be image (HxWxC)'
        assert in_tensor.shape[0] == in_tensor.shape[1], 'in_tensor should be square'
        resolution = in_tensor.shape[0]
        assert _power_of_two(resolution), 'in_tensor size should be power of two'
        out_shape = (resolution*resolution, in_tensor.shape[-1])
        if out is None:
            out = _vk.tensor(*out_shape, dtype=_torch.float)
        else:
            assert out.shape == out_shape
        self.in_tensor = _vk.wrap_gpu(in_tensor, 'in')
        self.out_tensor = _vk.wrap_gpu(out, 'out')
        self.resolution = resolution
        self.output_dim = out_shape[-1]
        return (resolution*resolution, 1, 1)

    def result(self) -> _torch.Tensor:
        self.out_tensor.mark_as_dirty()
        self.out_tensor.invalidate()
        return self.out_tensor.obj


def copy_img_to_morton(in_tensor: _torch.Tensor, out: _typing.Optional[_torch.Tensor] = None) -> _torch.Tensor:
    return _copy_img_to_morton.eval(in_tensor, out = out)


class _copy_morton_to_img(_internal.FunctionBase):
    __extension_info__ = dict(
        path=__FUNCTIONS_FOLDER__ + "/tools/copy_morton_to_img.comp.glsl",
        parameters = dict(
            in_tensor = _torch.int64,
            out_tensor = _torch.int64,
            resolution = int,
            output_dim = int
        )
    )

    def bind(self, in_tensor: _torch.Tensor, out: _typing.Optional[_torch.Tensor] = None) -> _typing.Tuple:
        import math
        assert len(in_tensor.shape) == 2
        resolution = int(math.sqrt(in_tensor.shape[0]))
        assert in_tensor.shape[0] == resolution * resolution, 'linearization doesnt correspond to a square image'
        out_shape = (resolution, resolution, in_tensor.shape[-1])
        assert _power_of_two(resolution), 'resolution implicit in in_tensor should be power of two'
        if out is None:
            out = _vk.tensor(*out_shape, dtype=_torch.float)
        else:
            assert out.shape == out_shape
        self.in_tensor = _vk.wrap_gpu(in_tensor, 'in')
        self.out_tensor = _vk.wrap_gpu(out, 'out')
        self.resolution = resolution
        self.output_dim = out_shape[-1]
        return (resolution*resolution, 1, 1)

    def result(self) -> _torch.Tensor:
        self.out_tensor.mark_as_dirty()
        self.out_tensor.invalidate()
        return self.out_tensor.obj


def copy_morton_to_img(in_tensor: _torch.Tensor, out: _typing.Optional[_torch.Tensor] = None) -> _torch.Tensor:
    return _copy_morton_to_img.eval(in_tensor, out = out)


def create_density_quadtree(densities: _torch.Tensor) -> _torch.Tensor:
    resolution = densities.shape[0]
    assert len(densities.shape) == 3
    assert densities.shape[-1] == 1
    assert densities.shape[1] == resolution, 'Must be square'
    assert _power_of_two(resolution), "Resolution must be power of two"
    sizes = []
    size = resolution
    while size > 1:
        sizes.append(size * size)
        size //= 2
    sizes.reverse()
    offsets = [0]
    for s in sizes:
        offsets.append(offsets[-1] + s)
    pdfs = _torch.zeros(offsets[-1], 1, device=_internal.device())
    offsets = offsets[:-1]
    offsets.reverse()
    for o in offsets:
        copy_img_to_morton(densities, out=pdfs[o:o + densities.numel()])
        densities = resample_img(densities, (int(densities.shape[0]) // 2, int(densities.shape[1]) // 2)) * 4
    return pdfs


def model_to_tensor(model, shape: _typing.Tuple[int, int, int], bmin: _vk.vec3, bmax: _vk.vec3) -> _torch.Tensor:
    dx = (bmax[0] - bmin[0]).item()/(shape[2] - 1)
    dy = (bmax[1] - bmin[1]).item()/(shape[1] - 1)
    dz = (bmax[2] - bmin[2]).item()/(shape[0] - 1)
    xs = _torch.arange(bmin[0].item(), bmax[0].item() + 0.0000001, dx, device=_internal.device())
    ys = _torch.arange(bmin[1].item(), bmax[1].item() + 0.0000001, dy, device=_internal.device())
    zs = _torch.arange(bmin[2].item(), bmax[2].item() + 0.0000001, dz, device=_internal.device())
    points = _torch.cartesian_prod(zs, ys, xs)[:, [2, 1, 0]]
    values = model(points)
    return values.view(*shape, -1)


class _DifferentiableClamp(_torch.autograd.Function):
    """
    In the forward pass this operation behaves like _torch.clamp.
    But in the backward pass its gradient is 1 everywhere, as if instead of clamp one had used the identity function.
    """
    @staticmethod
    def forward(ctx, *args):
        input, min, max = args
        ctx.save_for_backward(input)
        return input.clamp(min=min, max=max)

    @staticmethod
    def backward(ctx, *args):
        grad_output, = args
        return grad_output.clone(), None, None


def dclamp(grid, min = 0.0, max = 1.0):
    return _DifferentiableClamp.apply(grid, min, max)


class EnhancingLayer(_torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args):
        args = list(args)
        enhancing_process = args[-1]
        ctx.save_for_backward(*args[:-1])
        with _torch.no_grad():
            output = enhancing_process(*args[:-1])
        return output

    @staticmethod
    def backward(ctx, *args):
        return tuple([*args])+(None,)  # None because the process is not differentiable


def enhance_output(*outputs, enhance_process):
    return EnhancingLayer.apply(*outputs, enhance_process)


def generate_perlin_noise_3d(shape, res):
    def f(t):
        return 6*t**5 - 15*t**4 + 10*t**3
    _np._random.seed(_random.randint(0, 1 << 30))
    delta = (res[0] / shape[0], res[1] / shape[1], res[2] / shape[2])
    d = (shape[0] // res[0], shape[1] // res[1], shape[2] // res[2])
    grid = _np.mgrid[0:res[0]:delta[0],0:res[1]:delta[1],0:res[2]:delta[2]]
    grid = grid.transpose(1, 2, 3, 0) % 1
    # Gradients
    theta = 2*_np.pi*_np._random.rand(res[0]+1, res[1]+1, res[2]+1)
    phi = 2*_np.pi*_np._random.rand(res[0]+1, res[1]+1, res[2]+1)
    gradients = _np.stack((_np.sin(phi)*_np.cos(theta), _np.sin(phi)*_np.sin(theta), _np.cos(phi)), axis=3)
    gradients[-1] = gradients[0]
    g000 = gradients[0:-1,0:-1,0:-1].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g100 = gradients[1:  ,0:-1,0:-1].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g010 = gradients[0:-1,1:  ,0:-1].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g110 = gradients[1:  ,1:  ,0:-1].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g001 = gradients[0:-1,0:-1,1:  ].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g101 = gradients[1:  ,0:-1,1:  ].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g011 = gradients[0:-1,1:  ,1:  ].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g111 = gradients[1:  ,1:  ,1:  ].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    # Ramps
    n000 = _np.sum(_np.stack((grid[:,:,:,0]  , grid[:,:,:,1]  , grid[:,:,:,2]  ), axis=3) * g000, 3)
    n100 = _np.sum(_np.stack((grid[:,:,:,0]-1, grid[:,:,:,1]  , grid[:,:,:,2]  ), axis=3) * g100, 3)
    n010 = _np.sum(_np.stack((grid[:,:,:,0]  , grid[:,:,:,1]-1, grid[:,:,:,2]  ), axis=3) * g010, 3)
    n110 = _np.sum(_np.stack((grid[:,:,:,0]-1, grid[:,:,:,1]-1, grid[:,:,:,2]  ), axis=3) * g110, 3)
    n001 = _np.sum(_np.stack((grid[:,:,:,0]  , grid[:,:,:,1]  , grid[:,:,:,2]-1), axis=3) * g001, 3)
    n101 = _np.sum(_np.stack((grid[:,:,:,0]-1, grid[:,:,:,1]  , grid[:,:,:,2]-1), axis=3) * g101, 3)
    n011 = _np.sum(_np.stack((grid[:,:,:,0]  , grid[:,:,:,1]-1, grid[:,:,:,2]-1), axis=3) * g011, 3)
    n111 = _np.sum(_np.stack((grid[:,:,:,0]-1, grid[:,:,:,1]-1, grid[:,:,:,2]-1), axis=3) * g111, 3)
    # Interpolation
    t = f(grid)
    n00 = n000*(1-t[:,:,:,0]) + t[:,:,:,0]*n100
    n10 = n010*(1-t[:,:,:,0]) + t[:,:,:,0]*n110
    n01 = n001*(1-t[:,:,:,0]) + t[:,:,:,0]*n101
    n11 = n011*(1-t[:,:,:,0]) + t[:,:,:,0]*n111
    n0 = (1-t[:,:,:,1])*n00 + t[:,:,:,1]*n10
    n1 = (1-t[:,:,:,1])*n01 + t[:,:,:,1]*n11
    _np._random.seed(_random.randint(0, 1<<31))
    return _torch.from_numpy((1-t[:,:,:,2])*n0 + t[:,:,:,2]*n1).float()






