from vulky import vec3
from . import _internal
from . import _maps
from . import _functions
import torch as _torch
import typing as _typing
import numpy as _np


def medium_box_AABB(ds: _internal.DependencySet, *, bmin: vec3 = vec3(-1.0, -1.0, -1.0), bmax: vec3 = vec3(1.0, 1.0, 1.0)):
    ds.add_parameters(box=(bmin, bmax))

def medium_box_normalized(ds: _internal.DependencySet, *, t: _typing.Union[_torch.Tensor, _typing.List[int]]):
    ds.add_parameters(box=_maps.normalized_box(t))

def medium_box(ds: _internal.DependencySet):
    '''
    Ensures a box parameter with AABB limits for the volume.
    '''
    if ds.ensures('box', tuple):
        return
    try:
        ds.assert_ensures('sigma_tensor', _torch.Tensor)
        medium_box_normalized(ds, t = ds.sigma_tensor)
    except:
        try:
            ds.assert_ensures('sigma', _maps.Grid3D)
            medium_box_normalized(ds, t=ds.sigma.grid)
        except:
            print("[WARNING] Neither Grid3D sigma nor tensor was found, -1..1 box used instead.")
            medium_box_AABB(ds)

def medium_boundary_box(ds: _internal.DependencySet):
    ds.requires(medium_box)
    bmin, bmax = ds.box
    ds.add_parameters(boundary=_maps.RayBoxIntersection(bmin, bmax))

def medium_boundary(ds: _internal.DependencySet):
    '''
    Ensures a boundary map by means of raycast (x,w) -> (tMin, tMax).
    If necessary, creates a ray-box intersection with box.
    '''
    if ds.ensures('boundary', _maps.MapBase):
        return
    medium_boundary_box(ds)

def build_map_from_tensor(ds: _internal.DependencySet, *, field_name: str):
    ds.requires(medium_box)
    bmin, bmax = ds.box
    ds.assert_ensures(field_name + "_tensor", _torch.Tensor)
    tensor = getattr(ds, field_name + "_tensor")
    if len(tensor.shape) == 1:
        return _maps.const[tensor]
    else:
        return _maps.Grid3D(tensor, bmin, bmax)

def medium_sigma_tensor(ds: _internal.DependencySet):
    ds.add_parameters(sigma=build_map_from_tensor(ds, field_name='sigma'))

def medium_sigma(ds: _internal.DependencySet):
    if ds.ensures('sigma', _maps.MapBase):
        return
    medium_sigma_tensor(ds)

def medium_scattering_albedo_tensor(ds: _internal.DependencySet):
    ds.add_parameters(scattering_albedo=build_map_from_tensor(ds, field_name='scattering_albedo'))

def medium_scattering_albedo(ds: _internal.DependencySet):
    if ds.ensures('scattering_albedo', _maps.MapBase):
        return
    medium_scattering_albedo_tensor(ds)

def medium_emission_tensor(ds: _internal.DependencySet):
    emission_grid = build_map_from_tensor(ds, field_name='emission')
    ds.add_parameters(emission=_maps.SH_PDF(3, emission_grid))
    
def medium_emission(ds: _internal.DependencySet):
    if ds.ensures('emission', _maps.MapBase):
        return
    medium_emission_tensor(ds)
    
def medium_phase_g_tensor(ds: _internal.DependencySet):
    ds.add_parameters(phase_g=build_map_from_tensor(ds, field_name='phase_g'))

def medium_phase_g(ds: _internal.DependencySet):
    if ds.ensures('phase_g', _maps.MapBase):
        return
    medium_phase_g_tensor(ds)

def medium_majorant_tensor(ds):
    ds.assert_ensures('majorant_tensor', _torch.Tensor)
    ds.add_parameters(majorant=_maps.const[ds.majorant_tensor])

def medium_majorant_grid(ds: _internal.DependencySet):
    ds.requires(medium_sigma)
    ds.assert_ensures('sigma', _maps.Grid3D)
    g: _maps.Grid3D = ds.sigma
    ds.add_parameters(majorant_tensor=lambda: _torch.tensor([g.get_maximum(), 1000000.0], device=_internal.device()))
    ds.add_parameters(majorant=_maps.const[ds.majorant_tensor])

def medium_majorant(ds: _internal.DependencySet):
    '''
    Ensures a majorant map (x, w) -> (majorant, distance).
    By default assumes a Grid3D sigma can be reached.
    '''
    if ds.ensures('majorant', _maps.MapBase):
        return
    try:
        medium_majorant_tensor(ds)
    except:
        # print("[WARNING] majorant tensor not found, trying to get it from a Grid3D sigma")
        medium_majorant_grid(ds)

def medium_transmittance_GDT(ds: _internal.DependencySet):
    ds.requires(medium_sigma)
    ds.requires(medium_boundary)
    ds.assert_ensures('sigma', _maps.Grid3D)
    ds.add_parameters(transmittance=_maps.GridDeltatrackingTransmittance(ds.sigma, ds.boundary))

def medium_transmittance_GRT(ds: _internal.DependencySet):
    ds.requires(medium_sigma)
    ds.requires(medium_boundary)
    ds.assert_ensures('sigma', _maps.Grid3D)
    ds.add_parameters(transmittance=_maps.GridRatiotrackingTransmittance(ds.sigma, ds.boundary))

def medium_transmittance_DDA(ds: _internal.DependencySet):
    ds.requires(medium_sigma)
    ds.requires(medium_boundary)
    ds.assert_ensures('sigma', _maps.Grid3D)
    ds.add_parameters(transmittance=_maps.GridDDATransmittance(ds.sigma, ds.boundary))

def medium_transmittance_RM(ds: _internal.DependencySet, *, step: float = 0.05):
    ds.requires(medium_sigma)
    ds.requires(medium_boundary)
    ds.add_parameters(transmittance=_maps.RaymarchingTransmittance(ds.sigma, ds.boundary, step=step))

def medium_transmittance_DT(ds: _internal.DependencySet):
    ds.requires(medium_sigma)
    ds.requires(medium_boundary)
    ds.requires(medium_majorant)
    ds.add_parameters(transmittance=_maps.DeltatrackingTransmittance(ds.sigma, ds.boundary, ds.majorant))

def medium_transmittance_RT(ds: _internal.DependencySet):
    ds.requires(medium_sigma)
    ds.requires(medium_boundary)
    ds.requires(medium_majorant)
    ds.add_parameters(transmittance=_maps.RatiotrackingTransmittance(ds.sigma, ds.boundary, ds.majorant))

def medium_transmittance(ds: _internal.DependencySet):
    '''
    Ensures a transmittance map.
    By default Ratiotracking technique is used
    '''
    if ds.ensures('transmittance', _maps.MapBase):
        return
    try:
        medium_transmittance_RT(ds)
    except:
        print("[WARNING] fields for ratio-tracking transmittance not found, trying DDA instead")
        medium_transmittance_DDA(ds)

def medium_environment_tensor(ds: _internal.DependencySet, *, projection: _typing.Literal['sph', 'cyl', 'xr', 'cube'] = 'xr'):
    ds.assert_ensures("environment_tensor", _torch.Tensor)
    t = ds.environment_tensor
    if len(t.shape) == 1:  # constant
        ds.add_parameters(environment=_maps.const[ds.environment_tensor])
    else:
        projection = {
            'xr': _maps.xr_projection,
        }[projection]
        ds.add_parameters(environment=_maps.Image2D(ds.environment_tensor).after(projection))

def medium_environment(ds: _internal.DependencySet):
    if ds.ensures('environment', _maps.MapBase):
        return
    medium_environment_tensor(ds)

def medium_environment_sampler_quadtree(ds: _internal.DependencySet, *, projection: _typing.Literal['sph', 'cyl', 'xr', 'cube'] = 'xr', levels: int = 10):
    ds.requires(medium_environment_tensor, projection=projection)
    if projection == 'xr':
        skybox_img = ds.environment_tensor
        def build_quadtree():
            with _torch.no_grad():
                resolution = 1 << levels
                densities = _functions.resample_img(skybox_img.sum(-1, keepdim=True), (resolution, resolution))
                for py in range(resolution):
                    w = _np.cos(py * _np.pi / resolution) - _np.cos((py + 1) * _np.pi / resolution)
                    densities[py, :, :] *= w
                densities /= max(densities.sum(), 0.00000001)
                return _functions.create_density_quadtree(densities)

        ds.add_parameters(environment_sampler_quadtree=build_quadtree)
        dir_sampling = _maps.XRQuadtreeRandomDirection(input_dim=6, densities=ds.environment_sampler_quadtree,
                                                 levels=levels)
        ds.add_parameters(environment_sampler=_maps._functionsampler(point_sampler=dir_sampling, function_map=ds.environment))
    else:
        raise Exception(f'Not supported quadtree creation from projection {projection}')

def medium_environment_sampler_uniform(ds: _internal.DependencySet):
    ds.includes(medium_environment)
    dir_sampling = _maps.UniformDirectionSampler(3)
    ds.add_parameters(environment_sampler=_maps._functionsampler(point_sampler=dir_sampling, function_map=ds.environment))

def medium_environment_sampler(ds: _internal.DependencySet):
    if ds.ensures('environment_sampler', _maps.MapBase):
        return
    try:
        medium_environment_sampler_quadtree(ds)
    except Exception as e:
        print(f"[WARNING] Quadtree sampler for environment could not be created. {e}. Using uniform sampling instead.")

def medium_phase_iso(ds: _internal.DependencySet):
    ds.add_parameters(phase=_maps.const[1.0 / (4 * _np.pi)])

def medium_phase_HG(ds: _internal.DependencySet):
    ds.requires(medium_phase_g)
    ds.add_parameters(phase=_maps.HGPhase(ds.phase_g))

def medium_phase(ds: _internal.DependencySet):
    if not ds.ensures('phase', _maps.MapBase):
        try:
            medium_phase_HG(ds)
        except:
            print("[WARNING] a HG phase could not be created, assuming isotropic instead.")
            medium_phase_iso(ds)

def medium_phase_sampler_uniform(ds: _internal.DependencySet):
    dir_sampling = _maps.UniformDirectionSampler(6)
    ds.requires(medium_phase)
    ds.add_parameters(phase_sampler=_maps._functionsampler(_maps.identity(6) | dir_sampling, ds.phase.cast(9,1))[0, 7, 8, 9])

def medium_phase_sampler_HG(ds: _internal.DependencySet):
    ds.requires(medium_phase_g)
    ds.add_parameters(phase_sampler=_maps.VHGPhaseSampler(ds.phase_g))

def medium_phase_sampler(ds: _internal.DependencySet):
    if ds.ensures('phase_sampler', _maps.MapBase):
        return
    try:
        medium_phase_sampler_HG(ds)
    except:
        print("[WARNING] a HG phase sampler could not be created, assuming isotropic instead.")
        medium_phase_sampler_uniform(ds)

def medium_collision_sampler_DT(ds: _internal.DependencySet, *, ds_epsilon: float = 0.1):
    ds.requires(medium_sigma)
    ds.requires(medium_boundary)
    ds.requires(medium_majorant)
    ds.add_parameters(collision_sampler=_maps.DeltatrackingCollisionSampler(ds.sigma, ds.boundary, ds.majorant, ds_epsilon))

def medium_collision_sampler(ds: _internal.DependencySet):
    medium_collision_sampler_DT(ds)

def medium_radiance_transmitted(ds: _internal.DependencySet):
    ds.requires(medium_transmittance)
    ds.requires(medium_environment)
    ds.add_parameters(radiance=ds.transmittance * ds.environment.after(_maps.ray_direction))

def medium_exitance_radiance_emission(ds: _internal.DependencySet):
    ds.requires(medium_emission)
    ds.add_parameters(exitance_radiance=ds.emission)

def medium_exitance_radiance(ds: _internal.DependencySet):
    if ds.ensures('exitance_radiance', _maps.MapBase):
        return
    try:
        ds.requires(medium_scattering_albedo)
        has_scattering_albedo = True
    except:
        has_scattering_albedo = False
    try:
        ds.requires(medium_emission)
        has_emission = True
    except:
        has_emission = False
    if has_scattering_albedo:
        ds.requires(medium_phase_sampler)
        ds.requires(medium_radiance)
        if not has_emission:
            exitance_radiance = _maps.MCScatteredRadiance(ds.scattering_albedo, ds.phase_sampler, ds.radiance)
        else:
            exitance_radiance = _maps.MCScatteredEmittedRadiance(ds.scattering_albedo, ds.emission, ds.phase_sampler, ds.radiance)
    else:
        if not has_emission:
            exitance_radiance = _maps.ZERO
        else:
            exitance_radiance = ds.emission
    ds.add_parameters(exitance_radiance=exitance_radiance)

def medium_radiance_collision_integrator_MC(ds: _internal.DependencySet):
    ds.requires(medium_collision_sampler)
    ds.requires(medium_environment)
    ds.requires(medium_exitance_radiance)
    ds.add_parameters(radiance=_maps.MCCollisionIntegrator(ds.collision_sampler, ds.exitance_radiance, ds.environment))

def medium_radiance_collision_integrator_RM(ds: _internal.DependencySet):
    pass

def medium_radiance_collision_integrator_DDA(ds: _internal.DependencySet):
    ds.requires(medium_sigma)
    ds.requires(medium_boundary)
    ds.assert_ensures('sigma', _maps.Grid3D)
    ds.requires(medium_environment)
    ds.requires(medium_exitance_radiance)
    ds.add_parameters(radiance=_maps.GridDDACollisionIntegrator(ds.sigma, ds.exitance_radiance, ds.environment, ds.boundary))

def medium_radiance_path_integrator_DT(ds: _internal.DependencySet, *, ds_epsilon: float = 0.1):
    ds.requires(medium_sigma)
    try:
        ds.requires(medium_scattering_albedo)
    except:
        ds.add_parameters(scattering_albedo=_maps.ZERO)
    try:
        ds.requires(medium_emission)
    except:
        ds.add_parameters(emission=_maps.ZERO)
    ds.requires(medium_boundary)
    ds.requires(medium_majorant)
    ds.requires(medium_environment)
    ds.requires(medium_phase_sampler)
    ds.add_parameters(radiance = _maps.DeltatrackingPathIntegrator(
        ds.sigma,
        ds.scattering_albedo,
        ds.emission,
        ds.environment,
        ds.phase_sampler,
        ds.boundary,
        ds.majorant,
        ds_epsilon
    ))

def medium_radiance_path_integrator_NEE_DT(ds: _internal.DependencySet, *, ds_epsilon: float = 0.1):
    ds.requires(medium_sigma)
    try:
        ds.requires(medium_scattering_albedo)
    except:
        ds.add_parameters(scattering_albedo=_maps.ZERO)
    try:
        ds.requires(medium_emission)
    except:
        ds.add_parameters(emission=_maps.ZERO)
    ds.requires(medium_boundary)
    ds.requires(medium_majorant)
    ds.requires(medium_environment)
    ds.requires(medium_environment_sampler)
    ds.requires(medium_phase)
    ds.requires(medium_phase_sampler)
    ds.requires(medium_transmittance)
    ds.add_parameters(radiance = _maps.DeltatrackingNEEPathIntegrator(
        ds.sigma,
        ds.scattering_albedo,
        ds.emission,
        ds.environment,
        ds.environment_sampler,
        ds.phase,
        ds.phase_sampler,
        ds.boundary,
        ds.majorant,
        ds.transmittance,
        ds_epsilon
    ))


def medium_radiance_path_integrator_NEE_DRTDS(ds: _internal.DependencySet, *, ds_epsilon: float = 0.1):
    ds.requires(medium_sigma)
    try:
        ds.requires(medium_scattering_albedo)
    except:
        ds.add_parameters(scattering_albedo=_maps.ZERO)
    try:
        ds.requires(medium_emission)
    except:
        ds.add_parameters(emission=_maps.ZERO)
    ds.requires(medium_boundary)
    ds.requires(medium_majorant)
    ds.requires(medium_environment)
    ds.requires(medium_environment_sampler)
    ds.requires(medium_phase)
    ds.requires(medium_phase_sampler)
    ds.requires(medium_transmittance)
    ds.add_parameters(radiance = _maps.DRTDSPathIntegrator(
        ds.sigma,
        ds.scattering_albedo,
        ds.emission,
        ds.environment,
        ds.environment_sampler,
        ds.phase,
        ds.phase_sampler,
        ds.boundary,
        ds.majorant,
        ds.transmittance,
        ds_epsilon
    ))


def medium_radiance_path_integrator_NEE_DRT(ds: _internal.DependencySet):
    ds.requires(medium_sigma)
    try:
        ds.requires(medium_scattering_albedo)
    except:
        ds.add_parameters(scattering_albedo=_maps.ZERO)
    try:
        ds.requires(medium_emission)
    except:
        ds.add_parameters(emission=_maps.ZERO)
    ds.requires(medium_boundary)
    ds.requires(medium_majorant)
    ds.requires(medium_environment)
    ds.requires(medium_environment_sampler)
    ds.requires(medium_phase)
    ds.requires(medium_phase_sampler)
    ds.requires(medium_transmittance)
    ds.add_parameters(radiance = _maps.DRTPathIntegrator(
        ds.sigma,
        ds.scattering_albedo,
        ds.emission,
        ds.environment,
        ds.environment_sampler,
        ds.phase,
        ds.phase_sampler,
        ds.boundary,
        ds.majorant,
        ds.transmittance
    ))


def medium_radiance_path_integrator_NEE_DRTQ(ds: _internal.DependencySet):
    ds.requires(medium_sigma)
    try:
        ds.requires(medium_scattering_albedo)
    except:
        ds.add_parameters(scattering_albedo=_maps.ZERO)
    try:
        ds.requires(medium_emission)
    except:
        ds.add_parameters(emission=_maps.ZERO)
    ds.requires(medium_boundary)
    ds.requires(medium_majorant)
    ds.requires(medium_environment)
    ds.requires(medium_environment_sampler)
    ds.requires(medium_phase)
    ds.requires(medium_phase_sampler)
    ds.requires(medium_transmittance)
    ds.add_parameters(radiance = _maps.DRTQPathIntegrator(
        ds.sigma,
        ds.scattering_albedo,
        ds.emission,
        ds.environment,
        ds.environment_sampler,
        ds.phase,
        ds.phase_sampler,
        ds.boundary,
        ds.majorant,
        ds.transmittance
    ))



def medium_radiance_path_integrator_NEE_SPS(ds: _internal.DependencySet):
    ds.requires(medium_sigma)
    try:
        ds.requires(medium_scattering_albedo)
    except:
        ds.add_parameters(scattering_albedo=_maps.ZERO)
    try:
        ds.requires(medium_emission)
    except:
        ds.add_parameters(emission=_maps.ZERO)
    ds.requires(medium_boundary)
    ds.requires(medium_majorant)
    ds.requires(medium_environment)
    ds.requires(medium_environment_sampler)
    ds.requires(medium_phase)
    ds.requires(medium_phase_sampler)
    ds.requires(medium_transmittance)
    ds.add_parameters(radiance = _maps.SPSPathIntegrator(
        ds.sigma,
        ds.scattering_albedo,
        ds.emission,
        ds.environment,
        ds.environment_sampler,
        ds.phase,
        ds.phase_sampler,
        ds.boundary,
        ds.majorant,
        ds.transmittance
    ))


def medium_radiance(ds: _internal.DependencySet):
    if ds.ensures('radiance', _maps.MapBase):
        return
    try:
        ds.requires(medium_radiance_path_integrator_DT)
    except:
        raise NotImplemented('Not supported default radiance')



#
#
# def medium_collision_integrator(ds: _internal.DependencySet, *,
#                                 sampling: Literal['uniform', 'transmittance', 'collision']):
#
#
#
# def medium_radiance(ds: _internal.DependencySet, *,
#                     model: Literal['ao', 'ae', 'so', 'se', 'mo', 'me'],
#                     ps: Literal['dt', 'drt', 'ssp'], **kwargs):
#     ds.include(medium_environment, sampling=kwargs.get('environment_sampling', 'uniform'))
#     if model == 'ao':



def camera_sensors(ds: _internal.DependencySet, *, width: int, height: int, jittered: bool = False, fov: float = _np.pi/4):
    ds.assert_ensures('camera_poses', _torch.Tensor)
    camera_poses = ds.camera_poses
    ds.add_parameters(camera=_maps.PerspectiveCameraSensor(width, height, camera_poses, fov=fov, jittered=jittered))

#
#
# ds = DependencySet()
# ds.add_parameters(sigma_grid=_torch.tensor([]))
# ds.include(medium_fields, fields=['sigma'])
# ds.include(transmittance, mode='dt')
# ds.include(environment_map)
# ds.include(absorption_only_RF)
#
# im = camera.capture(ds.radiance)
#
# build_environment__maps(ds)
#
#
# class Environment(DependentObject):
#
#     def __init__(self, parameter_set: _internal.DependencySet):
#         self._skybox = parameter_set.skybox
#         self._skybox_sampler = parameter_set.skybox_sampler
#
#     @property
#     def skybox(self) -> _maps.MapBase:
#         return self._skybox
#
#     @property
#     def skybox_sampler(self) -> _maps.MapBase:
#         return self._skybox_sampler
#
#
# class XREnvironment (Environment):
#     def __init__(self, skybox_img: _torch.Tensor, quadtree_levels: int = 10):
#         def build_quadtree():
#             with _torch.no_grad():
#                 resolution = 1 << quadtree_levels
#                 densities = resample_img(skybox_img.sum(-1, keepdim=True), (resolution, resolution))
#                 for py in range(resolution):
#                     w = _np.cos(py * _np.pi / resolution) - _np.cos((py + 1) * _np.pi / resolution)
#                     densities[py, :, :] *= w
#                 densities /= max(densities.sum(), 0.00000001)
#                 return create_density_quadtree(densities)
#
#         quadtree_densities = self.register_parameter(build_quadtree)
#         skybox_img2d = Image2D(skybox_img)
#         skybox_map = skybox_img2d.after(xr_projection())
#         skybox_random_direction = XRQuadtreeRandomDirection(6, quadtree_densities, quadtree_levels)
#         # skybox_random_direction = UniformRandomDirection(6)
#         skybox_sampler = _functionsampler(skybox_random_direction, skybox_map)
#         super(XREnvironment, self).__init__(skybox_map, skybox_sampler)
#
#
# class ExtinctionField(BoundObject):
#     def __init__(self,
#                  sigma: _maps.MapBase,
#                  boundary: _maps.MapBase,
#                  transmittance: _maps.MapBase,
#                  collision_sampler: _maps.MapBase,
#                  transmittance_sampler: _maps.MapBase
#                  ):
#
#
# class Medium(BoundObject):
#     def __init__(self,
#                  sigma: _maps.MapBase,
#                  scattering_albedo: Optional[MapBase],
#                  emission: Optional[MapBase],
#                  boundary: Optional[MapBase]):
#         self._sigma = sigma
#         self._scattering_albedo = scattering_albedo
#         self._emission = emission
#         self._boundary = boundary
#         self._transmittance = self.build_transmittance()
#
#     @property
#     def sigma(self):
#         return self._sigma
#
#     @property
#     def scattering_albedo(self):
#         return constant(3, 0.0, 0.0, 0.0) if self._scattering_albedo is None else self._scattering_albedo
#
#     @property
#     def emission(self):
#         return constant(3, 0.0, 0.0, 0.0) if self._emission is None else self._emission
#
#     @property
#     def is_absorption_only(self):
#         return self._scattering_albedo is None and self._emission is None
#
#     @property
#     def is_emission_only(self):
#         return self._scattering_albedo is None and self._emission is not None
#
#
# class GridMedium(Medium):
#     def __init__(self,
#                  sigma: Union[_torch.Tensor, Callable[[], _torch.Tensor]],
#                  scattering_albedo: Union[None, _torch.Tensor, Callable[[], _torch.Tensor]],
#                  emission: Union[None, _torch.Tensor, Callable[[], _torch.Tensor]]
#                  ):
#         sigma = self.register_parameter(sigma)
#         scattering_albedo = self.register_parameter(scattering_albedo)
#         emission = self.register_parameter(emission)
#         bmin, bmax = normalized_box(sigma)
#         boundary_map = RayBoxIntersection(bmin, bmax)
#         sigma_map = Grid3D(sigma, bmin=bmin, bmax=bmax)
#         scattering_albedo_map = None if scattering_albedo is None else Grid3D(scattering_albedo, bmin=bmin, bmax=bmax)
#         emission_map = None if emission is None else Grid3D(emission, bmin=bmin, bmax=bmax)
#         super(GridMedium, self).__init__(sigma_map, scattering_albedo_map, emission_map, boundary_map)
#


