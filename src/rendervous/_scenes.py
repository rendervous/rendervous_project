import torch as _torch
import numpy as _np

from . import _internal
from . import _maps
from . import _functions
import vulky as _vk
import typing as _typing


class Singleton(object):

    __instance__ = None

    @classmethod
    def get_instance(cls):
        if cls.__instance__ is None:
            cls.__instance__ = cls()
        return cls.__instance__


class Boundary(_maps.MapBase):
    __extension_info__ = None  # abstract node

    def __len__(self):
        raise NotImplementedError()

    @staticmethod
    def signature():
        return (6, 3)


class MeshBoundary(Boundary):
    __extension_info__ = dict(
        parameters=dict(
            mesh_ads=_torch.int64,
        ),
        generics=dict(
            INPUT_DIM=6,  # x, w
            OUTPUT_DIM=3,  # t, is_entering, patch_index
        ),
        path=_internal.__INCLUDE_PATH__+'/maps/boundary_mesh.h'
    )

    def __init__(self, mesh: 'MeshGeometry'):
        super().__init__()
        self.mesh = mesh
        self.mesh_ads = mesh.scene_ads.handle

    def __len__(self):
        return 1

    def _pre_eval(self, include_grads: bool = False):
        self.mesh.update_ads_if_necessary()
        super()._pre_eval(include_grads)


class GroupBoundary(Boundary):
    """
    Wraps a geometry to retrieve a distance map efficiently.
    Given a ray x,w returns t, is_entering and patch_index.
    """
    __extension_info__ = dict(
        parameters=dict(
            group_ads=_torch.int64,
            patches=[-1, _torch.int64]
        ),
        dynamics=[(6, 16)],
        generics=dict(
            INPUT_DIM=6, OUTPUT_DIM=3
        ),
        path=_internal.__INCLUDE_PATH__ + '/maps/boundary_group.h'
    )

    def __init__(self, geometry: 'GroupGeometry'):
        patches_info = geometry.per_patch_info()
        super().__init__(len(patches_info), ONLY_MESHES=geometry.is_only_meshes)
        self.geometry = geometry
        self.group_ads = geometry.group_ads
        for i, patch in enumerate(patches_info):
            self.patches[i] = patch

    def __len__(self):
        return len(self.geometry)

    def _pre_eval(self, include_grads: bool = False):
        self.geometry.update_ads_if_necessary()
        super()._pre_eval(include_grads)



# GEOMETRIES

class Geometry(_maps.MapBase):
    __extension_info__ = None  # abstract node

    @staticmethod
    def signature():
        return (6, 16)

    # @staticmethod
    # def _build_corners_for_aabb(bmin, bmax):
    #     return _torch.tensor([
    #         [bmin[0], bmin[1], bmin[2]],
    #         [bmin[0], bmin[1], bmax[2]],
    #         [bmin[0], bmax[1], bmin[2]],
    #         [bmin[0], bmax[1], bmax[2]],
    #         [bmax[0], bmin[1], bmin[2]],
    #         [bmax[0], bmin[1], bmax[2]],
    #         [bmax[0], bmax[1], bmin[2]],
    #         [bmax[0], bmax[1], bmax[2]],
    #     ], device=bmin.device
    #     )
    #
    # @staticmethod
    # def _transform_aabb(bmin, bmax, transform):
    #     corners = Geometry._build_corners_for_aabb(bmin, bmax)
    #     pts = (corners @ transform[:3, :3]) + transform[3:4, :]
    #     world_min, _ = pts.min(dim=0)
    #     world_max, _ = pts.max(dim=0)
    #     return world_min, world_max

    # def __init__(self, *args, **generics):
    #     super().__init__(*args, **generics)
    #     # self.object_bmin = _torch.nn.Parameter(bmin, requires_grad=False)
    #     # self.object_bmax = _torch.nn.Parameter(bmax, requires_grad=False)

    # def object_aabb(self):
    #     '''
    #     When implemented, returns bmin, bmax in object space for this geometry
    #     '''
    #     return self.object_bmin, self.object_bmax

    # def top_aabb(self):
    #     '''
    #     When implemented returns bmin, bmax after transform for this geometry
    #     '''
    #     corners = Geometry._build_corners_for_aabb(self.object_bmin, self.object_bmax)
    #     pts = (corners @ self._transform[:3, :3]) + self._transform[3:4, :]
    #     world_min, _ = pts.min(dim=0)
    #     world_max, _ = pts.max(dim=0)
    #     return world_min, world_max

    # def get_transform(self):
    #     return self._transform

    @staticmethod
    def _create_patch_buffer(callable: _maps.MapBase, mesh_info_buffer: _typing.Optional[_vk.Buffer] = None):
        buf = _vk.object_buffer(layout=_vk.Layout.from_structure(
            _vk.LayoutAlignment.SCALAR,
            callable_map=_torch.int64,
            mesh_info=_torch.int64,
        )
        )
        with buf as b:
            b.callable_map = callable.__bindable__.device_ptr
            b.mesh_info = 0 if mesh_info_buffer is None else mesh_info_buffer.device_ptr
        return buf

    def __len__(self):
        '''
        When implemented, returns the number of patches for this geometry.
        '''
        return len(self.per_patch_info())

    def per_patch_info(self) -> _typing.List[int]:
        '''
        Gets the list of patch-descriptor references
        '''
        pass

    def per_patch_geometry(self) -> _typing.List[int]:
        '''
        Gets the list of patch geometry handles
        '''
        pass

    def update_patch_transforms(self, transforms: _torch.Tensor):
        pass

    def update_ads_if_necessary(self) -> bool:
        pass

    def get_boundary(self) -> 'Boundary':
        raise NotImplementedError()  # TODO: IMPLEMENT A DEFAULT WRAPPER

    def _pre_eval(self, include_grads: bool = False):
        self.update_ads_if_necessary()
        return super()._pre_eval(include_grads)

    def transformed(self, T: _torch.Tensor) -> 'TransformedGeometry':
        return TransformedGeometry(self, T)


class TransformedGeometry(Geometry):

    __extension_info__ = dict(
        generics=dict(INPUT_DIM=6, OUTPUT_DIM=16),
        parameters=dict(
            base_geometry=_maps.MapBase,
            transform=_maps.ParameterDescriptor
        ),
        path=_internal.__INCLUDE_PATH__ + '/maps/geometry_transformed.h'
    )

    def __init__(self, base_geometry: Geometry, initial_transform: _typing.Optional[_torch.Tensor] = None):
        super().__init__()
        self.base_geometry = base_geometry
        if initial_transform is None:
            initial_transform = _torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
        self.transform = _maps.parameter(initial_transform)
        self.transform_check = _maps.TensorCheck(initial_value=self.transform)

    def per_patch_geometry(self) -> _typing.List[int]:
        return self.base_geometry.per_patch_geometry()

    def per_patch_info(self) -> _typing.List[int]:
        return self.base_geometry.per_patch_info()

    def update_ads_if_necessary(self) -> bool:
        return self.transform_check.changed(self.transform) | self.base_geometry.update_ads_if_necessary()

    def update_patch_transforms(self, transforms: _torch.Tensor):
        with _torch.no_grad():
            for t in transforms:
                t.copy_(_vk.mat3x4.composite(self.transform.T, t))
        self.base_geometry.update_patch_transforms(transforms)


class MeshGeometry(Geometry):
    __extension_info__ = dict(
        parameters=dict(
            mesh_ads=_torch.int64,
            mesh_info=_torch.int64,
        ),
        generics=dict(
            INPUT_DIM=6,  # x, w
            OUTPUT_DIM=16,  # t, N, G, C, T, B, patch_index
        ),
        path=_internal.__INCLUDE_PATH__ + '/maps/geometry_mesh.h'
    )

    @staticmethod
    def compute_normals(positions: _torch.Tensor, indices: _torch.Tensor):
        normals = _torch.zeros_like(positions)
        indices = indices.long()  # to be used
        P0 = positions[indices[:, 0]]
        P1 = positions[indices[:, 1]]
        P2 = positions[indices[:, 2]]
        V1 = _vk.vec3.normalize(P1 - P0)
        V2 = _vk.vec3.normalize(P2 - P0)
        N = _torch.cross(V1, V2)  # do not normalize for proper weight
        indices0 = indices[:, 0].unsqueeze(-1).repeat(1, 3)
        normals.scatter_add_(0, indices0, N)
        indices1 = indices[:, 1].unsqueeze(-1).repeat(1, 3)
        normals.scatter_add_(0, indices1, N)
        indices2 = indices[:, 2].unsqueeze(-1).repeat(1, 3)
        normals.scatter_add_(0, indices2, N)
        return normals / _torch.sqrt((normals ** 2).sum(-1, keepdim=True))

    @staticmethod
    def load_obj(path: str) -> 'MeshGeometry':
        obj = _vk.load_obj(path)
        pos = obj['buffers']['P']
        bmin, _ = pos.min(dim=0)
        bmax, _ = pos.max(dim=0)
        pos = (pos - bmin - (bmax - bmin) * 0.5) * 2 / (bmax - bmin).max()
        return MeshGeometry(positions=pos, indices=obj['buffers']['P_indices'])

    @staticmethod
    def box():
        return MeshGeometry(
            positions=_torch.tensor(
                [
                    # Neg z
                    [-0.5, -0.5, -0.5],  # 0
                    [0.5, -0.5, -0.5],  # 1
                    [-0.5, 0.5, -0.5],  # 2
                    [0.5, 0.5, -0.5],  # 3
                    # Pos z

                    [-0.5, -0.5, 0.5],  # 4
                    [0.5, -0.5, 0.5],  # 5
                    [-0.5, 0.5, 0.5],  # 6
                    [0.5, 0.5, 0.5],  # 7

                    [-0.5, -0.5, -0.5],  # 8
                    [0.5, -0.5, -0.5],  # 9
                    [-0.5, -0.5, 0.5],  # 10
                    [0.5, -0.5, 0.5],  # 11

                    [-0.5, 0.5, -0.5],  # 12
                    [0.5, 0.5, -0.5],  # 13
                    [-0.5, 0.5, 0.5],  # 14
                    [0.5, 0.5, 0.5],  # 15

                    [-0.5, -0.5, -0.5],  # 16
                    [-0.5, 0.5, -0.5],  # 17
                    [-0.5, -0.5, 0.5],  # 18
                    [-0.5, 0.5, 0.5],  # 19

                    [0.5, -0.5, -0.5],  # 20
                    [0.5, 0.5, -0.5],  # 21
                    [0.5, -0.5, 0.5],  # 22
                    [0.5, 0.5, 0.5],  # 23
                ],
                device=_internal.device()
            ),
            # positions=_torch.tensor(
            #     [
            #         # Neg z
            #         [-1.0, -1.0, -1.0],  # 0
            #         [1.0, -1.0, -1.0],  # 1
            #         [-1.0, 1.0, -1.0],  # 2
            #         [1.0, 1.0, -1.0],  # 3
            #         # Pos z
            #
            #         [-1.0, -1.0, 1.0],  # 4
            #         [1.0, -1.0, 1.0],  # 5
            #         [-1.0, 1.0, 1.0],  # 6
            #         [1.0, 1.0, 1.0],  # 7
            #
            #         [-1.0, -1.0, -1.0],  # 8
            #         [1.0, -1.0, -1.0],  # 9
            #         [-1.0, -1.0, 1.0],  # 10
            #         [1.0, -1.0, 1.0],  # 11
            #
            #         [-1.0, 1.0, -1.0],  # 12
            #         [1.0, 1.0, -1.0],  # 13
            #         [-1.0, 1.0, 1.0],  # 14
            #         [1.0, 1.0, 1.0],  # 15
            #
            #         [-1.0, -1.0, -1.0],  # 16
            #         [-1.0, 1.0, -1.0],  # 17
            #         [-1.0, -1.0, 1.0],  # 18
            #         [-1.0, 1.0, 1.0],  # 19
            #
            #         [1.0, -1.0, -1.0],  # 20
            #         [1.0, 1.0, -1.0],  # 21
            #         [1.0, -1.0, 1.0],  # 22
            #         [1.0, 1.0, 1.0],  # 23
            #     ],
            #     device=device()
            # ),
            indices=_torch.tensor(
                [
                    [0, 2, 3],
                    [0, 3, 1],
                    [4, 7, 6],
                    [4, 5, 7],

                    [8, 11, 10],
                    [8, 9, 11],
                    [12, 14, 15],
                    [12, 15, 13],

                    [16, 18, 19],
                    [16, 19, 17],
                    [20, 23, 22],
                    [20, 21, 23],
                ], dtype=_torch.int32, device=_internal.device()
            ), normals=None, uvs=_torch.tensor([
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],

                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],

                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],

                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],

                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],

                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
            ], device=_internal.device()),
            normalize=True
        )

    def __init__(self, positions: _torch.Tensor, normals: _typing.Optional[_torch.Tensor] = None,
                 uvs: _typing.Optional[_torch.Tensor] = None, indices: _typing.Optional[_torch.Tensor] = None, normalize: bool = False, compute_normals: bool = True):
        # if indices is None:  # assume default indices for all positions
        #     assert len(positions) % 3 == 0, 'If no indices is provided, positions should be divisible by 3'
        #     indices = _torch.arange(0, len(positions), 1, dtype=_torch.int32, device=positions.device).view(-1, 3)
        if normals is None and compute_normals:
            normals = MeshGeometry.compute_normals(positions, indices)
        super().__init__()
        if normalize:
            with _torch.no_grad():
                bmin, _ = positions.min(dim=0)
                bmax, _ = positions.max(dim=0)
                positions = (positions - bmin - (bmax - bmin) * 0.5) * 2 / (bmax - bmin).max()
        self.mesh_info_buffer = _vk.object_buffer(layout=_vk.Layout.from_structure(
            _vk.LayoutAlignment.SCALAR,
            positions=_maps.ParameterDescriptorLayoutType,
            normals=_maps.ParameterDescriptorLayoutType,
            coordinates=_maps.ParameterDescriptorLayoutType,
            tangents=_maps.ParameterDescriptorLayoutType,
            binormals=_maps.ParameterDescriptorLayoutType,
            indices=_maps.ParameterDescriptorLayoutType
        ), memory=_vk.MemoryLocation.GPU, usage=_vk.BufferUsage.STORAGE)
        self.mesh_info_module = _maps.StructModule(self.mesh_info_buffer.accessor)
        self.mesh_info_module.positions = _maps.parameter(positions)
        self.mesh_info_module.normals = _maps.parameter(normals)
        self.mesh_info_module.coordinates = _maps.parameter(uvs)
        self.mesh_info_module.tangents = _maps.parameter(None)
        self.mesh_info_module.binormals = _maps.parameter(None)
        self.mesh_info_module.indices = _maps.parameter(indices)
        self.mesh_info = self.mesh_info_buffer.device_ptr
        self._patch_info = Geometry._create_patch_buffer(self, self.mesh_info_buffer)
        self._per_patch_info = [self._patch_info.device_ptr]
        # create bottom ads
        # Create vertices for the triangle
        vertices = _vk.structured_buffer(
            count=len(positions),
            element_description=dict(
                position=_vk.vec3
            ),
            usage=_vk.BufferUsage.RAYTRACING_RESOURCE,
            memory=_vk.MemoryLocation.GPU
        )
        # Create indices for the faces
        indices_buffer = _vk.structured_buffer(
            count=len(indices) * 3,
            element_description=int,
            usage=_vk.BufferUsage.RAYTRACING_RESOURCE,
            memory=_vk.MemoryLocation.GPU
        )
        # vertices.load([ vec3(-0.6, 0, 0), vec3(0.6, 0, 0), vec3(0, 1, 0)])
        with vertices.map('in') as v:
            # v.position refers to all positions at once...
            v.position = positions
        indices_buffer.load(indices.view(-1))
        # Create a triangle collection
        self.geometry = _vk.triangle_collection()
        self.geometry.append(vertices=vertices, indices=indices_buffer)
        # Create a bottom ads with the geometry
        self.geometry_ads = _vk.ads_model(self.geometry)

        self._per_patch_geometry = [self.geometry_ads.handle]

        # Create an instance buffer for the top level objects
        self.scene_buffer = _vk.instance_buffer(1, _vk.MemoryLocation.GPU)
        with self.scene_buffer.map('in', clear=True) as s:
            s.flags = 0
            s.mask8_idx24 = _vk.asint32(0xFF000000)
            # By default, all other values of the instance are filled
            # for instance, transform with identity transform and 0 offset.
            # mask with 255
            s.transform[0][0] = 1.0
            s.transform[1][1] = 1.0
            s.transform[2][2] = 1.0
            s.accelerationStructureReference = self.geometry_ads.handle

        # Create the top level ads
        self.scene_ads = _vk.ads_scene(self.scene_buffer)

        # scratch buffer used to build the ads shared for all ads'
        self.scratch_buffer = _vk.scratch_buffer(self.geometry_ads, self.scene_ads)
        self.position_checker = _maps.TensorCheck()
        self.mesh_ads = self.scene_ads.handle
        self.update_ads_if_necessary()
        self.mesh_info_buffer.update_gpu()

    # def _pre_eval(self, include_grads: bool = False):
    #     super()._pre_eval(include_grads)
    #     # self.mesh_info_buffer.update_gpu()

    def update_ads_if_necessary(self) -> bool:
        if self.position_checker.changed(self.mesh_info_module.positions):
            with _vk.raytracing_manager() as man:
                man.build_ads(self.geometry_ads, self.scratch_buffer)
                man.build_ads(self.scene_ads, self.scratch_buffer)
            return True
        return False

    def __len__(self):
        return 1

    def get_boundary(self) -> 'Boundary':
        return MeshBoundary(self)

    def per_patch_geometry(self) -> _typing.List[int]:
        return self._per_patch_geometry

    def per_patch_info(self) -> _typing.List[int]:
        return self._per_patch_info

    def update_patch_transforms(self, transforms: _torch.Tensor):
        return


class GroupGeometry(Geometry):
    __extension_info__ = dict(
        dynamics=[(6, 16)],  # can call dynamically to other geometries
        parameters=dict(
            group_ads=_torch.int64,
            patches=[-1, _torch.int64]  # flatten representation of all patches
        ),
        generics=dict(
            INPUT_DIM=6,  # x, w
            OUTPUT_DIM=16,  # t, N, G, C, T, B, patch_index
        ),
        path=_internal.__INCLUDE_PATH__ + '/maps/geometry_group.h'
    )


    @staticmethod
    def _flatten_patches(geometries: _typing.Iterable[Geometry]):
        patch_geometries = []
        patch_infos = []
        for g in geometries:
            patch_infos.extend(g.per_patch_info())
            patch_geometries.extend(g.per_patch_geometry())
        return patch_infos, patch_geometries

    def __init__(self, *geometries: Geometry):
        # Tag with a generic that a new geometry collection is needed to differentiate from others within nested dynamic calls
        _per_patch_infos, _per_patch_geometries = GroupGeometry._flatten_patches(geometries)
        def all_meshes(g):
            if isinstance(g, MeshGeometry):
                return True
            if isinstance(g, TransformedGeometry):
                return all_meshes(g.base_geometry)
            if isinstance(g, GroupGeometry):
                return all(all_meshes(geom) for geom in g.geometries)
            return False

        is_only_meshes = 1 if all(all_meshes(geom) for geom in geometries) else 0
        super().__init__(len(_per_patch_infos), ONLY_MESHES=is_only_meshes)
        self._per_patch_infos, self._per_patch_geometries = _per_patch_infos, _per_patch_geometries
        self._per_patch_geometries_tensor = _torch.tensor(_per_patch_geometries, dtype=_torch.int64).unsqueeze(-1)
        self.geometries_module_list = _torch.nn.ModuleList(geometries)
        self.geometries = geometries
        for i, patch in enumerate(self._per_patch_infos):
            self.patches[i] = patch
        self.is_only_meshes = is_only_meshes
        # Create an instance buffer for the top level objects
        self.scene_buffer = _vk.instance_buffer(len(self._per_patch_geometries), _vk.MemoryLocation.GPU)
        # Create the top level ads
        self.scene_ads = _vk.ads_scene(self.scene_buffer)
        # scratch buffer used to build the ads shared for all ads'
        self.scratch_buffer = _vk.scratch_buffer(self.scene_ads)
        self.group_ads = self.scene_ads.handle
        self.cached_transforms = _torch.empty(len(self._per_patch_geometries), 3, 4)
        self.cached_transforms[:] = GroupGeometry.__identity__
        self._update_ads()

    __identity__ = _torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])

    def per_patch_geometry(self) -> _typing.List[int]:
        return self._per_patch_geometries

    def per_patch_info(self) -> _typing.List[int]:
        return self._per_patch_infos

    def update_patch_transforms(self, transforms: _torch.Tensor):
        offset = 0
        for g in self.geometries:
            p = len(g)
            g.update_patch_transforms(transforms[offset:offset + p])
            offset += p

    def _update_ads(self):
        current_transforms = _torch.empty(len(self._per_patch_geometries), 3, 4)
        current_transforms[:] = GroupGeometry.__identity__
        self.update_patch_transforms(current_transforms)
        with self.scene_buffer.map('in') as s:
            # notice that these updates are serial
            s.flags = 0
            s.mask8_idx24 = _vk.asint32(0xFF000000)
            s.accelerationStructureReference = self._per_patch_geometries_tensor
            s.transform = current_transforms
        with _vk.raytracing_manager() as man:
            man.build_ads(self.scene_ads, self.scratch_buffer)
        self.cached_transforms = current_transforms

    def update_ads_if_necessary(self):
        any_geometry_changed = False
        for g in self.geometries:
            any_geometry_changed |= g.update_ads_if_necessary()
        if any_geometry_changed: # or not _torch.equal(current_transforms, self.cached_transforms):
            self._update_ads()

    def get_boundary(self) -> 'Boundary':
        return GroupBoundary(self)


# ENVIRONMENTS


class Environment(_maps.MapBase):
    __extension_info__ = None  # abstract node

    @staticmethod
    def create_extension_info(
            path_or_code: str,
            **parameters
    ):
        if ',' in path_or_code:
            code = """
    #include "environment_interface.h"
            """ + path_or_code
            path = None
        else:
            path = path_or_code
            code = None
        return dict(
            generics=dict(INPUT_DIM=3, OUTPUT_DIM=3),
            parameters=parameters,
            path=path,
            code=code,
            bw_implementations=_maps.BACKWARD_IMPLEMENTATIONS.DEFAULT
        )

    @staticmethod
    def signature():
        return (3, 3)


class EnvironmentSamplerPDF(_maps.MapBase):
    __extension_info__ = None  # abstract node

    @staticmethod
    def create_extension_info(
            path_or_code: str,
            **parameters
    ):
        if ',' in path_or_code:
            code = """
        #include "environment_sampler_pdf_interface.h"
                """ + path_or_code
            path = None
        else:
            path = path_or_code
            code = None
        return dict(
            generics=dict(INPUT_DIM=6, OUTPUT_DIM=1),
            parameters=parameters,
            bw_implementations=_maps.BACKWARD_IMPLEMENTATIONS.NONE, # sampler pdfs are not differentiable
            path=path,
            code=code
        )

    @staticmethod
    def signature():
        return (6, 1)


class EnvironmentSampler(_maps.MapBase):  # x -> we, E(we)/pdf(we), pdf(we)

    __extension_info__ = None  # abstract node

    @staticmethod
    def create_extension_info(
            path_or_code: str,
            **parameters
    ):
        if ',' in path_or_code:
            code = """
        #include "environment_sampler_interface.h"
                """ + path_or_code
            path = None
        else:
            path = path_or_code
            code = None
        return dict(
            generics = dict(INPUT_DIM=3, OUTPUT_DIM=7),
            parameters = parameters,
            bw_implementations=_maps.BACKWARD_IMPLEMENTATIONS.WITH_OUTPUT,
            path=path,
            code=code,
        )

    def get_environment(self) -> _maps.MapBase:  # we -> E(we)
        raise NotImplementedError()

    def get_pdf(self) -> _maps.MapBase:   # x, we -> pdf(we)
        raise NotImplementedError()

    @staticmethod
    def signature():
        return (3, 7)


class XREnvironment(Environment):
    __extension_info__ = Environment.create_extension_info(
        _internal.__INCLUDE_PATH__ + '/maps/environment_xr.h',
        environment_img = _maps.MapBase
    )
    def __init__(self, environment_img: _maps.MapBase):
        super().__init__()
        self.environment_img = environment_img.cast(2, 3)


class XREnvironmentSamplerPDF(EnvironmentSamplerPDF):
    __extension_info__ = EnvironmentSamplerPDF.create_extension_info(
        _internal.__INCLUDE_PATH__ + '/maps/environment_sampler_pdf_xr.h',
        densities=_torch.Tensor,  # quadtree probabilities
        levels=int,  # Number of levels of the quadtree
    )

    def __init__(self, densities: _torch.Tensor, levels: int):
        super().__init__()
        self.densities = densities
        self.levels = levels


class XREnvironmentSampler(EnvironmentSampler):
    __extension_info__ = EnvironmentSampler.create_extension_info(
        _internal.__INCLUDE_PATH__+'/maps/environment_sampler_xr.h',
        environment=_maps.MapBase,  # Map with ray-dependent radiances
        densities=_torch.Tensor,  # quadtree probabilities
        levels=int,  # Number of levels of the quadtree
    )

    def __init__(self, environment_img: _maps.MapBase, quadtree: _torch.Tensor, levels: int):
        environment_img = environment_img.cast(2, 3)
        super().__init__()
        self.environment_img = environment_img
        self.environment = XREnvironment(environment_img)
        self.densities = quadtree
        self.levels = levels
        self.pdf = XREnvironmentSamplerPDF(self.densities, self.levels)

    def get_environment(self) -> Environment:
        return self.environment

    def get_pdf(self) -> EnvironmentSamplerPDF:
        return self.pdf

    @staticmethod
    def from_tensor(environment_tensor: _torch.Tensor, levels: int = 10) -> 'XREnvironmentSampler':
        resolution = 1 << levels
        densities = _functions.resample_img(environment_tensor.sum(-1, keepdim=True), (resolution, resolution))
        for py in range(resolution):
            w = _np.cos(py * _np.pi / resolution) - _np.cos((py + 1) * _np.pi / resolution)
            densities[py, :, :] *= w
        densities /= max(densities.sum(), 0.00000001)
        quadtree = _functions.create_density_quadtree(densities)
        return XREnvironmentSampler(_maps.Image2D(environment_tensor), quadtree, levels)


class DeltaEnvironmentSampler(EnvironmentSampler):
    __extension_info__ = EnvironmentSampler.create_extension_info(
        _internal.__INCLUDE_PATH__ + '/maps/environment_sampler_delta.h',
        light_direction=_maps.ParameterDescriptor,
        light_intensity=_maps.ParameterDescriptor
    )
    def __init__(self, light_direction: _vk.vec3, light_intensity: _vk.vec3):
        super().__init__()
        self.light_direction = _maps.parameter(light_direction)
        self.light_intensity = _maps.parameter(light_intensity)

    def get_pdf(self) -> EnvironmentSamplerPDF:
        return _maps.ONE.cast(*EnvironmentSamplerPDF.signature())

    def get_environment(self) -> _maps.MapBase:
        return _maps.ZERO.cast(*Environment.signature())


class UniformEnvironmentSampler(EnvironmentSampler):
    __extension_info__ = EnvironmentSampler.create_extension_info(
        _internal.__INCLUDE_PATH__ + '/maps/environment_sampler_uniform.h',
        environment=_maps.MapBase
    )

    def __init__(self, environment: _maps.MapBase):
        super().__init__()
        self.environment = environment.cast(*Environment.signature())

    def get_pdf(self) -> EnvironmentSamplerPDF:
        return _maps.const[0.25 / _np.pi]

    def get_environment(self) -> _maps.MapBase:
        return self.environment
# class UniformEnvironmentSampler(EnvironmentSampler):
#     __extension_info__ = EnvironmentSampler.create_extension_info("""
#         vec3 we = randomDirection();
#         pdf = 1/(4 * pi);
#         E = environment(object, x, we);
#     """, parameters=dict(
#         environment=MapBase
#     ))


# SURFACES


class SurfaceScattering(_maps.MapBase):  # wi, wo, P, N, G, C, T, B-> W
    """
    Represents a cosine weighted BSDF
    """
    __extension_info__ = None  # Abstract Node
    @staticmethod
    def create_extension_info(
        path_or_code: str,
        **parameters
    ):
        if ',' in path_or_code:
            code = """
        #include "surface_scattering_interface.h"
                """ + path_or_code
            path = None
        else:
            path = path_or_code
            code = None
        return dict(
            generics=dict(INPUT_DIM=23, OUTPUT_DIM=3),
            bw_implementations=_maps.BACKWARD_IMPLEMENTATIONS.DEFAULT,
            parameters=parameters,
            code=code,
            path=path,
        )

    @staticmethod
    def signature():
        return (23, 3)


class SurfaceScatteringSamplerPDF(_maps.MapBase):  # wi, wo, P, N, G, C, T, B-> pdf(wo)
    """
    Represents the outgoing direction pdf of a cosine weighted BSDF sampler
    """
    __extension_info__ = None  # Abstract Node
    @staticmethod
    def create_extension_info(
        path_or_code: str,
        **parameters
    ):
        if ',' in path_or_code:
            code = """
        #include "surface_scattering_sampler_pdf_interface.h"
                """ + path_or_code
            path = None
        else:
            path = path_or_code
            code = None
        return dict(
            generics=dict(INPUT_DIM=23, OUTPUT_DIM=1),
            bw_implementations=_maps.BACKWARD_IMPLEMENTATIONS.NONE,
            parameters=parameters,
            code=code,
            path=path
        )

    @staticmethod
    def signature():
        return (23, 1)


class SurfaceScatteringSampler(_maps.MapBase):  # win, N, C, P, T, B -> wout, W/pdf(wo), pdf(W,wo)
    __extension_info__ = None  # Abstract Node
    @staticmethod
    def create_extension_info(
        path_or_code: str,
        **parameters
    ):
        if ',' in path_or_code:
            code = """
        #include "surface_scattering_sampler_interface.h"
                """ + path_or_code
            path = None
        else:
            path = path_or_code
            code = None
        return dict(
            generics=dict(INPUT_DIM=20, OUTPUT_DIM=7),
            parameters=parameters,
            bw_implementations=_maps.BACKWARD_IMPLEMENTATIONS.WITH_OUTPUT,
            code=code,
            path=path,
        )

    @staticmethod
    def signature():
        return (20, 7)

    def get_scattering(self) -> _typing.Union[_maps.MapBase, SurfaceScattering]:
        raise NotImplementedError()

    def get_pdf(self) -> _typing.Union[_maps.MapBase, SurfaceScatteringSamplerPDF]:
        raise NotImplementedError()


class LambertSurfaceScattering(SurfaceScattering, Singleton):
    __extension_info__ = SurfaceScattering.create_extension_info(
        f"""
    void surface_scattering(map_object, vec3 win, vec3 wout, Surfel surfel, out vec3 W) {{
        INCLUDE_ADDITIONAL_COLLISION_VARS
        float cosine_theta = dot(wout, fN);
        W = vec3(cosine_theta <= 0 ? 0.0 : cosine_theta / pi);
    }}
    void surface_scattering_bw(map_object, vec3 win, vec3 wout, Surfel surfel, vec3 dL_dW) {{
    }}
        """
    )


class LambertSurfaceScatteringSamplerPDF(SurfaceScatteringSamplerPDF, Singleton):
    __extension_info__ = SurfaceScatteringSamplerPDF.create_extension_info(f"""
    void surface_scattering_sampler_pdf(map_object, vec3 win, vec3 wout, Surfel surfel, out float pdf) {{
        INCLUDE_ADDITIONAL_COLLISION_VARS
        float cosine_theta = dot(wout, fN);
        pdf = cosine_theta <= 0 ? 0.0 : cosine_theta / pi;
    }}
    """)


class LambertSurfaceScatteringSampler(SurfaceScatteringSampler, Singleton):
    __extension_info__ = SurfaceScatteringSampler.create_extension_info(
        f"""
    void surface_scattering_sampler(map_object, vec3 win, Surfel surfel, out vec3 wout, out vec3 W, out float pdf)
    {{
        INCLUDE_ADDITIONAL_COLLISION_VARS
        float cosine_theta;
        if (!from_outside)
        {{
            wout = vec3(0.0);
            pdf = 0.0;
            W = vec3(0.0);
        }}
        else {{
            wout = randomHSDirectionCosineWeighted(fN, cosine_theta);
            pdf = cosine_theta / pi;
            W = vec3(1.0);
        }}
    }}
    
    void surface_scattering_sampler_bw(map_object, vec3 win, Surfel surfel, vec3 out_wout, vec3 out_W, vec3 dL_dwout, vec3 dL_dW)
    {{
        // TODO: Implement backward here!
    }}    
        """
    )

    def get_scattering(self):
        return LambertSurfaceScattering.get_instance()

    def get_pdf(self):
        return LambertSurfaceScatteringSamplerPDF.get_instance()


class DeltaSurfaceScattering(SurfaceScattering, Singleton):
    __extension_info__ = dict(
        generics=dict(INPUT_DIM=23, OUTPUT_DIM=3),
        code="""
    FORWARD {
        _output = float[3] (0.0, 0.0, 0.0);
    }
        """,
        bw_implementations=_maps.BACKWARD_IMPLEMENTATIONS.NONE,
    )


class DeltaSurfaceScatteringPDF(SurfaceScatteringSamplerPDF, Singleton):
    __extension_info__ = dict(
        generics=dict(INPUT_DIM=23, OUTPUT_DIM=1),
        code="""
        FORWARD {
            _output = float[1] (0.0);
        }
            """,
        bw_implementations=_maps.BACKWARD_IMPLEMENTATIONS.NONE,
    )


class FresnelSurfaceScatteringSampler(SurfaceScatteringSampler):
    __extension_info__ = SurfaceScatteringSampler.create_extension_info(f"""
    void surface_scattering_sampler(map_object, vec3 win, Surfel surfel, out vec3 wout, out vec3 W, out float pdf)
    {{
        INCLUDE_ADDITIONAL_COLLISION_VARS
        vec3 reflect_w = reflect(win, fN);
        float eta = from_outside ? 1 / parameters.refraction_index : parameters.refraction_index;
        vec3 refract_w = refract(win, fN, eta);
        float theta = abs(dot(win, fN));
        float f = pow((1 - eta)/(1 + eta), 2);
        float R = f + (1 - f) * pow(1 - theta, 5);
        if (refract_w == vec3(0.0) || random() < R) {{
            wout = reflect_w;
        }}
        else {{
            wout = refract_w;
        }}
        W = vec3(1.0/(eta*eta));
        pdf = -1.0;
    }}
    
    void surface_scattering_sampler_bw(map_object, vec3 win, Surfel surfel, vec3 out_wout, vec3 out_W, vec3 dL_dwout, vec3 dL_dW)
    {{
    }}
    """,
    refraction_index=float
    )

    def __init__(self, refraction_index: float = 1.5):
        super().__init__()
        self.refraction_index = refraction_index

    def get_scattering(self):
        return DeltaSurfaceScattering.get_instance()

    def get_pdf(self):
        return DeltaSurfaceScatteringPDF.get_instance()


class MirrorSurfaceScatteringSampler(SurfaceScatteringSampler, Singleton):
    __extension_info__ = SurfaceScatteringSampler.create_extension_info(f"""
    void surface_scattering_sampler(map_object, vec3 win, Surfel surfel, out vec3 wout, out vec3 W, out float pdf)
    {{
        INCLUDE_ADDITIONAL_COLLISION_VARS
        wout = reflect(win, fN);
        W = vec3(1.0);
        pdf = -1.0;
    }}
    
    void surface_scattering_sampler_bw(map_object, vec3 win, Surfel surfel, vec3 out_wout, vec3 out_W, vec3 dL_dwout, vec3 dL_dW)
    {{
    }}
    """)

    def get_scattering(self):
        return DeltaSurfaceScattering.get_instance()

    def get_pdf(self):
        return DeltaSurfaceScatteringPDF.get_instance()


class SpectralFresnelSurfaceScatteringSampler(SurfaceScatteringSampler):
    __extension_info__ = SurfaceScatteringSampler.create_extension_info(f"""
    void surface_scattering_sampler(map_object, vec3 win, Surfel surfel, out vec3 wout, out vec3 W, out float pdf)
    {{
        INCLUDE_ADDITIONAL_COLLISION_VARS
        vec3 reflect_w = reflect(win, fN);
        float spectral_alpha = random() * 6;
        int index = int(spectral_alpha);
        float refraction_index = parameters.refraction_index*(1 + 0.01*spectral_alpha);
        vec3 colors [6] = {{ 
            vec3(3.0, 0.0, 0.0), 
            vec3(1.5, 1.5, 0.0),
            vec3(0.0, 3.0, 0.0),
            vec3(0.0, 1.5, 1.5),
            vec3(0.0, 0.0, 3.0), 
            vec3(1.5, 0.0, 1.5)
        }};
        vec3 Cr = colors[index];
        float eta = from_outside ? 1/refraction_index : refraction_index;
        vec3 refract_w = refract(win, fN, eta);
    
        float theta = -dot(win, fN);
        float f = pow((1 - 1/refraction_index)/(1 + 1/refraction_index), 2);
        float R = f + (1 - f) * pow(1 - theta, 5);
    
        if (refract_w == vec3(0.0) || random() < R) {{
            wout = reflect_w;
        }}
        else {{
            wout = refract_w;
        }}
        W = Cr / vec3(eta*eta);
        pdf = -1.0;
    }}
    
    void surface_scattering_sampler_bw(map_object, vec3 win, Surfel surfel, vec3 out_wout, vec3 out_W, vec3 dL_dwout, vec3 dL_dW)
    {{
    }}
    """,
                                                                        parameters=dict(
                                                                            refraction_index=float
                                                                        )
                                                                        )

    def __init__(self, refraction_index: float = 1.5):
        super().__init__()
        self.refraction_index = refraction_index

    def get_scattering(self):
        return DeltaSurfaceScattering.get_instance()

    def get_pdf(self):
        return DeltaSurfaceScatteringPDF.get_instance()


class AlbedoSurfaceScattering(SurfaceScattering):
    __extension_info__ = dict(
        parameters=dict(
            base_scattering=_maps.MapBase,
            albedo=_maps.MapBase,
        ),
        generics=dict(INPUT_DIM=23, OUTPUT_DIM=3),
        bw_implementations=_maps.BACKWARD_IMPLEMENTATIONS.DEFAULT,
        code=f"""
    FORWARD {{
        forward(parameters.base_scattering, _input, _output);
        float[3] albedo;
    #if ALBEDO_MAP_DIM == 2
        float[2] C = float[2](_input[15],_input[16]);
        forward(parameters.albedo, C, albedo);
    #endif
    #if ALBEDO_MAP_DIM == 3
        float[3] P = float[3](_input[6], _input[7], _input[8]);
        forward(parameters.albedo, P, albedo);
    #endif
        _output[0] *= albedo[0];
        _output[1] *= albedo[1];
        _output[2] *= albedo[2];
    }}
    
    BACKWARD {{
        float _output_scattering[3];
        forward(parameters.base_scattering, _input, _output_scattering);
        float[3] albedo;
    #if ALBEDO_MAP_DIM == 2
        float[2] C = float[2](_input[15],_input[16]);
        forward(parameters.albedo, C, albedo);
    #endif
    #if ALBEDO_MAP_DIM == 3
        float[3] P = float[3](_input[6], _input[7], _input[8]);
        forward(parameters.albedo, P, albedo);
    #endif
        float _grad[3];
        _grad = float[3](_output_grad[0] * albedo[0], _output_grad[1] * albedo[1], _output_grad[2] * albedo[2]);
        backward(parameters.base_scattering, _input, _output_scattering, _grad, _input_grad);
        _grad = float[3](_output_grad[0] * _output_scattering[0], _output_grad[1] * _output_scattering[1], _output_grad[2] * _output_scattering[2]);
    #if ALBEDO_MAP_DIM == 2
        float[2] C_grad;
        backward(parameters.albedo, C, albedo, _grad, C_grad);
        // TODO: From C_grad to _input_grad
    #endif
    #if ALBEDO_MAP_DIM == 3
        float[3] P_grad;
        backward(parameters.albedo, P, albedo, _grad, P_grad);
        // TODO: From P_grad to _input_grad
    #endif
    }}
        """
    )

    def __init__(self, base: SurfaceScattering, albedo: _maps.MapBase):
        assert albedo.input_dim == 2 or albedo.input_dim == 3, 'Can not use a map to albedo that is not 2D or 3D'
        albedo = albedo.cast(output_dim=3)
        super().__init__(ALBEDO_MAP_DIM = albedo.input_dim)
        self.base_scattering = base
        self.albedo = albedo


class AlbedoSurfaceScatteringSampler(SurfaceScatteringSampler):
    __extension_info__ = dict(
        parameters=dict(
            base_scattering_sampler=_maps.MapBase,
            albedo=_maps.MapBase
        ),
        generics=dict(INPUT_DIM=20, OUTPUT_DIM=7),
        bw_implementations = _maps.BACKWARD_IMPLEMENTATIONS.NONE,
        code=f"""
    FORWARD {{
        forward(parameters.base_scattering_sampler, _input, _output);
        float[3] albedo;
    #if ALBEDO_MAP_DIM == 2
        float[2] C = float[2](_input[12],_input[13]);
        forward(parameters.albedo, C, albedo);
    #endif
    #if ALBEDO_MAP_DIM == 3
        float[3] P = float[3](_input[3], _input[4], _input[5]);
        forward(parameters.albedo, P, albedo);
    #endif
        _output[3] *= albedo[0];
        _output[4] *= albedo[1];
        _output[5] *= albedo[2];
    }}
        """
    )

    def __init__(self, base: SurfaceScatteringSampler, albedo: _maps.MapBase):
        if albedo.is_generic_input:
            albedo = albedo.cast(input_dim=2)
        assert albedo.input_dim == 2 or albedo.input_dim == 3, 'Can not use a map to albedo that is not 2D or 3D'
        albedo = albedo.cast(output_dim = 3)
        super().__init__(ALBEDO_MAP_DIM = albedo.input_dim)
        self.base_scattering_sampler = base
        self.albedo = albedo

    def get_scattering(self):
        return AlbedoSurfaceScattering(self.base_scattering_sampler.get_scattering(), self.albedo)

    def get_pdf(self):
        return self.base_scattering_sampler.get_pdf()


class MixtureSurfaceScatter(SurfaceScattering):
    __extension_info__ = dict(
        parameters=dict(
            scattering_a=_maps.MapBase,
            scattering_b=_maps.MapBase,
            alpha=_maps.MapBase
        ),
        generics=dict(INPUT_DIM=23, OUTPUT_DIM=3),
        bw_implementations=_maps.BACKWARD_IMPLEMENTATIONS.NONE,
        code=f"""
    FORWARD {{
        float alpha[1];
    #if ALPHA_MAP_DIM == 2
        float[2] C = float[2](_input[15],_input[16]);
        forward(parameters.alpha, C, alpha);
    #endif
    #if ALPHA_MAP_DIM == 3
        float[3] P = float[3](_input[6], _input[7], _input[8]);
        forward(parameters.alpha, P, alpha);
    #endif
        float Wa[3];
        forward(parameters.scattering_a, _input, Wa);
        float Wb[3];
        forward(parameters.scattering_b, _input, Wb);
        _output = float[3](mix(Wa[0], Wb[0], alpha[0]), mix(Wa[1], Wb[1], alpha[0]), mix(Wa[2], Wb[2], alpha[0]));
    }}
        """
    )
    def __init__(self, scattering_a: SurfaceScattering, scattering_b: SurfaceScattering, alpha: _maps.MapBase):
        assert alpha.input_dim == 2 or alpha.input_dim == 3
        alpha = alpha.cast(output_dim = 1)
        super().__init__(ALPHA_MAP_DIM=alpha.input_dim)
        self.scattering_a = scattering_a
        self.scattering_b = scattering_b
        self.alpha = alpha


class MixtureSurfaceScatterSamplerPDF(SurfaceScatteringSamplerPDF):
    __extension_info__ = dict(
        parameters=dict(
            scattering_a_pdf=_maps.MapBase,
            scattering_b_pdf=_maps.MapBase,
            alpha=_maps.MapBase
        ),
        generics=dict(INPUT_DIM=23, OUTPUT_DIM=1),
        bw_implementations=_maps.BACKWARD_IMPLEMENTATIONS.NONE,
        code=f"""
    FORWARD {{
        float alpha[1];
    #if ALPHA_MAP_DIM == 2
        float[2] C = float[2](_input[15],_input[16]);
        forward(parameters.alpha, C, alpha);
    #endif
    #if ALPHA_MAP_DIM == 3
        float[3] P = float[3](_input[6], _input[7], _input[8]);
        forward(parameters.alpha, P, alpha);
    #endif
        float pdf_a[1];
        forward(parameters.scattering_a_pdf, _input, pdf_a);
        float pdf_b[1];
        forward(parameters.scattering_b_pdf, _input, pdf_b);
        _output[0] = mix(pdf_a[0], pdf_b[0], alpha[0]);
    }}
        """
    )
    def __init__(self, scattering_a_pdf: SurfaceScatteringSamplerPDF, scattering_b_pdf: SurfaceScatteringSamplerPDF, alpha: _maps.MapBase):
        assert alpha.input_dim == 2 or alpha.input_dim == 3
        alpha = alpha.cast(output_dim = 1)
        super().__init__(ALPHA_MAP_DIM=alpha.input_dim)
        self.scattering_a_pdf = scattering_a_pdf
        self.scattering_b_pdf = scattering_b_pdf
        self.alpha = alpha


class MixtureSurfaceScatteringSampler(SurfaceScatteringSampler):
    __extension_info__ = dict(
        parameters=dict(
            scattering_sampler_a=_maps.MapBase,
            scattering_sampler_b=_maps.MapBase,
            alpha=_maps.MapBase
        ),
        generics=dict(INPUT_DIM=20, OUTPUT_DIM=7),
        bw_implementations=_maps.BACKWARD_IMPLEMENTATIONS.NONE,
        code=f"""
        FORWARD {{
            float alpha[1];
        #if ALPHA_MAP_DIM == 2
            float[2] C = float[2](_input[12],_input[13]);
            forward(parameters.alpha, C, alpha);
        #endif
        #if ALPHA_MAP_DIM == 3
            float[3] P = float[3](_input[3], _input[4], _input[5]);
            forward(parameters.alpha, P, alpha);
        #endif
            if (random() < alpha[0]) // sample from a
                forward(parameters.scattering_sampler_a, _input, _output);
            else  // sample from b
                forward(parameters.scattering_sampler_b, _input, _output);
        }}
            """
    )
    def __init__(self, scattering_sampler_a: SurfaceScatteringSampler, scattering_sampler_b: SurfaceScatteringSampler, alpha: _maps.MapBase):
        if alpha.is_generic_input:
            alpha = alpha.cast(input_dim=2)
        assert alpha.input_dim == 2 or alpha.input_dim == 3
        alpha = alpha.cast(output_dim = 1)
        super().__init__(ALPHA_MAP_DIM=alpha.input_dim)
        self.scattering_sampler_a = scattering_sampler_a
        self.scattering_sampler_b = scattering_sampler_b
        self.alpha = alpha

    def get_scattering(self):
        return MixtureSurfaceScatter(
            self.scattering_sampler_a.get_scattering(),
            self.scattering_sampler_b.get_scattering(),
            self.alpha
        )

    def get_pdf(self):
        return MixtureSurfaceScatterSamplerPDF(
            self.scattering_sampler_a.get_pdf(),
            self.scattering_sampler_b.get_pdf(),
            self.alpha
        )


class NullSurfaceScatteringSampler(SurfaceScatteringSampler, Singleton):
    __extension_info__ = dict(
        generics=dict(INPUT_DIM=20, OUTPUT_DIM=7),
        bw_implementations=_maps.BACKWARD_IMPLEMENTATIONS.NONE,
        code=f"""
    FORWARD {{
        _output = float[7](_input[0], _input[1], _input[2], 1.0, 1.0, 1.0, -1.0);
    }}
        """
    )
    def get_scattering(self):
        return DeltaSurfaceScattering.get_instance()

    def get_pdf(self):
        return DeltaSurfaceScatteringPDF.get_instance()


class NoSurfaceScatteringSampler(SurfaceScatteringSampler):
    __extension_info__ = dict(
        generics=dict(INPUT_DIM=20, OUTPUT_DIM=7),
        bw_implementations=_maps.BACKWARD_IMPLEMENTATIONS.NONE,
        code=f"""
    FORWARD {{
        _output = float[7](0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }}
        """
    )

    def get_scattering(self):
        return DeltaSurfaceScattering.get_instance()

    def get_pdf(self):
        return DeltaSurfaceScatteringPDF.get_instance()


class SurfaceEmission(_maps.MapBase):
    __extension_info__ = None  # Abstract Node

    @staticmethod
    def create_extension_info(
            fw_code: str,
            bw_code: _typing.Optional[str] = None,
            external_code: _typing.Optional[str] = None,
            parameters: _typing.Optional[_typing.Dict] = None,
    ):
        if parameters is None:
            parameters = { }
        if bw_code is None:
            bw_code = ''
        if external_code is None:
            external_code = ''
        return dict(
            generics=dict(INPUT_DIM=20, OUTPUT_DIM=3),
            parameters=parameters,
            bw_implementations=_maps.BACKWARD_IMPLEMENTATIONS.NONE,
            code = f"""
{external_code if external_code is not None else ''}

FORWARD {{
    vec3 win = vec3(_input[0], _input[1], _input[2]);
    Surfel surfel = Surfel(
        vec3(_input[3], _input[4], _input[5]),
        vec3(_input[6], _input[7], _input[8]),
        vec3(_input[9], _input[10], _input[11]),
        vec2(_input[12], _input[13]),
        vec3(_input[14], _input[15], _input[16]),
        vec3(_input[17], _input[18], _input[19]));
    bool from_outside = dot(win, surfel.G) < 0;
    bool correct_hemisphere = (dot(win, surfel.N) < 0) == from_outside;
    vec3 N = correct_hemisphere ? surfel.N : surfel.G; 
    vec3 fN = from_outside ? N : -N;
    vec3 E;
    {fw_code}
    _output = float[3](E.x, E.y, E.z); 
}}
            """,

        )

    @staticmethod
    def signature():
        return (20, 3)


class IsotropicSurfaceEmission(SurfaceEmission):
    __extension_info__ = SurfaceEmission.create_extension_info(
        f"""
        #if EMISSION_MAP_DIM == 2
        forward(parameters.emission_map, float[2](_input[12], _input[13]), _output);    
        #endif
        #if EMISSION_MAP_DIM == 3
        forward(parameters.emission_map, float[3](_input[3], _input[4], _input[5]), _output);    
        #endif    
        float weight = 0.5 / pi;
        E = vec3(_output[0], _output[1], _output[2])*weight;
        """,
        parameters=dict(
            emission_map=_maps.MapBase
        )
    )
    def __init__(self, emission_map: _maps.MapBase):
        assert emission_map.input_dim == 2 or emission_map.input_dim == 3
        emission_map = emission_map.cast(output_dim = 3)
        super().__init__(EMISSION_MAP_DIM = emission_map.input_dim)
        self.emission_map = emission_map


class LambertSurfaceEmission(SurfaceEmission):
    __extension_info__ = SurfaceEmission.create_extension_info(
        f"""
        #if EMISSION_MAP_DIM == 2
        forward(parameters.emission_map, float[2](_input[12], _input[13]), _output);    
        #endif
        #if EMISSION_MAP_DIM == 3
        forward(parameters.emission_map, float[3](_input[3], _input[4], _input[5]), _output);    
        #endif
        float weight = dot(-win, fN) / pi;
        E = vec3(_output[0], _output[1], _output[2])*weight;
        """,
        parameters=dict(
            emission_map=_maps.MapBase
        )
    )
    def __init__(self, emission_map: _maps.MapBase):
        assert emission_map.input_dim == 2 or emission_map.input_dim == 3
        emission_map = emission_map.cast(output_dim = 3)
        super().__init__(EMISSION_MAP_DIM = emission_map.input_dim)
        self.emission_map = emission_map


class NoSurfaceEmission(SurfaceEmission):
    __extension_info__ = dict(
        generics=dict(INPUT_DIM=20, OUTPUT_DIM=3),
        bw_implementations=_maps.BACKWARD_IMPLEMENTATIONS.NONE,
        code=f"""
    FORWARD {{
        _output = float[3](0.0, 0.0, 0.0);
    }}
        """
    )

    __instance__ = None
    @classmethod
    def get_instance(cls):
        if cls.__instance__ is None:
            cls.__instance__ = NoSurfaceEmission()
        return cls.__instance__


class SurfaceGathering(_maps.MapBase): # win, P, N, G, C, T, B -> R
    __extension_info__ = None

    @staticmethod
    def create_extension_info(
            path_or_code: str,
            **parameters
    ):
        if ',' in path_or_code:
            code = """
    #include "surface_gathering_interface.h"
                """ + path_or_code
            path = None
        else:
            path = path_or_code
            code = None
        return dict(
            parameters=parameters,
            generics=dict(INPUT_DIM=20, OUTPUT_DIM=3),
            path=path,
            code=code,
            bw_implementations=_maps.BACKWARD_IMPLEMENTATIONS.WITH_OUTPUT
        )

    @staticmethod
    def signature():
        return (20, 3)


class NoSurfaceGathering(SurfaceGathering, Singleton):
    __extension_info__ = dict(
        bw_implementations=_maps.BACKWARD_IMPLEMENTATIONS.NONE,
        code = """
    FORWARD {
        _output = float[3](0.0, 0.0, 0.0);
    }
    """
    )


class EnvironmentGathering(SurfaceGathering):
    __extension_info__ = SurfaceGathering.create_extension_info(
        """
    #include "sc_environment_sampler.h"
    #include "sc_environment.h"
    #include "sc_surface_scattering.h"
    #include "sc_surface_scattering_sampler.h"
    #include "sc_visibility.h"

    void surface_gathering(map_object, vec3 win, Surfel surfel, out vec3 A) {
        INCLUDE_ADDITIONAL_COLLISION_VARS
        A = vec3(0.0);
        // Check scattering if 
        // x, s -> w, W, pdf
        vec3 x; 
        vec3 W, we; float we_pdf;
        vec3 E;
        surface_scattering_sampler(object, win, surfel, we, W, we_pdf);
        if (we_pdf == -1) // delta scattering
        {
            x = surfel.P + we * 0.0001;
            environment(object, we, E);
        }
        else 
        {
            x = fN * 0.0001 + surfel.P;
            float pdf;
            environment_sampler(object, x, we, E, pdf);
            surface_scattering(object, win, we, surfel, W);
        }
        if (E == vec3(0.0) || W == vec3(0.0))
            return;
        float Tr = ray_visibility(object, x, we); 
        A = Tr * W * E;
    }
    
    void surface_gathering_bw(map_object, vec3 win, Surfel surfel, vec3 out_A, vec3 dL_dA)
    {
        //TODO: Implement grad backprop here!
    }
    
        """,
        environment=_maps.MapBase,
        environment_sampler=_maps.MapBase,
        visibility=_maps.MapBase,
        surface_scattering=_maps.MapBase,
        surface_scattering_sampler=_maps.MapBase,
    )

    def __init__(self, environment_sampler: EnvironmentSampler, visibility: _maps.MapBase, surface_scattering_sampler: _typing.Optional[SurfaceScatteringSampler] = None):
        super().__init__()
        self.environment = environment_sampler.get_environment()
        self.environment_sampler = environment_sampler
        self.visibility = visibility.cast(6, 1)
        self.surface_scattering = _maps.ZERO.cast(*SurfaceScattering.signature()) if surface_scattering_sampler is None else surface_scattering_sampler.get_scattering()
        self.surface_scattering_sampler = _maps.const[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0].cast(*SurfaceScatteringSampler.signature()) if surface_scattering_sampler is None else surface_scattering_sampler


class SurfaceMaterial:
    def __init__(self, sampler: _typing.Optional[SurfaceScatteringSampler] = None, emission: _typing.Optional[SurfaceEmission] = None):
        self.sampler = sampler
        self.sampler_pdf = None if sampler is None else sampler.get_pdf()
        self.scattering = None if sampler is None else sampler.get_scattering()
        self.emission = emission

    def emission_gathering(self):
        return self.emission

    def environment_gathering(self, environment_sampler: EnvironmentSampler, visibility: _maps.MapBase):
        return None if self.scattering is None else (
            EnvironmentGathering(
                environment_sampler,
                visibility,
                self.sampler
            ))

    def environment_and_emission_gathering(self, environment_sampler: EnvironmentSampler, visibility: _maps.MapBase):
        env = self.environment_gathering(environment_sampler, visibility)
        em = self.emission_gathering()
        if em is None:
            return env
        if env is None:
            return em
        return env + em

    def photon_gathering(self, incomming_radiance: _maps.MapBase):
        pass


class MediumPathIntegrator(_maps.MapBase):
    __extension_info__ = None
    @staticmethod
    def create_extension_info(
            path: str,
            **parameters
    ):
        parameters.update(dict(gathering=_maps.MapBase))
        return dict(
            parameters = parameters,
            bw_implementations=_maps.BACKWARD_IMPLEMENTATIONS.DEFAULT,
            generics = dict(INPUT_DIM=7, OUTPUT_DIM=12),
            path=path
        )

    @staticmethod
    def signature():
        return (7, 12)

    def __init__(self, gathering: _typing.Optional[_maps.MapBase] = None, medium_filter: int = 3):
        if gathering is None:
           medium_filter &= 1  # only scatter
        super().__init__(MEDIUM_FILTER = medium_filter)
        # self.gathering_module = gathering
        # self.gathering = 0 if gathering is None else gathering.__bindable__.device_ptr # constant(6, 0., 0., 0.) if gathering is None else gathering
        self.gathering = (_maps.ZERO if gathering is None else gathering).cast(*VolumeGathering.signature())


class DTMediumPathIntegrator(MediumPathIntegrator):
    __extension_info__ = MediumPathIntegrator.create_extension_info(
        _internal.__INCLUDE_PATH__+"/maps/medium_path_integrator_DT.h",
        sigma=_maps.MapBase,
        scattering_albedo=_maps.MapBase,
        phase_sampler=_maps.MapBase,
        majorant=_maps.ParameterDescriptor,
    )

    def __init__(self, sigma: _maps.MapBase, majorant: _torch.Tensor, scattering_albedo: _typing.Optional[_maps.MapBase] = None, phase_sampler: _typing.Optional[_maps.MapBase] = None, gathering: _typing.Optional[_maps.MapBase] = None, medium_filter: int = 3):
        if scattering_albedo is None or phase_sampler is None:
            medium_filter &= 2
            scattering_albedo = _maps.ZERO
            phase_sampler = _maps.ZERO
        super().__init__(gathering, medium_filter=medium_filter)
        self.sigma = sigma.cast(3, 1)
        self.scattering_albedo = scattering_albedo.cast(3, 3)
        self.phase_sampler = phase_sampler.cast(*PhaseSampler.signature())
        self.majorant = _maps.parameter(majorant)


class DTVolumePathIntegrator(MediumPathIntegrator):
    __extension_info__ = MediumPathIntegrator.create_extension_info(
        _internal.__INCLUDE_PATH__ + '/maps/volume_path_integrator_DT.h',
        sigma=_maps.MapBase,
        scattering_albedo=_maps.MapBase,
        phase_sampler=_maps.MapBase,
        majorant=_maps.ParameterDescriptor,
        boundaries=_maps.MapBase
    )

    def __init__(self,
                 sigma: _maps.MapBase,
                 majorant: _torch.Tensor,
                 boundaries: Boundary,
                 scattering_albedo: _typing.Optional[_maps.MapBase] = None,
                 phase_sampler: _typing.Optional[_maps.MapBase] = None,
                 gathering: _typing.Optional[_maps.MapBase] = None,
                 medium_filter: int = 3):
        if scattering_albedo is None or phase_sampler is None:
            medium_filter &= 2
            scattering_albedo = _maps.ZERO
            phase_sampler = _maps.ZERO
        super().__init__(gathering, medium_filter=medium_filter)
        self.sigma = sigma.cast(3, 1)
        self.scattering_albedo = scattering_albedo.cast(3,3)
        self.phase_sampler = phase_sampler.cast(*PhaseSampler.signature())
        self.majorant = _maps.parameter(majorant)
        self.boundaries = boundaries


class DSVolumePathIntegrator(MediumPathIntegrator):
    __extension_info__ = MediumPathIntegrator.create_extension_info(
        _internal.__INCLUDE_PATH__ + '/maps/volume_path_integrator_DS.h',
        sigma=_maps.MapBase,
        scattering_albedo=_maps.MapBase,
        phase_sampler=_maps.MapBase,
        majorant=_maps.ParameterDescriptor,
        boundaries=_maps.MapBase,
        ds_epsilon=float
    )

    def __init__(self,
                 sigma: _maps.MapBase,
                 majorant: _torch.Tensor,
                 boundaries: Boundary,
                 scattering_albedo: _typing.Optional[_maps.MapBase] = None,
                 phase_sampler: _typing.Optional[_maps.MapBase] = None,
                 gathering: _typing.Optional[_maps.MapBase] = None,
                 ds_epsilon: float = 0.1,
                 medium_filter: int = 3):
        if scattering_albedo is None or phase_sampler is None:
            medium_filter &= 2
            scattering_albedo = _maps.ZERO
            phase_sampler = _maps.ZERO
        super().__init__(gathering, medium_filter=medium_filter)
        self.sigma = sigma
        self.scattering_albedo = scattering_albedo
        self.phase_sampler = phase_sampler
        self.majorant = _maps.parameter(majorant)
        self.boundaries = boundaries
        self.ds_epsilon = ds_epsilon



class RTMediumPathIntegrator(MediumPathIntegrator):
    __extension_info__ = MediumPathIntegrator.create_extension_info(
        _internal.__INCLUDE_PATH__ + "/maps/medium_path_integrator_RT.h",
        sigma=_maps.MapBase,
        scattering_albedo=_maps.MapBase,
        phase_sampler=_maps.MapBase,
        majorant=_maps.ParameterDescriptor,
    )

    def __init__(self, sigma: _maps.MapBase, majorant: _torch.Tensor, scattering_albedo: _typing.Optional[_maps.MapBase] = None, phase_sampler: _typing.Optional[_maps.MapBase] = None, gathering: _typing.Optional[_maps.MapBase] = None, medium_filter: int = 3):
        if scattering_albedo is None or phase_sampler is None:
            medium_filter &= 2
            scattering_albedo = _maps.ZERO
            phase_sampler = _maps.ZERO
        super().__init__(gathering, medium_filter=medium_filter)
        self.sigma = sigma.cast(3,1)
        self.scattering_albedo = scattering_albedo.cast(3,3)
        self.phase_sampler = phase_sampler.cast(*PhaseSampler.signature())
        self.majorant = _maps.parameter(majorant)



class HomogeneousMediumPathIntegrator(MediumPathIntegrator):
    __extension_info__ = MediumPathIntegrator.create_extension_info(
        _internal.__INCLUDE_PATH__ + "/maps/medium_path_integrator_CF.h",
        sigma=_maps.ParameterDescriptor,
        scattering_albedo=_maps.ParameterDescriptor,
        phase_sampler=_maps.MapBase
    )

    def __init__(self, sigma: _torch.Tensor, scattering_albedo: _typing.Optional[_torch.Tensor] = None,
                 phase_sampler: _typing.Optional[_maps.MapBase] = None, gathering: _typing.Optional[_maps.MapBase] = None, medium_filter: int = 3):
        if scattering_albedo is None or phase_sampler is None:
            medium_filter &= 2
            scattering_albedo = _torch.zeros(3, device=_internal.device())
            phase_sampler = _maps.ZERO
        super().__init__(gathering, medium_filter)
        self.sigma = _maps.parameter(sigma)
        self.scattering_albedo = _maps.parameter(scattering_albedo)
        self.phase_sampler = phase_sampler.cast(*PhaseSampler.signature())


# class VolumeEmissionGathering(MapBase):
#     __extension_info__ = dict(
#         generics=dict(INPUT_DIM=6, OUTPUT_DIM=3),
#         dynamics=[(3, 3), (6, 3)],
#         parameters=dict(
#             scattering_albedo=_torch.int64,
#             emission=_torch.int64
#         ),
#         code=f"""
#     FORWARD {{
#         _output = float[3](0.0, 0.0, 0.0);
#         vec3 E = vec3(0.0);
#         if (parameters.emission != 0)
#         {{
#             float [3] em;
#             dynamic_forward(object, parameters.emission, _input, em);
#             E = vec3(em[0], em[1], em[2]);
#         }}
#         else
#         return;
#
#         vec3 scattering_albedo = vec3(0.0);
#         if (parameters.scattering_albedo != 0)
#         {{
#             float[3] sa;
#             dynamic_forward(object, parameters.scattering_albedo, float[3](_input[0], _input[1], _input[2]), sa);
#             scattering_albedo = vec3(sa[0], sa[1], sa[2]);
#         }}
#
#         vec3 o = (vec3(1.0) - scattering_albedo)*E;
#         _output = float[3](o.x, o.y, o.z);
#     }}
#
#     BACKWARD {{  }}
#         """
#     )
#
#     def __init__(self, scattering_albedo: Optional[MapBase], emission: Optional[MapBase]):
#         super().__init__()
#         self.scattering_albedo_module = scattering_albedo
#         self.scattering_albedo = 0 if scattering_albedo is None else scattering_albedo.__bindable__.device_ptr
#         self.emission_module = emission
#         self.emission = 0 if emission is None else emission.__bindable__.device_ptr


class VolumeGathering(_maps.MapBase):
    __extension_info__ = None
    @staticmethod
    def create_extension_info(
            path_or_code: str,
            **parameters
    ):
        if ',' in path_or_code:
            code = """
        #include "volume_gathering_interface.h"
                    """ + path_or_code
            path = None
        else:
            path = path_or_code
            code = None
        return dict(
            parameters=parameters,
            generics=dict(INPUT_DIM=6, OUTPUT_DIM=3),
            bw_implementations=_maps.BACKWARD_IMPLEMENTATIONS.WITH_OUTPUT,
            code=code,
            path=path
        )

    @staticmethod
    def signature():
        return (6, 3)


class VolumeEmissionGathering(VolumeGathering):
    __extension_info__ = VolumeGathering.create_extension_info(
        _internal.__INCLUDE_PATH__ + '/maps/volume_emission_gathering.h',
        scattering_albedo=_maps.MapBase,
        emission=_maps.MapBase
    )

    def __init__(self, scattering_albedo: _typing.Optional[_maps.MapBase], emission: _typing.Optional[_maps.MapBase]):
        super().__init__()
        self.scattering_albedo = _maps.ZERO if scattering_albedo is None else scattering_albedo
        self.emission = _maps.ZERO if emission is None else emission


class VolumeEnvironmentGathering(VolumeGathering):
    __extension_info__ = VolumeGathering.create_extension_info(
        _internal.__INCLUDE_PATH__ + '/maps/volume_environment_gathering.h',
        environment_sampler=_maps.MapBase,
        visibility=_maps.MapBase,
        scattering_albedo=_maps.MapBase,
        homogeneous_phase=_maps.MapBase,
    )

    def __init__(self, environment_sampler: EnvironmentSampler, visibility: _maps.MapBase, scattering_albedo: _typing.Optional[_maps.MapBase], homogeneous_phase: _typing.Optional[_maps.MapBase]):
        super().__init__()
        self.environment_sampler = environment_sampler
        self.visibility = visibility.cast(6, 1)
        self.scattering_albedo = (_maps.ZERO if scattering_albedo is None else scattering_albedo).cast(3, 3)
        self.homogeneous_phase = (_maps.ZERO if homogeneous_phase is None else homogeneous_phase).cast(6, 1)



# class VolumeNEEGathering(MapBase):
#     __extension_info__ = dict(
#         generics=dict(INPUT_DIM=6, OUTPUT_DIM=3),
#         dynamics=[(3, 3), (6, 3)],
#         parameters=dict(
#             scattering_albedo=_torch.int64,
#             emission=_torch.int64
#         ),
#         code=f"""
#         FORWARD {{
#             if (parameters.emission == 0)
#             {{
#                 _output = float[3](0.0, 0.0, 0.0);
#                 return;
#             }}
#             vec3 scattering_albedo = vec3(0.0);
#             if (parameters.scattering_albedo != 0)
#             {{
#                 float[3] sa;
#                 dynamic_forward(object, parameters.scattering_albedo, float[3](_input[0], _input[1], _input[2]), sa);
#                 scattering_albedo = vec3(sa[0], sa[1], sa[2]);
#             }}
#             vec3 o = vec3(1.0) - scattering_albedo;
#             float[3] em;
#             dynamic_forward(object, parameters.emission, _input, em);
#             o *= vec3(em[0], em[1], em[2]);
#             _output = float[3](o.x, o.y, o.z);
#         }}
#             """
#     )
#
#     def __init__(self, scattering_albedo: Optional[MapBase], phase_sampler: Optional[MapBase], emission: Optional[MapBase]):
#         super().__init__()
#         self.scattering_albedo_module = scattering_albedo
#         self.scattering_albedo = 0 if scattering_albedo is None else scattering_albedo.__bindable__.device_ptr
#         self.emission_module = emission
#         self.emission = 0 if emission is None else emission.__bindable__.device_ptr


class PhaseSamplerPDF(_maps.MapBase):
    __extension_info__ = None
    @staticmethod
    def create_extension_info(fw_code: str,
                              bw_code: _typing.Optional[str] = None,
                              parameters: _typing.Optional[_typing.Dict] = None):
        if bw_code is None:
            bw_code = ''
        if parameters is None:
            parameters = {}
        return dict(
            parameters=parameters,
            bw_implementations=_maps.BACKWARD_IMPLEMENTATIONS.DEFAULT,
            generics=dict(INPUT_DIM = 6, OUTPUT_DIM=1),
            code=f"""
    FORWARD {{
        vec3 win = vec3(_input[0], _input[1], _input[2]);
        vec3 wout = vec3(_input[3], _input[4], _input[5]);
        float pdf;
        {fw_code}
        _output[0] = pdf;
    }}
    BACKWARD {{
        {bw_code}
    }}
            """
        )


class PhaseSampler(_maps.MapBase):
    __extension_info__ = None
    @staticmethod
    def create_extension_info(fw_code: str, bw_code: _typing.Optional[str] = None, parameters: _typing.Optional[_typing.Dict]= None):
        if bw_code is None:
            bw_code = ''
        if parameters is None:
            parameters = {}
        return dict(
            generics=dict(INPUT_DIM=3, OUTPUT_DIM=5),
            bw_implementations=_maps.BACKWARD_IMPLEMENTATIONS.DEFAULT,
            parameters = parameters,
            code=f"""
    FORWARD {{
        vec3 win = vec3(_input[0], _input[1], _input[2]);
        vec3 wout; float weight; float pdf;
        {fw_code}
        _output = float[5](wout.x, wout.y, wout.z, weight, pdf);
    }}
    BACKWARD {{
        {bw_code}
    }}
            """
        )

    def get_pdf(self) -> PhaseSamplerPDF:
        raise NotImplementedError()

    def get_phase(self) -> _maps.MapBase:
        raise NotImplementedError()

    @staticmethod
    def signature():
        return (3, 5)


class IsotropicPhaseSamplerPdf(PhaseSamplerPDF, Singleton):
    __extension_info__ = PhaseSamplerPDF.create_extension_info(f"""
    pdf = 0.25/pi;
    """)


class IsotropicPhaseSampler(PhaseSampler, Singleton):
    __extension_info__ = PhaseSampler.create_extension_info("""
    wout = randomDirection(win);
    pdf = 0.25 / pi;
    weight = 1.0;
    """)

    def get_pdf(self) -> PhaseSamplerPDF:
        return IsotropicPhaseSamplerPdf.get_instance()

    def get_phase(self) -> _maps.MapBase:
        return IsotropicPhaseSamplerPdf.get_instance()


class HGPhaseSamplerPDF(PhaseSamplerPDF):
    __extension_info__ = PhaseSamplerPDF.create_extension_info(
        parameters=dict(
            g=_maps.ParameterDescriptor
        ),
        fw_code="""
    float g = param_float(parameters.g);
    pdf = hg_phase_eval(win, wout, g);
    """)

    def __init__(self, g: _torch.Tensor):
        super().__init__()
        self.g = _maps.parameter(g)


class HGPhaseSampler(PhaseSampler):
    __extension_info__ = PhaseSampler.create_extension_info(
        parameters=dict(
            g=_maps.ParameterDescriptor
        ),
        fw_code="""
    float g = param_float(parameters.g);
    wout = hg_phase_sample(win, g, pdf);
    weight = 1.0; // Importance sampled function
        """
    )
    def __init__(self, g: _torch.Tensor):
        super().__init__()
        self.g = _maps.parameter(g)
        self.pdf = HGPhaseSamplerPDF(self.g)

    def get_pdf(self) -> PhaseSamplerPDF:
        return self.pdf

    def get_phase(self) -> _maps.MapBase:
        return self.get_pdf()  # perfect importance sampled


class MediumMaterial:
    def __init__(self,
                 path_integrator_factory: _typing.Callable[[_typing.Optional[_maps.MapBase],
                 _typing.Optional[Boundary], int], _maps.MapBase],
                 scattering_albedo: _typing.Optional[_maps.MapBase] = None,
                 phase_sampler: _typing.Optional['PhaseSampler'] = None,
                 emission: _typing.Optional[_maps.MapBase] = None,
                 enclosed: bool = False,
                 custom_boundary: _typing.Optional[Boundary] = None,
                 volume_track: bool = True
                 ):
        self.path_integrator_factory = path_integrator_factory
        self.scattering_albedo = scattering_albedo
        self.emission = emission
        self.phase_sampler = phase_sampler
        self.phase_sampler_pdf = None if phase_sampler is None else phase_sampler.get_pdf()
        self.enclosed = enclosed  # means that it will be assumed that the boundary it's not null
        self.custom_boundary = custom_boundary
        self.volume_track = volume_track

    def transmittance(self):
        return self.path_integrator_factory(None, None, 0)

    # def only_emission(self):
    #     return self.path_integrator_factory(None if self.emission is None else VolumeEmissionGathering(self.scattering_albedo, self.emission), 1)
    #
    # def only_scattering(self):
    #     return self.path_integrator_factory(None, 2)
    #
    def default_gathering(self, boundaries: Boundary):
        if self.custom_boundary is not None:
            boundaries = self.custom_boundary
        if not self.volume_track:
            boundaries = None
        return self.path_integrator_factory(
            None if self.emission is None else VolumeEmissionGathering(
                self.scattering_albedo, self.emission), boundaries, 3
        )

    def environment_gathering(self, environment_sampler: EnvironmentSampler, visibility: _maps.MapBase, boundaries: Boundary):
        if self.custom_boundary is not None:
            boundaries = self.custom_boundary
        if not self.volume_track:
            boundaries = None
        nee_gathering = None if self.scattering_albedo is None or self.enclosed else VolumeEnvironmentGathering(
                                                environment_sampler,
                                                visibility,
                                                self.scattering_albedo,
                                                self.phase_sampler.get_phase()
                                            )
        if nee_gathering is None:
            gathering = self.emission
        elif self.emission is None:
            gathering = nee_gathering
        else:
            gathering = nee_gathering + self.emission
        return self.path_integrator_factory(gathering, boundaries, 3)
        # return self.path_integrator_factory(None, 0)

    def photon_gathering(self, incomming_radiance: _maps.MapBase):
        pass


class HomogeneousMediumMaterial(MediumMaterial):
    def __init__(self, sigma: _typing.Union[_torch.Tensor, float], scattering_albedo: _typing.Optional[_torch.Tensor] = None, phase_sampler: _typing.Optional['PhaseSampler'] = None, emission: _typing.Optional[_maps.MapBase] = None, enclosed: bool = False):
        super().__init__(
            lambda gathering, boundaries, filter: HomogeneousMediumPathIntegrator(
                sigma=sigma,
                scattering_albedo=scattering_albedo,
                phase_sampler=phase_sampler,
                gathering=gathering,
                medium_filter=filter
            ),
            scattering_albedo=None if scattering_albedo is None else _maps.const[scattering_albedo],
            phase_sampler=phase_sampler,
            emission=emission,
            enclosed=enclosed
        )


class HeterogeneousMediumMaterial(MediumMaterial):
    def __init__(self,
                 sigma: _maps.MapBase,
                 scattering_albedo: _typing.Optional[_maps.MapBase] = None,
                 emission: _typing.Optional[_maps.MapBase]=None,
                 phase_sampler: _typing.Optional['PhaseSampler'] = None,
                 technique: _typing.Literal['dt', 'ds', 'rt'] = 'dt',
                 enclosed: bool = False,
                 custom_boundary: _typing.Optional[Boundary] = None,
                 volume_track: bool = True,
                 **kwargs):
        if technique == 'dt':
            assert 'majorant' in kwargs, "Delta tracking technique requires a majorant: _torch.Tensor parameter"
            def factory(gathering, boundaries, filter):
                if boundaries is None:
                    return DTMediumPathIntegrator(
                        sigma=sigma,
                        scattering_albedo=scattering_albedo,
                        phase_sampler=phase_sampler,
                        majorant=kwargs.get('majorant'),
                        gathering=gathering,
                        medium_filter=filter
                    )
                else:
                    return DTVolumePathIntegrator(
                        sigma=sigma,
                        majorant=kwargs.get('majorant'),
                        boundaries=boundaries,
                        scattering_albedo=scattering_albedo,
                        phase_sampler=phase_sampler,
                        gathering=gathering,
                        medium_filter=filter
                    )
        elif technique == 'ds':
            assert 'majorant' in kwargs, "Defensive tracking technique requires a majorant: _torch.Tensor parameter"
            # assert 'ds_epsilon' in kwargs, "Defensive tracking technique requires a ds_epsilon: float parameter"
            def factory(gathering, boundaries, filter):
                if boundaries is None:
                    return DTMediumPathIntegrator(
                        sigma=sigma,
                        scattering_albedo=scattering_albedo,
                        phase_sampler=phase_sampler,
                        majorant=kwargs.get('majorant'),
                        gathering=gathering,
                        medium_filter=filter
                    )
                else:
                    return DSVolumePathIntegrator(
                        sigma=sigma,
                        majorant=kwargs.get('majorant'),
                        boundaries=boundaries,
                        scattering_albedo=scattering_albedo,
                        phase_sampler=phase_sampler,
                        gathering=gathering,
                        medium_filter=filter,
                        ds_epsilon=kwargs.get('ds_epsilon', 0.1)
                    )
        elif technique == 'rt':
            assert 'majorant' in kwargs, "Delta tracking technique requires a majorant: _torch.Tensor parameter"
            def factory(gathering, boundaries, filter):
                return RTMediumPathIntegrator(
                    sigma=sigma,
                    scattering_albedo=scattering_albedo,
                    phase_sampler=phase_sampler,
                    majorant=kwargs.get('majorant'),
                    gathering=gathering,
                    medium_filter=filter
                )
        else:
            raise NotImplementedError()
        super().__init__(
            path_integrator_factory=factory,
            scattering_albedo=scattering_albedo,
            phase_sampler=phase_sampler,
            emission=emission,
            custom_boundary=custom_boundary,
            enclosed=enclosed,
            volume_track=volume_track,
        )


PTPatchInfo = _maps.map_struct(
    'PTPatchInfo',
    surface_scattering_sampler=_torch.int64,
    surface_gathering=_torch.int64,
    inside_medium=_torch.int64,
    outside_medium=_torch.int64
)


VSPatchInfo = _maps.map_struct(
    'VSPatchInfo',
    inside_medium=_torch.int64,
    outside_medium=_torch.int64,
    surface_scatters = int,
    pad0 = int,
    pad1=int,
    pad2=int,
)


class SceneVisibility(_maps.MapBase):
    __extension_info__ = dict(
        parameters=dict(
            boundaries=_maps.MapBase,
            patch_info=[-1, VSPatchInfo]
        ),
        generics=dict(INPUT_DIM=6, OUTPUT_DIM=1),
        dynamics=[
            MediumPathIntegrator.signature()
        ],
        path=_internal.__INCLUDE_PATH__+'/maps/scene_visibility.h',
        bw_implementations=_maps.BACKWARD_IMPLEMENTATIONS.WITH_OUTPUT,
    )
    def __init__(self,
        boundaries: Boundary,
        surface_scatters: _typing.Optional[_typing.List[bool]] = None,
        medium_integrator: _typing.Optional[_typing.List[MediumPathIntegrator]] = None,
        inside_medium_indices: _typing.Optional[_typing.List[int]] = None,
        outside_medium_indices: _typing.Optional[_typing.List[int]] = None,
    ):
        super().__init__(len(boundaries))
        self.boundaries = boundaries
        for i in range(len(boundaries)):
            self.patch_info[i].surface_scatters = 0
            self.patch_info[i].inside_medium = 0
            self.patch_info[i].outside_medium = 0
        assert surface_scatters is None or len(surface_scatters) == len(boundaries)
        if surface_scatters is not None:
            for i, ss in enumerate(surface_scatters):
                self.patch_info[i].surface_scatters = 0 if not ss else 1
        self.media_modules = _torch.nn.ModuleList(medium_integrator)
        assert inside_medium_indices is None or len(inside_medium_indices) == len(boundaries)
        if inside_medium_indices is not None:
            for i, imi in enumerate(inside_medium_indices):
                assert imi == -1 or (medium_integrator is not None and imi < len(medium_integrator))
                self.patch_info[i].inside_medium = 0 if imi == -1 else medium_integrator[imi].__bindable__.device_ptr
        assert outside_medium_indices is None or len(outside_medium_indices) == len(boundaries)
        if outside_medium_indices is not None:
            for i, omi in enumerate(outside_medium_indices):
                assert omi == -1 or (medium_integrator is not None and omi < len(medium_integrator))
                self.patch_info[i].outside_medium = 0 if omi == -1 else medium_integrator[omi].__bindable__.device_ptr


class SceneTransmittance(_maps.MapBase):
    __extension_info__ = dict(
        parameters=dict(
            visibility=_maps.MapBase,
            environment=_maps.MapBase
        ),
        generics=dict(INPUT_DIM=6, OUTPUT_DIM=3),
        path=_internal.__INCLUDE_PATH__ + '/maps/scene_transmittance.h',
        bw_implementations=_maps.BACKWARD_IMPLEMENTATIONS.WITH_OUTPUT,
    )

    def __init__(self,
                 visibility: _maps.MapBase,
                 environment: _maps.MapBase
                 ):
        super().__init__()
        self.visibility = visibility.cast(6, 1)
        self.environment = environment.cast(*Environment.signature())


class PathtracedScene(_maps.MapBase):
    __extension_info__ = dict(
        parameters=dict(
            surfaces=_maps.MapBase,
            boundaries=_maps.MapBase,
            environment=_maps.MapBase,
            direct_gathering=_maps.MapBase,
            patch_info=[-1, PTPatchInfo]
        ),
        generics=dict(INPUT_DIM=6, OUTPUT_DIM=3),
        bw_implementations=_maps.BACKWARD_IMPLEMENTATIONS.DEFAULT,
        dynamics=[
            SurfaceScatteringSampler.signature(),
            SurfaceGathering.signature(),
            MediumPathIntegrator.signature()
        ],
        path=_internal.__INCLUDE_PATH__+'/maps/pathtraced_scene.h'
    )
    def __init__(self,
        surfaces: Geometry,
        environment: _maps.MapBase,
        direct_gathering: _maps.MapBase,
        surface_gathering: _typing.Optional[_typing.List[SurfaceGathering]] = None,
        surface_scattering_sampler: _typing.Optional[_typing.List[SurfaceScatteringSampler]] = None,
        medium_integrator: _typing.Optional[_typing.List[MediumPathIntegrator]] = None,
        inside_medium_indices: _typing.Optional[_typing.List[int]] = None,
        outside_medium_indices: _typing.Optional[_typing.List[int]] = None
    ):
        assert not surfaces.is_generic
        if surface_gathering is not None:
            assert all(not s.is_generic for s in surface_gathering if s is not None), f"Found a generic surface gathering"
        if medium_integrator is not None:
            assert all(not m.is_generic for m in medium_integrator if m is not None), f"Found a generic medium integrator"
        super().__init__(len(surfaces))
        self.surfaces = surfaces.cast(*Geometry.signature())
        self.boundaries = surfaces.get_boundary().cast(*Boundary.signature())
        self.environment = environment.cast(*Environment.signature())
        self.direct_gathering = direct_gathering.cast(6, 3)  # TODO: abstract RadianceMap?
        for i in range(len(surfaces)):
            self.patch_info[i].surface_scattering_sampler = 0
            self.patch_info[i].surface_gathering = 0
            self.patch_info[i].inside_medium = 0
            self.patch_info[i].outside_medium = 0
        assert surface_scattering_sampler is None or len(surface_scattering_sampler) == len(surfaces)
        self.surface_scatterers = _torch.nn.ModuleList(surface_scattering_sampler)
        if surface_scattering_sampler is not None:
            for i, sss in enumerate(surface_scattering_sampler):
                self.patch_info[i].surface_scattering_sampler = 0 if sss is None else sss.__bindable__.device_ptr
        self.surface_gatherers = _torch.nn.ModuleList(surface_gathering)
        if surface_gathering is not None:
            for i, sg in enumerate(surface_gathering):
                self.patch_info[i].surface_gathering = 0 if sg is None else sg.__bindable__.device_ptr
        self.media_modules = _torch.nn.ModuleList(medium_integrator)
        assert inside_medium_indices is None or len(inside_medium_indices) == len(surfaces)
        if inside_medium_indices is not None:
            for i, imi in enumerate(inside_medium_indices):
                assert imi == -1 or (medium_integrator is not None and imi < len(medium_integrator))
                self.patch_info[i].inside_medium = 0 if imi == -1 else medium_integrator[imi].__bindable__.device_ptr
        if outside_medium_indices is not None:
            for i, omi in enumerate(outside_medium_indices):
                assert omi == -1 or (medium_integrator is not None and omi < len(medium_integrator))
                self.patch_info[i].outside_medium = 0 if omi == -1 else medium_integrator[omi].__bindable__.device_ptr


class ScenePathSampler(_maps.MapBase):
    __extension_info__ = dict(
        parameters=dict(
            surfaces=_maps.MapBase,
            boundaries=_maps.MapBase,
            patch_info=[-1, PTPatchInfo]
        ),
        generics=dict(INPUT_DIM=6, OUTPUT_DIM=6),
        bw_implementations=_maps.BACKWARD_IMPLEMENTATIONS.NONE,
        dynamics=[
            SurfaceScatteringSampler.signature(),
            MediumPathIntegrator.signature()
        ],
        path=_internal.__INCLUDE_PATH__+'/maps/scene_path_sampler.h'
    )
    def __init__(self,
        surfaces: Geometry,
        surface_scattering_sampler: _typing.Optional[_typing.List[SurfaceScatteringSampler]] = None,
        medium_integrator: _typing.Optional[_typing.List[MediumPathIntegrator]] = None,
        inside_medium_indices: _typing.Optional[_typing.List[int]] = None
    ):
        assert not surfaces.is_generic
        if medium_integrator is not None:
            assert all(not m.is_generic for m in medium_integrator if m is not None), f"Found a generic medium integrator"
        super().__init__(len(surfaces))
        self.surfaces = surfaces.cast(*Geometry.signature())
        self.boundaries = surfaces.get_boundary().cast(*Boundary.signature())
        self.medium_integrators = _torch.nn.ModuleList(medium_integrator)
        for i in range(len(surfaces)):
            self.patch_info[i].surface_scattering_sampler = 0
            self.patch_info[i].surface_gathering = 0
            self.patch_info[i].inside_medium = 0
            self.patch_info[i].outside_medium = 0
        assert surface_scattering_sampler is None or len(surface_scattering_sampler) == len(surfaces)
        self.surface_scatterers = _torch.nn.ModuleList(surface_scattering_sampler)
        if surface_scattering_sampler is not None:
            for i, sss in enumerate(surface_scattering_sampler):
                self.patch_info[i].surface_scattering_sampler = 0 if sss is None else sss.__bindable__.device_ptr
        self.media_modules = _torch.nn.ModuleList(medium_integrator)
        assert inside_medium_indices is None or len(inside_medium_indices) == len(surfaces)
        if inside_medium_indices is not None:
            for i, imi in enumerate(inside_medium_indices):
                assert imi == -1 or (medium_integrator is not None and imi < len(medium_integrator))
                self.patch_info[i].inside_medium = 0 if imi == -1 else medium_integrator[imi].__bindable__.device_ptr



class Scene:
    def __init__(self,
                 surfaces: Geometry,
                 environment_sampler: EnvironmentSampler,
                 materials: _typing.Optional[_typing.List[SurfaceMaterial]] = None,
                 inside_media: _typing.Optional[_typing.List[MediumMaterial]] = None,
                 outside_media: _typing.Optional[_typing.List[MediumMaterial]] = None
                 ):
        self.surfaces = surfaces
        self.boundaries = surfaces.get_boundary()
        self.environment_sampler = environment_sampler
        self.materials = materials
        self.media = []
        if inside_media is not None:
            for m in inside_media:
                if m is not None and m not in self.media:
                    self.media.append(m)
        if outside_media is not None:
            for m in outside_media:
                if m is not None and m not in self.media:
                    self.media.append(m)
        self.inside_media = None if inside_media is None else [-1 if m is None else self.media.index(m) for m in inside_media]
        self.outside_media = None if outside_media is None else [-1 if m is None else self.media.index(m) for m in outside_media]
        self.scene_visibility = SceneVisibility(
            boundaries=self.boundaries,
            surface_scatters=None if self.materials is None else [
                False if m is None or isinstance(m, NoSurfaceScatteringSampler) else True for m in self.materials
            ],
            medium_integrator=None if self.media is None else [
                None if m is None else m.transmittance() for m in self.media
            ],
            inside_medium_indices=self.inside_media,
            outside_medium_indices=self.outside_media
        )
        self.transmittance_map = SceneTransmittance(self.scene_visibility, self.environment_sampler.get_environment())

    @staticmethod
    def from_graph(
            geometries: _typing.Dict[str, Geometry],
            environment_sampler: EnvironmentSampler,
            materials: _typing.Optional[_typing.Dict[str, SurfaceMaterial]] = None,
            inside_media: _typing.Optional[_typing.Dict[str, MediumMaterial]] = None,
            outside_media: _typing.Optional[_typing.Dict[str, MediumMaterial]] = None
    ):
        # if len(geometries) == 1:
        #     surfaces = next(iter(geometries.values()))
        # else:
        surfaces = GroupGeometry(*(v for k, v in geometries.items()))
        ranges = {}
        offset = 0
        for k, v in geometries.items():
            ranges[k] = (offset, offset + len(v))
            offset += len(v)
        if materials is None:
            surface_materials = None
        else:
            surface_materials = [None] * offset
            for k, m in materials.items():
                if k in geometries:
                    for i in range(*ranges[k]):
                        surface_materials[i] = m
        if inside_media is None:
            scene_inside_media = None
        else:
            scene_inside_media = [None] * offset
            for k, m in inside_media.items():
                if k == '__all__':
                    for i in range(offset):
                        scene_inside_media[i] = m
                elif k in geometries:
                    for i in range(*ranges[k]):
                        scene_inside_media[i] = m
        if outside_media is None:
            scene_outside_media = None
        else:
            scene_outside_media = [None] * offset
            for k, m in outside_media.items():
                if k == '__all__':
                    for i in range(offset):
                        scene_outside_media[i] = m
                elif k in geometries:
                    for i in range(*ranges[k]):
                        scene_outside_media[i] = m
        return Scene (surfaces, environment_sampler, surface_materials, scene_inside_media, scene_outside_media)

    def visibility(self) -> _maps.MapBase:
        """
        Creates a visibility map using the scene surfaces and transmittance of volumes
        """
        return self.scene_visibility


    def pathtrace(self) -> _maps.MapBase:
        if not hasattr(self, 'pathtrace_cache'):
            self.pathtrace_cache = PathtracedScene(
            surfaces=self.surfaces,
            environment=self.environment_sampler.get_environment(),
            direct_gathering=_maps.ZERO,
            surface_gathering=None if self.materials is None else [
                None if m is None else m.emission_gathering() for m in self.materials
            ],
            surface_scattering_sampler=None if self.materials is None else [
                None if m is None else m.sampler for m in self.materials
            ],
            medium_integrator=None if self.media is None else [
                None if m is None else m.default_gathering(boundaries=self.boundaries) for m in self.media
            ],
            inside_medium_indices=self.inside_media,
            outside_medium_indices=self.outside_media
        )
        return self.pathtrace_cache

    def pathtrace_environment_nee(self) -> _maps.MapBase:
        if not hasattr(self, 'pathtrace_environment_nee_cache'):
            self.pathtrace_environment_nee_cache = PathtracedScene(
                surfaces=self.surfaces,
                # environment=ray_direction(),
                environment=_maps.ZERO,
                direct_gathering=self.transmittance(),
                surface_gathering=None if self.materials is None else [
                    # None if m is None else m.emission_gathering() for m in self.materials
                    # None for m in self.materials
                    None if m is None else m.environment_gathering(self.environment_sampler, self.visibility()) for m in self.materials
                ],
                surface_scattering_sampler=None if self.materials is None else [
                    None if m is None else m.sampler for m in self.materials
                ],
                medium_integrator=None if self.media is None else [
                    None if m is None else m.environment_gathering(
                        self.environment_sampler, self.visibility(), boundaries=self.boundaries
                    ) for m in self.media
                    # None if m is None else m.default_gathering() for m in self.media
            ],
                inside_medium_indices=self.inside_media,
                outside_medium_indices=self.outside_media
            )
        return self.pathtrace_environment_nee_cache

    def path_sampler(self) -> _maps.MapBase:
        if not hasattr(self, 'path_sampler_cache'):
            self.path_sampler_cache = ScenePathSampler(
                surfaces=self.surfaces,
                surface_scattering_sampler=None if self.materials is None else [
                    None if m is None else m.sampler for m in self.materials
                ],
                medium_integrator=None if self.media is None else [
                    None if m is None else m.environment_gathering(
                        self.environment_sampler, self.visibility(), boundaries=self.boundaries
                    ) for m in self.media
                    # None if m is None else m.default_gathering() for m in self.media
            ],
                inside_medium_indices=self.inside_media,
            )
        return self.path_sampler_cache

    def transmittance(self) -> _maps.MapBase:
        return self.transmittance_map
        # return self.visibility().after(ray_to_segment(constant(6, 10000.0))).promote(3) * self.environment_sampler.get_environment()


# class Medium(MapBase):
#     __extension_info__ = None  # Abstract Node
#     @staticmethod
#     def create_extension_info(
#         fw_code: str,
#         bw_code: Optional[str] = None,
#         external_code: Optional[str] = None,
#         parameters: Optional[Dict] = None
#     ):
#         return dict(
#             generics=dict(INPUT_DIM=7, OUTPUT_DIM=13),
#             parameters=parameters if parameters is not None else {},
#             code=f"""
# {external_code if external_code is not None else ''}
#
# FORWARD {{
#     vec3 x = vec3(_input[0], _input[1], _input[2]);
#     vec3 w = vec3(_input[3], _input[4], _input[5]);
#     float d = _input[6];
#     float Tr = 1.0;
#     vec3 xout, wout, W = vec3(0.0), A = vec3(0.0);
#
#     {fw_code}
#
#     _output = float[13] (Tr, xout.x, xout.y, xout.z, wout.x, wout.y, wout.z, W.x, W.y, W.z, A.x, A.y, A.z);
# }}
#
# BACKWARD {{
#     {bw_code if bw_code is not None else ''}
# }}
#             """
#         )
#
#     def __init__(self):
#         super().__init__()  # x, w, d -> T, xe, we, W, A
#
#
# class NoMedium(Medium):
#     __extension_info__ = Medium.create_extension_info(f"""
#         Tr = 1.0; // full transmittance
#         W = vec3(0.0); // no scattering
#         A = vec3(0.0); // no accumulation
#         xout = w * d + x;
#         wout = w;
#     """)
#
#     __instance__ = None
#     @classmethod
#     def get_instance(cls):
#         if cls.__instance__ is None:
#             cls.__instance__ = NoMedium()
#         return cls.__instance__
#
#
# class HomogeneousEmissionMedium(Medium):
#     __extension_info__ = Medium.create_extension_info(
#         parameters=dict(
#             sigma=ParameterDescriptor,
#             emission=ParameterDescriptor,
#         ),
#         fw_code=f"""
#     float sigma = param_float(parameters.sigma);
#     vec3 emission = param_vec3(parameters.emission);
#     Tr = exp(-d * sigma);
#     W = vec3(0.0);
#     A = (1 - Tr) * emission;
#     """)
#
#     def __init__(self, sigma: _torch.Tensor, emission: _torch.Tensor):
#         super().__init__()
#         self.sigma = parameter(sigma)
#         self.emission = parameter(emission)
#
#
#
#
#
# class HomogeneousMedium(Medium):
#     __extension_info__ = Medium.create_extension_info(
#         parameters=dict(
#             sigma=ParameterDescriptor,
#             scattering_albedo=ParameterDescriptor,
#             phase_sampler=MapBase
#         ),
#         external_code="#include \"sc_phase_sampler.h\"",
#         fw_code=f"""
#         float sigma = param_float(parameters.sigma);
#         vec3 albedo = param_vec3(parameters.scattering_albedo);
#         Tr = exp(-d * sigma);
#         float t = -log(1 - random()*(1 - Tr)) / max(0.0000001, sigma);
#         xout = x + w * t;
#         float weight, pdf;
#         wout = sample_phase(object, w, weight, pdf);
#         W = albedo * weight;
#         """)
#
#     def __init__(self, sigma: _torch.Tensor, scattering_albedo: _torch.Tensor, phase_sampler: PhaseSampler):
#         super().__init__()
#         self.sigma = parameter(sigma)
#         self.scattering_albedo = parameter(scattering_albedo)
#         self.phase_sampler = phase_sampler
#
#
# class DTHeterogeneousMedium(Medium):
#     __extension_info__ = Medium.create_extension_info(
#         parameters=dict(
#             sigma=MapBase,
#             scattering_albedo=MapBase,
#             phase_sampler=MapBase,
#             majorant=ParameterDescriptor,
#         ),
#         external_code=f"""
#         #include "sc_phase_sampler.h"
#         #include "vr_sigma.h"
#         #include "vr_scattering_albedo.h"
#         """,
#         fw_code=f"""
#     float m = max(0.0001, param_float(parameters.majorant));
#     A = vec3(0.0);
#     W = vec3(0.0);
#     Tr = 1.0;
#     while (true)
#     {{
#         float t = -log(1 - random()) / m;
#         if (t > d - 0.00001)
#             break;
#         x += w * t;
#         float s = sigma(object, x);
#         if (random() < s/m) // real interaction
#         {{
#             float weight, pdf;
#             wout = sample_phase(object, w, weight, pdf);
#             W = scattering_albedo(object, x) * weight;
#             xout = x;
#             Tr = 0.0;
#             break;
#         }}
#         d -= t;
#     }}
#         """
#     )
#
#     def __init__(self, sigma: MapBase, scattering_albedo: MapBase, phase_sampler: MapBase, majorant: _torch.Tensor):
#         super().__init__()
#         self.sigma = sigma
#         self.scattering_albedo = scattering_albedo
#         self.phase_sampler = phase_sampler
#         self.majorant = parameter(majorant)
#
#
# class EnvironmentEmission(MapBase):
#     __extension_info__ = None
#     def __init__(self):
#         super().__init__(INPUT_DIM=3, OUTPUT_DIM=3)  # w -> R
#
#
# class Pathtracer(MapBase):
#     __extension_info__ = dict(
#         parameters=dict(
#             surfaces=MapBase,  # Scene to raycast
#             environment=MapBase,  # Radiance map directional emitted from infinity
#             surface_emission=_torch.Tensor,
#             surface_scattering_sampler=_torch.Tensor,
#             inside_media=_torch.Tensor,
#             outside_media=_torch.Tensor,
#         ),
#         dynamics=[
#             (20, 3),  #surface emitters
#             (20, 7),  #surface scattering samplers
#             (7, 13),  #volume scattering
#         ],
#         generics=dict(
#             INPUT_DIM=6,  # x, w
#             OUTPUT_DIM=3,  # R
#         ),
#         path = __INCLUDE_PATH__+'/maps/scene_radiance.h'
#     )
#
#     def __init__(self,
#                  surfaces: Geometry,
#                  environment: MapBase,
#                  surface_emission: Optional[List[MapBase]] = None,
#                  surface_scattering_sampler: Optional[List[MapBase]] = None,
#                  inside_media: List[MapBase] = None,
#                  outside_media: List[MapBase] = None,
#                  media_filter: Literal['none', 'transmitted', 'emitted', 'scattered'] = 'none'
#                  ):
#         super().__init__(surfaces.get_number_of_patches(), MEDIA_FILTER = {'none': 0, 'transmitted': 1, 'emitted': 2, 'scattered': 3}[media_filter])
#         self.surfaces = surfaces
#         self.environment = environment
#         self.sss_tensor = None if surface_scattering_sampler is None else _torch.zeros(surfaces.get_number_of_patches(), dtype=_torch.int64, device=device())
#         self.se_tensor = None if surface_emission is None else _torch.zeros(surfaces.get_number_of_patches(), dtype=_torch.int64, device=device())
#         self.im_tensor = None if inside_media is None else _torch.zeros(surfaces.get_number_of_patches(), dtype=_torch.int64, device=device())
#         self.om_tensor = None if outside_media is None else _torch.zeros(surfaces.get_number_of_patches(), dtype=_torch.int64, device=device())
#         if surface_scattering_sampler is not None:
#             for i, sss in enumerate(surface_scattering_sampler):
#                 self.sss_tensor[i] = 0 if sss is None else sss.__bindable__.device_ptr
#         if surface_emission is not None:
#             for i, e in enumerate(surface_emission):
#                 self.se_tensor[i] = 0 if e is None else e.__bindable__.device_ptr
#         if inside_media is not None:
#             for i, mi in enumerate(inside_media):
#                 self.im_tensor[i] = 0 if mi is None else mi.__bindable__.device_ptr
#         if outside_media is not None:
#             for i, mo in enumerate(outside_media):
#                 self.om_tensor[i] = 0 if mo is None else mo.__bindable__.device_ptr
#         self.surface_scattering_sampler = self.sss_tensor
#         self.surface_emission = self.se_tensor
#         self.inside_media = self.im_tensor
#         self.outside_media = self.om_tensor
#         modules = _torch.nn.ModuleList()
#         if surface_emission is not None:
#             modules.extend(surface_emission)
#         if surface_scattering_sampler is not None:
#             modules.extend(surface_scattering_sampler)
#         if inside_media is not None:
#             modules.extend(inside_media)
#         if outside_media is not None:
#             modules.extend(outside_media)
#         self.used_modules = modules


# def create_emitters(
#         surfaces: MapBase,
#         media_filter: Literal['none', 'transmittance', 'emitted', 'scattered'] = 'transmittance',
#         environment: Optional[MapBase] = None,
#         surface_emission: Optional[List[MapBase]] = None,
#         inside_media: Optional[List[MapBase]] = None,
#         outside_media: Optional[List[MapBase]] = None
# )

# class Scene(MapBase):
#     pass
