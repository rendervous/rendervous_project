import vulky as _vk
import typing as _typing
import os as _os
import torch as _torch


__MANUAL_SEED__ : _typing.Optional[int] = None
__SEEDS_TENSOR__ : _typing.Optional[_torch.Tensor] = None

__INCLUDE_PATH__ = _os.path.dirname(__file__).replace('\\','/') + '/include'


__TORCH_DEVICE__ = _torch.device('cuda:0') if _torch.cuda.is_available() else _torch.device('cpu')  #TODO: Check with AMD


def device() -> _torch.device:
    return __TORCH_DEVICE__


def seed(manual_seed: _typing.Optional[int] = None):
    global __MANUAL_SEED__
    __MANUAL_SEED__ = manual_seed


def get_seeds() -> _vk.ivec4:
    global __MANUAL_SEED__, __SEEDS_TENSOR__
    if __MANUAL_SEED__ is not None:
        _torch.manual_seed(__MANUAL_SEED__)
        __MANUAL_SEED__ = None
    if __SEEDS_TENSOR__ is None:
        __SEEDS_TENSOR__ = _vk.ivec4(0, 0, 0, 0)
    _torch.randint(low=129, high=1 << 30, size=(4,), dtype=_torch.int32, out=__SEEDS_TENSOR__)
    return __SEEDS_TENSOR__


class RendererModule(_torch.nn.Module):
    """
    Module that performs gradient-based operations on compute, graphics or raytracing pipelines.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._setup(*args, **kwargs)

    def _setup(self, *args, **kwargs):
        """
        When implemented, creates the pipelines and resources necessary for the rendezvous process.
        """
        pass

    def _forward_render(self, input: _typing.List[_torch.Tensor], **kwargs) -> _typing.List[_torch.Tensor]:
        """
        Computes the output given the parameters
        """
        pass

    def _backward_render(self, input: _typing.List[_torch.Tensor],
                        output_gradients: _typing.List[_typing.Optional[_torch.Tensor]], **kwargs) -> _typing.List[_typing.Optional[_torch.Tensor]]:
        """
        Computes the gradient of parameters given the original inputs and the gradients of outputs
        """
        return [None for _ in input]  # assume by default no differentiable options for input

    def forward(self, *args, **kwargs):
        outputs = _AutogradRendererFunction.apply(
            *(list(args) + [kwargs, self]))
        return outputs[0] if len(outputs) == 1 else outputs


class _AutogradRendererFunction(_torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args):
        renderer: RendererModule
        args = list(args)
        renderer = args[-1]
        kwargs = args[-2]
        inputs = args[0:-2]  # to skip renderer
        tensor_mask = [isinstance(a, _torch.Tensor) for a in inputs]
        only_tensors = [t if is_tensor else None for t, is_tensor in zip(inputs, tensor_mask)]
        only_non_tensors = [t if not is_tensor else None for t, is_tensor in zip(inputs, tensor_mask)]
        ctx.save_for_backward(*only_tensors)  # properly save tensors for backward
        ctx.non_tensors = only_non_tensors  # save other values in a list
        ctx.tensor_mask = tensor_mask  # mask to merge inputs in backward
        ctx.renderer = renderer
        ctx.kwargs = kwargs
        # print(f"- FW: {seed}")
        outputs = renderer._forward_render(inputs, **kwargs)
        return tuple(outputs)

    @staticmethod
    def backward(ctx, *args):
        only_tensors = list(ctx.saved_tensors)  # Just check for inplace operations in input tensors
        only_non_tensors = ctx.non_tensors
        tensor_mask = ctx.tensor_mask
        inputs = [t if is_tensor else nt for t, nt, is_tensor in zip(only_tensors, only_non_tensors, tensor_mask)]
        renderer = ctx.renderer
        kwargs = ctx.kwargs
        # print(f"< BW: {seed}")
        grad_outputs = list(args)
        grad_inputs = renderer._backward_render(seed, inputs, grad_outputs, **kwargs)
        # print(f"[DEBUG] Backward grads from renderer {grad_inputs[0].mean()}")
        # assert grad_inputs[0] is None or _torch.isnan(grad_inputs[0]).sum() == 0, "error in generated grads."
        return (*(None if g is None else g for g in grad_inputs),
                None, None)  # append None to refer to renderer object passed in forward


class FunctionMeta(type):
    def __new__(cls, name, bases, dct):
        ext_class = super().__new__(cls, name, bases, dct)
        assert '__extension_info__' in dct, 'Extension functions requires a dict __extension_info__ with path, parameters'
        extension_info = dct['__extension_info__']
        if extension_info is not None:  # is not an abstract node
            extension_path = extension_info.get('path', None)
            assert isinstance(extension_path, str) and _os.path.isfile(extension_path), 'path must be a valid file path str'
            include_dirs = extension_info.get('include_dirs', [])
            include_dirs.append(_os.path.dirname(extension_path))
            shader_path = _vk.compile_shader_file(extension_path, include_dirs)
            parameters = extension_info.get('parameters', {})
            ext_class.rdv_group_size = extension_info.get('group_size', (1024, 1, 1))
            if parameters is None or len(parameters) == 0:
                parameters = dict(foo=int)
            parameters_layout = _vk.Layout.from_description(mode=_vk.LayoutAlignment.STD430, description=parameters)
            ext_class.rdv_parameters_buffer = _vk.object_buffer(layout=parameters_layout)
            ext_class.rdv_parameters_buffer_accessor = ext_class.rdv_parameters_buffer.accessor
            ext_class.rdv_system_buffer = _vk.object_buffer(layout=_vk.Layout.from_structure(
                _vk.LayoutAlignment.STD430,
                seeds = _vk.ivec4,
                dim_x = int,
                dim_y = int,
                dim_z = int
            ))
            pipeline = _vk.pipeline_compute()
            pipeline.load_shader(shader_path)
            pipeline.layout(set=0, binding=0, system_buffer=_vk.DescriptorType.UNIFORM_BUFFER)
            pipeline.layout(set=0, binding=1, parameters_buffer=_vk.DescriptorType.UNIFORM_BUFFER)
            pipeline.close()
            global_bindings = pipeline.create_descriptor_set_collection(set=0, count=1)
            global_bindings[0].update(
                system_buffer=ext_class.rdv_system_buffer,
                parameters_buffer=ext_class.rdv_parameters_buffer
            )
            ext_class.rdv_pipeline = pipeline
            ext_class.rdv_global_bindings = global_bindings

        return ext_class


class FunctionBase(object, metaclass=FunctionMeta):
    __extension_info__ = None  # Abstract node
    rdv_man_cache = None

    __instance__ = None

    @classmethod
    def instance(cls):
        if cls.__instance__ is None:
            cls.__instance__ = cls()
        return cls.__instance__

    def __init__(self):
        object.__setattr__(self, 'rdv_man_cache', dict())

    def __getattr__(self, item):
        accessor = super().__getattribute__('rdv_parameters_buffer_accessor')
        if item in accessor._rdv_layout.fields_layout:
            return getattr(accessor, item)
        return super(FunctionBase, self).__getattribute__(item)

    def __setattr__(self, key, value):
        accessor = super().__getattribute__('rdv_parameters_buffer_accessor')
        setattr(accessor, key, value)
        super(FunctionBase, self).__setattr__(key, value)

    @classmethod
    def eval(cls, *args, **kwargs):
        instance = cls.instance()
        invocations = instance.bind(*args, **kwargs)
        man = instance._resolve_dispatcher(invocations)
        _vk.submit(man)
        return instance.result()

    def bind(self, *args, **kwargs) -> _typing.Tuple:
        '''
        sets the arguments to the object and return the number of threads to dispatch
        '''
        raise NotImplementedError()

    def result(self) -> _typing.Any:
        '''
        :param parameters: accessor to get bound tensors
        :return: resultant output, can be directly tensors or a postprocessing on them.
        '''
        raise NotImplementedError()

    def _resolve_dispatcher(self, threads: _typing.Tuple) -> _vk.ComputeManager:
        dim_x, dim_y, dim_z = threads
        with self.rdv_system_buffer as b:
            b.seeds = get_seeds()
            b.dim_x = dim_x
            b.dim_y = dim_y
            b.dim_z = dim_z

        if threads not in self.rdv_man_cache:
            man = _vk.compute_manager()
            man.set_pipeline(self.rdv_pipeline)
            man.bind(self.rdv_global_bindings[0])
            man.dispatch_threads(*threads, *self.rdv_group_size)
            man.freeze()
            self.rdv_man_cache[threads] = man
        else:
            man = self.rdv_man_cache[threads]
        return man


class DependentObject:
    def forward_dependency(self):
        raise NotImplementedError()

    def backward_dependency(self):
        raise NotImplementedError()


class Reparameter(_torch.nn.Parameter, DependentObject):
    __torch_function__ = _torch._C._disabled_torch_function_impl

    def __init__(self, *args):
        super(Reparameter, self).__init__()

    def initialize(self, forward_exp: _typing.Callable[[], _torch.Tensor]):
        self.forward_exp = forward_exp
        self.diff_tensor = None
        self.requires_grad = False

    def forward_dependency(self):
        self.diff_tensor = self.forward_exp()
        self.requires_grad = self.diff_tensor.requires_grad
        if self.grad is not None:
            self.grad.zero_()
        with _torch.no_grad():
            self.copy_(self.diff_tensor)

    def backward_dependency(self):
        if self.diff_tensor is not None and self.diff_tensor.requires_grad and self.grad is not None:
            self.diff_tensor.backward(self.grad)

    @staticmethod
    def from_lambda(exp: _typing.Callable[[], _torch.Tensor]):
        initial_value = exp()
        p = Reparameter(initial_value)
        p.initialize(exp)
        return p


class DependencySet(DependentObject):

    def __init__(self, parent: 'DependencySet' = None):
        self.__initialize(parent)

    def __initialize(self, parent: 'DependencySet' = None):
        object.__setattr__(self, '_reparameters', [])
        object.__setattr__(self, '_fields', {})
        object.__setattr__(self, '_parent', parent)

    def ensures(self, field_name, field_type) -> bool:
        return field_name in self._fields and isinstance(self._fields[field_name], field_type) or \
            (self._parent is not None and self._parent.ensures(field_name, field_type))

    def assert_ensures(self, field_name, field_type):
        b = self.ensures(field_name, field_type)
        assert b, f"Required field {field_type} of type {field_type} not found."

    def branch(self) -> 'DependencySet':
        b = DependencySet(parent=self)
        self._reparameters.append(b)
        return b

    def add_parameters(self, **kwargs):
        for k, v in kwargs.items():
            if not isinstance(v, _torch.nn.Module) and hasattr(v, '__call__'):
                v = Reparameter.from_lambda(v)
            if isinstance(v, DependentObject):
                self._reparameters.append(v)
            self._fields[k] = v

    def copy_parameters(self, ds: 'DependencySet'):
        self.add_parameters(**ds._fields)
        if ds._parent is not None:
            self.copy_parameters(ds._parent)

    def requires(self, builder: _typing.Callable[['DependencySet', _typing.Dict[str, _typing.Any]], None], **kwargs):
        builder(self, **kwargs)

    def _forward_dependency(self):
        for p in self._reparameters:
            if isinstance(p, DependencySet):
                p._forward_dependency()
            else:
                p.forward_dependency()

    def forward_dependency(self):
        if self._parent is not None:
            self._parent.forward_dependency()
            return
        else:
            self._forward_dependency()

    def _backward_dependency(self):
        for p in self._reparameters:
            if isinstance(p, DependencySet):
                p._backward_dependency()
            else:
                p.backward_dependency()

    def backward_dependency(self):
        if self._parent is not None:
            self._parent.backward_dependency()
            return
        self._backward_dependency()

    def __getattr__(self, item):
        if item in self._fields:
            return self._fields[item]
        if self._parent is not None:
            return getattr(self._parent, item)
        return object.__getattribute__(self, item)

    def __repr__(self):
        s = ""
        for k, p in self._fields.items():
            s += k+": "+repr(p) + "\n"
        return s

