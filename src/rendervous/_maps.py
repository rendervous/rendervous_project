import vulky as _vk
from . import _internal
from . import _functions
import typing as _typing
import enum as _enum
import torch as _torch
import os as _os
from vulky import vec2, vec3
import numpy as _np
import math as _math
import threading as _threading


class BACKWARD_IMPLEMENTATIONS(_enum.IntEnum):
    NONE = 0
    """
    The map source code doesn't contains any backward function
    """
    DEFAULT = 1
    """
    The map source code contains only the default backward function (input, output_grad, input_grad)
    """
    WITH_OUTPUT = 2
    """
    The map source code contains only the output provided backward function (input, output, output_grad, input_grad)
    """
    ALL = 3
    """
    The map source code contains both backward functions, with and without output.
    """


class DispatcherEngine:
    __REGISTERED_MAPS__ = []  # Each map type
    __MAP_INSTANCES__ = {}  # All instances, from tuple with signature to codename.
    __MAP_BY_SIGNATURE__ = { } # All different-signature maps grouped by parameter signature
    # __RAYCASTER_INSTANCES__ = { }  # All instances, from tuple with signature to codename.
    __CS_SUPER_KERNEL__ = ""  # Current engine code
    __INCLUDE_DIRS__ = []
    __DEFINED_STRUCTS__ = {}
    __ENGINE_OBJECTS__ = None  # Objects to dispatch map evaluation and raycasting

    __FW_RT_ENGINE_PIPELINES__ = {}  # Pipelines and RT Programs for fw map evaluation
    __FW_CS_ENGINE_PIPELINES__ = {}  # Pipelines and RT Programs for fw map evaluation
    __FW_DISPATCHER_CACHED_MAN__ = {}  # command buffers for dispatching fw map evaluations
    __FW_RAYCASTER_CACHED_MAN__ = {}  # command buffers for dispatching fw raycast evaluations
    __FW_CAPTURE_CACHED_MAN__ = {}  # command buffers for dispatching fw capture evaluations

    __BW_RT_ENGINE_PIPELINES__ = {}  # Pipelines and RT Programs for bw map evaluation
    __BW_CS_ENGINE_PIPELINES__ = {}  # Pipelines and RT Programs for bw map evaluation
    __BW_DISPATCHER_CACHED_MAN__ = {}  # command buffers for dispatching bw map evaluations
    __BW_RAYCASTER_CACHED_MAN__ = {}  # command buffers for dispatching bw raycast evaluations
    __BW_CAPTURE_CACHED_MAN__ = {}  # command buffers for dispatching bw capture evaluations

    __LOCKER__ = _threading.Lock()

    @classmethod
    def start(cls):
        _, inner_structs, _ = cls.create_code_for_struct_declaration(ParameterDescriptor)
        cls.__DEFINED_STRUCTS__.update(inner_structs)
        _, inner_structs, _ = cls.create_code_for_struct_declaration(MeshInfo)
        cls.__DEFINED_STRUCTS__.update(inner_structs)
        _, inner_structs, _ = cls.create_code_for_struct_declaration(RaycastableInfo)
        cls.__DEFINED_STRUCTS__.update(inner_structs)

    @classmethod
    def register_map(cls, map_type: 'MapMeta') -> int:
        code = len(cls.__REGISTERED_MAPS__) + 1
        cls.__REGISTERED_MAPS__.append(map_type)
        return code

    @classmethod
    def create_support_code(cls):
        # Gets vulkan device used
        caps = _vk.support()
        code = ""
        if caps.ray_query:
            code += "#define SUPPORTED_RAY_QUERY\n"
        if caps.atom_float:
            code += "#define SUPPORTED_FLOAT_ATOM_ADD\n"
        return code

    @classmethod
    def create_code_for_struct_declaration(cls, type_definition) -> _typing.Tuple[
        str, dict, list]:  # Code, new structures, sizes
        if type_definition == MapBase:
            raise Exception('Basic structs can not contain map references')
        if type_definition == _torch.Tensor:
            return 'GPUPtr', {}, []
        if _vk.Layout.is_scalar_type(type_definition):
            if type_definition == int:
                return 'int', {}, []
            if type_definition == 'float':
                return 'float', {}, []
            return {
                       _torch.int32: 'int',
                       _torch.float32: 'float',
                       _torch.int64: 'GPUPtr'
                   }[type_definition], {}, []
        if isinstance(type_definition, list):
            size = type_definition[0]
            t = type_definition[1]
            element_decl, inner_structures, element_sizes = cls.create_code_for_struct_declaration(t)
            return element_decl, inner_structures, [size] + element_sizes
        if isinstance(type_definition, dict):
            assert '__name__' in type_definition, 'Basic structs must be named, include a key name: str'
            inner_structures = {}
            struct_code = f"struct {type_definition['__name__']} {{"
            for field_id, field_type in type_definition.items():
                if field_id != '__name__':
                    t, field_inner_structures, sizes = cls.create_code_for_struct_declaration(field_type)
                    struct_code += t + " " + field_id + ''.join(f"[{size}]" for size in sizes) + '; \n'
                    inner_structures.update(field_inner_structures)
            struct_code += '};'
            inner_structures[type_definition['__name__']] = struct_code
            return type_definition['__name__'], inner_structures, []
        return type_definition.__name__, {}, []  # vec and mats

    @classmethod
    def create_code_for_map_declaration(cls, type_definition, field_value, allow_block: bool = False) -> _typing.Tuple[
        str, dict, list]:  # Code, new structures, sizes
        if type_definition == MapBase:
            return cls.register_instance(field_value.obj)[1], {}, []
        if type_definition == _torch.Tensor:
            return 'GPUPtr', {}, []
        if _vk.Layout.is_scalar_type(type_definition):
            if type_definition == int:
                return 'int', {}, []
            if type_definition == float:
                return 'float', {}, []
            return {
                       _torch.int32: 'int',
                       _torch.float32: 'float',
                       _torch.int64: 'GPUPtr'
                   }[type_definition], {}, []
        if isinstance(type_definition, list):
            size = type_definition[0]
            t = type_definition[1]
            field_value: _vk.ObjectBufferAccessor
            element_decl, inner_structures, element_sizes = cls.create_code_for_map_declaration(t, field_value[0] if len(field_value) > 0 else None)
            return element_decl, inner_structures, [size] + element_sizes
        if isinstance(type_definition, dict):
            inner_structures = {}
            if '__name__' in type_definition:  # external struct
                struct_code = f"struct {type_definition['__name__']} {{"
                for field_id, field_type in type_definition.items():
                    if field_id != '__name__':
                        t, field_inner_structures, sizes = cls.create_code_for_map_declaration(field_type,
                                                                                               getattr(field_value,
                                                                                                       field_id))
                        struct_code += t + " " + field_id + ''.join(f"[{size}]" for size in sizes) + '; \n'
                        inner_structures.update(field_inner_structures)
                struct_code += '};'
                inner_structures[type_definition['__name__']] = struct_code
                return type_definition['__name__'], inner_structures, []
            else:  # block
                assert allow_block, 'Can not create a nested block. Add a name attribute to the dictionary to make it a struct'
                code = "{"
                for field_id, field_type in type_definition.items():
                    f = getattr(field_value, field_id)
                    assert field_type != MapBase or f.obj is not None, f'Field {field_id} must be bound to a non-null map'
                    t, field_inner_structures, sizes = cls.create_code_for_map_declaration(field_type, f)
                    code += t + " " + field_id + ''.join(f"[{size if size > 0 else ''}]" for size in sizes) + '; \n'
                    inner_structures.update(field_inner_structures)
                code += '}'
                return code, inner_structures, []
        return type_definition.__name__, {}, []  # vec and mats

    @classmethod
    def create_code_for_dynamic_map(cls, input_dim, output_dim):
        fw_cases = ""
        bw_cases = ""
        bw_using_output_cases = ""
        sig = (input_dim, output_dim)
        if sig in cls.__MAP_BY_SIGNATURE__:
            for (id, code_name) in cls.__MAP_BY_SIGNATURE__[sig]:
                fw_cases += f"""
                case {id}: forward({code_name}(buffer_{code_name}(dynamic_map)), _input, _output); break;
                """
                bw_cases += f"""
                case {id}: backward({code_name}(buffer_{code_name}(dynamic_map)), _input, _output_grad, _input_grad); break;
                """
                bw_using_output_cases += f"""
                case {id}: backward({code_name}(buffer_{code_name}(dynamic_map)), _input, _output, _output_grad, _input_grad); break;
                """
        return f"""
void dynamic_forward (map_object, GPUPtr dynamic_map, in float _input[{input_dim}], out float _output[{output_dim}]) {{
    for (int i=0; i<{output_dim}; i++) _output[i] = 0.0;//((i^13+15 + int(random()*17))%{output_dim})/float({output_dim});
    if (dynamic_map == 0) {{
        return;
    }}
    int map_id = int_ptr(dynamic_map).data[0];
    switch(map_id)
    {{
    {fw_cases}
    }}  
}}

void dynamic_backward(map_object, GPUPtr dynamic_map, in float _input[{input_dim}], in float _output_grad[{output_dim}], inout float _input_grad[{input_dim}])  {{
    if (dynamic_map == 0) return;
    int map_id = int_ptr(dynamic_map).data[0];
    switch(map_id)
    {{
    {bw_cases}
    }}  
}}

void dynamic_backward(map_object, GPUPtr dynamic_map, in float _input[{input_dim}], in float _output[{output_dim}], in float _output_grad[{output_dim}], inout float _input_grad[{input_dim}])  {{
    if (dynamic_map == 0) return;
    int map_id = int_ptr(dynamic_map).data[0];
    switch(map_id)
    {{
    {bw_using_output_cases}
    }}  
}}
    
        """

    @classmethod
    def append_map_instance_source_code(cls, map: 'MapBase', instance_id: int, codename: str):
        code = ""
        map_object_parameters_code, external_structs, _ = cls.create_code_for_map_declaration(map.map_object_definition,
                                                                                              map._rdv_accessor,
                                                                                              allow_block=True)
        for struct_name, struct_code in external_structs.items():
            if struct_name in cls.__DEFINED_STRUCTS__:
                assert cls.__DEFINED_STRUCTS__[
                           struct_name] == struct_code, f'A different body was already defined for {struct_name}'
            else:
                code += struct_code + "\n"
                cls.__DEFINED_STRUCTS__[struct_name] = struct_code
        # Add buffer_reference definition with codename and map object layout
        code += f"""
layout(buffer_reference, scalar, buffer_reference_align=8) buffer buffer_{codename} {map_object_parameters_code};
struct {codename} {{ buffer_{codename} data; }};
"""
        for g, v in map.generics.items():
            code += f"#define {g} {v} \n"
        code += f"#define map_object in {codename} object \n"
        code += f"#define parameters object.data \n"

        for s in map.dynamic_requires:  # Generate dynamic access code for all required signatures
            code += cls.create_code_for_dynamic_map(*s)

        code += map.map_source_code + "\n"
        if (map.input_dim, map.output_dim) not in cls.__MAP_BY_SIGNATURE__:
            cls.__MAP_BY_SIGNATURE__[(map.input_dim, map.output_dim)] = []
        cls.__MAP_BY_SIGNATURE__[(map.input_dim, map.output_dim)].append((instance_id, codename))

        if map.bw_implementations == BACKWARD_IMPLEMENTATIONS.NONE:
            # Add default implementation for both bw modes
            code += """
            void backward (map_object, in float _input[INPUT_DIM], in float _output_grad[OUTPUT_DIM], inout float _input_grad[INPUT_DIM]) {  }
            void backward (map_object, in float _input[INPUT_DIM], in float _output[OUTPUT_DIM], in float _output_grad[OUTPUT_DIM], inout float _input_grad[INPUT_DIM]) {  }
            """
        elif map.bw_implementations == BACKWARD_IMPLEMENTATIONS.WITH_OUTPUT:
            # If only default implementation is provided, add an implementation of bw based on bw using output, re-computing the output
            code += """
            void backward (map_object, in float _input[INPUT_DIM], in float _output_grad[OUTPUT_DIM], inout float _input_grad[INPUT_DIM]) {
                float _output[OUTPUT_DIM];
                SAVE_SEED(before_fw)
                forward(object, _input, _output);
                SET_SEED(before_fw)
                backward(object, _input, _output, _output_grad, _input_grad);  
            }
            """
        elif map.bw_implementations == BACKWARD_IMPLEMENTATIONS.DEFAULT:
            # add an implementation of a bw using output, ignoring the output
            code += """
            void backward (map_object, in float _input[INPUT_DIM], in float _output[OUTPUT_DIM], in float _output_grad[OUTPUT_DIM], inout float _input_grad[INPUT_DIM]) {
                backward(object, _input, _output_grad, _input_grad);  
            }
            """

        # Add a reduced overload to omit input grad propagation
        code += """
void backward (map_object, in float _input[INPUT_DIM], in float _output_grad[OUTPUT_DIM]) {
    float _input_grad[INPUT_DIM];
    backward(object, _input, _output_grad, _input_grad);  
}
        """

        code += f"#undef map_object\n"
        code += f"#undef parameters\n"

        for g in map.generics:
            code += f"#undef {g}\n"

        cls.__INCLUDE_DIRS__.extend(map.include_dirs)
        return code

    @classmethod
    def register_instance(cls, map: 'MapBase') -> _typing.Tuple[int, str]:  # new or existing instance id for the object
        signature = map.signature
        if signature not in cls.__MAP_INSTANCES__:
            instance_id = len(cls.__MAP_INSTANCES__) + 1
            codename = f'{type(map).__name__}_{instance_id}' # 'rdv_map_' + str(instance_id)
            cls.__CS_SUPER_KERNEL__ += cls.append_map_instance_source_code(map, instance_id, codename)
            cls.__MAP_INSTANCES__[signature] = (instance_id, codename)
        return cls.__MAP_INSTANCES__[signature]

    @classmethod
    def ensure_engine_objects(cls):
        if cls.__ENGINE_OBJECTS__ is not None:
            return

        map_fw_eval = _vk.object_buffer(layout=_vk.Layout.from_structure(_vk.LayoutAlignment.STD430,
                                                                                main_map=_torch.int64,
                                                                                input=_torch.int64,
                                                                                output=_torch.int64,
                                                                                seeds=_vk.ivec4,
                                                                                start_index=int,
                                                                                total_threads=int,
                                                                                debug_ptr=_torch.int64
                                                                                ))

        map_bw_eval = _vk.object_buffer(layout=_vk.Layout.from_structure(_vk.LayoutAlignment.STD430,
                                                                                main_map=_torch.int64,
                                                                                input=_torch.int64,
                                                                                output_grad=_torch.int64,
                                                                                input_grad=_torch.int64,
                                                                                seeds=_vk.ivec4,
                                                                                start_index=int,
                                                                                total_threads=int,
                                                                                debug_ptr=_torch.int64
                                                                                ))

        capture_fw_eval = _vk.object_buffer(layout=_vk.Layout.from_structure(_vk.LayoutAlignment.STD430,
                                                                                    capture_sensor=_torch.int64,
                                                                                    main_map=_torch.int64,
                                                                                    sensors=_torch.int64,
                                                                                    tensor=_torch.int64,
                                                                                    seeds=_vk.ivec4,
                                                                                    sensor_shape=[4, int],
                                                                                    start_index=int,
                                                                                    total_threads=int,
                                                                                    samples=int,
                                                                                    debug_ptr=_torch.int64
                                                                                    ))

        capture_bw_eval = _vk.object_buffer(layout=_vk.Layout.from_structure(_vk.LayoutAlignment.STD430,
                                                                                    capture_sensor=_torch.int64,
                                                                                    main_map=_torch.int64,
                                                                                    sensors=_torch.int64,
                                                                                    tensor_grad=_torch.int64,
                                                                                    seeds=_vk.ivec4,
                                                                                    sensor_shape=[4, int],
                                                                                    start_index=int,
                                                                                    total_threads=int,
                                                                                    samples=int,
                                                                                    debug_ptr=_torch.int64
                                                                                    ))
        cls.__ENGINE_OBJECTS__ = {
            'map_fw_eval': map_fw_eval,
            'map_bw_eval': map_bw_eval,
            'capture_fw_eval': capture_fw_eval,
            'capture_bw_eval': capture_bw_eval,
        }

    @classmethod
    def build_map_fw_eval_objects(cls, map: 'MapBase'):
        """
        Creates a raytracing or compute pipeline with the required buffers to eval map
        """
        if map.signature in cls.__FW_CS_ENGINE_PIPELINES__:
            return cls.__FW_CS_ENGINE_PIPELINES__[map.signature]

        full_code = """
#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_debug_printf : enable
""" + cls.create_support_code() + """
#include "common.h"

layout (local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

int DEBUG_COUNTER = 0;

        """ + cls.__CS_SUPER_KERNEL__ + f"""

layout(set = 0, std430, binding = 0) uniform RayGenMainDispatching {{
    {cls.register_instance(map)[1]} main_map; // Map model to execute
    GPUPtr input_tensor_ptr; // Input tensor (forward and backward stage)
    GPUPtr output_tensor_ptr; // Output tensor (forward stage)
    uvec4 seeds; // seeds for the batch randoms
    int start_index;
    int total_threads;
    GPUPtr debug_tensor_ptr;
}};        

layout(buffer_reference, scalar, buffer_reference_align=4) buffer rdv_input_data {{ float data [{map.input_dim}]; }};
layout(buffer_reference, scalar, buffer_reference_align=4) buffer rdv_output_data {{ float data [{map.output_dim}]; }};

void main()
{{
    int index = int(gl_GlobalInvocationID.x) + start_index;
    if (index >= total_threads) return;

    uvec4 current_seeds = seeds ^ uvec4(index ^ 17, index * 123111171, index + 11, index ^ (seeds.x + 13 * seeds.y));
    set_seed(current_seeds);
    random();
    random();
    random();

    int input_dim = {map.input_dim};
    int output_dim = {map.output_dim};
    rdv_input_data input_buf = rdv_input_data(input_tensor_ptr + index * input_dim * 4);
    rdv_output_data output_buf = rdv_output_data(output_tensor_ptr + index * output_dim * 4);
    forward(main_map, input_buf.data, output_buf.data);
    
    if (debug_tensor_ptr != 0)
    {{
        int_ptr d_buf = int_ptr(debug_tensor_ptr);
        d_buf.data[index] = DEBUG_COUNTER;
    }}
}}
        """

        cls.ensure_engine_objects()
        # Build pipeline for forward map evaluation
        pipeline = _vk.pipeline_compute()
        pipeline.layout(set=0, binding=0, settings=_vk.DescriptorType.UNIFORM_BUFFER)
        pipeline.load_shader_from_source(full_code, include_dirs=[_internal.__INCLUDE_PATH__] + cls.__INCLUDE_DIRS__)
        pipeline.close()

        global_bindings = pipeline.create_descriptor_set_collection(set=0, count = 1)
        global_bindings[0].update(
            settings = cls.__ENGINE_OBJECTS__['map_fw_eval']
        )

        cls.__FW_CS_ENGINE_PIPELINES__[map.signature] = (pipeline, global_bindings)

        return (pipeline, global_bindings)

    @classmethod
    def build_map_bw_eval_objects(cls, map: 'MapBase'):
        """
        Creates a raytracing or compute pipeline with the required buffers to eval map
        """
        if map.signature in cls.__BW_CS_ENGINE_PIPELINES__:
            return cls.__BW_CS_ENGINE_PIPELINES__[map.signature]

        full_code = """
    #version 460
    #extension GL_GOOGLE_include_directive : require
    #extension GL_EXT_debug_printf : enable
    """ + cls.create_support_code() + """
    #include "common.h"
    layout (local_size_x = 32, local_size_y = 1, local_size_z = 1) in;
    
    int DEBUG_COUNTER = 0;

            """ + cls.__CS_SUPER_KERNEL__ + f"""

    layout(set = 0, std430, binding = 0) uniform BackwardMapEval {{
        {cls.register_instance(map)[1]} main_map; // Map model to execute
        GPUPtr input_tensor_ptr; // Input tensor (forward and backward stage)
        GPUPtr output_tensor_grad_ptr; // Output tensor (backward stage)
        GPUPtr input_tensor_grad_ptr; // Input tensor gradients (backward stage)
        uvec4 seeds; // seeds for the batch randoms
        int start_index;
        int total_threads;
        GPUPtr debug_tensor_ptr;
    }};        

    layout(buffer_reference, scalar, buffer_reference_align=4) buffer rdv_input_data {{ float data [{map.input_dim}]; }};
    layout(buffer_reference, scalar, buffer_reference_align=4) buffer rdv_output_data {{ float data [{map.output_dim}]; }};

    void main()
    {{
        int index = int(gl_GlobalInvocationID.x) + start_index;
        if (index >= total_threads) return;

        uvec4 current_seeds = seeds ^ uvec4(index ^ 17, index * 123111171, index + 11, index ^ (seeds.x + 13 * seeds.y));
        set_seed(current_seeds);
        random();
        random();
        random();

        int input_dim = {map.input_dim};
        int output_dim = {map.output_dim};
        rdv_input_data input_buf = rdv_input_data(input_tensor_ptr + index * input_dim * 4);
        rdv_output_data output_grad_buf = rdv_output_data(output_tensor_grad_ptr + index * output_dim * 4);
        
        if (input_tensor_grad_ptr == 0) // no input gradient
            backward(main_map, input_buf.data, output_grad_buf.data);
        else
        {{
            rdv_input_data input_grad_buff = rdv_input_data(input_tensor_grad_ptr + index * input_dim * 4);
            backward(main_map, input_buf.data, output_grad_buf.data, input_grad_buff.data);
        }}
        
        if (debug_tensor_ptr != 0)
        {{
            int_ptr d_buf = int_ptr(debug_tensor_ptr);
            d_buf.data[index] = DEBUG_COUNTER;
        }}
    }}
            """

        cls.ensure_engine_objects()
        # Build pipeline for forward map evaluation
        pipeline = _vk.pipeline_compute()
        pipeline.layout(set=0, binding=0, settings=_vk.DescriptorType.UNIFORM_BUFFER)
        pipeline.load_shader_from_source(full_code, include_dirs=[_internal.__INCLUDE_PATH__] + cls.__INCLUDE_DIRS__)
        pipeline.close()

        global_bindings = pipeline.create_descriptor_set_collection(set=0, count=1)
        global_bindings[0].update(
            settings=cls.__ENGINE_OBJECTS__['map_bw_eval']
        )

        cls.__BW_CS_ENGINE_PIPELINES__[map.signature] = (pipeline, global_bindings)

        return (pipeline, global_bindings)

    @classmethod
    def build_capture_fw_eval_objects(cls, capture_object: 'SensorsBase', field: 'MapBase'):
        """
        Creates a raytracing or compute pipeline with the required buffers to eval the specific capture of map
        """
        capture_signature = (capture_object.signature, field.signature)
        if capture_signature in cls.__FW_CS_ENGINE_PIPELINES__:
            return cls.__FW_CS_ENGINE_PIPELINES__[capture_signature]

        full_code = """
        #version 460
        #extension GL_GOOGLE_include_directive : require
        #extension GL_EXT_debug_printf : enable
        """ + cls.create_support_code() + """
        #include "common.h"
        layout (local_size_x = 32, local_size_y = 1, local_size_z = 1) in;
        
        int DEBUG_COUNTER = 0;

                """ + cls.__CS_SUPER_KERNEL__ + f"""

        layout(set = 0, std430, binding = 0) uniform RayGenMainDispatching {{
            {cls.register_instance(capture_object)[1]} capture_object; // Map model to execute
            {cls.register_instance(field)[1]} main_map; // Map model to execute
            GPUPtr sensors_ptr; // Input tensor for sensors batch
            GPUPtr output_tensor_ptr; // Output tensor (forward stage)
            uvec4 seeds; // seeds for the batch randoms
            int sensor_shape[4]; // up to 4 lengths for each index dimension
            int start_index;
            int total_threads;
            int samples;
            GPUPtr debug_tensor_ptr;
        }};        

        layout(buffer_reference, scalar, buffer_reference_align=4) buffer rdv_output_data {{ float data [{field.output_dim}]; }};

        void main()
        {{
            int index = int(gl_GlobalInvocationID.x) + start_index;
            if (index >= total_threads) return;

            uvec4 current_seeds = seeds + uvec4(0x23F1+index,0x3137+(index+14)*index*17923113,129*index,index + 129) ;//^ uvec4(int(cos(index)*1000000), index ^ 1231231, index + 1234122, index + seeds.w * 100202021);//seeds ^ uvec4(index ^ 17, index * 123111171, index + 11, index ^ (seeds.x + 13 * seeds.y));

            //uvec4 current_seeds = seeds ^ uvec4(index * 78182311, index ^ 1231231, index + 1234122, index + seeds.w * 100202021);//seeds ^ uvec4(index ^ 17, index * 123111171, index + 11, index ^ (seeds.x + 13 * seeds.y));
            set_seed(current_seeds);
            advance_random();
            advance_random();
            uvec4 new_seed = floatBitsToUint(vec4(random(), random(), random(), random()));
            set_seed(new_seed);
            advance_random();
            advance_random();
            advance_random();
            advance_random();

            float indices[{capture_object.input_dim}];

            if (sensors_ptr == 0) {{
                int current_index_component = index;
                for (int i={capture_object.input_dim} - 1; i>=0; i--)
                {{
                    int d = sensor_shape[i];
                    indices[i] = intBitsToFloat(current_index_component % d);
                    current_index_component /= d;
                }}
            }}
            else
            {{
                int_ptr sensors_buf = int_ptr(sensors_ptr + 8 * {capture_object.input_dim} * index);
                for (int i=0; i<{capture_object.input_dim}; i++)
                    indices[i] = intBitsToFloat(sensors_buf.data[i*2]);
            }}

            if (samples == 1) {{
                float sensor_position[{field.input_dim}];
                forward(capture_object, indices, sensor_position);

                int output_dim = {field.output_dim};
                rdv_output_data output_buf = rdv_output_data(output_tensor_ptr + index * output_dim * 4);
                forward(main_map, sensor_position, output_buf.data);
            }}
            else {{
                int output_dim = {field.output_dim};
                rdv_output_data output_buf = rdv_output_data(output_tensor_ptr + index * output_dim * 4);
                for (int i=0; i<output_dim; i++)
                    output_buf.data[i] = 0.0;
                for (int s = 0; s < samples; s ++)
                {{
                    float sensor_position[{field.input_dim}];
                    forward(capture_object, indices, sensor_position);
                    float temp_output [{field.output_dim}];
                    forward(main_map, sensor_position, temp_output);
                    for (int i=0; i<output_dim; i++)
                        output_buf.data[i] += temp_output[i];
                }}
                for (int i=0; i<output_dim; i++)
                    output_buf.data[i] /= samples;
            }}
            
            if (debug_tensor_ptr != 0)
            {{
                int_ptr d_buf = int_ptr(debug_tensor_ptr);
                d_buf.data[index] = DEBUG_COUNTER;
            }}
        }}
                """

        cls.ensure_engine_objects()
        # Build pipeline for forward map evaluation
        pipeline = _vk.pipeline_compute()
        pipeline.layout(set=0, binding=0, settings=_vk.DescriptorType.UNIFORM_BUFFER)
        pipeline.load_shader_from_source(full_code, include_dirs=[_internal.__INCLUDE_PATH__]+cls.__INCLUDE_DIRS__)
        pipeline.close()

        global_bindings = pipeline.create_descriptor_set_collection(set=0, count=1)
        global_bindings[0].update(
            settings = cls.__ENGINE_OBJECTS__['capture_fw_eval']
        )

        cls.__FW_CS_ENGINE_PIPELINES__[capture_signature] = (pipeline, global_bindings)

        return (pipeline, global_bindings)

    @classmethod
    def build_capture_bw_eval_objects(cls, capture_object: 'SensorsBase', field: 'MapBase'):
        """
        Creates a raytracing or compute pipeline with the required buffers to eval the specific capture of map
        """
        capture_signature = (capture_object.signature, field.signature)
        if capture_signature in cls.__BW_CS_ENGINE_PIPELINES__:
            return cls.__BW_CS_ENGINE_PIPELINES__[capture_signature]

        full_code = """
        #version 460
        #extension GL_GOOGLE_include_directive : require
        #extension GL_EXT_debug_printf : enable
        """ + cls.create_support_code() + """
        #include "common.h"
        layout (local_size_x = 32, local_size_y = 1, local_size_z = 1) in;
        
        int DEBUG_COUNTER = 0;

                """ + cls.__CS_SUPER_KERNEL__ + f"""

        layout(set = 0, std430, binding = 0) uniform BackwardCaptureEval {{
            {cls.register_instance(capture_object)[1]} capture_object; // Map model to execute
            {cls.register_instance(field)[1]} main_map; // Map model to execute
            GPUPtr sensors_ptr; // Input tensor for sensors batch
            GPUPtr output_grad_tensor_ptr; // Output gradients tensor (backward stage)
            uvec4 seeds; // seeds for the batch randoms
            int sensor_shape[4]; // up to 4 lengths for each index dimension
            int start_index;
            int total_threads;
            int samples;
            GPUPtr debug_tensor_ptr;
        }};        

        layout(buffer_reference, scalar, buffer_reference_align=4) buffer rdv_output_data {{ float data [{field.output_dim}]; }};

        void main()
        {{
            int index = int(gl_GlobalInvocationID.x) + start_index;
            if (index >= total_threads) return;

            uvec4 current_seeds = seeds + uvec4(0x23F1,0x3137,129,index + 129) ;//^ uvec4(int(cos(index)*1000000), index ^ 1231231, index + 1234122, index + seeds.w * 100202021);//seeds ^ uvec4(index ^ 17, index * 123111171, index + 11, index ^ (seeds.x + 13 * seeds.y));

            //uvec4 current_seeds = seeds ^ uvec4(index * 78182311, index ^ 1231231, index + 1234122, index + seeds.w * 100202021);//seeds ^ uvec4(index ^ 17, index * 123111171, index + 11, index ^ (seeds.x + 13 * seeds.y));
            set_seed(current_seeds);
            advance_random();
            advance_random();
            advance_random();
            advance_random();
            uvec4 new_seed = floatBitsToUint(vec4(random(), random(), random(), random()));
            set_seed(new_seed);

            float indices[{capture_object.input_dim}];
            if (sensors_ptr == 0) {{
                int current_index_component = index;
                for (int i={capture_object.input_dim} - 1; i>=0; i--)
                {{
                    int d = sensor_shape[i];
                    indices[i] = intBitsToFloat(current_index_component % d);
                    current_index_component /= d;
                }}
            }}
            else
            {{
                int_ptr sensors_buf = int_ptr(sensors_ptr + 8 * {capture_object.input_dim} * index);
                for (int i=0; i<{capture_object.input_dim}; i++)
                    indices[i] = intBitsToFloat(sensors_buf.data[i*2]);
            }}

            float dL_dp [{field.input_dim}];
            for (int i = 0; i < {field.input_dim}; i++) dL_dp[i] = 0.0;
            
            int output_dim = {field.output_dim};
            rdv_output_data output_grad_buf = rdv_output_data(output_grad_tensor_ptr + index * output_dim * 4);
            float output_grad[{field.output_dim}];
            for (int i=0; i< {field.output_dim}; i++) output_grad[i] = output_grad_buf.data[i] / samples;
            
            float sensor_position[{field.input_dim}];

            for (int s = 0; s < samples; s++)
            {{
                forward(capture_object, indices, sensor_position);
                backward(main_map, sensor_position, output_grad, dL_dp);
            }}
            backward(capture_object, indices, dL_dp); // TODO: Check ways to update sensor parameters from ray differentials
            
            if (debug_tensor_ptr != 0)
            {{
                int_ptr d_buf = int_ptr(debug_tensor_ptr);
                d_buf.data[index] = DEBUG_COUNTER;
            }}
        }}
                """

        cls.ensure_engine_objects()
        # Build pipeline for forward map evaluation
        pipeline = _vk.pipeline_compute()
        pipeline.layout(set=0, binding=0, settings=_vk.DescriptorType.UNIFORM_BUFFER)
        pipeline.load_shader_from_source(full_code, include_dirs=[_internal.__INCLUDE_PATH__] + cls.__INCLUDE_DIRS__)
        pipeline.close()

        global_bindings = pipeline.create_descriptor_set_collection(set=0, count=1)
        global_bindings[0].update(
            settings=cls.__ENGINE_OBJECTS__['capture_bw_eval']
        )

        cls.__BW_CS_ENGINE_PIPELINES__[capture_signature] = (pipeline, global_bindings)

        return (pipeline, global_bindings)

    @classmethod
    def eval_capture_forward(cls, capture_object: 'SensorsBase', field: 'MapBase',
                             sensors: _typing.Optional[_torch.Tensor] = None, batch_size: _typing.Optional[int] = None,
                             fw_samples: int = 1, debug_out: _typing.Optional[_torch.Tensor] = None) -> _torch.Tensor:
        with cls.__LOCKER__:

            if sensors is not None:
                total_threads = sensors.numel() // sensors.shape[-1]
            else:
                total_threads = _math.prod(capture_object.index_shape[
                                          :capture_object.input_dim]).item()  # capture_object.screen_width * capture_object.screen_height

            assert debug_out is None or debug_out.numel() == total_threads, f"Debug tensor provided has not the required size {total_threads}"

            if batch_size is None:
                batch_size = total_threads

            cache_key = (batch_size, capture_object.signature, field.signature)

            # create man if not cached
            if cache_key not in cls.__FW_CAPTURE_CACHED_MAN__:
                pipeline, global_bindings = cls.build_capture_fw_eval_objects(capture_object, field)
                man = _vk.compute_manager()
                man.set_pipeline(pipeline)
                man.bind(global_bindings[0])
                man.dispatch_threads_1D(batch_size, group_size_x=32)
                man.freeze()
                cls.__FW_CAPTURE_CACHED_MAN__[cache_key] = man

            man = cls.__FW_CAPTURE_CACHED_MAN__[cache_key]

            # assert input.shape[-1] == map_object.input_dim, f'Wrong last dimension for the input tensor, must be {map_object.input_dim}'

            if sensors is not None:
                output = _vk.tensor(*sensors.shape[:-1], field.output_dim, dtype=_torch.float)
            else:
                output = _vk.tensor(*capture_object.index_shape[:capture_object.input_dim], field.output_dim, dtype=_torch.float)

            capture_object._pre_eval(False)
            field._pre_eval(False)

            output_ptr = _vk.wrap_gpu(output, 'out')

            with cls.__ENGINE_OBJECTS__['capture_fw_eval'] as b:
                b.capture_sensor = _vk.wrap_gpu(capture_object)
                b.main_map = _vk.wrap_gpu(field)
                b.sensors = _vk.wrap_gpu(sensors)
                b.tensor = output_ptr
                b.seeds[:] = _internal.get_seeds()
                shape = b.sensor_shape
                shape[0] = capture_object.index_shape[0].item()
                shape[1] = capture_object.index_shape[1].item()
                shape[2] = capture_object.index_shape[2].item()
                shape[3] = capture_object.index_shape[3].item()
                b.start_index = 0
                b.total_threads = total_threads
                b.samples = fw_samples
                b.debug_ptr = 0 if debug_out is None else _vk.wrap_gpu(debug_out, 'out')

            for batch in range((total_threads + batch_size - 1) // batch_size):
                with cls.__ENGINE_OBJECTS__['capture_fw_eval'] as b:
                    b.start_index = batch * batch_size
                _vk.submit(man)

            output_ptr.mark_as_dirty()
            output_ptr.invalidate()

            capture_object._pos_eval(False)
            field._pos_eval(False)

            # if sensors is None:
            #     output = output.view(*capture_object.index_shape[:capture_object.input_dim],-1)

            return output.clone()

    @classmethod
    def eval_capture_backward(cls, capture_object: 'SensorsBase', field: 'MapBase', output_grad: _torch.Tensor,
                              sensors: _typing.Optional[_torch.Tensor] = None, batch_size: _typing.Optional[int] = None,
                              bw_samples: int = 1, debug_out: _torch.Tensor = None):
        with cls.__LOCKER__:
            if sensors is not None:
                total_threads = sensors.numel() // sensors.shape[-1]
            else:
                total_threads = _math.prod(capture_object.index_shape[
                                          :capture_object.input_dim]).item()  # capture_object.screen_width * capture_object.screen_height

            assert debug_out is None or debug_out.numel() == total_threads, f"Debug tensor provided has not the required size {total_threads}"

            if batch_size is None:
                batch_size = total_threads

            cache_key = (batch_size, capture_object.signature, field.signature)

            # create man if not cached
            if cache_key not in cls.__BW_CAPTURE_CACHED_MAN__:
                pipeline, global_bindings = cls.build_capture_bw_eval_objects(capture_object, field)
                man = _vk.compute_manager()
                man.set_pipeline(pipeline)
                man.bind(global_bindings[0])
                man.dispatch_threads_1D(batch_size, group_size_x=32)
                man.freeze()
                cls.__BW_CAPTURE_CACHED_MAN__[cache_key] = man

            man = cls.__BW_CAPTURE_CACHED_MAN__[cache_key]

            # assert input.shape[-1] == map_object.input_dim, f'Wrong last dimension for the input tensor, must be {map_object.input_dim}'

            capture_object._pre_eval(True)
            field._pre_eval(True)
            _torch.cuda.synchronize()

            with cls.__ENGINE_OBJECTS__['capture_bw_eval'] as b:
                b.capture_sensor = _vk.wrap_gpu(capture_object)
                b.main_map = _vk.wrap_gpu(field)
                b.sensors = _vk.wrap_gpu(sensors)
                b.tensor_grad = _vk.wrap_gpu(output_grad)
                b.seeds[:] = _internal.get_seeds()
                shape = b.sensor_shape
                shape[0] = capture_object.index_shape[0].item()
                shape[1] = capture_object.index_shape[1].item()
                shape[2] = capture_object.index_shape[2].item()
                shape[3] = capture_object.index_shape[3].item()
                b.start_index = 0
                b.total_threads = total_threads
                b.samples = bw_samples
                b.debug_ptr = 0 if debug_out is None else _vk.wrap_gpu(debug_out, 'out')

            for batch in range((total_threads + batch_size - 1) // batch_size):
                with cls.__ENGINE_OBJECTS__['capture_bw_eval'] as b:
                    b.start_index = batch * batch_size
                _vk.submit(man)

            capture_object._pos_eval(True)
            field._pos_eval(True)
            _torch.cuda.synchronize()

    @classmethod
    def eval_map_forward(cls, map_object: 'MapBase', input: _torch.Tensor) -> _torch.Tensor:
        total_threads = _math.prod(input.shape[:-1])

        cache_key = (total_threads, map_object.signature)

        # create man if not cached
        if cache_key not in cls.__FW_DISPATCHER_CACHED_MAN__:
            pipeline, global_bindings = cls.build_map_fw_eval_objects(map_object)
            man = _vk.compute_manager()
            man.set_pipeline(pipeline)
            man.bind(global_bindings[0])
            man.dispatch_threads_1D(total_threads, group_size_x=32)
            man.freeze()
            cls.__FW_DISPATCHER_CACHED_MAN__[cache_key] = man

        man = cls.__FW_DISPATCHER_CACHED_MAN__[cache_key]

        assert input.shape[
                   -1] == map_object.input_dim, f'Wrong last dimension for the input tensor, must be {map_object.input_dim}'
        output = _vk.tensor(*input.shape[:-1], map_object.output_dim, dtype=_torch.float)

        map_object._pre_eval(False)

        output_ptr = _vk.wrap_gpu(output, 'out')

        with cls.__ENGINE_OBJECTS__['map_fw_eval'] as b:
            b.main_map = _vk.wrap_gpu(map_object)
            b.input = _vk.wrap_gpu(input)
            b.output = output_ptr
            b.seeds[:] = _internal.get_seeds()
            b.start_index = 0
            b.total_threads = total_threads

        _vk.submit(man)

        map_object._pos_eval(False)

        output_ptr.mark_as_dirty()
        output_ptr.invalidate()

        return output.clone()

    @classmethod
    def eval_map_backward(cls, map_object: 'MapBase', input: _torch.Tensor, output_grad: _torch.Tensor):
        with cls.__LOCKER__:
            total_threads = _math.prod(input.shape[:-1])

            cache_key = (total_threads, map_object.signature)

            # create man if not cached
            if cache_key not in cls.__BW_DISPATCHER_CACHED_MAN__:
                pipeline, global_bindings = cls.build_map_bw_eval_objects(map_object)
                man = _vk.compute_manager()
                man.set_pipeline(pipeline)
                man.bind(global_bindings[0])
                man.dispatch_threads_1D(total_threads, group_size_x=32)
                man.freeze()
                cls.__BW_DISPATCHER_CACHED_MAN__[cache_key] = man

            man = cls.__BW_DISPATCHER_CACHED_MAN__[cache_key]

            assert input.shape[
                       -1] == map_object.input_dim, f'Wrong last dimension for the input tensor, must be {map_object.input_dim}'
            assert output_grad.shape[
                       -1] == map_object.output_dim, f'Wrong last dimension for the output_grad tensor, must be {map_object.output_dim}'

            if input.requires_grad:  #
                input_grad = _torch.zeros_like(input)
            else:
                input_grad = None

            map_object._pre_eval(True)

            input_grad_ptr = _vk.wrap_gpu(input_grad, 'inout')

            with cls.__ENGINE_OBJECTS__['map_bw_eval'] as b:
                b.main_map = _vk.wrap_gpu(map_object)
                b.input = _vk.wrap_gpu(input)
                b.output_grad = _vk.wrap_gpu(output_grad)
                b.input_grad = input_grad_ptr
                b.seeds[:] = _internal.get_seeds()
                b.start_index = 0
                b.total_threads = total_threads

            _vk.submit(man)

            map_object._pos_eval(True)

            input_grad_ptr.mark_as_dirty()
            input_grad_ptr.invalidate()

            return input_grad


def start_engine():
    if _torch.cuda.is_available():
        _torch.cuda.init()
    DispatcherEngine.start()


def map_struct(
        struct_name: str,
        **fields
):
    return dict(
        __name__ = struct_name,
        **fields
    )


ParameterDescriptor = map_struct(
    'Parameter',
    data=_torch.Tensor,
    stride=[4, int],
    shape=[4, int],
    grad_data=_torch.Tensor
)


ParameterDescriptorLayoutType = dict(
    __name__='Parameter',
    data=_torch.int64,
    stride=[4, int],
    shape=[4, int],
    grad_data=_torch.int64
)



MeshInfo = map_struct(
    'MeshInfo',
    positions=ParameterDescriptor,
    normals=ParameterDescriptor,
    coordinates=ParameterDescriptor,
    tangents=ParameterDescriptor,
    binormals=ParameterDescriptor,
    indices=ParameterDescriptor
)


RaycastableInfo = map_struct(
    'RaycastableInfo',
    callable_map=_torch.int64,
    explicit_info=_torch.int64,
)


def parameter(p: _typing.Union[None, _torch.Tensor, _torch.nn.Parameter]):
    if p is None:
        return None
    if isinstance(p, _torch.nn.Parameter):
        return p
    assert not p.requires_grad, 'Tensors used as parameters can no require grads. Use Parameter instead.'
    return _torch.nn.Parameter(p, requires_grad=False)


def bind_parameter(field: _vk.ObjectBufferAccessor, t: _torch.Tensor):
    field.data = _vk.wrap_gpu(t, 'in')
    if t is not None:
        for i, s in enumerate(t.shape):
            field.stride[i] = t.stride(i)
            field.shape[i] = s


def bind_parameter_grad(field: _vk.ObjectBufferAccessor):
    if field.data.obj is None:
        field.grad_data = _vk.wrap_gpu(None, 'inout')
        return
    t: _torch.Tensor = field.data.obj
    if t.requires_grad:
        if t.grad is None:
            t.grad = _vk.tensor(*t.shape, dtype=t.dtype).zero_()
            # t.grad = _torch.zeros_like(t)
        field.grad_data = _vk.wrap_gpu(t.grad, 'inout')


class TensorCheck(object):
    def __init__(self, initial_value: _typing.Optional[_torch.Tensor] = None):
        self.cached_tensor = initial_value
        self.cached_tensor_version = -1 if initial_value is None else initial_value._version

    def changed(self, t: _torch.Tensor):
        if not (t is self.cached_tensor):
            self.cached_tensor = t
            return True
        if t._version != self.cached_tensor_version:
            self.cached_tensor_version = t._version
            return True
        return False


class MapMeta(type):
    __DYNAMIC_ID__  = 0   # This is an autoincremental id for dynamic maps
    def __new__(cls, name, bases, dct):
        ext_class = super().__new__(cls, name, bases, dct)
        assert '__extension_info__' in dct, 'Extension maps requires a dict __extension_info__ with path, parameters, [optional] bw_implementations'
        extension_info = dct['__extension_info__']
        if extension_info is not None:  # is not an abstract node
            extension_path = extension_info.get('path', None)
            extension_code = extension_info.get('code', None)
            extension_generics = extension_info.get('generics', {})
            extension_dynamic_requires = extension_info.get('dynamics', [])  # List with list of map signatures that can be dispatched dynamically by this map
            parameters = extension_info.get('parameters', {})
            assert (extension_path is None or isinstance(extension_path, str) and _os.path.isfile(
                extension_path)), 'path must be a valid file path str'
            include_dirs = extension_info.get('include_dirs', [])
            assert (extension_path is None) != (extension_code is None), 'Either path or code must be provided'
            if extension_path is not None:
                include_dirs.append(_os.path.dirname(extension_path))
                extension_code = f"#include \"{_os.path.basename(extension_path)}\"\n"
                # with open(extension_path) as f:
                #     extension_code = f.readlines()
            bw_implementations = extension_info.get('bw_implementations', BACKWARD_IMPLEMENTATIONS.NONE)

            def from_type_2_layout_description(p, dynamic_array_size = 0):
                if p == MapBase:
                    return _torch.int64
                if p == _torch.Tensor:
                    return _torch.int64
                if isinstance(p, list):
                    return [p[0] if p[0] > 0 else dynamic_array_size, from_type_2_layout_description(p[1])]
                if isinstance(p, dict):
                    return {'__name__': p.get('__name__'), ** {k: from_type_2_layout_description(v, dynamic_array_size) for k, v in p.items() if k != '__name__'}}
                return p

            parameters = {'rdv_map_id': int, 'rdv_map_pad0': int, 'rdv_map_pad1': int, 'rdv_map_pad2': int, **parameters}
            parameters_layout = lambda s: _vk.Layout.from_description(_vk.LayoutAlignment.SCALAR, description=from_type_2_layout_description(parameters, s))
            ext_class.default_generics = extension_generics
            ext_class.dynamic_requires = extension_dynamic_requires
            ext_class.map_object_layout = parameters_layout
            ext_class.map_object_definition = parameters
            ext_class.map_source_code = extension_code
            ext_class.bw_implementations = bw_implementations # Determines if the bw uses a cached output evaluation to improve efficiency while replaying, or is default or none
            ext_class.include_dirs = include_dirs
            ext_class.map_code = DispatcherEngine.register_map(ext_class)
        return ext_class

    def __call__(self, *args, **kwargs):
        map_instance: MapBase = super(MapMeta, self).__call__(*args, **kwargs)
        if not map_instance.is_generic_input and not map_instance.is_generic_output:
            assert not map_instance.has_generic_submap(), f'A non-generic map {type(map_instance)} can not contains generic submaps'
        if not map_instance.is_generic:
            generic_for_dynamic_id = {}
            if len(self.dynamic_requires) != 0:  # if dynamic module is used
                MapMeta.__DYNAMIC_ID__  += 1
                generic_for_dynamic_id = {'RDV_DYNAMIC_ID': MapMeta.__DYNAMIC_ID__ }
            map_instance.generics.update(generic_for_dynamic_id)
            map_instance._create_signature()
            map_id, map_codename = DispatcherEngine.register_instance(map_instance)
            map_instance.rdv_map_id = map_id
        return map_instance


class GPUDirectModule(_torch.nn.Module):
    def __init__(self, accessor: _typing.Optional[_vk.ObjectBufferAccessor]):
        assert accessor is None or accessor._rdv_layout.is_structure
        super().__init__()
        object.__setattr__(self, '_rdv_accessor', accessor)
        if accessor is not None:
            # object used to access attributes from the Module directly to the gpu
            for k, (offset, field_type) in accessor._rdv_layout.fields_layout.items():
                if field_type.is_array:  # lists
                    object.__setattr__(self, k, getattr(accessor, k))
                if field_type.is_structure and field_type.declaration['__name__'] != ParameterDescriptor['__name__']:
                    # Wrap field structures with a struct module
                    super().__setattr__(k, StructModule(getattr(accessor, k)))

    def __setattr__(self, key, value):
        a: _vk.ObjectBufferAccessor = self._rdv_accessor
        if a is not None:
            # Update gpu info if field is part of the accessed object
            if key in a._rdv_fields:
                field_layout = a._rdv_layout.fields_layout[key][1]
                if field_layout.scalar_format == 'Q':
                    if isinstance(value, int):
                        setattr(a, key, value)
                    else:
                        setattr(a, key, _vk.wrap_gpu(value))
                elif field_layout.is_structure and field_layout.declaration['__name__'] == ParameterDescriptor['__name__']: # key in self._parameters or isinstance(value, _torch.nn.Parameter):
                    bind_parameter(getattr(a, key), value)
                else:
                    setattr(a, key, value)
        super().__setattr__(key, value)

    def _pre_eval(self, include_grads: bool = False):
        if include_grads:
            for k,v in self._parameters.items():
                if v.requires_grad:
                    # print(f'Bound parameter with grad for {k} in {type(self)}')
                    bind_parameter_grad(getattr(self._rdv_accessor, k))
        def deep_pre_eval(m: _typing.Union[_torch.nn.Module]):
            if isinstance(m, GPUDirectModule):
                m._pre_eval(include_grads)
            if isinstance(m, _torch.nn.ModuleList):
                for c in m:
                    deep_pre_eval(c)
        for k, m in self._modules.items():
            deep_pre_eval(m)
        # iterate over all potential wrapped objects that needs to update their info on the gpu
        for r in self._rdv_accessor.references():
            if r is not None:
                r.flush()

    def _pos_eval(self, include_grads: bool = False):
        def deep_pos_eval(m: _typing.Union[_torch.nn.Module]):
            # if isinstance(m, GPUDirectModule):
            #     m._pos_eval(include_grads)
            if isinstance(m, _torch.nn.ModuleList):
                for c in m:
                    deep_pos_eval(c)
            for k, mo in m._modules.items():
                deep_pos_eval(mo)
            # iterate over all potential wrapped objects that needs to update their info from the gpu
            if isinstance(m, GPUDirectModule):
                for r in self._rdv_accessor.references():
                    if r is not None:
                        if isinstance(r.obj, _torch.Tensor):
                            assert _torch.isnan(r.obj).sum() == 0, f'Object {type(r.obj)} with nans, shape: {r.obj.shape}'
                        r.mark_as_dirty()
                        r.invalidate()
                        if isinstance(r.obj, _torch.Tensor):
                            assert _torch.isnan(r.obj).sum() == 0, f'Object {type(r.obj)} with nans, shape: {r.obj.shape}'

        deep_pos_eval(self)


class MapBase(GPUDirectModule, metaclass=MapMeta):
    __extension_info__ = None  # none extension info marks the node as abstract
    __bindable__ = None
    map_object_layout: _typing.Callable[[int], _vk.Layout] = None

    def __init__(self, *args, **generics):
        array_size = 0 if len(args) == 0 else args[0]
        generics = {**self.default_generics, **generics}
        is_generic = 'INPUT_DIM' not in generics or 'OUTPUT_DIM' not in generics
        # if not is_generic:
        instance_layout = type(self).map_object_layout(array_size)
        map_buffer = _vk.object_buffer(layout=instance_layout, usage=_vk.BufferUsage.STORAGE,
                                   memory=_vk.MemoryLocation.CPU)
        object.__setattr__(self, '__bindable__', map_buffer)
        object.__setattr__(self, '_rdv_map_buffer', map_buffer)
        object.__setattr__(self, '_rdv_trigger_bw', _torch.tensor([0.0], requires_grad=True))
        object.__setattr__(self, '_rdv_no_trigger_bw', _torch.tensor([0.0], requires_grad=False))
        object.__setattr__(self, 'generics', generics)
        super().__init__(map_buffer.accessor)

    def has_generic_submap(self):
        def deep_search(module: _torch.nn.Module):
            for k, m in module._modules.items():
                if isinstance(m, MapBase):
                    if m.is_generic:
                        return True
                    continue
                if m is None:
                    return False
                return deep_search(m)
            return False
        return deep_search(self)

    @_vk.lazy_constant
    def is_generic(self):
        return self.input_dim is None or self.output_dim is None or self.has_generic_submap()

    @_vk.lazy_constant
    def is_generic_input(self):
        return self.input_dim is None

    @_vk.lazy_constant
    def is_generic_output(self):
        return self.output_dim is None

    @staticmethod
    def match_input(*maps: 'MapBase') -> _typing.Tuple['MapBase', ...]:
        input_dim = None
        for m in maps:
            input_dim = input_dim or m.input_dim
        if input_dim is None:
            return maps
        return tuple(m.cast(input_dim=input_dim) for m in maps)

    @staticmethod
    def match_output(*maps: 'MapBase'):
        output_dim = None
        for m in maps:
            output_dim = output_dim or (m.output_dim if m.output_dim != 1 else None)
        if output_dim is None or output_dim == 1:
            return maps
        return tuple(m.cast(output_dim=output_dim) for m in maps)

    @_vk.lazy_constant
    def is_scalar_output(self):
        return self.output_dim is not None and self.output_dim == 1

    def cast(self, input_dim: _typing.Optional[int] = None, output_dim: _typing.Optional[int] = None):
        """
        This method sould be overriden in all maps that support generics to create
        a generic instance. Always check if the map type satisfies input and output dimensions.
        """
        assert not self.is_generic, f"Missing cast at {type(self)}: all generic maps must implement properly a cast method"
        assert input_dim is None or self.input_dim == input_dim, f"Required input {input_dim} was not satisfied {self.input_dim}"
        assert output_dim is None or self.output_dim == output_dim or self.output_dim == 1, f"Required output {output_dim} was not satisfied {self.output_dim}"
        if self.is_scalar_output and output_dim is not None and output_dim > 1:
            return self.promote(output_dim)  # Automatic map promotion
        return self

    @_vk.lazy_constant
    def input_dim(self) -> _typing.Optional[int]:
        return self.generics.get('INPUT_DIM', None)

    @_vk.lazy_constant
    def output_dim(self) -> _typing.Optional[int]:
        return self.generics.get('OUTPUT_DIM', None)

    def get_maximum(self) -> _torch.Tensor:
        raise NotImplementedError()

    def support_maximum_query(self) -> bool:
        return False

    def _create_signature(self):
        submodules = []

        def collect_submodules(field_name, t, v):
            if t == MapBase:
                assert v is not None, f'Expected module {field_name} to be bound'
                submodules.append(v.obj)
                return
            if isinstance(t, list):
                size = t[0]
                element_type = t[1]
                if size > 0:
                    collect_submodules(field_name, element_type, v[0])  # only collect from zero index, TODO: CHECK all array derivations have the same signature!
            if isinstance(t, dict):
                for field_name, field_type in t.items():
                    if field_name != '__name__':
                        collect_submodules(field_name, field_type, getattr(v, field_name))

        collect_submodules('', self.map_object_definition, self._rdv_accessor)

        object.__setattr__(self, 'signature', (
        self.map_code, *[0 if s is None else DispatcherEngine.register_instance(s)[0] for s in submodules],
        *[v for k, v in self.generics.items()]))

    def _pre_eval(self, include_grads: bool = False):
        super()._pre_eval(include_grads)
        self._rdv_map_buffer.update_gpu()

    # def forward_torch(self, *args):
    #     raise Exception(f"Not implemented torch engine in type {type(self)}")

    def forward(self, *args):
        input_tensor, = args
        assert not self.is_generic_output, "Can not eval a map with generic output. Use cast specifying desired output dimension"
        assert self.is_generic_input or self.input_dim == input_tensor.shape[-1], "Map input dimension and tensor last dim missmatch"
        if self.is_generic_input:
            map_object = self.cast(input_dim=input_tensor.shape[-1])
        else:
            map_object = self
        assert not map_object.is_generic, f'Evaluated map is still generic input:{map_object.is_generic_input} output:{map_object.is_generic_output} children: {map_object.has_generic_submap()}'
        trigger_bw = map_object._rdv_trigger_bw if any(True for _ in map_object.parameters()) else map_object._rdv_no_trigger_bw
        return AutogradMapFunction.apply(input_tensor, trigger_bw, map_object)

    def after(self, prev_map: 'MapBase') -> 'MapBase':
        return CompositionMap(prev_map, self)

    def then(self, next_map: 'MapBase') -> 'MapBase':
        if isinstance(self, Identity):
            return next_map
        if isinstance(next_map, Identity):
            return self
        if isinstance(next_map, InputPromoteMap):
            return ComposePromoteMap(self, dim=next_map.output_dim)
        return CompositionMap(self, next_map)

    def promote(self, output_dim: _typing.Optional[int] = None) -> 'MapBase':
        return ComposePromoteMap(self, output_dim)

    @staticmethod
    def as_map(data: _typing.Union[int, float, _torch.Tensor, 'MapBase']):
        if isinstance(data, MapBase):
            return data
        if isinstance(data, int) or isinstance(data, float):
            t = _vk.tensor(1, dtype=_torch.float32)
            t[:] = data
            return ConstantMap(t)
        if isinstance(data, _torch.Tensor):
            return ConstantMap(data)
        raise Exception(f"Can not convert type {type(data)} into a map")

    def like_this(self, o):
        if isinstance(o, int) or isinstance(o, float):
            t = _vk.tensor(self.output_dim, dtype=_torch.float32)
            t[:] = o
            return ConstantMap(t, input_dim=self.input_dim)
        if isinstance(o, _torch.Tensor):
            o = ConstantMap(o, input_dim=self.input_dim)
        if isinstance(o, MapBase):
            if o.output_dim == self.output_dim:
                return o
            return o.promote(self.output_dim)
        raise Exception(f'Can not cast type {type(o)} to this map')

    @staticmethod
    def make_match(v1, v2):
        if not isinstance(v1, MapBase):
            assert isinstance(v2, MapBase)
            v2, v1 = MapBase.make_match(v2, v1)
            return v1, v2
        if not isinstance(v2, MapBase) or v2.output_dim < v1.output_dim:
            return v1, v1.like_this(v2)
        return v2.like_this(v1), v2

    def custom_gradient(self, map: 'MapBase') -> 'MapBase':
        return CustomGradMap(self, map)

    def __add__(self, other):
        return AdditionMap(self, MapBase.as_map(other))

    def __radd__(self, other):
        return AdditionMap(self, MapBase.as_map(other))

    def __sub__(self, other):
        return SubtractionMap(self, MapBase.as_map(other))

    def __rsub__(self, other):
        return SubtractionMap(self, MapBase.as_map(other))

    def __mul__(self, other):
        return MultiplicationMap(self, MapBase.as_map(other))

    def __rmul__(self, other):
        return MultiplicationMap(self, MapBase.as_map(other))

    def __truediv__(self, other):
        return DivisionMap(self, MapBase.as_map(other))

    def __rtruediv__(self, other):
        return DivisionMap(self, MapBase.as_map(other))

    def __getitem__(self, item):
        if isinstance(item, int):
            return ComposeSelectMap(self, [item])
        if isinstance(item, tuple) or isinstance(item, list):
            return ComposeSelectMap(self, list(item))
        if isinstance(item, slice):
            if self.output_dim is None:
                indices = list(range(10000)) # sufficently large to index
            else:
                indices = [i for i in range(self.output_dim)]
            return ComposeSelectMap(self, indices[item])
        raise Exception(f"Not supported index/slice object {type(item)}")

    def __or__(self, other):
        return ConcatMap(self, MapBase.as_map(other))

    def __matmul__(self, other):
        return self.then(MatrixMultMap(other))

    def __rmatmul__(self, other):
        return self.then(MatrixMultMap(other, premul=True))


# class ParameterizedAutograd(_torch.autograd.Function):
#     @staticmethod
#     def forward(ctx: _typing.Any, *args: _typing.Any, **kwargs: _typing.Any) -> _typing.Any:
#         process, *p = args
#         input_p = [_torch.nn.Parameter(t, requires_grad=t.requires_grad) for t in p]
#         with _torch.enable_grad():
#             outputs = process(*input_p)
#             if isinstance(outputs, _torch.Tensor):
#                 outputs = [outputs]
#             os = [o.clone() for o in outputs]
#         ctx.input_count = len(p)
#         ctx.save_for_backward(*p, *input_p, *os)
#         if not isinstance(outputs, _torch.Tensor) and len(outputs) == 1:
#             return outputs[0]
#         return outputs
#
#     @staticmethod
#     def backward(ctx: _typing.Any, *grad_outputs: _typing.Any) -> _typing.Any:
#         input_count = ctx.input_count
#         p = ctx.saved_tensors[:input_count]
#         input_p = ctx.saved_tensors[input_count:input_count*2]
#         outputs = ctx.saved_tensors[input_count*2:]
#         with _torch.enable_grad():
#             # grads = _torch.autograd.grad(outputs, input_p, grad_outputs, allow_unused=True)
#             _torch.autograd.backward(outputs, grad_outputs)
#         return None, *tuple(None if t.grad is None else t.grad.clone() for t in input_p)



class ParameterizedAutograd(_torch.autograd.Function):
    @staticmethod
    def forward(ctx: _typing.Any, *args: _typing.Any, **kwargs: _typing.Any) -> _typing.Any:
        process, *p = args
        input_p = [_torch.nn.Parameter(t, requires_grad=t.requires_grad) for t in p]
        ctx.process = process
        ctx.save_for_backward(*p)
        return process(*input_p)

        # input_p = [_torch.nn.Parameter(t, requires_grad=t.requires_grad) for t in p]
        # with _torch.enable_grad():
        #     outputs = process(*input_p)
        #     if isinstance(outputs, _torch.Tensor):
        #         outputs = [outputs]
        #     os = [o.clone() for o in outputs]
        # ctx.input_count = len(p)
        # ctx.save_for_backward(*p, *input_p, *os)
        # if not isinstance(outputs, _torch.Tensor) and len(outputs) == 1:
        #     return outputs[0]
        # return outputs

    @staticmethod
    def backward(ctx: _typing.Any, *grad_outputs: _typing.Any) -> _typing.Any:
        process = ctx.process
        p = list(ctx.saved_tensors)
        with _torch.enable_grad():
            input_p = [_torch.nn.Parameter(t, requires_grad=t.requires_grad) for t in p]
            outputs = process(*input_p)
            grads = _torch.autograd.grad(outputs, input_p, grad_outputs, allow_unused=True)
            # _torch.autograd.backward(outputs, grad_outputs)
        return None, *grads  #tuple(None if t.grad is None else t.grad.clone() for t in input_p)



def parameterized_call(callable, *args):
    assert callable is not None
    return ParameterizedAutograd.apply(callable, *args)



class StructModule(GPUDirectModule):
    def __init__(self, accessor: _vk.ObjectBufferAccessor):
        super().__init__(accessor)


class Sampler(MapBase):
    """
    Represents a distribution of x~q(x|C) by means of a map
    C -> x, w(x) where x~p(x|C) and w(x) is a weighted sampled of x (q(x|C)/p(x|C)).
    """
    __extension_info__ = None  # Abstract Node

    def get_pdf(self) -> 'MapBase':
        '''
        Gets the map that represents the pdf q(x|C)
        '''
        raise NotImplementedError()

    @staticmethod
    def mixture(alpha: MapBase, sampler_a: 'Sampler', sampler_b: 'Sampler') -> 'Sampler':
        return MixtureSampler(alpha, sampler_a, sampler_b)

    @staticmethod
    def mis(self, *samplers):
        pass


class MixtureSampler(Sampler):
    __extension_info__ = dict(
        parameters=dict(
            alpha=MapBase,
            sampler_a=MapBase,
            sampler_b=MapBase
        ),
        code=f"""
FORWARD
{{
    float alpha[1];
    forward(parameters.alpha, _input, alpha);
    if (random() < alpha[0])    
    {{
        forward(parameters.sampler_a, _input, _output);
    }}
    else {{
        forward(parameters.sampler_b, _input, _output);
    }}
}}
        """
    )

    def __init__(self, alpha: MapBase, sampler_a: Sampler, sampler_b: Sampler):
        super().__init__()
        self.alpha = alpha
        self.sampler_a = sampler_a
        self.sampler_b = sampler_b

    def get_pdf(self) -> 'MapBase':
        return MixturePDF(self.alpha, self.sampler_a.get_pdf(), self.sampler_b.get_pdf())


class MixturePDF(MapBase):
    __extension_info__ = dict(
        parameters=dict(
            alpha=MapBase,
            sampler_a_pdf=MapBase,
            sampler_b_pdf=MapBase
        ),
        code=f"""
FORWARD {{
    float alpha[1];
    float _condition[CONDITION_DIM];
    for (int i=0; i<CONDITION_DIM; i++)
        _condition[i] = _input[i];
    forward(parameters.alpha, _condition, alpha);
    float _pdf_a[1];
    forward(parameters.sampler_a_pdf, _input, _pdf_a);
    float _pdf_b[1];
    forward(parameters.sampler_b_pdf, _input, _pdf_b);
    _output[0] = _pdf_a[0] * alpha[0] + _pdf_b[0] * (1 - alpha[0]);
}}
        """
    )


class AutogradMapFunction(_torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args):
        input_tensor, triggering, map_object = args
        ctx.save_for_backward(input_tensor)  # properly save tensors for backward
        ctx.map_object = map_object
        return DispatcherEngine.eval_map_forward(map_object, input_tensor)

    @staticmethod
    def backward(ctx, *args):
        output_grad, = args
        input_tensor, = ctx.saved_tensors  # Just check for inplace operations in input tensors
        map_object = ctx.map_object
        input_grad = DispatcherEngine.eval_map_backward(map_object, input_tensor, output_grad)
        return (input_grad, None, None)  # append None to refer to renderer object passed in forward


class AutogradCaptureFunction(_torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args):
        sensors, batch_size, fw_samples, bw_samples, field, capture_object, debug_out, *parameters = args
        ctx.save_for_backward(*parameters)
        ctx.field = field
        ctx.sensors = sensors
        ctx.capture_object = capture_object
        ctx.batch_size = batch_size
        ctx.bw_samples = bw_samples
        ctx.debug_out = debug_out
        return DispatcherEngine.eval_capture_forward(capture_object, field, sensors, batch_size, fw_samples, debug_out)

    @staticmethod
    def backward(ctx, *args):
        output_grad, = args
        capture = ctx.capture_object
        sensors = ctx.sensors
        field = ctx.field
        batch_size = ctx.batch_size
        bw_samples = ctx.bw_samples
        debug_out = ctx.debug_out
        DispatcherEngine.eval_capture_backward(capture, field, output_grad, sensors, batch_size, bw_samples, debug_out)
        # print(f"[DEBUG] Backward grads from renderer {grad_inputs[0].mean()}")
        # assert grad_inputs[0] is None or _torch.isnan(grad_inputs[0]).sum() == 0, "error in generated grads."
        return (None, None, None, None, None, None, None) + tuple(p.grad for p in ctx.saved_tensors)  # append None to refer to renderer object passed in forward


class ActivationMap(MapBase):
    __extension_info__ = None

    def __init__(self, dimension: _typing.Optional[int] = None):
        dims = { } if dimension is None else dict(INPUT_DIM=dimension, OUTPUT_DIM=dimension)
        super().__init__(**dims)

    def cast(self, input_dim: _typing.Optional[int] = None, output_dim: _typing.Optional[int] = None):
        assert input_dim is None or output_dim is None or input_dim == output_dim
        cast_dim = input_dim or output_dim
        if cast_dim is None:
            return self
        if not self.is_generic:
            assert self.input_dim == cast_dim or self.input_dim == 1
            if self.is_scalar_output and output_dim is not None and output_dim > 1:
                return self.promote(output_dim)  # Automatic map promotion
            return self
        return type(self)(cast_dim)



class Identity(ActivationMap):
    __extension_info__ = dict(
        path=_internal.__INCLUDE_PATH__+"/maps/identity.h",
        bw_implementations=BACKWARD_IMPLEMENTATIONS.DEFAULT
    )
    #
    # def __init__(self, dimension: _typing.Optional[int] = None):
    #     dims = { } if dimension is None else dict(INPUT_DIM=dimension, OUTPUT_DIM=dimension)
    #     super(Identity, self).__init__(**dims)
    #
    # def cast(self, input_dim: _typing.Optional[int] = None, output_dim: _typing.Optional[int] = None):
    #     assert input_dim is None or output_dim is None or input_dim == output_dim
    #     cast_dim = input_dim or output_dim
    #     if cast_dim is None:
    #         return self
    #     if not self.is_generic:
    #         assert self.input_dim == cast_dim or self.input_dim == 1
    #         if self.is_scalar_output and output_dim is not None and output_dim > 1:
    #             return self.promote(output_dim)  # Automatic map promotion
    #         return self
    #     return Identity(cast_dim)


class SensorsBase(MapBase):
    __extension_info__ = None  # Abstract node

    def __init__(self, index_shape: _typing.List[int], **generics):
        super(SensorsBase, self).__init__(INPUT_DIM=len(index_shape), **generics)
        object.__setattr__(self, 'identity', Identity(self.output_dim))
        object.__setattr__(self, 'index_shape', _torch.tensor(index_shape + ([0] * (4 - len(index_shape)))))
        object.__setattr__(self, '_rdv_trigger_bw', _torch.tensor([0.0], requires_grad=True))
        object.__setattr__(self, '_rdv_no_trigger_bw', _torch.tensor([0.0], requires_grad=False))

    # def measurement_point_dim(self):
    #     return self.output_dim

    # def generate_measuring_points_torch(self, indices):
    #     raise NotImplementedError()

    # def forward_torch(self, *args):
    #     if len(args) == 0 or args[0] is None:
    #         dims = self.index_shape[:self.input_dim]
    #         sensors = _torch.cartesian_prod(
    #             *[_torch.arange(0, d, dtype=_torch.long, device=_torch.device('cuda:0')) for d in dims])
    #     else:
    #         sensors, = args
    #     return self.generate_measuring_points_torch(sensors)

    def forward(self, *args):
        '''
        Generates random measurement points for all or selected batch of sensors.
        :param args: empty set or batch of sensors.
        :return: a tensor with all points where measurement should be taken.
        '''
        if len(args) == 0 or args[0] is None:
            sensors = None
        else:
            sensors, = args
        return self.capture(self.identity, sensors, None, 1, 1)

    # def capture_torch(self, field: 'MapBase', sensors_batch: Optional[_torch.Tensor] = None, fw_samples: int = 1):
    #     # TODO: Implement a torch base replay backpropagation for stochastic processes
    #     output = None
    #     for _ in range(fw_samples):
    #         points = self.forward_torch(sensors_batch)
    #         if output is None:
    #             output = field.forward_torch(points)
    #         else:
    #             output += field.forward_torch(points)
    #     output /= fw_samples
    #     if sensors_batch is None:
    #         # reshape
    #         output = output.view(*self.index_shape[:self.input_dim], -1)
    #     return output

    def capture(self, field: 'MapBase', sensors_batch: _typing.Optional[_torch.Tensor] = None, batch_size: _typing.Optional[int] = None,
                fw_samples: int = 1, bw_samples: int = 1, debug_out: _typing.Optional[_torch.Tensor] = None):
        # if not __USE_VULKAN_DISPATCHER__:
        #     return self.capture_torch(field, sensors_batch, fw_samples)
        field = field.cast(input_dim=self.output_dim)
        triggers = [p for p in field.parameters() if p.requires_grad]
        trigger_bw = self._rdv_trigger_bw #if any(True for _ in self.parameters()) or any(
            # True for _ in field.parameters()) else self._rdv_no_trigger_bw
        return AutogradCaptureFunction.apply(sensors_batch, batch_size, fw_samples, bw_samples, field, self, debug_out, *triggers)

    def random_sensors(self, batch_size: int, out: _typing.Optional[_torch.Tensor] = None) -> _torch.Tensor:
        # if out is not None:
        #     out.copy_((_torch.rand(batch_size, self.input_dim, device=device()) * self.index_shape[:self.input_dim].to(
        #         device())).long())
        # else:
        #     out = (_torch.rand(batch_size, self.input_dim, device=device()) * self.index_shape[:self.input_dim].to(
        #         device())).long()
        # return out
        if out is None:
            # out = _torch.zeros(batch_size, self.input_dim, dtype=_torch.long, device=device())
            out = _vk.tensor(batch_size, self.input_dim, dtype=_torch.long)
        return _functions.random_ids(batch_size, self.index_shape[:self.input_dim], out=out)


class PerspectiveCameraSensor(SensorsBase):
    __extension_info__ = dict(
        parameters=dict(
            poses=_torch.Tensor,
            width=int,
            height=int,
            generation_mode=int,
            fov=float,
            znear=float
        ),
        generics=dict(OUTPUT_DIM=6),
        path=_internal.__INCLUDE_PATH__+"/maps/sensor_perspective_camera.h"
    )

    def __init__(self, width: int, height: int, poses: _torch.Tensor, *, fov: float = _np.pi / 4, jittered: bool = False):
        super(PerspectiveCameraSensor, self).__init__([poses.numel() // 9, height, width])
        self.poses = poses
        self.fov = fov
        self.znear = 0.001
        self.width = width
        self.height = height
        self.generation_mode = 0 if not jittered else 1

    # def generate_measuring_points_torch(self, indices):
    #     o = self.poses[indices[:, 0], 0:3]
    #     d = self.poses[indices[:, 0], 3:6]
    #     n = self.poses[indices[:, 0], 6:9]
    #     dim = _torch.tensor([self.height, self.width], dtype=_torch.float, device=indices.device)
    #     s = ((indices[:, 1:3] + 0.5) * 2 - dim) * self.znear / self.height
    #     t = _np.float32(self.znear / _np.float32(_np.tan(_np.float32(self.fov) * _np.float32(0.5))))
    #     zaxis = vec3.normalize(d)
    #     xaxis = vec3.normalize(vec3.cross(n, zaxis))
    #     yaxis = vec3.cross(zaxis, xaxis)
    #     w = xaxis * s[:, 1:2] + yaxis * s[:, 0:1] + zaxis * t
    #     x = o + w
    #     w = vec3.normalize(w)
    #     return _torch.cat([x, w], dim=-1)


class CompositionMap(MapBase):
    __extension_info__ = dict(
        parameters=dict(
            inner=MapBase,
            outter=MapBase,
        ),
        bw_implementations=BACKWARD_IMPLEMENTATIONS.DEFAULT,
        path=_internal.__INCLUDE_PATH__ + '/maps/composition.h'
    )

    def __init__(self, inner: MapBase, outter: MapBase):
        if inner.is_generic_output:
            inner = inner.cast(output_dim=outter.input_dim)
        if outter.is_generic_input:
            outter = outter.cast(input_dim=inner.output_dim)
        assert inner.is_generic_output or outter.is_generic_input or inner.output_dim == outter.input_dim
        dims = {}
        if not inner.is_generic_input:
            dims.update(INPUT_DIM=inner.input_dim)
        if not outter.is_generic_output:
            dims.update(OUTPUT_DIM=outter.output_dim)
        if not inner.is_generic_output:
            dims.update(INTERMEDIATE_DIM=inner.output_dim)
        super(CompositionMap, self).__init__(**dims)
        self.inner = inner
        self.outter = outter

    def cast(self, input_dim: _typing.Optional[int] = None, output_dim: _typing.Optional[int] = None):
        inner = self.inner.cast(input_dim=input_dim)
        outter = self.outter.cast(output_dim=output_dim)
        if self.inner is inner and self.outter is outter:  # cast didnt change
            return self
        return CompositionMap(inner, outter)


class CustomGradMap(MapBase):
    __extension_info__ = dict(
        parameters=dict(
            fw=MapBase,
            bw=MapBase
        ),
        bw_implementations=BACKWARD_IMPLEMENTATIONS.ALL,
        path=_internal.__INCLUDE_PATH__ + '/maps/custom_grad.h'
    )

    def __init__(self, fw: MapBase, bw: MapBase):
        assert fw.is_generic_input or bw.is_generic_input or fw.input_dim == bw.input_dim
        assert fw.is_generic_output or bw.is_generic_output or fw.output_dim == bw.output_dim
        input_dim = fw.input_dim or bw.input_dim
        output_dim = fw.output_dim or bw.output_dim
        fw = fw.cast(input_dim, output_dim)
        bw = bw.cast(input_dim, output_dim)
        dims = {}
        if input_dim is not None:
            dims.update(INPUT_DIM=input_dim)
        if output_dim is not None:
            dims.update(OUTPUT_DIM=output_dim)
        super().__init__(**dims)
        self.fw = fw
        self.bw = bw

    def cast(self, input_dim: _typing.Optional[int] = None, output_dim: _typing.Optional[int] = None):
        fw = self.fw.cast(input_dim, output_dim)
        bw = self.bw.cast(input_dim, output_dim)
        if fw is self.fw and bw is self.bw:
            return self
        return CustomGradMap(fw, bw)



class BinaryOpMap(MapBase):
    __extension_info__ = None  # Mark as an abstract map

    @staticmethod
    def create_extension_info(path: str):
        return dict(
            parameters=dict(
                map_a=MapBase,
                map_b=MapBase
            ),
            bw_implementations=BACKWARD_IMPLEMENTATIONS.DEFAULT,
            path=path
        )

    def __init__(self, map_a: MapBase, map_b: MapBase):
        map_a, map_b = MapBase.match_input(map_a, map_b)
        map_a, map_b = MapBase.match_output(map_a, map_b)
        assert map_a.is_generic_input or map_b.is_generic_input or map_a.input_dim == map_b.input_dim
        assert map_a.is_generic_output or map_b.is_generic_output or map_a.output_dim == map_b.output_dim
        dims = {}
        if not map_a.is_generic_input and not map_b.is_generic_input:
            dims.update(INPUT_DIM = map_a.input_dim)
        if not map_a.is_generic_output and not map_b.is_generic_output:
            dims.update(OUTPUT_DIM = map_a.output_dim)
        super(BinaryOpMap, self).__init__(**dims)
        self.map_a = map_a
        self.map_b = map_b

    def cast(self, input_dim: _typing.Optional[int] = None, output_dim: _typing.Optional[int] = None):
        map_a = self.map_a.cast(input_dim=input_dim, output_dim=output_dim)
        map_b = self.map_b.cast(input_dim=input_dim, output_dim=output_dim)
        if map_a is self.map_a and map_b is self.map_b:
            return self
        return type(self)(map_a, map_b)


class AdditionMap(BinaryOpMap):
    __extension_info__ = BinaryOpMap.create_extension_info(_internal.__INCLUDE_PATH__+'/maps/operator_add.h')


class SubtractionMap(BinaryOpMap):
    __extension_info__ = BinaryOpMap.create_extension_info(_internal.__INCLUDE_PATH__+'/maps/operator_sub.h')


class MultiplicationMap(BinaryOpMap):
    __extension_info__ = BinaryOpMap.create_extension_info(_internal.__INCLUDE_PATH__+'/maps/operator_mul.h')


class DivisionMap(BinaryOpMap):
    __extension_info__ = BinaryOpMap.create_extension_info(_internal.__INCLUDE_PATH__+'/maps/operator_div.h')


class ComposePromoteMap(MapBase):
    __extension_info__ = dict(
        parameters=dict(map=MapBase),
        bw_implementations=BACKWARD_IMPLEMENTATIONS.ALL,
        path=_internal.__INCLUDE_PATH__+'/maps/compose_promote.h'
    )

    def __init__(self, map: MapBase, dim: _typing.Optional[int] = None):
        map = map.cast(output_dim=1)
        # assert map.output_dim == 1, 'Promotion is only valid for single valued maps'
        dims = { 'INPUT_DIM': map.input_dim }
        if dim is not None:
            dims.update(OUTPUT_DIM=dim)
        super(ComposePromoteMap, self).__init__(**dims)
        self.map = map

    def cast(self, input_dim: int = None, output_dim: int = None):
        map = self.map.cast(input_dim=input_dim, output_dim=1)
        assert self.input_dim is None or input_dim is None or input_dim == self.input_dim, f'Promotion cast to invalid input {input_dim}'
        if not self.is_generic_output:
            assert output_dim is None or self.output_dim == output_dim, f'Non-generic promotion cast to invalid output {output_dim}'
            if map is self.map:
                return self
            return ComposePromoteMap(map, self.output_dim)
        if output_dim is None:
            if map is self.map:
                return self
            return ComposePromoteMap(map, self.output_dim)
        return ComposePromoteMap(self.map, output_dim)


class ConstantMap(MapBase):
    __extension_info__ = dict(
        parameters=dict(
            value=ParameterDescriptor
        ),
        bw_implementations=BACKWARD_IMPLEMENTATIONS.DEFAULT,
        path=_internal.__INCLUDE_PATH__+'/maps/constant.h'
    )

    def __init__(self, value: _typing.Union[_torch.Tensor, _torch.nn.Parameter], input_dim: _typing.Optional[int] = None):
        assert len(value.shape) <= 1, f"Error {value.shape}"
        assert value.dtype == _torch.float
        dims = {'OUTPUT_DIM': value.numel()}
        if input_dim is not None:
            dims.update(INPUT_DIM=input_dim)
        super(ConstantMap, self).__init__(**dims)
        self.value = parameter(value)

    def cast(self, input_dim: int = None, output_dim: int = None):
        assert output_dim is None or output_dim == self.output_dim or self.output_dim == 1, f'Constant map cast to an invalid output dim {output_dim} from {self.output_dim}'
        c = self
        if input_dim is not None:
            assert self.is_generic_input or self.input_dim == input_dim
            if self.is_generic_input:
                c = ConstantMap(c.value, input_dim)
        if output_dim is not None:
            if self.output_dim == 1 and output_dim > 1:
                c = c.promote(output_dim)
        return c


class ConcatMap(MapBase):
    __extension_info__ = dict(
        parameters=dict(
            map_a=MapBase,
            map_b=MapBase
        ),
        bw_implementations=BACKWARD_IMPLEMENTATIONS.ALL,
        path=_internal.__INCLUDE_PATH__ + '/maps/concat.h'
    )

    def __init__(self, map_a: MapBase, map_b: MapBase):
        input_dim = None
        if map_a.input_dim is not None and map_b.input_dim is None:
            map_b = map_b.cast(input_dim = map_a.input_dim)

        if map_b.input_dim is not None and map_a.input_dim is None:
            map_a = map_a.cast(input_dim= map_b.input_dim)

        assert map_a.input_dim == map_b.input_dim
        super(ConcatMap, self).__init__(
            INPUT_DIM=map_a.input_dim,
            OUTPUT_DIM=map_a.output_dim + map_b.output_dim,
            A_OUTPUT_DIM=map_a.output_dim,
            B_OUTPUT_DIM=map_b.output_dim
        )
        self.map_a = map_a
        self.map_b = map_b


class MatrixMultMap(MapBase):
    __extension_info__ = dict(
        parameters=dict(
            matrix=ParameterDescriptor,
            is_pre=int
        ),
        bw_implementations=BACKWARD_IMPLEMENTATIONS.DEFAULT,
        path=_internal.__INCLUDE_PATH__+'/maps/matmul.h'
    )

    def __init__(self, matrix: _torch.Tensor, premul: bool = False):
        assert len(matrix.shape) == 2
        if premul:
            dims = dict(INPUT_DIM=matrix.shape[1], OUTPUT_DIM=matrix.shape[0])
        else:
            dims = dict(INPUT_DIM=matrix.shape[0], OUTPUT_DIM=matrix.shape[1])
        super().__init__(**dims)
        self.matrix = parameter(matrix)
        self.is_pre=int(premul)


class ComposeSelectMap(MapBase):
    __extension_info__ = dict(
        parameters=dict(
            map=MapBase,
            indices=[32, int]
        ),
        bw_implementations=BACKWARD_IMPLEMENTATIONS.DEFAULT,
        path=_internal.__INCLUDE_PATH__ + '/maps/compose_select.h'
    )

    def __init__(self, map: MapBase, indices: _typing.List[int]):
        if map.output_dim is not None:
            assert all(i >= 0 and i<map.output_dim for i in indices)
        super(ComposeSelectMap, self).__init__(
            INPUT_DIM=map.input_dim,
            OUTPUT_DIM=len(indices),
            MAP_OUTPUT_DIM=map.output_dim
        )
        self.map = map
        for i in range(len(indices)):
            self.indices[i] = indices[i]

    # def cast(self, input_dim=None, output_dim=None):
    #     m = self
    #     if output_dim is not None:
    #         assert output_dim == self.output_dim or self.output_dim == 1
    #         if self.output_dim != output_dim:
    #             m = self.promote(output_dim)


class InputSelectMap(MapBase):
    __extension_info__ = dict(
        parameters=dict(
            indices=[32, int]
        ),
        path=_internal.__INCLUDE_PATH__+'/maps/input_select.h'
    )

    def __init__(self, input_dim: int, indices: _typing.List[int]):
        assert all(i >= 0 and i<input_dim for i in indices)
        super(InputSelectMap, self).__init__(
            INPUT_DIM=input_dim,
            OUTPUT_DIM=len(indices)
        )
        for i in range(len(indices)):
            self.indices[i] = indices[i]


class InputPromoteMap(MapBase):
    __extension_info__ = dict(
        parameters=dict(
            indices=[32, int]
        ),
        path=_internal.__INCLUDE_PATH__+'/maps/input_promote.h',
        bw_implementations=BACKWARD_IMPLEMENTATIONS.DEFAULT
    )

    def __init__(self, output_dim: int):
        super(InputPromoteMap, self).__init__(
            INPUT_DIM=1,
            OUTPUT_DIM=output_dim
        )


class ReluMap(ActivationMap):
    # __extension_info__ = ActivationMap.create_info(
    #     fw = "max(0, x)",
    #     bw = "dL_dx = dL_dy * (x > 0) 1: 0"
    # )
    __extension_info__ = dict(
        path=_internal.__INCLUDE_PATH__ + '/maps/f_relu.h',
        bw_implementations = BACKWARD_IMPLEMENTATIONS.DEFAULT
    )

class SinMap(ActivationMap):
    __extension_info__ = dict(
        path=_internal.__INCLUDE_PATH__ + '/maps/f_sin.h',
        bw_implementations = BACKWARD_IMPLEMENTATIONS.DEFAULT
    )


class CosMap(ActivationMap):
    __extension_info__ = dict(
        path=_internal.__INCLUDE_PATH__ + '/maps/f_cos.h',
        bw_implementations = BACKWARD_IMPLEMENTATIONS.DEFAULT
    )


# class Sequential(MapBase):
#     """
#     Experimental: Dynamic maps slow down performance.
#     """
#     __extension_info__ = dict(
#         parameters=dict(
#             initial_map=MapBase,
#             final_map=MapBase,
#             # -1 means unbounded array, real value used will come from the initialization
#             # _torch.int64 is to represent a pointer (dynamic map), not a specific map (typed)
#             maps = [-1, _torch.int64]
#         ),
#         path=_internal.__INCLUDE_PATH__ + "/maps/sequential.h",
#         bw_implementations = BACKWARD_IMPLEMENTATIONS.NONE
#     )
#
#     def __init__(self, *maps: MapBase):
#         maps = list(maps)
#         assert len(maps) >= 3, "At least 3 maps are required. initial, intermediates and final."
#         intermediate_dim = None
#         for i, m in enumerate(maps):
#             if i == 0:
#                 intermediate_dim = m.output_dim
#             elif i == len(maps) - 1:
#                 intermediate_dim = m.input_dim if m.input_dim is not None else intermediate_dim
#             else:
#                 intermediate_dim = m.input_dim if m.input_dim is not None else intermediate_dim
#                 intermediate_dim = m.output_dim if m.output_dim is not None else intermediate_dim
#         if intermediate_dim is not None:
#             for i in range(len(maps)):
#                 if i == 0:
#                     maps[i] = maps[i].cast(output_dim=intermediate_dim)
#                 elif i == len(maps)-1:
#                     maps[i] = maps[i].cast(input_dim=intermediate_dim)
#                 else:
#                     maps[i] = maps[i].cast(input_dim=intermediate_dim, output_dim=intermediate_dim)
#         input_dim = maps[0].input_dim
#         output_dim = maps[-1].output_dim
#         assert all(m.input_dim == intermediate_dim and m.output_dim == intermediate_dim for m in maps[1:-1])
#         dims = {}
#         if input_dim is not None:
#             dims.update(INPUT_DIM=input_dim)
#         if output_dim is not None:
#             dims.update(OUTPUT_DIM=output_dim)
#         if intermediate_dim is not None:
#             dims.update(INTERMEDIATE_DIM=intermediate_dim)
#         number_of_maps = len(maps) - 2
#         dims.update(NUMBER_OF_MAPS=number_of_maps)
#         super().__init__(
#             len(maps)-2,  # *args with single argument will be used to fill the size of unbounded arrays
#             **dims,  # generics
#         )
#         self.initial_map = maps[0]
#         self.final_map = maps[-1]
#         for i in range(1, len(maps)-1):
#             self.maps[i-1] = maps[i].__bindable__.device_ptr  # dynamic maps
#
#         self.map_objects = maps  # used to cast
#         if intermediate_dim is not None:
#             self.dynamic_requires = [(intermediate_dim, intermediate_dim)]
#
#     def cast(self, input_dim: _typing.Optional[int] = None, output_dim: _typing.Optional[int] = None):
#         if self.input_dim == input_dim and self.output_dim == output_dim:
#             return self  # no cast necessary
#         maps = self.map_objects
#         maps[0] = maps[0].cast(input_dim=input_dim)
#         maps[-1] = maps[-1].cast(output_dim=output_dim)
#         return Sequential(*maps)


# class CompiledMap(MapBase):
#
#     __extension_info__ = None
#
#     @staticmethod
#     def build_extension_info(cls) -> dict:
#         pass
#
# class MLP(CompiledMap):
#     @classmethod
#     def build_extension_info(cls) -> dict:
#         pass
#
#     __extension_info__ = CompiledMap.build_extension_info(MLP)


# class CompiledMap(MapBase):
#     __extension_info__ = dict(
#
#     )
#
#     @classmethod
#     def





# def build_custom_map(
#     code: str,
#     bw_implementations: BACKWARD_IMPLEMENTATIONS,
#     **parameters
# ):
#     class CustomMap(MapBase):
#         __extension_info__ = dict(
#             parameters=parameters,
#             code=code,
#             bw_implementations=bw_implementations
#         )
#     return CustomMap
#
# mlp_type = build_custom_map(mlp_dsl.build(), )


# Sensors

# class CameraSensor(SensorsBase):
#     __extension_info__ = dict(
#         parameters=dict(
#             origin=_torch.Tensor,
#             direction=_torch.Tensor,
#             normal=_torch.Tensor,
#             width=int,
#             height=int,
#             generation_mode=int,
#             fov=float,
#             znear=float
#         ),
#         generics=dict(OUTPUT_DIM=6),
#         code=f"""
# FORWARD
# {{
#     ivec3 index = floatBitsToInt(vec3(_input[0], _input[1], _input[2]));
#     vec3_ptr origin_buf = vec3_ptr(parameters.origin);
#     vec3_ptr direction_buf = vec3_ptr(parameters.direction);
#     vec3_ptr normal_buf = vec3_ptr(parameters.normal);
#     vec3 o = origin_buf.data[index[0]];
#     vec3 d = direction_buf.data[index[0]];
#     vec3 n = normal_buf.data[index[0]];
#
#     vec2 subsample = vec2(0.5);
#     if (parameters.generation_mode == 1)
#         subsample = vec2(random(), random());
#
#     float sx = ((index[2] + subsample.x) * 2 - parameters.width) * parameters.znear / parameters.height;
#     float sy = ((index[1] + subsample.y) * 2 - parameters.height) * parameters.znear / parameters.height;
#     float sz = parameters.znear / tan(parameters.fov * 0.5);
#
#     vec3 zaxis = normalize(d);
#     vec3 xaxis = normalize(cross(n, zaxis));
#     vec3 yaxis = cross(zaxis, xaxis);
#
#     vec3 x, w;
#
#     w = xaxis * sx + yaxis * sy + zaxis * sz;
#     x = o + w;
#     w = normalize(w);
#
#     _output = float[6]( x.x, x.y, x.z, w.x, w.y, w.z );
# }}
# """
#     )
#
#     def __init__(self, width: int, height: int, cameras: int = 1, jittered: bool = False):
#         super(CameraSensor, self).__init__([cameras, height, width])
#         self.origin = tensor(cameras, 3, dtype=_torch.float32)
#         self.direction = tensor(cameras, 3, dtype=_torch.float32)
#         self.normal = tensor(cameras, 3, dtype=_torch.float32)
#         self.origin[:] = 0.0
#         self.origin[:, 2] = -1.0
#         self.direction[:] = 0.0
#         self.direction[:, 2] = 1.0
#         self.normal[:] = 0.0
#         self.normal[:, 1] = 1.0
#         self.fov = _np.pi / 4
#         self.znear = 0.001
#         self.width = width
#         self.height = height
#         self.generation_mode = 0 if not jittered else 1
#
#     def generate_measuring_points_torch(self, indices):
#         o = self.origin[indices[:, 0]]
#         d = self.direction[indices[:, 0]]
#         n = self.normal[indices[:, 0]]
#         dim = _torch.tensor([self.height, self.width], dtype=_torch.float, device=indices.device)
#         s = ((indices[:, 1:3] + 0.5) * 2 - dim) * self.znear / self.height
#         t = _np.float32(self.znear / _np.float32(_np.tan(_np.float32(self.fov) * _np.float32(0.5))))
#         zaxis = vec3.normalize(d)
#         xaxis = vec3.normalize(vec3.cross(n, zaxis))
#         yaxis = vec3.cross(zaxis, xaxis)
#         w = xaxis * s[:, 1:2] + yaxis * s[:, 0:1] + zaxis * t
#         x = o + w
#         w = vec3.normalize(w)
#         return _torch.cat([x, w], dim=-1)


class Grid3DSensor(SensorsBase):
    __extension_info__ = dict(
        parameters=dict(
            width=int,
            height=int,
            depth=int,
            box_min=vec3,
            box_max=vec3,
            sd=float
        ),
        generics=dict(OUTPUT_DIM=3),
        path=_internal.__INCLUDE_PATH__ + '/maps/sensor_grid3d.h'
    )

    def __init__(self, width: int, height: int, depth: int = 1, box_min: vec3 = vec3(-1.0, -1.0, -1.0),
                 box_max: vec3 = vec3(1.0, 1.0, 1.0), sd: float = 0.0):
        super(Grid3DSensor, self).__init__([depth, height, width])
        self.width = width
        self.height = height
        self.depth = depth
        self.sd = sd
        self.box_min = box_min.clone()
        self.box_max = box_max.clone()

    def generate_measuring_points_torch(self, indices):
        dim = _torch.tensor([self.depth - 1, self.height - 1, self.width - 1], dtype=_torch.float, device=indices.device)
        subsample = _torch.randn(*indices.shape, device=indices.device) * self.sd
        bmin = self.box_min.to(indices.device)
        bmax = self.box_max.to(indices.device)
        return ((indices + subsample) / dim) * (bmax - bmin) + bmin


class Box3DSensor(SensorsBase):
    __extension_info__ = dict(
        parameters=dict(
            count=int,
            box_min=vec3,
            box_max=vec3
        ),
        generics=dict(OUTPUT_DIM=3),
        path=_internal.__INCLUDE_PATH__ + '/maps/sensor_box.h'
    )

    def __init__(self, samples, box_min: vec3 = vec3(-1.0, -1.0, -1.0), box_max: vec3 = vec3(1.0, 1.0, 1.0)):
        super(Box3DSensor, self).__init__([samples])
        self.box_min = box_min.clone()
        self.box_max = box_max.clone()


class DummyExample(MapBase):
    __extension_info__ = dict(
        parameters=dict(
            alpha=float
        ),
        code="""
FORWARD {
    for (int i=0; i<INPUT_DIM; i++)
        _output[i] = (_input[i] * parameters.alpha + _input[i]) * parameters.alpha;
}               
        """,
        use_raycast=False,
    )
    
    def __init__(self, dimension: int, alpha: float = 1.0):
        super(DummyExample, self).__init__(INPUT_DIM=dimension, OUTPUT_DIM=dimension)
        # parameter fields requires a 'proxy' attribute in the module
        self.alpha = alpha

    def forward_torch(self, *args):
        return (args[0] * self.alpha + args[0])*self.alpha


class Grid2D(MapBase):
    __extension_info__ = dict(
        parameters=dict(
            grid=ParameterDescriptor,
            bmin=vec2,
            inv_bsize=vec2
        ),
        generics=dict(INPUT_DIM=2),
        path=_internal.__INCLUDE_PATH__ + '/maps/grid2d.h'
    )

    def __init__(self, grid: _typing.Union[_torch.Tensor, _torch.nn.Parameter], bmin: vec2 = vec2(-1.0, -1.0), bmax: vec2 = vec2(1.0, 1.0)):
        assert len(grid.shape) == 3
        super(Grid2D, self).__init__(OUTPUT_DIM=grid.shape[-1])
        self.grid = parameter(grid)
        self.bmin = bmin.clone()
        self.inv_bsize = (1.0/(bmax- bmin)).clone()

    def forward_torch(self, *args):
        x, = args
        x = 2 * (x - self.bmin.to(self.grid.device)) * self.inv_bsize.to(self.grid.device) - 1
        return _torch.nn.functional.grid_sample(
            self.grid.unsqueeze(0).permute(0, 3, 1, 2),
            x.reshape(1, len(x), 1, -1),
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        ).permute(0, 2, 3, 1).reshape(-1, self.output_dim)


class Image2D(MapBase):
    __extension_info__ = dict(
        parameters=dict(
            grid=ParameterDescriptor,
            bmin=vec2,
            inv_bsize=vec2
        ),
        generics=dict(INPUT_DIM=2),
        bw_implementations=BACKWARD_IMPLEMENTATIONS.DEFAULT,
        path=_internal.__INCLUDE_PATH__ + '/maps/image2d.h'
    )

    def __init__(self, grid: _typing.Union[_torch.Tensor, _torch.nn.Parameter], bmin: vec2 = vec2(-1.0, -1.0), bmax: vec2 = vec2(1.0, 1.0)):
        assert len(grid.shape) == 3
        super(Image2D, self).__init__(OUTPUT_DIM=grid.shape[-1])
        self.grid = parameter(grid)
        self.bmin = bmin.clone()
        self.inv_bsize = (1.0/(bmax- bmin)).clone()

    def forward_torch(self, *args):
        x, = args
        x = 2 * (x - self.bmin.to(self.grid.device)) * self.inv_bsize.to(self.grid.device) - 1
        return _torch.nn.functional.grid_sample(
            self.grid.unsqueeze(0).permute(0, 3, 1, 2),
            x.reshape(1, len(x), 1, -1),
            mode='bilinear',
            padding_mode='border',
            align_corners=False
        ).permute(0, 2, 3, 1).reshape(-1, self.output_dim)


class Grid3D(MapBase):
    __extension_info__ = dict(
        generics=dict(INPUT_DIM=3),
        parameters=dict(
            grid=ParameterDescriptor,
            bmin=vec3,
            pad0=float,
            inv_bsize=vec3,
            pad1=float
        ),
        bw_implementations=BACKWARD_IMPLEMENTATIONS.DEFAULT,
        path=_internal.__INCLUDE_PATH__ + '/maps/grid3d.h'
    )

    def __init__(self, grid: _torch.Tensor, bmin: vec3 = vec3(-1.0, -1.0, -1.0), bmax: vec3 = vec3(1.0, 1.0, 1.0)):
        assert len(grid.shape) == 4
        super(Grid3D, self).__init__(OUTPUT_DIM=grid.shape[-1])
        self.grid = parameter(grid)
        self.bmin = bmin.clone()
        self.inv_bsize = (1.0/(bmax- bmin)).clone()
        self.bmax = bmax.clone()
        self.width = grid.shape[2]
        self.height = grid.shape[1]
        self.depth = grid.shape[0]

    def forward_torch(self, *args):
        x, = args
        x = 2 * (x - self.bmin.to(self.grid.device)) * self.inv_bsize.to(self.grid.device) - 1
        return _torch.nn.functional.grid_sample(
            self.grid.unsqueeze(0).permute(0, 4, 1, 2, 3),
            x.reshape(1, len(x), 1, 1, -1),
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        ).permute(0, 2, 3, 4, 1).view(-1, self.output_dim)

    def line_integral(self):
        # return LineIntegrator(self)
        # return old_Grid3DLineIntegral(self)
        return Grid3DLineIntegral(self)

    def to_transmittance(self):
        return TransmittanceFromTau(self.line_integral())

    def get_maximum(self) -> float:
        return self.grid.max().item()

    def support_maximum_query(self) -> bool:
        return True


class Transformed3DMap(MapBase):
    __extension_info__ = dict(
        parameters=dict(
            base_map=MapBase,
            transform=_vk.mat4,
            inverse_transform=_vk.mat4
        ),
        generics=dict(INPUT_DIM=3),
        bw_implementations = BACKWARD_IMPLEMENTATIONS.DEFAULT,
        path = _internal.__INCLUDE_PATH__ + '/maps/transform_3D_map.h'
    )

    def __init__(self, base_map: MapBase, transform: _vk.mat4 ):
        assert base_map.input_dim == 3
        super().__init__(OUTPUT_DIM=base_map.output_dim)
        self.base_map = base_map
        self.transform = transform
        self.inverse_transform = _vk.mat4.inverse(transform)


class TransformedRayMap(MapBase):
    __extension_info__ = dict(
        parameters=dict(
            base_map=MapBase,
            transform=_vk.mat4,
            inverse_transform=_vk.mat4
        ),
        generics=dict(INPUT_DIM=6),
        bw_implementations = BACKWARD_IMPLEMENTATIONS.DEFAULT,
        path = _internal.__INCLUDE_PATH__ + '/maps/transform_ray_map.h'
    )

    def __init__(self, base_map: MapBase, transform: _vk.mat4 ):
        assert base_map.input_dim == 6
        super().__init__(OUTPUT_DIM=base_map.output_dim)
        self.base_map = base_map
        self.transform = transform
        self.inverse_transform = _vk.mat4.inverse(transform)


class XRProjection(MapBase):
    __extension_info__ = dict(
        parameters=dict(
        ),
        generics=dict(
            OUTPUT_DIM=2
        ),
        path=_internal.__INCLUDE_PATH__ + '/maps/xr_projection.h'
    )

    def __init__(self, ray_input: bool = False):
        super(XRProjection, self).__init__(INPUT_DIM=6 if ray_input else 3)
        object.__setattr__(self, 'input_ray', 1 if ray_input else 0)

    def forward_torch(self, *args):
        if self.input_ray:
            xw, = args
            w = xw[:, 3:6]
        else:
            w, = args
        #    vec2 c = vec2((atan(w.z, w.x) + pi) / (2 * pi), acos(clamp(w.y, -1.0, 1.0)) / pi); // two floats for coordinates
        a = (_torch.atan2(w[:,0:1], w[:,2:3])) / _np.pi
        b = 2 * _torch.acos(_torch.clamp(w[:, 1:2], -1.0, 1.0)) / _np.pi - 1
        return _torch.cat([a, b], dim=-1)


class OctProjection(MapBase):
    __extension_info__ = dict(
        parameters=dict(
        ),
        generics=dict(
            OUTPUT_DIM=2
        ),
        path=_internal.__INCLUDE_PATH__ + '/maps/oct_projection.h'
    )

    def __init__(self, ray_input: bool = False):
        super().__init__(INPUT_DIM=6 if ray_input else 3)


class OctUnprojection(MapBase):
    __extension_info__ = dict(
        parameters=dict(
        ),
        generics=dict(
            INPUT_DIM=2,
            OUTPUT_DIM=3
        ),
        path=_internal.__INCLUDE_PATH__ + '/maps/oct_unprojection.h'
    )


class UniformSampler(MapBase):
    __extension_info__ = dict(
        parameters=dict(
        ),
        path=_internal.__INCLUDE_PATH__+'/maps/uniform_sampler.h'
    )

    def __init__(self, input_dim, point_dim):
        super(UniformSampler, self).__init__(INPUT_DIM=input_dim, OUTPUT_DIM=point_dim + 1)


class GaussianSampler(MapBase):
    __extension_info__ = dict(
        parameters=dict(
        ),
        path=_internal.__INCLUDE_PATH__ + '/maps/gaussian_sampler.h'
    )

    def __init__(self, input_dim, output_dim):
        super(GaussianSampler, self).__init__(INPUT_DIM=input_dim, OUTPUT_DIM=output_dim)


class UniformDirectionSampler(MapBase):
    __extension_info__ = dict(
        parameters=dict(
        ),
        generics=dict(OUTPUT_DIM=4),
        path=_internal.__INCLUDE_PATH__ +'/maps/uniform_direction_sampler.h'
    )

    def __init__(self, input_dim):
        super(UniformDirectionSampler, self).__init__(INPUT_DIM=input_dim)


class XRQuadtreeRandomDirection(MapBase):
    __extension_info__ = dict(
        parameters=dict(
            densities=_torch.Tensor,
            levels=int
        ),
        generics=dict(OUTPUT_DIM=4),
        code="""
    FORWARD
    {
        float_ptr densities_buf = float_ptr(parameters.densities);
        float sel = random();
        int current_node = 0;
        vec2 p0 = vec2(0,0);
        vec2 p1 = vec2(1,1);
        float prob = 1;
        for (int i=0; i<parameters.levels; i++)
        {
            int offset = current_node * 4;
            int selected_child = 3;
            prob = densities_buf.data[offset + 3];
            for (int c = 0; c < 3; c ++)
                if (sel < densities_buf.data[offset + c])
                {
                    selected_child = c;
                    prob = densities_buf.data[offset + c];
                    break;
                }
                else
                    sel -= densities_buf.data[offset + c];;
            float xmed = (p1.x + p0.x)/2;
            float ymed = (p1.y + p0.y)/2;
            if (selected_child % 2 == 0) // left
                p1.x = xmed;
            else
                p0.x = xmed;
            if (selected_child / 2 == 0) // top
                p1.y = ymed;
            else
                p0.y = ymed;
            current_node = current_node * 4 + 1 + selected_child;
        }
        float pixel_area = 2 * pi * (cos(p0.y*pi) - cos(p1.y*pi)) / (1 << parameters.levels);
        float weight = pixel_area / max(0.000000001, prob);
        vec3 w_out = randomDirection((p0.x * 2 - 1) * pi, (p1.x * 2 - 1) * pi, p0.y * pi, p1.y * pi);
        _output = float[4](w_out.x, w_out.y, w_out.z, weight);
    }
            """,
    )

    def __init__(self, input_dim: int, densities: _torch.Tensor, levels: int):
        super(XRQuadtreeRandomDirection, self).__init__(INPUT_DIM=input_dim)
        self.densities = densities
        self.levels = levels


class _functionsampler(MapBase):
    __extension_info__ = dict(
        parameters=dict(
            point_sampler=MapBase,
            function_map=MapBase
        ),
        generics=dict(),
        code = """
FORWARD
{
    float x_wx [FUNCTION_INPUT_DIM + 1]; // x: FUNCTION_INPUT_DIM, wx: 1
    forward(parameters.point_sampler, _input, x_wx);
    
    float function_in[FUNCTION_INPUT_DIM];
    for (int i=0; i<FUNCTION_INPUT_DIM; i++) function_in[i] = x_wx[i];
    float wx = x_wx[FUNCTION_INPUT_DIM];
    float function_out[FUNCTION_OUTPUT_DIM];
    forward(parameters.function_map, function_in, function_out);
    for (int i = 0; i<FUNCTION_OUTPUT_DIM; i++) _output[i] = function_out[i] * wx;
    for (int i = 0; i<FUNCTION_INPUT_DIM; i++) _output[i + FUNCTION_OUTPUT_DIM] = function_in[i];
}
        """,
    )
    def __init__(self, point_sampler: MapBase, function_map: MapBase):
        assert point_sampler.output_dim - 1 == function_map.input_dim
        super(_functionsampler, self).__init__(
            INPUT_DIM=point_sampler.input_dim,
            OUTPUT_DIM=point_sampler.output_dim - 1 + function_map.output_dim,
            FUNCTION_INPUT_DIM=function_map.input_dim,
            FUNCTION_OUTPUT_DIM=function_map.output_dim
        )
        self.point_sampler = point_sampler
        self.function_map = function_map


class GridRatiotrackingTransmittance(MapBase):
    __extension_info__ = dict(
        generics=dict(INPUT_DIM=6, OUTPUT_DIM=1),
        parameters=dict(
            grid=ParameterDescriptor,
            box_min=vec3,
            box_max=vec3,
            boundary=MapBase
        ),
        path=_internal.__INCLUDE_PATH__ + "/maps/transmittance_grt.h",
    )

    def __init__(self, grid: Grid3D, boundary: MapBase):
        super(GridRatiotrackingTransmittance, self).__init__()
        self.grid_model = grid
        self.grid = grid.grid
        self.box_min = grid.bmin.clone()
        self.box_max = grid.bmax.clone()
        self.boundary = boundary.cast(6, 2)

    def update_grid(self, grid: _typing.Optional[Grid3D] = None):
        if grid is None:
            assert self.grid_model is not None
            grid = self.grid_model
        else:
            self.grid_model = grid
        self.grid = grid.grid
        self.box_min = grid.bmin.clone()
        self.box_max = grid.bmax.clone()


class GridDeltatrackingTransmittance(MapBase):
    __extension_info__ = dict(
        generics=dict(INPUT_DIM=6, OUTPUT_DIM=1),
        parameters=dict(
            grid=ParameterDescriptor,
            box_min=vec3,
            box_max=vec3,
            boundary=MapBase
        ),
        path=_internal.__INCLUDE_PATH__ + "/maps/transmittance_gdt.h",
    )

    def __init__(self, grid: Grid3D, boundary: MapBase):
        super(GridDeltatrackingTransmittance, self).__init__()
        self.grid_model = grid
        self.grid = grid.grid
        self.box_min = grid.bmin.clone()
        self.box_max = grid.bmax.clone()
        self.boundary = boundary


class GridDDATransmittance(MapBase):
    __extension_info__ = dict(
        generics=dict(INPUT_DIM=6, OUTPUT_DIM=1),
        parameters=dict(
            grid=ParameterDescriptor,
            box_min=vec3,
            box_max=vec3,
            boundary=MapBase
        ),
        path=_internal.__INCLUDE_PATH__ + "/maps/transmittance_dda.h",
        bw_implementations=BACKWARD_IMPLEMENTATIONS.DEFAULT
    )

    def __init__(self, grid: Grid3D, boundary: MapBase):
        super(GridDDATransmittance, self).__init__()
        self.boundary = boundary
        self.grid_model = None
        self.update_grid(grid)

    def update_grid(self, grid: _typing.Optional[Grid3D] = None):
        if grid is None:
            assert self.grid_model is not None
            grid = self.grid_model
        else:
            self.grid_model = grid
        self.grid = grid.grid
        self.box_min = grid.bmin.clone()
        self.box_max = grid.bmax.clone()


class RatiotrackingTransmittance(MapBase):
    __extension_info__ = dict(
        path=_internal.__INCLUDE_PATH__ + "/maps/transmittance_rt.h",
        generics=dict(INPUT_DIM=6, OUTPUT_DIM=1),
        parameters=dict(sigma=MapBase, boundary=MapBase, majorant=MapBase),
        bw_implementations=BACKWARD_IMPLEMENTATIONS.DEFAULT
    )

    def __init__(self, sigma: MapBase, boundary: MapBase, majorant: MapBase):
        super(RatiotrackingTransmittance, self).__init__()
        self.sigma = sigma.cast(3, 1)
        self.boundary = boundary.cast(6, 2)
        self.majorant = majorant.cast(6, 2)


class DeltatrackingTransmittance(MapBase):
    __extension_info__ = dict(
        path=_internal.__INCLUDE_PATH__ + "/maps/transmittance_dt.h",
        generics=dict(INPUT_DIM=6, OUTPUT_DIM=1),
        parameters=dict(sigma=MapBase, boundary=MapBase, majorant=MapBase),
        bw_implementations=BACKWARD_IMPLEMENTATIONS.DEFAULT
    )

    def __init__(self, sigma: MapBase, boundary: MapBase, majorant: MapBase):
        super(DeltatrackingTransmittance, self).__init__()
        self.sigma = sigma.cast(3,1)
        self.boundary = boundary.cast(6, 2)
        self.majorant = majorant.cast(6, 2)


class RaymarchingTransmittance(MapBase):
    __extension_info__ = dict(
        path=_internal.__INCLUDE_PATH__ + "/maps/transmittance_rm.h",
        generics=dict(INPUT_DIM=6, OUTPUT_DIM=1),
        parameters=dict(sigma=MapBase, boundary=MapBase, step=float),
    )

    def __init__(self, sigma: MapBase, boundary: MapBase, step: float = 0.005):
        super(RaymarchingTransmittance, self).__init__()
        self.sigma = sigma.cast(3, 1)
        self.boundary = boundary.cast(6, 2)
        self.step = step


class DeltatrackingCollisionSampler(MapBase):
    __extension_info__ = dict(
        path=_internal.__INCLUDE_PATH__ + "/maps/collision_sampler_dt.h",
        generics=dict(INPUT_DIM=6, OUTPUT_DIM=3),
        parameters=dict(sigma=MapBase, boundary=MapBase, majorant=MapBase, ds_epsilon=float),
        bw_implementations=BACKWARD_IMPLEMENTATIONS.DEFAULT
    )

    def __init__(self, sigma: MapBase, boundary: MapBase, majorant: MapBase, ds_epsilon:float):
        super(DeltatrackingCollisionSampler, self).__init__()
        self.sigma = sigma
        self.boundary = boundary
        self.majorant = majorant
        self.ds_epsilon = ds_epsilon


class MCScatteredRadiance(MapBase):
    __extension_info__ = dict(
        path=_internal.__INCLUDE_PATH__ + "/maps/scattered_radiance_mc.h",
        generics=dict(INPUT_DIM=6, OUTPUT_DIM=3),
        parameters=dict(scattering_albedo=MapBase, phase_sampler=MapBase, radiance=MapBase),
        bw_implementations=BACKWARD_IMPLEMENTATIONS.DEFAULT
    )

    def __init__(self, scattering_albedo: MapBase, phase_sampler: MapBase, radiance: MapBase):
        super(MCScatteredRadiance, self).__init__()
        self.scattering_albedo = scattering_albedo
        self.phase_sampler = phase_sampler
        self.radiance = radiance


class MCScatteredEmittedRadiance(MapBase):
    __extension_info__ = dict(
        path=_internal.__INCLUDE_PATH__ + "/maps/scattered_emitted_radiance_mc.h",
        generics=dict(INPUT_DIM=6, OUTPUT_DIM=3),
        parameters=dict(scattering_albedo=MapBase, emission=MapBase, phase_sampler=MapBase, radiance=MapBase),
        bw_implementations=BACKWARD_IMPLEMENTATIONS.DEFAULT
    )

    def __init__(self, scattering_albedo: MapBase, emission: MapBase, phase_sampler: MapBase, radiance: MapBase):
        super(MCScatteredEmittedRadiance, self).__init__()
        self.scattering_albedo = scattering_albedo
        self.emission = emission
        self.phase_sampler = phase_sampler
        self.radiance = radiance


class SingleScatteredRadiance(MapBase):
    __extension_info__ = dict(
        path=_internal.__INCLUDE_PATH__ + "/maps/single_scattered_radiance_mc.h",
        generics=dict(INPUT_DIM=6, OUTPUT_DIM=3),
        parameters=dict(
            scattering_albedo=MapBase,
            phase=MapBase,
            environment_sampler=MapBase,
            transmittance=MapBase
        ),
        bw_implementations=BACKWARD_IMPLEMENTATIONS.DEFAULT
    )

    def __init__(self,
                 scattering_albedo: MapBase,
                 phase: MapBase,
                 environment_sampler: MapBase,
                 transmittance: MapBase
                 ):
        super(SingleScatteredRadiance, self).__init__()
        self.scattering_albedo = scattering_albedo.cast(3, 3)
        self.phase = phase.cast(9, 1)
        self.environment_sampler = environment_sampler.cast(6, 6)
        self.transmittance = transmittance.cast(6, 1)



class MCCollisionIntegrator(MapBase):
    '''
    Montecarlo solver for
    :math:`\int^{x_{d}}_{x_{0}} T(x, x_{t})\sigma(x_{t}) R(x_{t} , -w) dt + T(x, x_d)B(w)`
    using \n
    collision_sampler: :math:`<T(x, x_{t})\sigma(x_{t})>/p(t), t, <T(x, x_d)>` \n
    exitance_radiance: :math:`R(x, w)` \n
    environment: :math:`B(w)`
    '''
    __extension_info__ = dict(
        path=_internal.__INCLUDE_PATH__ + '/maps/collision_integrator_mc.h',
        generics=dict(INPUT_DIM=6, OUTPUT_DIM=3),
        parameters=dict(collision_sampler=MapBase, exitance_radiance=MapBase, environment=MapBase),
        bw_implementations=BACKWARD_IMPLEMENTATIONS.DEFAULT
    )

    def __init__(self, collision_sampler: MapBase, exitance_radiance: MapBase, environment: MapBase):
        super(MCCollisionIntegrator, self).__init__()
        self.collision_sampler = collision_sampler
        self.exitance_radiance = exitance_radiance
        self.environment = environment


class GridDDACollisionIntegrator(MapBase):
    '''
    DDA solver for
    :math:`\int^{x_{d}}_{x_{0}} T(x, x_{t})\sigma(x_{t}) R(x_{t} , -w) dt + T(x, x_d)B(w)`
    using \n
    collision_sampler: :math:`<T(x, x_{t})\sigma(x_{t})>/p(t), t, <T(x, x_d)>` \n
    exitance_radiance: :math:`R(x, w)` \n
    environment: :math:`B(w)`
    '''
    __extension_info__ = dict(
        path=_internal.__INCLUDE_PATH__ + '/maps/collision_integrator_dda.h',
        generics=dict(INPUT_DIM=6, OUTPUT_DIM=3),
        parameters=dict(
            grid=ParameterDescriptor,
            exitance_radiance=MapBase,
            environment=MapBase,
            boundary=MapBase,
            box_min=vec3,
            box_max=vec3
        ),
        bw_implementations=BACKWARD_IMPLEMENTATIONS.DEFAULT
    )

    def __init__(self, sigma_grid: Grid3D, exitance_radiance: MapBase, environment: MapBase, boundary: MapBase):
        super(GridDDACollisionIntegrator, self).__init__()
        self.sigma_grid = sigma_grid.cast(3, 1)
        self.grid = sigma_grid.grid
        self.box_min = sigma_grid.bmin.clone()
        self.box_max = sigma_grid.bmax.clone()
        self.exitance_radiance = exitance_radiance.cast(6, 3)
        self.environment = environment.cast(3, 3)
        self.boundary = boundary.cast(6, 2)

    def update_grid(self, grid: _typing.Optional[Grid3D] = None):
        if grid is None:
            assert self.sigma_grid is not None
            grid = self.sigma_grid
        self.sigma_grid = grid
        self.grid = grid.grid
        self.box_min = grid.bmin.clone()
        self.box_max = grid.bmax.clone()


class DeltatrackingScatteringSampler(MapBase):
    __extension_info__ = dict(
        path=_internal.__INCLUDE_PATH__ + '/maps/path_sampler_DT.h',
        generics=dict(INPUT_DIM=6, OUTPUT_DIM=3),  # x,w -> Wo * Env(wo)
        parameters=dict(
            sigma=MapBase,
            scattering_albedo=MapBase,
            phase_sampler=MapBase,
            environment=MapBase,
            boundary=MapBase,
            majorant=MapBase
        ),
        bw_implementations=BACKWARD_IMPLEMENTATIONS.DEFAULT
    )

    def __init__(self,
                 sigma: MapBase,
                 scattering_albedo: MapBase,
                 phase_sampler: MapBase,
                 environment: MapBase,
                 boundary: MapBase,
                 majorant: MapBase
                 ):
        super().__init__()
        self.sigma = sigma.cast(3, 1)
        self.scattering_albedo = scattering_albedo.cast(3, 3)
        self.phase_sampler = phase_sampler.cast(6, 4)
        self.environment = environment.cast(3, 3)
        self.boundary = boundary.cast(6, 2)
        self.majorant = majorant.cast(6, 2)


class DeltatrackingPathIntegrator(MapBase):
    __extension_info__ = dict(
        path=_internal.__INCLUDE_PATH__ + '/maps/radiance_DT.h',
        generics=dict(INPUT_DIM=6, OUTPUT_DIM=3),
        parameters=dict(
            sigma=MapBase,
            scattering_albedo=MapBase,
            emission=MapBase,
            environment=MapBase,
            phase_sampler=MapBase,
            boundary=MapBase,
            majorant=MapBase,
            ds_epsilon=float
        ),
        bw_implementations=BACKWARD_IMPLEMENTATIONS.DEFAULT
    )

    def __init__(self,
                 sigma: MapBase,
                 scattering_albedo: MapBase,
                 emission: MapBase,
                 environment: MapBase,
                 phase_sampler: MapBase,
                 boundary: MapBase,
                 majorant: MapBase,
                 ds_epsilon: float
                 ):
        super(DeltatrackingPathIntegrator, self).__init__()
        self.sigma = sigma
        self.scattering_albedo = scattering_albedo
        self.emission = emission
        self.environment = environment
        self.phase_sampler = phase_sampler
        self.boundary = boundary
        self.majorant = majorant
        self.ds_epsilon = ds_epsilon


class DeltatrackingNEEPathIntegrator(MapBase):
    __extension_info__ = dict(
        path=_internal.__INCLUDE_PATH__ + '/maps/radiance_NEE_DS.h',
        generics=dict(INPUT_DIM=6, OUTPUT_DIM=3),
        parameters=dict(
            sigma=MapBase,
            scattering_albedo=MapBase,
            emission=MapBase,
            environment=MapBase,
            environment_sampler=MapBase,
            phase=MapBase,
            phase_sampler=MapBase,
            boundary=MapBase,
            majorant=MapBase,
            transmittance=MapBase,
            ds_epsilon=float
        ),
        bw_implementations=BACKWARD_IMPLEMENTATIONS.DEFAULT
    )

    def __init__(self,
                 sigma: MapBase,
                 scattering_albedo: MapBase,
                 emission: MapBase,
                 environment: MapBase,
                 environment_sampler: MapBase,
                 phase: MapBase,
                 phase_sampler: MapBase,
                 boundary: MapBase,
                 majorant: MapBase,
                 transmittance: MapBase,
                 ds_epsilon: float
                 ):
        sigma = sigma.cast(3, 1)
        scattering_albedo = scattering_albedo.cast(3, 3)
        emission = emission.cast(6, 3)
        environment = environment.cast(3, 3)
        environment_sampler = environment_sampler.cast(6, 6)
        phase = phase.cast(9, 1)
        phase_sampler = phase_sampler.cast(6, 4)
        boundary = boundary.cast(6, 2)
        majorant = majorant.cast(6, 2)
        transmittance = transmittance.cast(6, 1)
        super(DeltatrackingNEEPathIntegrator, self).__init__()
        self.sigma = sigma
        self.scattering_albedo = scattering_albedo
        self.emission = emission
        self.environment = environment
        self.environment_sampler = environment_sampler
        self.phase = phase
        self.phase_sampler = phase_sampler
        self.boundary = boundary
        self.majorant = majorant
        self.transmittance = transmittance
        self.ds_epsilon = ds_epsilon


class DRTPathIntegrator(MapBase):
    __extension_info__ = dict(
        path=_internal.__INCLUDE_PATH__ + '/maps/radiance_NEE_DRT.h',
        generics=dict(INPUT_DIM=6, OUTPUT_DIM=3),
        parameters=dict(
            sigma=MapBase,
            scattering_albedo=MapBase,
            emission=MapBase,
            environment=MapBase,
            environment_sampler=MapBase,
            phase=MapBase,
            phase_sampler=MapBase,
            boundary=MapBase,
            majorant=MapBase,
            transmittance=MapBase
        ),
        bw_implementations=BACKWARD_IMPLEMENTATIONS.DEFAULT
    )

    def __init__(self,
                 sigma: MapBase,
                 scattering_albedo: MapBase,
                 emission: MapBase,
                 environment: MapBase,
                 environment_sampler: MapBase,
                 phase: MapBase,
                 phase_sampler: MapBase,
                 boundary: MapBase,
                 majorant: MapBase,
                 transmittance: MapBase
                 ):
        super(DRTPathIntegrator, self).__init__()
        self.sigma = sigma.cast(input_dim=3, output_dim=1)
        self.scattering_albedo = scattering_albedo.cast(input_dim=3, output_dim=3)
        self.emission = emission.cast(input_dim=6, output_dim=3)
        self.environment = environment.cast(input_dim=3, output_dim=3)
        self.environment_sampler = environment_sampler.cast(input_dim=6, output_dim=6)
        self.phase = phase.cast(9, 1)
        self.phase_sampler = phase_sampler.cast(6, 4)
        self.boundary = boundary.cast(6, 2)
        self.majorant = majorant.cast(6, 2)
        self.transmittance = transmittance.cast(6, 1)


class DRTQPathIntegrator(MapBase):
    __extension_info__ = dict(
        path=_internal.__INCLUDE_PATH__ + '/maps/radiance_NEE_DRTQ.h',
        generics=dict(INPUT_DIM=6, OUTPUT_DIM=3),
        parameters=dict(
            sigma=MapBase,
            scattering_albedo=MapBase,
            emission=MapBase,
            environment=MapBase,
            environment_sampler=MapBase,
            phase=MapBase,
            phase_sampler=MapBase,
            boundary=MapBase,
            majorant=MapBase,
            transmittance=MapBase
        ),
        bw_implementations=BACKWARD_IMPLEMENTATIONS.DEFAULT
    )

    def __init__(self,
                 sigma: MapBase,
                 scattering_albedo: MapBase,
                 emission: MapBase,
                 environment: MapBase,
                 environment_sampler: MapBase,
                 phase: MapBase,
                 phase_sampler: MapBase,
                 boundary: MapBase,
                 majorant: MapBase,
                 transmittance: MapBase
                 ):
        super(DRTQPathIntegrator, self).__init__()
        self.sigma = sigma
        self.scattering_albedo = scattering_albedo
        self.emission = emission
        self.environment = environment
        self.environment_sampler = environment_sampler
        self.phase = phase
        self.phase_sampler = phase_sampler
        self.boundary = boundary
        self.majorant = majorant
        self.transmittance = transmittance



class DRTDSPathIntegrator(MapBase):
    __extension_info__ = dict(
        path=_internal.__INCLUDE_PATH__ + '/maps/radiance_NEE_DRTDS.h',
        generics=dict(INPUT_DIM=6, OUTPUT_DIM=3),
        parameters=dict(
            sigma=MapBase,
            scattering_albedo=MapBase,
            emission=MapBase,
            environment=MapBase,
            environment_sampler=MapBase,
            phase=MapBase,
            phase_sampler=MapBase,
            boundary=MapBase,
            majorant=MapBase,
            transmittance=MapBase,
            ds_epsilon=float
        ),
        bw_implementations=BACKWARD_IMPLEMENTATIONS.DEFAULT
    )

    def __init__(self,
                 sigma: MapBase,
                 scattering_albedo: MapBase,
                 emission: MapBase,
                 environment: MapBase,
                 environment_sampler: MapBase,
                 phase: MapBase,
                 phase_sampler: MapBase,
                 boundary: MapBase,
                 majorant: MapBase,
                 transmittance: MapBase,
                 ds_epsilon: float
                 ):
        super(DRTDSPathIntegrator, self).__init__()
        self.sigma = sigma
        self.scattering_albedo = scattering_albedo
        self.emission = emission
        self.environment = environment
        self.environment_sampler = environment_sampler
        self.phase = phase
        self.phase_sampler = phase_sampler
        self.boundary = boundary
        self.majorant = majorant
        self.transmittance = transmittance
        self.ds_epsilon = ds_epsilon



class SPSPathIntegrator(MapBase):
    __extension_info__ = dict(
        path=_internal.__INCLUDE_PATH__ + '/maps/radiance_NEE_SPS.h',
        generics=dict(INPUT_DIM=6, OUTPUT_DIM=3),
        parameters=dict(
            sigma=MapBase,
            scattering_albedo=MapBase,
            emission=MapBase,
            environment=MapBase,
            environment_sampler=MapBase,
            phase=MapBase,
            phase_sampler=MapBase,
            boundary=MapBase,
            majorant=MapBase,
            transmittance=MapBase
        ),
        bw_implementations=BACKWARD_IMPLEMENTATIONS.DEFAULT
    )

    def __init__(self,
                 sigma: MapBase,
                 scattering_albedo: MapBase,
                 emission: MapBase,
                 environment: MapBase,
                 environment_sampler: MapBase,
                 phase: MapBase,
                 phase_sampler: MapBase,
                 boundary: MapBase,
                 majorant: MapBase,
                 transmittance: MapBase
                 ):
        super(SPSPathIntegrator, self).__init__()
        self.sigma = sigma.cast(input_dim=3, output_dim=1)
        self.scattering_albedo = scattering_albedo.cast(input_dim=3, output_dim=3)
        self.emission = emission.cast(input_dim=6, output_dim=3)
        self.environment = environment.cast(input_dim=3, output_dim=3)
        self.environment_sampler = environment_sampler.cast(input_dim=6, output_dim=6)
        self.phase = phase.cast(9, 1)
        self.phase_sampler = phase_sampler.cast(6, 4)
        self.boundary = boundary.cast(6, 2)
        self.majorant = majorant.cast(6, 2)
        self.transmittance = transmittance.cast(6, 1)

# --------------------


class RayDirection(MapBase):
    __extension_info__ = dict(
        parameters=dict(
        ),
        generics=dict(
            INPUT_DIM=6,
            OUTPUT_DIM=3
        ),
        code = """
FORWARD
{
    for (int i=0; i<3; i++)
        _output[i] = _input[3 + i];
}
""",
    )

    def __init__(self):
        super(RayDirection, self).__init__()

    def forward_torch(self, *args):
        xw, = args
        return xw[...,3:6]


class RayPosition(MapBase):
    __extension_info__ = dict(
        parameters=dict(
        ),
        generics=dict(
            INPUT_DIM=6,
            OUTPUT_DIM=3
        ),
        code = """
FORWARD
{
    for (int i=0; i<3; i++)
        _output[i] = _input[i];
}
""",
    )

    def __init__(self):
        super(RayPosition, self).__init__()

    def forward_torch(self, *args):
        xw, = args
        return xw[...,0:3]


class RayToSegment(MapBase):
    __extension_info__ = dict(
        parameters=dict(
            distance_field=MapBase
        ),
        generics=dict(
            INPUT_DIM=6,
            OUTPUT_DIM=6
        ),
        code="""
    FORWARD
    {
        for (int i=0; i<3; i++)
            _output[i] = _input[i]; // x0 = x
        float d[1];
        forward(parameters.distance_field, _input, d);
        for (int i=0; i<3; i++)
            _output[i+3] = _input[i+3]*d[0] + _input[i]; // x1 = x + w*d
    }
    """,
    )

    def __init__(self, distance_field: 'MapBase'):
        distance_field = distance_field.cast(input_dim=6, output_dim=1)
        super(RayToSegment, self).__init__()
        self.distance_field = distance_field

    def forward_torch(self, *args):
        xw, = args
        x = xw[..., 0:3]
        w = xw[..., 3:6]
        d = self.distance_field(xw)
        return _torch.cat([x, x + w*d], dim=-1)


class LineIntegrator(MapBase):

    __extension_info__ = dict(
        parameters=dict(
            map=MapBase,
            step=float
        ),
        code = """
FORWARD
{
    float x0[INPUT_DIM/2];
    for (int i = 0; i < INPUT_DIM/2; i++)
        x0[i] = _input[i];
    float dx[INPUT_DIM/2];
    float d = 0.0;
    for (int i = 0; i < INPUT_DIM/2; i++)
    {
        dx[i] = _input[INPUT_DIM/2 + i] - x0[i];
        d += dx[i] * dx[i];
    }
    d = sqrt(d);
    for (int i=0; i<OUTPUT_DIM; i++)
        _output[i] = 0.0;
    int samples = int(d / (parameters.step + 0.00000001)) + 1;
    for (int s = 0; s < samples; s++)
    {
        float xt[INPUT_DIM / 2];
        float alpha = random();
        for (int i=0; i<INPUT_DIM/2; i++)
            xt[i] = dx[i] * alpha + x0[i];
        float o[OUTPUT_DIM];
        forward(parameters.map, xt, o);
        for (int i=0; i<OUTPUT_DIM; i++)
            _output[i] += o[i];
    }
    for (int i=0; i<OUTPUT_DIM; i++)
        _output[i] /= samples;
}
        """,
    )

    def __init__(self, map: 'MapBase', step: float = 0.005):
        super(LineIntegrator, self).__init__(INPUT_DIM=map.input_dim * 2, OUTPUT_DIM=map.output_dim)
        self.map = map
        self.step = step


class TransmittanceDT(MapBase):
    __extension_info__ = dict(
        parameters=dict(
            sigma=MapBase,
            majorant=float
        ),
        generics=dict(INPUT_DIM=6, OUTPUT_DIM=1),
        code="""
    FORWARD
    {
        vec3 x0 = vec3(_input[0], _input[1], _input[2]);
        vec3 x1 = vec3(_input[3], _input[4], _input[5]);
        vec3 dx = x1 - x0;
        float d = length(dx);
        while(true)
        {
            float t = -log(1 - random()) / parameters.majorant;
            if (t >= d)
            {
                _output[0] = 1.0;
                return;
            }
            float sigma_value[1];
            x0 += dx * t;
            forward(parameters.sigma, float[3]( x0.x, x0.y, x0.z ), sigma_value);
            if (random() < sigma_value[0] / parameters.majorant)
                break;
            d -= t; 
        }
        _output[0] = 0.0;
    }
            """,
    )

    def __init__(self, sigma: 'MapBase', majorant: float):
        super(TransmittanceDT, self).__init__()
        self.sigma = sigma
        self.majorant = majorant


class TransmittanceRT(MapBase):
    __extension_info__ = dict(
        parameters=dict(
            sigma=MapBase,
            majorant=float
        ),
        generics=dict(INPUT_DIM=6, OUTPUT_DIM=1),
        code="""
    FORWARD
    {
        vec3 x0 = vec3(_input[0], _input[1], _input[2]);
        vec3 x1 = vec3(_input[3], _input[4], _input[5]);
        vec3 dx = x1 - x0;
        float d = length(dx);
        float T = 1.0;
        while(true)
        {
            float t = -log(1 - random()) / parameters.majorant;
            if (t >= d)
                break;
            float sigma_value[1];
            x0 += dx * t;
            forward(parameters.sigma, float[3]( x0.x, x0.y, x0.z ), sigma_value);
            T *= (1 - sigma_value[0] / parameters.majorant);
            d -= t; 
        }
        _output[0] = T;
    }
            """,
    )

    def __init__(self, sigma: 'MapBase', majorant: float):
        super(TransmittanceRT, self).__init__()
        self.sigma = sigma
        self.majorant = majorant


class TotalVariation(MapBase):
    __extension_info__ = dict(
        parameters=dict(
            map=MapBase,
            expected_dx=float
        ),
        bw_implementations=BACKWARD_IMPLEMENTATIONS.DEFAULT,
        code="""
    // Total Variation - input: x, output: (map(x + dx * random_w) - map(x))/dx
    FORWARD
    {
        float w[INPUT_DIM];
        float dx = 0;
        for (int i=0; i<INPUT_DIM; i++)
        {
            w[i] = gauss() * parameters.expected_dx;
            dx += w[i] * w[i];
            w[i] = w[i] + _input[i];
        }
        dx = sqrt(dx);

        dx = max (dx, 0.00000001);
        
        forward(parameters.map, _input, _output); // eval map at current pos
        float adj_output[OUTPUT_DIM];
        forward(parameters.map, w, adj_output);
        for (int i = 0; i < OUTPUT_DIM; i++)
            _output[i] = abs(adj_output[i] - _output[i]) / dx;
    }
    
    BACKWARD
    {
        float w[INPUT_DIM];
        float dx = 0;
        for (int i=0; i<INPUT_DIM; i++)
        {
            w[i] = gauss() * parameters.expected_dx;
            dx += w[i] * w[i];
            w[i] = w[i] + _input[i];
        }
        dx = sqrt(dx);

        dx = max (dx, 0.00000001);
        
        float _output[OUTPUT_DIM];
        forward(parameters.map, _input, _output); // eval map at current pos
        float adj_output[OUTPUT_DIM];
        forward(parameters.map, w, adj_output);

        float tmp_output_grad[OUTPUT_DIM];
        for (int i = 0; i < OUTPUT_DIM; i++)
            tmp_output_grad[i] = sign(adj_output[i] - _output[i]) * _output_grad[i] / dx;
        backward(parameters.map, w, tmp_output_grad, _input_grad);
        for (int i = 0; i < OUTPUT_DIM; i++)
            tmp_output_grad[i] = -tmp_output_grad[i];
        backward(parameters.map, _input, tmp_output_grad, _input_grad);
    }
            """
    )

    def __init__(self, map: 'MapBase', expected_dx: float = 0.005):
        super(TotalVariation, self).__init__(INPUT_DIM=map.input_dim, OUTPUT_DIM=map.output_dim)
        self.map = map
        self.expected_dx = expected_dx


class TransmittanceFromTau(MapBase):
    __extension_info__ = dict(
        parameters=dict(
            tau=MapBase
        ),
        generics=dict(INPUT_DIM=6, OUTPUT_DIM=1),
        bw_implementations=BACKWARD_IMPLEMENTATIONS.DEFAULT,
        code="""
    FORWARD
    {
        forward(parameters.tau, _input, _output);
        _output[0] = exp(-_output[0]);
    }
    
    BACKWARD
    {
        float dL_dT = _output_grad[0];
        float tau[1];
        forward(parameters.tau, _input, tau);
        float dL_dtau = - dL_dT * exp(-tau[0]);
        // dT/dtau = -exp(-tau) * dtau
        backward(parameters.tau, _input, float[1](dL_dtau), _input_grad);
    }
            """
    )

    def __init__(self, tau: 'MapBase'):
        super(TransmittanceFromTau, self).__init__()
        self.tau = tau


class Grid3DLineIntegral(MapBase):
    __extension_info__ = dict(
        parameters=dict(
            grid=ParameterDescriptor,
            box_min=vec3,
            box_max=vec3
        ),
        generics=dict(INPUT_DIM=6),
        bw_implementations=BACKWARD_IMPLEMENTATIONS.DEFAULT,
        code="""

void load_tensor_at(map_object, ivec3 cell, out float[OUTPUT_DIM] values)
{
    ivec3 dim = ivec3(parameters.grid.shape[2] - 1, parameters.grid.shape[1] - 1, parameters.grid.shape[0] - 1);
    cell = clamp(cell, ivec3(0), dim - ivec3(1));

    float_ptr buf = param_buffer(parameters.grid, cell);
    for (int i=0; i<OUTPUT_DIM; i++)
        values[i] = buf.data[i]; 
}

void add_grad_tensor_at(map_object, ivec3 cell, float[OUTPUT_DIM] values)
{
    ivec3 dim = ivec3(parameters.grid.shape[2] - 1, parameters.grid.shape[1] - 1, parameters.grid.shape[0] - 1);
    cell = clamp(cell, ivec3(0), dim - ivec3(1));

    //if (parameters.grid.grad_data == 0)
    //return;
    
    float_ptr buf = param_grad_buffer(parameters.grid, cell);
    for (int i=0; i<OUTPUT_DIM; i++)
        atomicAdd_f(buf, i, values[i]);
}


void load_cell(map_object, ivec3 cell, out float[2][2][2][OUTPUT_DIM] values)
{
    for (int dz = 0; dz < 2; dz ++)
        for (int dy = 0; dy < 2; dy ++)
            for (int dx = 0; dx < 2; dx ++) 
                load_tensor_at(object, cell + ivec3(dx, dy, dz), values[dz][dy][dx]);
}

void add_grad_cell(map_object, ivec3 cell, float[2][2][2][OUTPUT_DIM] values)
{
    for (int dz = 0; dz < 2; dz ++)
        for (int dy = 0; dy < 2; dy ++)
            for (int dx = 0; dx < 2; dx ++) 
                add_grad_tensor_at(object, cell + ivec3(dx, dy, dz), values[dz][dy][dx]);
}

void get_alphas(map_object, vec3 nx0, vec3 nx1, out float[2][2][2] alphas)
{
    alphas[0][0][0] = 0.125;
    alphas[0][0][1] = 0.125;
    alphas[0][1][0] = 0.125;
    alphas[0][1][1] = 0.125;
    alphas[1][0][0] = 0.125;
    alphas[1][0][1] = 0.125;
    alphas[1][1][0] = 0.125;
    alphas[1][1][1] = 0.125;
}

void add_cell_integral(map_object, float[2][2][2][OUTPUT_DIM] cell, vec3 nx0, vec3 nx1, float dt, inout float _output[OUTPUT_DIM])
{
    float alphas[2][2][2];
    get_alphas(object, nx0, nx1, alphas);

    for (int i=0; i<OUTPUT_DIM; i++)
        _output[i] += dt * (
            cell[0][0][0][i] * alphas[0][0][0] + 
            cell[0][0][1][i] * alphas[0][0][1] + 
            cell[0][1][0][i] * alphas[0][1][0] + 
            cell[0][1][1][i] * alphas[0][1][1] + 
            cell[1][0][0][i] * alphas[1][0][0] + 
            cell[1][0][1][i] * alphas[1][0][1] + 
            cell[1][1][0][i] * alphas[1][1][0] + 
            cell[1][1][1][i] * alphas[1][1][1]); 
}

void add_cell_dL_dI(map_object, ivec3 cell, float[OUTPUT_DIM] dL_dI, vec3 nx0, vec3 nx1, float dt)
{
    float alphas [2][2][2];
    get_alphas(object, nx0, nx1, alphas);

    float grads_cell[2][2][2][OUTPUT_DIM];

    for (int i=0; i<OUTPUT_DIM; i++)
    {
        float dtdLdI = dt * dL_dI[i];
        grads_cell[0][0][0][i] = dtdLdI * alphas[0][0][0];
        grads_cell[0][0][1][i] = dtdLdI * alphas[0][0][1];
        grads_cell[0][1][0][i] = dtdLdI * alphas[0][1][0];
        grads_cell[0][1][1][i] = dtdLdI * alphas[0][1][1];
        grads_cell[1][0][0][i] = dtdLdI * alphas[1][0][0];
        grads_cell[1][0][1][i] = dtdLdI * alphas[1][0][1];
        grads_cell[1][1][0][i] = dtdLdI * alphas[1][1][0];
        grads_cell[1][1][1][i] = dtdLdI * alphas[1][1][1];
    }
    
    add_grad_cell(object, cell, grads_cell);
}

FORWARD
{
    for (int i=0; i < OUTPUT_DIM; i++)
        _output[i] = 0.0;
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 x_end = vec3(_input[3], _input[4], _input[5]);
    vec3 dx = x_end - x;
    float d = length(dx);
    if (d == 0.0)
        return; // 0 integral value if domain is empty
    vec3 w = dx / d;
    float tMin, tMax;
    if (!intersect_ray_box(x, w, parameters.box_min, parameters.box_max, tMin, tMax))
        return; // 0 integral value outside bounding box
    x += w * tMin;  
    d = tMax - tMin;
    
    vec3 box_size = parameters.box_max - parameters.box_min;
    ivec3 dim = ivec3(parameters.grid.shape[2] - 1, parameters.grid.shape[1] - 1, parameters.grid.shape[0] - 1);
    vec3 cell_size = box_size / dim;
    ivec3 cell = ivec3((x - parameters.box_min) * dim / box_size);
    cell = clamp(cell, ivec3(0), dim - ivec3(1));
    vec3 alpha_inc = cell_size / max(vec3(0.00001), abs(w));
    ivec3 side = ivec3(sign(w));
    vec3 corner = (cell + side * 0.5 + vec3(0.5)) * cell_size + parameters.box_min;
    vec3 alpha = abs(corner - x) / max(vec3(0.00001), abs(w));
    float current_t = 0;
    float[2][2][2][OUTPUT_DIM] cell_values;
    
    while(current_t < d - 0.00001){
        float next_t = min(d, min(alpha.x, min(alpha.y, alpha.z)));
        
        load_cell(object, cell, cell_values);
        add_cell_integral(object, cell_values, vec3(0.0), vec3(1.0), next_t - current_t, _output); 
        
        ivec3 cell_inc = ivec3(
            alpha.x <= alpha.y && alpha.x <= alpha.z,
            alpha.x > alpha.y && alpha.y <= alpha.z,
            alpha.x > alpha.z && alpha.y > alpha.z);

        current_t = next_t;
        alpha += cell_inc * alpha_inc;
        cell += cell_inc * side;
    }
}

BACKWARD
{
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 x_end = vec3(_input[3], _input[4], _input[5]);

    // TODO: Add dL_dx0 and dL_dx1 here... boundary terms
    
    if (parameters.grid.grad_data == 0)
        return; // No grad
        
    vec3 dx = x_end - x;
    float d = length(dx);
    if (d == 0.0)
        return; // 0 integral value if domain is empty

    vec3 w = dx / d;
    float tMin, tMax;
    if (!intersect_ray_box(x, w, parameters.box_min, parameters.box_max, tMin, tMax))
        return; // 0 integral value outside bounding box
    x += w * tMin;  
    d = tMax - tMin;
    
    vec3 box_size = parameters.box_max - parameters.box_min;
    ivec3 dim = ivec3(parameters.grid.shape[2] - 1, parameters.grid.shape[1] - 1, parameters.grid.shape[0] - 1);
    vec3 cell_size = box_size / dim;
    ivec3 cell = ivec3((x - parameters.box_min) * dim / box_size);
    cell = clamp(cell, ivec3(0), dim - ivec3(1));
    vec3 alpha_inc = cell_size / max(vec3(0.00001), abs(w));
    ivec3 side = ivec3(sign(w));
    vec3 corner = (cell + side * 0.5 + vec3(0.5)) * cell_size + parameters.box_min;
    vec3 alpha = abs(corner - x) / max(vec3(0.00001), abs(w));
    float current_t = 0;
    
    while(current_t < d - 0.00001){
        float next_t = min(d, min(alpha.x, min(alpha.y, alpha.z)));
        
        add_cell_dL_dI(object, cell, _output_grad, vec3(0.0), vec3(1.0), next_t - current_t); 
        
        ivec3 cell_inc = ivec3(
            alpha.x <= alpha.y && alpha.x <= alpha.z,
            alpha.x > alpha.y && alpha.y <= alpha.z,
            alpha.x > alpha.z && alpha.y > alpha.z);

        current_t = next_t;
        alpha += cell_inc * alpha_inc;
        cell += cell_inc * side;
    }
}

                """
    )

    def __init__(self, grid_model: Grid3D):
        super(Grid3DLineIntegral, self).__init__(OUTPUT_DIM=grid_model.output_dim)
        self.grid_model = grid_model
        self.grid = grid_model.grid
        self.box_min = grid_model.bmin.clone()
        self.box_max = grid_model.bmax.clone()


class Grid3DTransmittanceDDARayIntegral(MapBase):
    __extension_info__ = dict(
        parameters=dict(
            grid=ParameterDescriptor,
            out_radiance=MapBase,
            boundary_radiance=MapBase,
            box_min=vec3,
            box_max=vec3
        ),
        generics=dict(INPUT_DIM=6),
        code="""

float sigma_at(map_object, ivec3 cell)
{
    ivec3 dim = ivec3(parameters.grid.shape[2] - 1, parameters.grid.shape[1] - 1, parameters.grid.shape[0] - 1);
    cell = clamp(cell, ivec3(0), dim - ivec3(1));

    float_ptr buf = param_buffer(parameters.grid, cell);
    return buf.data[0];
}

void load_cell(map_object, ivec3 cell, out float[2][2][2] sigmas)
{
    for (int dz = 0; dz < 2; dz ++)
        for (int dy = 0; dy < 2; dy ++)
            for (int dx = 0; dx < 2; dx ++) 
                sigmas[dz][dy][dx] = sigma_at(object, cell + ivec3(dx, dy, dz));
}

float interpolated_sigma(map_object, vec3 alpha, float[2][2][2] sigmas)
{
    return mix(mix(
        mix(sigmas[0][0][0], sigmas[0][0][1], alpha.x),
        mix(sigmas[0][1][0], sigmas[0][1][1], alpha.x), alpha.y),
        mix(
        mix(sigmas[1][0][0], sigmas[1][0][1], alpha.x),
        mix(sigmas[1][1][0], sigmas[1][1][1], alpha.x), alpha.y), alpha.z);
}

FORWARD
{
    for (int i=0; i < OUTPUT_DIM; i++)
        _output[i] = 0.0;
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);

    float radiance_values[OUTPUT_DIM];
    float T = 1.0; // full transmittance start

    float tMin, tMax;
    if (intersect_ray_box(x, w, parameters.box_min, parameters.box_max, tMin, tMax))
    {
        x += w * tMin;  
        float d = tMax - tMin;

        vec3 box_size = parameters.box_max - parameters.box_min;
        ivec3 dim = ivec3(parameters.grid.shape[2] - 1, parameters.grid.shape[1] - 1, parameters.grid.shape[0] - 1);
        vec3 cell_size = box_size / dim;
        ivec3 cell = ivec3((x - parameters.box_min) * dim / box_size);
        cell = clamp(cell, ivec3(0), dim - ivec3(1));
        vec3 alpha_inc = cell_size / max(vec3(0.00001), abs(w));
        ivec3 side = ivec3(sign(w));
        vec3 corner = (cell + side * 0.5 + vec3(0.5)) * cell_size + parameters.box_min;
        vec3 alpha = abs(corner - x) / max(vec3(0.00001), abs(w));
        float current_t = 0;
        vec3 vn = (x - parameters.box_min) * dim / box_size;
        vec3 vm = w * dim / box_size;

        float[2][2][2] sigma_values;

        while(current_t < d - 0.0001){
            float next_t = min(d, min(alpha.x, min(alpha.y, alpha.z)));

            load_cell(object, cell, sigma_values);
            float cell_t = mix(current_t, next_t, 0.5);
            // ** Accumulate interaction
            // * sample sigma
            vec3 interpolation_alpha = fract(vm * cell_t + vn);
            float sigma_value = interpolated_sigma(object, interpolation_alpha, sigma_values);
            float emission_integral = 1 - exp(-sigma_value * (next_t - current_t));
            // if (emission_integral > 0.9) emission_integral = 1.0; else emission_integral = 0.0;
            if (emission_integral > 0.0001){
                vec3 xc = cell_t*w + x;
                forward(parameters.out_radiance, float[6](xc.x, xc.y, xc.z, w.x, w.y, w.z), radiance_values);
                for (int i=0; i<OUTPUT_DIM; i++)
                    _output[i] += T * emission_integral * radiance_values[i];
            }
            T *= (1 - emission_integral);

            if (T < 0.001) break;

            ivec3 cell_inc = ivec3(
                alpha.x <= alpha.y && alpha.x <= alpha.z,
                alpha.x > alpha.y && alpha.y <= alpha.z,
                alpha.x > alpha.z && alpha.y > alpha.z);

            current_t = next_t;
            alpha += cell_inc * alpha_inc;
            cell += cell_inc * side;
        }
        x += w * d;
    }
    forward(parameters.boundary_radiance, float[6](x.x, x.y, x.z, w.x, w.y, w.z), radiance_values);
    for (int i=0; i<OUTPUT_DIM; i++)
        _output[i] += T * radiance_values[i];
}

                """
    )

    def __init__(self, grid_model: Grid3D, out_radiance: MapBase, boundary_radiance: MapBase):
        super(Grid3DTransmittanceDDARayIntegral, self).__init__(OUTPUT_DIM=out_radiance.output_dim)
        assert out_radiance.output_dim == boundary_radiance.output_dim
        self.grid_model = grid_model
        self.grid = grid_model.grid
        self.out_radiance = out_radiance
        self.boundary_radiance = boundary_radiance
        self.box_min = grid_model.bmin.clone()
        self.box_max = grid_model.bmax.clone()


class Grid3DTransmittanceRayIntegral(MapBase):
    __extension_info__ = dict(
        parameters=dict(
            grid=ParameterDescriptor,
            out_radiance=MapBase,
            boundary_radiance=MapBase,
            box_min=vec3,
            box_max=vec3
        ),
        generics=dict(INPUT_DIM=6),
        code="""

float sigma_at(map_object, ivec3 cell)
{
    ivec3 dim = ivec3(parameters.grid.shape[2] - 1, parameters.grid.shape[1] - 1, parameters.grid.shape[0] - 1);
    cell = clamp(cell, ivec3(0), dim - ivec3(1));

    float_ptr buf = param_buffer(parameters.grid, cell);
    return buf.data[0];
}

void load_cell(map_object, ivec3 cell, out float[2][2][2] sigmas)
{
    for (int dz = 0; dz < 2; dz ++)
        for (int dy = 0; dy < 2; dy ++)
            for (int dx = 0; dx < 2; dx ++) 
                sigmas[dz][dy][dx] = sigma_at(object, cell + ivec3(dx, dy, dz));
}

float interpolated_sigma(map_object, vec3 alpha, float[2][2][2] sigmas)
{
    return mix(mix(
        mix(sigmas[0][0][0], sigmas[0][0][1], alpha.x),
        mix(sigmas[0][1][0], sigmas[0][1][1], alpha.x), alpha.y),
        mix(
        mix(sigmas[1][0][0], sigmas[1][0][1], alpha.x),
        mix(sigmas[1][1][0], sigmas[1][1][1], alpha.x), alpha.y), alpha.z);
}

FORWARD
{
    for (int i=0; i < OUTPUT_DIM; i++)
        _output[i] = 0.0;
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);
    
    float radiance_values[OUTPUT_DIM];
    float T = 1.0; // full transmittance start
    
    float tMin, tMax;
    if (intersect_ray_box(x, w, parameters.box_min, parameters.box_max, tMin, tMax))
    {
        x += w * tMin;  
        float d = tMax - tMin;
    
        vec3 box_size = parameters.box_max - parameters.box_min;
        ivec3 dim = ivec3(parameters.grid.shape[2] - 1, parameters.grid.shape[1] - 1, parameters.grid.shape[0] - 1);
        vec3 cell_size = box_size / dim;
        ivec3 cell = ivec3((x - parameters.box_min) * dim / box_size);
        cell = clamp(cell, ivec3(0), dim - ivec3(1));
        vec3 alpha_inc = cell_size / max(vec3(0.00001), abs(w));
        ivec3 side = ivec3(sign(w));
        vec3 corner = (cell + side * 0.5 + vec3(0.5)) * cell_size + parameters.box_min;
        vec3 alpha = abs(corner - x) / max(vec3(0.00001), abs(w));
        float current_t = 0;
        vec3 vn = (x - parameters.box_min) * dim / box_size;
        vec3 vm = w * dim / box_size;
        
        float[2][2][2] sigma_values;
    
        while(current_t < d - 0.0001){
            float next_t = min(d, min(alpha.x, min(alpha.y, alpha.z)));
    
            load_cell(object, cell, sigma_values);
            float majorant = max(max(
                max (sigma_values[0][0][0], sigma_values[0][0][1]),
                max (sigma_values[0][1][0], sigma_values[0][1][1])),
                max(
                max (sigma_values[1][0][0], sigma_values[1][0][1]),
                max (sigma_values[1][1][0], sigma_values[1][1][1])));
            
            float cell_t = current_t;
            while (true)
            {
                float dt = -log(1 - random()) / max(0.00001, majorant);
                if (cell_t + dt > next_t)
                break;
                cell_t += dt;
                // ** Accumulate interaction
                // * sample sigma
                vec3 interpolation_alpha = fract(vm * cell_t + vn);
                float sigma_value = interpolated_sigma(object, interpolation_alpha, sigma_values);
                float Pc = min(1.0, sigma_value / majorant);
                vec3 xc = cell_t*w + x;
                forward(parameters.out_radiance, float[6](xc.x, xc.y, xc.z, w.x, w.y, w.z), radiance_values);
                for (int i=0; i<OUTPUT_DIM; i++)
                    _output[i] += T * Pc * radiance_values[i];
                T *= (1 - Pc);
                if (T < 0.001) break;
            }
    
            if (T < 0.001) break;
            
            ivec3 cell_inc = ivec3(
                alpha.x <= alpha.y && alpha.x <= alpha.z,
                alpha.x > alpha.y && alpha.y <= alpha.z,
                alpha.x > alpha.z && alpha.y > alpha.z);
    
            current_t = next_t;
            alpha += cell_inc * alpha_inc;
            cell += cell_inc * side;
        }
        x += w * d;
    }
    forward(parameters.boundary_radiance, float[6](x.x, x.y, x.z, w.x, w.y, w.z), radiance_values);
    for (int i=0; i<OUTPUT_DIM; i++)
        _output[i] += T * radiance_values[i];
}

                """
    )

    def __init__(self, grid_model: Grid3D, out_radiance: MapBase, boundary_radiance: MapBase):
        super(Grid3DTransmittanceRayIntegral, self).__init__(OUTPUT_DIM=out_radiance.output_dim)
        assert out_radiance.output_dim == boundary_radiance.output_dim
        self.grid_model = grid_model
        self.grid = grid_model.grid
        self.out_radiance = out_radiance
        self.boundary_radiance = boundary_radiance
        self.box_min = grid_model.bmin.clone()
        self.box_max = grid_model.bmax.clone()


class TransmittanceDDA(MapBase):
    __extension_info__ = dict(
        parameters=dict(
            grid=ParameterDescriptor,
            box_min=vec3,
            box_max=vec3
        ),
        generics=dict(INPUT_DIM=6, OUTPUT_DIM=1),
        code="""
        
    float get_grid_float(Parameter grid, ivec3 dim, ivec3 cell) {
        // cell = clamp(cell, ivec3(0), dim - 1);
        if (any(lessThan(cell, ivec3(0))) || any(greaterThanEqual(cell, dim)))
            return 0.0f;
        float_ptr voxel_buf = param_buffer(grid, cell);
        return voxel_buf.data[0];
        /* int voxel_index = cell.x + (cell.y + cell.z * grid.dim.y) * grid.dim.x;
        float x[1];
        tensorLoad(grid.ptr, voxel_index, x);
        return x[0]; */
    }    
    
    float sample_grid_float(Parameter grid, ivec3 dim, vec3 p, vec3 b_min, vec3 b_max){
        vec3 fcell = dim * (p - b_min)/(b_max - b_min) - vec3(0.5);
        ivec3 cell = ivec3(floor(fcell));
        vec3 alpha = (fcell - cell);
        float sigma = 0;
        sigma += get_grid_float(grid, dim, cell + ivec3(0, 0, 0))*(1 - alpha.x)*(1 - alpha.y)*(1 - alpha.z);
        sigma += get_grid_float(grid, dim, cell + ivec3(1, 0, 0))*(alpha.x)*(1 - alpha.y)*(1 - alpha.z);
        sigma += get_grid_float(grid, dim, cell + ivec3(0, 1, 0))*(1 - alpha.x)*(alpha.y)*(1 - alpha.z);
        sigma += get_grid_float(grid, dim, cell + ivec3(1, 1, 0))*(alpha.x)*(alpha.y)*(1 - alpha.z);
        sigma += get_grid_float(grid, dim, cell + ivec3(0, 0, 1))*(1 - alpha.x)*(1 - alpha.y)*(alpha.z);
        sigma += get_grid_float(grid, dim, cell + ivec3(1, 0, 1))*(alpha.x)*(1 - alpha.y)*(alpha.z);
        sigma += get_grid_float(grid, dim, cell + ivec3(0, 1, 1))*(1 - alpha.x)*(alpha.y)*(alpha.z);
        sigma += get_grid_float(grid, dim, cell + ivec3(1, 1, 1))*(alpha.x)*(alpha.y)*(alpha.z);
        return sigma;
    }
    
        
    float DDA_Transmittance(Parameter grid, vec3 x, vec3 w, float d, vec3 b_min, vec3 b_max)
    {
        ivec3 dim = ivec3(grid.shape[2] - 1, grid.shape[1] - 1, grid.shape[0] - 1);
        vec3 b_size = b_max - b_min;
        vec3 cell_size = b_size / dim;
        ivec3 cell = ivec3((x - b_min) * dim / b_size);
        cell = clamp(cell, ivec3(0), dim - ivec3(1));
        vec3 alpha_inc = cell_size / max(vec3(0.00001), abs(w));
        ivec3 side = ivec3(sign(w));
        vec3 corner = (cell + side * 0.5 + vec3(0.5)) * cell_size + b_min;
        vec3 alpha = abs(corner - x) / max(vec3(0.00001), abs(w));
        float tau = 0;
        float current_t = 0;
        while(current_t < d - 0.00001){
            float next_t = min(d, min(alpha.x, min(alpha.y, alpha.z)));
            ivec3 cell_inc = ivec3(
                alpha.x <= alpha.y && alpha.x <= alpha.z,
                alpha.x > alpha.y && alpha.y <= alpha.z,
                alpha.x > alpha.z && alpha.y > alpha.z);
            float a = 0.5;//random(seed);
            vec3 xt = x + (next_t*a + current_t*(1-a))*w;
            float voxel_density = sample_grid_float(grid, dim, xt, b_min, b_max);
    
            tau += (next_t - current_t) * voxel_density;
            current_t = next_t;
            alpha += cell_inc * alpha_inc;
            cell += cell_inc * side;
        }
        return exp(-tau);
    }

    FORWARD
    {
        for (int i=0; i < OUTPUT_DIM; i++)
            _output[i] = 0.0;
        vec3 x = vec3(_input[0], _input[1], _input[2]);
        vec3 w = vec3(_input[3], _input[4], _input[5]);
        float tMin, tMax;
        if (!intersect_ray_box(x, w, parameters.box_min, parameters.box_max, tMin, tMax))
            return; // 0 integral value outside bounding box
        x += w * tMin;  
        float d = tMax - tMin;
        float T = DDA_Transmittance (parameters.grid, x, w, d, parameters.box_min, parameters.box_max);
        _output[0] = T;
    }
                    """,
    )

    def __init__(self, grid: _torch.Tensor, box_min: vec3 = vec3(-1.0, -1.0, -1.0), box_max: vec3 = vec3(1.0, 1.0, 1.0)):
        super(TransmittanceDDA, self).__init__()
        self.grid = _torch.nn.Parameter(grid)
        self.box_min = box_min.clone()
        self.box_max = box_max.clone()


class SH_PDF(MapBase):
    __extension_info__ = dict(
        generics=dict(INPUT_DIM=6, SH_DIM=1),
        parameters=dict(
            coefficients=MapBase
        ),
        bw_implementations=BACKWARD_IMPLEMENTATIONS.DEFAULT,
        code="""
    FORWARD
    {
        float Y[SH_DIM];
        eval_sh(vec3(_input[3], _input[4], _input[5]), Y);
        
        float c[SH_DIM * OUTPUT_DIM];
        forward(parameters.coefficients, float[3](_input[0], _input[1], _input[2]), c);
        
        for (int i=0; i<OUTPUT_DIM; i++)
        {
            _output[i] = 0.0;
            for (int j=0; j<SH_DIM; j++)
                _output[i] += Y[j] * c[i*SH_DIM + j];
            _output[i] = max(_output[i], 0.001);
        }
    }

    BACKWARD
    {
        // NOT EASY TASK to backprop input here.
        
        float Y[SH_DIM];
        eval_sh(vec3(_input[3], _input[4], _input[5]), Y);
        
        float dL_dc[SH_DIM * OUTPUT_DIM];
        for (int i=0; i<OUTPUT_DIM; i++)
        {
            for (int j=0; j<SH_DIM; j++)
                dL_dc[i*SH_DIM + j] = _output_grad[i] * Y[j];
        }

        float dL_dx[3];
        backward(parameters.coefficients, float[3](_input[0], _input[1], _input[2]), dL_dc, dL_dx);
        
        for (int i=0; i<3; i++)
            _input_grad[i] += dL_dx[i];
    }
    """
    )

    def __init__(self, output_dim, coefficients_map: 'MapBase'):
        assert coefficients_map.output_dim % output_dim == 0, f'SH coefficients must divide number of channels {output_dim}'
        SH_DIM = coefficients_map.output_dim // output_dim
        assert SH_DIM in [1, 4, 9], 'Not supported higher order SH than 3'
        assert coefficients_map.input_dim == 3
        super(SH_PDF, self).__init__(OUTPUT_DIM=output_dim, SH_DIM=SH_DIM)
        self.coefficients = coefficients_map


class RayBoxIntersection(MapBase):
    __extension_info__ = dict(
        generics=dict(
            INPUT_DIM=6,
            OUTPUT_DIM=2
        ),
        parameters=dict(
            box_min=vec3, pad0=float,
            box_max=vec3, pad1=float
        ),
        code="""
FORWARD
{
    ray_box_intersection(
        vec3(_input[0], _input[1], _input[2]), 
        vec3(_input[3], _input[4], _input[5]), 
        parameters.box_min, 
        parameters.box_max, 
        _output[0], _output[1]);
}
        """,
    )

    def __init__(self, box_min:vec3, box_max: vec3, **kwargs):
        super(RayBoxIntersection, self).__init__()
        self.box_min = box_min
        self.box_max = box_max


class VolumeRadianceIntegratorBase(MapBase):
    __extension_info__ = None  # abstract node

    @staticmethod
    def create_extension_info(radiance_code: str, *requires: str, **parameters):
        return dict(
            generics=dict(
                INPUT_DIM=6,
                OUTPUT_DIM=3,
                **{ 'VR_'+k.upper(): 1 for k in requires }
            ),
            parameters={ **{k: MapBase for k in requires}, **parameters },
            bw_implementations=BACKWARD_IMPLEMENTATIONS.DEFAULT,
            code = f"""
#include "common_vr.h"

{radiance_code}

FORWARD {{
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);
    vec3 R = volume_radiance(object, x, w);
    _output = float[3](R.x, R.y, R.z);
}}

BACKWARD {{
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);
    vec3 dL_dR = vec3(_output_grad[0], _output_grad[1], _output_grad[2]);
    volume_radiance_bw(object, x, w, dL_dR);
}}
            """
        )

    def __init__(self, **maps):
        required_maps = type(self).__extension_info__['parameters'].keys()
        assert all(r in required_maps for r in maps)
        assert all(r in maps for r in required_maps)
        assert all(v is None or isinstance(v, MapBase) for v in maps.values())
        super(VolumeRadianceIntegratorBase, self).__init__()
        for k,v in maps.items():
            setattr(self, k, v)


class DTCollisionIntegrator(VolumeRadianceIntegratorBase):
    '''
    I(x, w) = \int_0^d(x,w) T(x_0, x_t) \sigma(x_t) F(x_t, w) dt + T(x_0, x_d) B(x_t, w)
    '''
    __extension_info__ = VolumeRadianceIntegratorBase.create_extension_info(
"""
vec3 volume_radiance(map_object, vec3 x, vec3 w)
{
    float tMin, tMax;
    if (boundary(object, x, w, tMin, tMax))
    {
        x += w * tMin;
        tMax -= tMin;
        vec3 xd = x + w * tMax;
        tMax -= empty_space(object, xd, -w);
        tMin = empty_space(object, x, w);
        x += w * tMin;
        float d = tMax - tMin;
                 
        while (d > 0.00000001)
        {
            float md;
            float maj = majorant(object, x, w, md);
            float local_d = min(d, md);

            float t = -log(1 - random())/maj;

            x += min(t, local_d) * w;

            if (t > local_d)
            {
                d -= local_d;
                continue;
            }

            if (random() < sigma(object, x) / maj)
                return out_radiance(object, x, -w);

            d -= t;
        }
    }
    
    return boundary_radiance(object, x, w);
}

void volume_radiance_bw(map_object, vec3 x, vec3 w, vec3 dL_dR) {

}
        """,
        "sigma", "majorant", "boundary", "empty_space", "out_radiance", "boundary_radiance"
    )
#         code = """
# #include "common_vr.h"
#
# FORWARD
# {
#     vec3 x = vec3(_input[0], _input[1], _input[2]);
#     vec3 w = vec3(_input[3], _input[4], _input[5]);
#     // Get boundary
#     float tMin, tMax;
#     float adding_radiance[OUTPUT_DIM];
#     if (ray_volume_intersection(object, x, w, tMin, tMax))
#     {
#         x += w * tMin;
#         float d = tMax - tMin;
#         float t = ray_empty_space(object, x, w);
#         x += w * t; // advance to first non-zero density
#         d -= t;
#         while (d > 0)
#         {
#             float md ;
#             float majorant = ray_majorant(object, x, w, md);
#             float local_d = min(d, md);
#
#             t = -log(1 - random())/majorant;
#
#             x += min(t, local_d) * w;
#
#             if (t > local_d)
#             {
#                 d -= local_d;
#                 continue;
#             }
#
#             if (random() < sigma(object, x) / majorant)
#             {
#                 out_radiance(object, x, -w, adding_radiance);
#                 for (int i=0; i<OUTPUT_DIM; i++)
#                     _output[i] = adding_radiance[i];
#                 return;
#             }
#
#             d -= t;
#         }
#     }
#     boundary_radiance(object, x, w, adding_radiance);
#     for (int i=0; i<OUTPUT_DIM; i++)
#         _output[i] = adding_radiance[i];
# }
#         """, nodiff=True

    def __init__(self,
                 sigma: MapBase,
                 boundary: MapBase,
                 empty_space: MapBase,
                 majorant: MapBase,
                 out_radiance: MapBase,
                 boundary_radiance: MapBase
                 ):
        assert out_radiance.output_dim == boundary_radiance.output_dim
        super(DTCollisionIntegrator, self).__init__(sigma=sigma, boundary=boundary, empty_space=empty_space, majorant=majorant, out_radiance=out_radiance, boundary_radiance=boundary_radiance)


class DTRadianceIntegrator(VolumeRadianceIntegratorBase):
    __extension_info__ = VolumeRadianceIntegratorBase.create_extension_info(
        """
vec3 volume_radiance(map_object, vec3 x, vec3 w)
{
    float tMin, tMax;
    if (!boundary(object, x, w, tMin, tMax))
        return environment(object, w);
    
    vec3 W = vec3(1.0); // importance of the path
    vec3 A = vec3(0.0); // radiance accumulation
    
    x += w * tMin;
    tMax -= tMin;
    vec3 xd = x + w * tMax;
    tMax -= empty_space(object, xd, -w);
    tMin = empty_space(object, x, w);
    x += w * tMin;
    float d = tMax - tMin;
             
    bool some_collision = false;
             
    while (d > 0.00000001)
    {
        float md;
        float maj = majorant(object, x, w, md);
        float local_d = min(d, md);

        float t = -log(1 - random())/maj;

        x += min(t, local_d) * w;

        if (t > local_d)
        {
            d -= local_d;
            continue;
        }
        
        if (random() < sigma(object, x) / maj)
        { 
            // ** Accumulate collision contribution, scatter and continue
            vec3 s = scattering_albedo(object, x);
            vec3 a = vec3(1.0) - s;
            // add emitted radiance
            A += W * a * emission(object, x); 
            // add NEE
            W *= s;
            vec3 w_wenv;
            vec3 wenv = environment_sampler(object, x, w, w_wenv);
            float nee_T = transmittance(object, x, x + wenv * neetMax); 
            A += W * nee_T * phase(object, x, w, wenv) * w_wenv;
            // continue with indirect contribution
            float w_wph;
            vec3 wph = phase_sampler(object, x, w, w_wph);
            w = wph;
            W *= w_wph;
            
            boundary(object, x, w, tMin, tMax);
            x += w * tMin; 
            d = tMax - tMin;
            some_collision = true;
            continue;
        }
        
        // Null collide
        d -= t;
    }
    
    if (!some_collision)
        A += W * environment(object, x, w); 
    
    return A;
}
        """,
        "sigma", "boundary", "empty_space", "majorant", "scattering_albedo", "emission", "environment",
        "phase", "environment_sampler", "phase_sampler", "transmittance"
    )

    def __init__(self,
                 sigma: MapBase,
                 boundary: MapBase,
                 empty_space: MapBase,
                 majorant: MapBase,
                 scattering_albedo: MapBase,
                 emission: MapBase,
                 environment: MapBase,
                 phase: MapBase,
                 environment_sampler: MapBase,
                 phase_sampler: MapBase,
                 transmittance: MapBase
                 ):
        assert emission.output_dim == environment.output_dim == scattering_albedo.output_dim == 3
        super(DTRadianceIntegrator, self).__init__(
            sigma=sigma,
            boundary=boundary,
            empty_space=empty_space,
            majorant=majorant,
            scattering_albedo=scattering_albedo,
            emission=emission,
            environment=environment,
            phase=phase,
            environment_sampler=environment_sampler,
            phase_sampler=phase_sampler,
            transmittance=transmittance
        )

# class RTCollisionIntegrator(VolumeRadianceIntegratorBase):



# class Concat(MapBase):
#     __extension_info__ = dict(
#         parameters=dict(
#             map_a = MapBase,
#             map_b = MapBase
#         )
#     )
#     def __init__(self, map_a: MapBase, map_b: MapBase):
#         assert map_a.input_dim == map_b.input_dim
#         super(Concat, self).__init__(INPUT_DIM = map_a.input_dim, OUTPUT_DIM = map_a.output_dim + map_b.output_dim)


class HGPhase(MapBase):
    __extension_info__ = dict(
        generics=dict(
            INPUT_DIM=9,
            OUTPUT_DIM=1,
        ),
        parameters=dict(
            phase_g=MapBase,
        ),
        code = """
FORWARD
{
    float _g[1];
    forward(parameters.phase_g, float[3](_input[0], _input[1], _input[2]), _g);
    float g = _g[0];
    vec3 w1 = vec3(_input[3], _input[4], _input[5]);
    vec3 w2 = vec3(_input[6], _input[7], _input[8]);
    _output[0] = mix(hg_phase_eval(w1, w2, g), hg_phase_eval(w1, w2, g/5), 0.2); 
}
        """
    )

    def __init__(self, phase_g: MapBase):
        super(HGPhase, self).__init__()
        self.phase_g = phase_g.cast(3, 1)


class VHGPhaseSampler(MapBase):
    __extension_info__ = dict(
        parameters=dict(
            phase_g=MapBase
        ),
        generics=dict(INPUT_DIM=6, OUTPUT_DIM=4),
        code="""
FORWARD
{
    float g[1];
    forward(parameters.phase_g, float[3](_input[0], _input[1], _input[2]), g);
    
    if (random() < 0.8)
        g[0] /= 5;
    vec3 w_out = hg_phase_sample(vec3(_input[3], _input[4], _input[5]), g[0]);
    _output = float[4](1.0, w_out.x, w_out.y, w_out.z);
}
        """
    )
    def __init__(self, phase_g: MapBase):
        super().__init__()
        self.phase_g = phase_g.cast(3, 1)


class HGDirectionSampler(MapBase):
    __extension_info__ = dict(
        parameters=dict(
            phase_g=MapBase
        ),
        generics=dict(INPUT_DIM=6, OUTPUT_DIM=4),
        code="""
FORWARD
{
    float g[1];
    forward(parameters.phase_g, float[3](_input[0], _input[1], _input[2]), g);
    vec3 w_in = vec3(_input[3], _input[4], _input[5]);
    vec3 w_out = hg_phase_sample(w_in, g[0]);
    _output = float[4](w_out.x, w_out.y, w_out.z, 1.0/hg_phase_eval(w_in, w_out, g[0]));
}
        """
    )

    def __init__(self, phase_g: MapBase):
        super(HGDirectionSampler, self).__init__()
        self.phase_g = phase_g


__endline__ = '\n'


def map_to_generic(map_name: str):
    return 'VR_'+map_name.upper()


# class VolumeRadianceFieldBase(MapBase):
#     __extension_info__ = None  # abstract node
#
#     @staticmethod
#     def create_extension_info(compute_radiance_code, *maps):
#         return dict(
#             generics=dict(
#                 INPUT_DIM=6,
#                 OUTPUT_DIM=3,
#             ),
#             parameters=dict(
#                 **{k: MapBase for k in maps },
#                 box_min=vec3,
#                 box_max=vec3
#             ),
#             code = f"""
# #include "common_vr.h"
#
# void compute_radiance(map_object, vec3 x, vec3 w, out vec3 wout, out float A[OUTPUT_DIM], out float W[OUTPUT_DIM]);
#
# FORWARD
# {{
#     vec3 x = vec3(_input[0], _input[1], _input[2]);
#     vec3 w = vec3(_input[3], _input[4], _input[5]);
#     float W[OUTPUT_DIM];
#     vec3 ws;
#     compute_radiance(object, x, w, ws, _output, W); // it is missing to retrieve proper jaccobians for x,w -> ws
#     float env[OUTPUT_DIM];
#     environment(object, ws, env);
#     for (int i = 0; i < OUTPUT_DIM; i++) _output[i] += W[i] * env[i];
# }}
#
# // TODO: the jaccobian for the ray transitions is missing. This can not propagate gradients wrt camera rays
# void compute_radiance_bw(map_object, vec3 x, vec3 w, float o_A[OUTPUT_DIM], float o_W[OUTPUT_DIM], float dL_dA[OUTPUT_DIM], float dL_dW[OUTPUT_DIM]);
#
# BACKWARD
# {{
#     float[OUTPUT_DIM] dL_dA = _output_grad;
#
#     uvec4 seed = get_seed(); // save current seed for later replay
#
#     vec3 x = vec3(_input[0], _input[1], _input[2]);
#     vec3 w = vec3(_input[3], _input[4], _input[5]);
#     float A[OUTPUT_DIM];
#     float W[OUTPUT_DIM];
#     vec3 ws;
#     compute_radiance(object, x, w, ws, A, W);
#
#     //A += W * environment(object, ws);
#
#     float dL_dW[OUTPUT_DIM];
#     float env[OUTPUT_DIM];
#     environment(object, ws, env);
#     for (int i=0; i<OUTPUT_DIM; i++) dL_dW[i] = dL_dA[i] * env[i];
#
#     //vec3 dL_denv = dL_dA * W;
#     //vec3 dL_dw;
#     //environment_bw(object, ws, dL_denv, dL_dw);
#
#     set_seed(seed); // start replaying
#
#     // precomputed output values
#     compute_radiance_bw(object, x, w, A, W, dL_dA, dL_dW);
#
#     // TODO: in a future when ray jaccobian is considered input_grad can be updated
# }}
#
#
# """ + compute_radiance_code
#         )
#
#     def __init__(self, box_min:vec3, box_max: vec3, **kwargs):
#         super(VolumeRadianceFieldBase, self).__init__(**{map_to_generic(k): v.output_dim for k,v in kwargs.items()})
#         for k, v in kwargs.items():
#             setattr(self, k, v)
#         self.box_min = box_min
#         self.box_max = box_max
#
#     def intersect_ray_box(self, x: _torch.Tensor, w: _torch.Tensor) -> Tuple[_torch.Tensor, _torch.Tensor, _torch.Tensor]:
#         w = _torch.where(_torch.abs(w) <= 0.000001, _torch.full_like(w, fill_value=0.000001), w)
#         C_Min = (self.box_min.to(x.device) - x)/w
#         C_Max = (self.box_max.to(x.device) - x)/w
#         min_C = _torch.minimum(C_Min, C_Max)
#         max_C = _torch.maximum(C_Min, C_Max)
#         tMin = _torch.clamp_min(_torch.max(min_C, dim=-1, keepdim=True)[0], 0.0)
#         tMax = _torch.min(max_C, dim=-1, keepdim=True)[0]
#         return (tMax > tMin)*(tMax > 0), tMin, tMax
#
#     def ray_enter(self, x: _torch.Tensor, w: _torch.Tensor):
#         mask, tMin, tMax = self.intersect_ray_box(x, w)
#         return mask, x + w*tMin
#
#     def ray_exit(self, x: _torch.Tensor, w: _torch.Tensor):
#         mask, tMin, tMax = self.intersect_ray_box(x, w)
#         return tMax
#
#     def compute_radiance_torch(self, alive: _torch.Tensor, x: _torch.Tensor, w: _torch.Tensor) -> Tuple[_torch.Tensor, _torch.Tensor]:
#         pass
#
#     def forward_torch(self, *args):
#         xw, = args
#         x = xw[...,0:3]
#         w = xw[...,3:6]
#         A = _torch.zeros_like(x)
#         W = _torch.ones_like(x)
#         entered, x = self.ray_enter(x, w)
#         Av, Wv = self.compute_radiance_torch(entered, x, w)
#         W = _torch.where(entered, Wv, W)
#         A = _torch.where(entered, Av, A)
#         A += W * self.environment(w)
#         return A
#

# class AbsorptionOnlyVolume(VolumeRadianceFieldBase):
#     __extension_info__ = VolumeRadianceFieldBase.create_extension_info("""
# float compute_tau(map_object, inout vec3 x, vec3 w, float d)
# {
#     float total_sigma = 0;
#     float dt = 0.005;
#     int samples = int(d / dt) + 1;
#     vec3 dw = w * dt;
#     x += dw * 0.5;
#     for (int i=0; i<samples; i++)
#     {
#         total_sigma += sigma(object, x);
#         x += dw;
#     }
#     return total_sigma * d / samples;
# }
#
# void compute_radiance(map_object, inout vec3 x, inout vec3 w, out vec3 A, out vec3 W)
# {
#     A = vec3(0.0);
#     float d = ray_exit(object, x, w);
#     float tau = compute_tau(object, x, w, d);
#     W = vec3(1.0) * exp(-tau);
# }
#     """, ['sigma', 'environment'])
#
#     def __init__(self, sigma: 'MapBase', environment: 'MapBase', box_min:vec3, box_max: vec3):
#         super(AbsorptionOnlyVolume, self).__init__(box_min, box_max, sigma = sigma, environment = environment)
#
#     def compute_tau(self, alive: _torch.Tensor, x: _torch.Tensor, w: _torch.Tensor, d: _torch.Tensor):
#         tau = _torch.zeros(*x.shape[:-1], 1, device=x.device)
#         i = _torch.zeros(*x.shape[:-1], 1, device=x.device)
#         dt = 0.005
#         samples = (d / dt).int() + 1
#         dw = w * dt
#         x += dw * 0.5
#
#         while alive.any():
#             tau += self.sigma(x)
#             x += dw
#             i += 1
#             alive *= i < samples
#
#         return tau * d / samples
#
#     def compute_radiance_torch(self, alive: _torch.Tensor, x: _torch.Tensor, w: _torch.Tensor) -> Tuple[_torch.Tensor, _torch.Tensor]:
#         """
#         A = vec3(0.0);
#         float d = ray_exit(x, w);
#         W = vec3(1.0) * exp(-d);
#         """
#         d = self.ray_exit(x, w)
#         A = _torch.zeros_like(x)
#         W = _torch.ones_like(x) * _torch.exp(-self.compute_tau(alive.clone(), x, w, d))
#         return A, W
#

class AbsorptionOnlyVolume(VolumeRadianceIntegratorBase):
    __extension_info__ = VolumeRadianceIntegratorBase.create_extension_info("""
vec3 volume_radiance(map_object, vec3 x, vec3 w)
{
    float T = transmittance(object, x, x + w*INF_DISTANCE);
    vec3 R = T * environment(object, w);
    return R;
}

void volume_radiance_bw(map_object, vec3 x, vec3 w, vec3 dL_dR)
{
    float dL_dT = dot(dL_dR, environment(object, w));
    transmittance_bw(object, x, x + w * INF_DISTANCE, dL_dT);
}
    """, 'transmittance', 'environment')

    def __init__(self, transmittance: 'MapBase', environment: 'MapBase'):
        super(AbsorptionOnlyVolume, self).__init__(transmittance=transmittance, environment = environment)





#
# class AbsorptionOnlyXVolume(VolumeRadianceFieldXBase):
#     __extension_info__ = dict(
#         path=__DISPATCHING_FOLDER__ + '/vrf/ao_tensor',
#         nodiff=True,
#         supports_express=True,
#         force_compilation=True,
#         parameters=Layout.create_structure('scalar',
#                                            sigma=_torch.int64,
#                                            sigma_shape=ivec3,
#                                            environment=_torch.int64,
#                                            environment_shape=ivec2,
#                                            box_min=vec3,
#                                            box_max=vec3,
#                                            custom=dict( foo=int )
#                                            )
#     )
#
#     def __init__(self, sigma: _torch.Tensor, environment: _torch.Tensor, box_min:vec3, box_max: vec3):
#         super(AbsorptionOnlyXVolume, self).__init__(sigma, environment, box_min, box_max)
#
#     def compute_tau(self, alive: _torch.Tensor, x: _torch.Tensor, w: _torch.Tensor, d: _torch.Tensor):
#         tau = _torch.zeros(*x.shape[:-1], 1, device=x.device)
#         i = _torch.zeros(*x.shape[:-1], 1, device=x.device)
#         dt = 0.005
#         samples = (d / dt).int() + 1
#         dw = w * dt
#         x += dw * 0.5
#
#         while alive.any():
#             tau += self.sigma(x)
#             x += dw
#             i += 1
#             alive *= i < samples
#
#         return tau * d / samples
#
#     def compute_radiance_torch(self, alive: _torch.Tensor, x: _torch.Tensor, w: _torch.Tensor) -> Tuple[_torch.Tensor, _torch.Tensor]:
#         """
#         A = vec3(0.0);
#         float d = ray_exit(x, w);
#         W = vec3(1.0) * exp(-d);
#         """
#         d = self.ray_exit(x, w)
#         A = _torch.zeros_like(x)
#         W = _torch.ones_like(x) * _torch.exp(-self.compute_tau(alive.clone(), x, w, d))
#         return A, W


class ParametricMap(object):
    def __init__(self, t):
        object.__setattr__(self, '_factory', t)

    def __getitem__(self, item):
        t = object.__getattribute__(self, '_factory')
        if not isinstance(item, tuple):
            item = [item]
        if isinstance(item[-1], dict):
            kwargs = item[-1]
            item = item[:-1]
        else:
            kwargs = {}
        return t(*item, **kwargs)

    def __getattr__(self, item):
        raise Exception('Parameters to a parametric map must be provided before using, e.g.: map[1.0, 2.0]')



def normalized_box(s: _typing.Union[_torch.Tensor, _typing.List]) -> _typing.Tuple[vec3, vec3]:
    if isinstance(s, _torch.Tensor):
        shape = s.shape
    else:
        shape = s
    max_dim = max(shape[0], shape[1], shape[2]) - 1
    b_max : vec3 = vec3(shape[2] - 1, shape[1] - 1, shape[0] - 1) / max_dim
    b_min = -b_max
    return (b_min, b_max)


def _const_factory(*args, input_dim: _typing.Optional[int] = None):
    """
    args: float values for the constant or a single _torch.Tensor object
    """
    if len(args) == 1 and isinstance(args[0], _torch.Tensor):
        return ConstantMap(value=args[0], input_dim=input_dim)
    return ConstantMap(value=_torch.tensor([*args], device=_internal.device(), dtype=_torch.float), input_dim=input_dim)

const = ParametricMap(_const_factory)
"""
Constant map with a specific value.
examples:
const[1.0, 2.0]
const[p]  # with p a tensor or a parameter
const[1.0, 2.0, 3.0, dict(input_dim=5)]  # Non generic constant map for R^5 inputs evaluating (1.0, 2.0, 3.0)
"""

ray_to_segment = ParametricMap(RayToSegment) # (distance_field: 'MapBase'):
"""
Given a ray xw and a distance field d(xw) return x + w * d(xw)
Example:
ray_to_segment[ray_box_intersect[1]]
"""
    # return RayToSegment(distance_field)

def grid2d(t: _torch.Tensor, bmin: vec2 = vec2(-1.0, -1.0), bmax: vec2 = vec2(1.0, 1.0)):
    return Grid2D(t, bmin, bmax)

def image2d(t: _torch.Tensor, bmin: vec2 = vec2(-1.0, -1.0), bmax: vec2 = vec2(1.0, 1.0)):
    return Image2D(t, bmin, bmax)

def grid3d(t: _torch.Tensor, bmin: vec3 = vec3(-1.0, -1.0, -1.0), bmax: vec3 = vec3(1.0, 1.0, 1.0)):
    return Grid3D(t, bmin, bmax)

def transmittance(sigma: 'MapBase', majorant: float = None, mode: _typing.Literal['dt', 'rt'] = 'rt'):
    if majorant is None:
        assert isinstance(sigma, Grid3D)
        majorant = sigma.grid.max().item()
    if mode == 'rt':
        return TransmittanceRT(sigma, majorant)
    if mode == 'dt':
        return TransmittanceDT(sigma, majorant)
    raise Exception()

xr_projection = XRProjection(False)
"""
Equirectangular projection of a direction. 
"""

xr_ray_projection = XRProjection(True)
"""
Equirectangular projection of a ray. 
"""

# def xr_projection(ray_input: bool = False):
#     return XRProjection(ray_input=ray_input)

oct_inv_projection = OctUnprojection()
"""
Octahedral inverse projection, from square to direction.
"""

oct_projection = OctProjection()
"""
Octahedral projection, from direction to square.
"""

oct_ray_projection = OctProjection(ray_input=True)
"""
Octahedral projection, from ray to square.
"""

def tsr(cls, translate: vec3 = vec3(0.0, 0.0, 0.0), scale: vec3 = vec3(1.0, 1.0, 1.0), rotatation_axis: vec3 = vec3(0.0, 1.0, 0.0), rotation_angle: float = 0.0):
    raise NotImplementedError()

def tsr_normal(cls, translate: vec3 = vec3(0.0, 0.0, 0.0), scale: vec3 = vec3(1.0, 1.0, 1.0), rotatation_axis: vec3 = vec3(0.0, 1.0, 0.0), rotation_angle: float = 0.0):
    raise NotImplementedError()

def tsr_position_normal(cls, translate: vec3 = vec3(0.0, 0.0, 0.0), scale: vec3 = vec3(1.0, 1.0, 1.0), rotatation_axis: vec3 = vec3(0.0, 1.0, 0.0), rotation_angle: float = 0.0):
    raise NotImplementedError()

def tsr_ray(cls, translate: vec3 = vec3(0.0, 0.0, 0.0), scale: vec3 = vec3(1.0, 1.0, 1.0), rotatation_axis: vec3 = vec3(0.0, 1.0, 0.0), rotation_angle: float = 0.0):
    raise NotImplementedError()

def spherical_projection(cls):
    raise NotImplementedError()

def cylindrical_projection(cls):
    raise NotImplementedError()

# def identity(input_dim: _typing.Optional[int] = None):
#     return Identity(input_dim)

X = Identity()
"""
Represents an identity map. Used to represent y=x and compose and operate with other maps.
example: doubled_map = 2 * X
"""

ZERO = const[0.0]
"""
Constant map 0.0
"""

ONE = const[1.0]
"""
Constant map 1.0
"""

ray_position = RayPosition()
"""
Assumes the input in the form x|w and return x
"""

ray_direction = RayDirection()
"""
Assumes the input in the form x|w and return w
"""

ray_box_intersection = ParametricMap(RayBoxIntersection)
"""
Given a ray returns the two intersection distance with a specific box bmin: vec3, bmax: vec3.
"""


relu = ReluMap()
"""
Performs relu activation on all values of input.
"""

sin = SinMap()
"""
Performs sin function on all values of input.
"""

cos = CosMap()
"""
Performs cos function on all values of input.
"""


def _sampling_map(t: _torch.Tensor):
    if len(t.shape) == 3:  # 2D grid
        return Grid2D(t)
    elif len(t.shape) == 4:
        return Grid3D(t)
    raise NotImplemented()


sample = ParametricMap(_sampling_map)
"""
Represents a piece-wise linear interpolated field (2D or 3D) from -1.0 to 1.0
using values of a tensor as a regular aligned grid.
"""


def look_at_poses(
    camera: _typing.Union[vec3, _typing.Tuple[float, float, float]],
    target: _typing.Optional[_typing.Union[vec3, _typing.Tuple[float, float, float]]] = None,
    up: _typing.Optional[_typing.Union[vec3, _typing.Tuple[float, float, float]]] = None
):
    if target is None:
        target = (0.0, 0.0, 0.0)
    if up is None:
        up = (0.0, 1.0, 0.0)
    camera, target, up = _vk.broadcast_args_to_max_batch((camera, (3,)), (target, (3,)), (up, (3,)))
    camera = camera.to(_internal.device())
    target = target.to(_internal.device())
    up = up.to(_internal.device())
    camera_poses = _torch.cat([camera, vec3.normalize(target - camera), up], dim=-1)
    return camera_poses





# def ray_box_intersection(bmin: vec3, bmax: vec3):
#     return RayBoxIntersection(box_min=bmin, box_max=bmax)