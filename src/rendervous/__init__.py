rdv_device = 0
# command = "nvidia-smi --query-gpu=memory.used --format=csv"
# mems = sp.check_output(command.split()).decode('ascii').split('\n')[1:-1]
# for i, m in enumerate(mems):
#     if i not in [1]:
#         if int(m.split()[0]) < 20:
#             rdv_device = i
#             break
# assert rdv_device>=0, "All devices where in use!"

# If is cuda available for torch, use same device that cuda:0
try:
    import os
    __devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',') # = str(rdv_device)
    rdv_device = int(__devices[0])
except:
    # print("[WARNING] Multiple cuda devices not found.")
    rdv_device = 0


from vulky import (
    create_device as _create_device,
    vec2, vec3, vec4, ivec2, ivec3, ivec4, mat2, mat3, mat3x4, mat4x3, mat4,
    broadcast_args_to_max_batch,
    load_texture, load_obj, load_image, load_video, save_video, save_image,
    tensor, tensor_to_vec, tensor_like, tensor_copy,
    tensor_to_mat, tensor_to_gtensor_if_possible,
)

__DEBUG__ = bool(os.environ.get('RDV_DEBUG', 'False') == 'True')

_create_device(device=rdv_device, debug=__DEBUG__)

from ._functions import *
from ._objects import *
from ._maps import *
from ._scenes import *
from ._internal import *
from ._compiling import *

start_engine()



