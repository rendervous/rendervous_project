from rendervous import _internal
from rendervous import _maps
import rendervous as rdv
import torch
import numpy as np
from enum import Enum, unique
from typing import Optional

# from rich import print


@unique
class ForwardTechnique(Enum):
    technique_1 = 0
    technique_2 = 1


@unique
class BackwardTechnique(Enum):
    technique_1 = 0
    technique_2 = 1
    technique_3 = 2


class BufferManager:
    def __init__(self, code_list: list[str]):
        self.code_list: list[str] = code_list
        self.buffers: dict[str, dict[int, list[str]]] = {}
        self._buffers_created: dict[str, int] = {}
        self.locked_buffers: set[str] = set()
        self.last_requested_buffer: dict[str, str] = {}
        self.checkpoints: dict[int, str] = {}

    def _create_buffer(self, namespace: str, dimension: int):
        if namespace not in self._buffers_created:
            self._buffers_created[namespace] = 0
        buffer_name = f"{namespace}_{self._buffers_created[namespace]}"
        self._buffers_created[namespace] += 1
        self.code_list.append(f"float {buffer_name}[{dimension}];")
        if namespace not in self.buffers:
            self.buffers[namespace] = {}
        if dimension not in self.buffers[namespace]:
            self.buffers[namespace][dimension] = []
        self.buffers[namespace][dimension].append(buffer_name)
        return buffer_name

    def get_free_buffer_with_dimension(
        self,
        namespace: str,
        dimension: int,
        update_last_req_buff: bool = True,
    ):
        if namespace not in self.buffers or dimension not in self.buffers[namespace]:
            buffer_name = self._create_buffer(namespace, dimension)
        else:
            for buffer_name in self.buffers[namespace][dimension]:
                if buffer_name not in self.locked_buffers:
                    if update_last_req_buff:
                        self.last_requested_buffer[namespace] = buffer_name
                    return buffer_name
            # If all buffers are locked, create a new one
            buffer_name = self._create_buffer(namespace, dimension)
        if update_last_req_buff:
            self.last_requested_buffer[namespace] = buffer_name
        return buffer_name

    def lock_buffer_with_name(self, name: str):
        for namespace_buffers in self.buffers.values():
            for buffers in namespace_buffers.values():
                if name in buffers:
                    self.locked_buffers.add(name)
                    return
        return

    def free_buffer_with_name(self, name: str):
        if name not in self.locked_buffers:
            return
        self.locked_buffers.remove(name)

    def get_last_requested_buffer(self, namespace: str):
        return self.last_requested_buffer.get(namespace, None)

    def set_last_requested_buffer(self, namespace: str, buffer_name: str):
        self.last_requested_buffer[namespace] = buffer_name

    def create_checkpoint(
        self, checkpoint_number: int, namespace: str, checkpoint_dim: int
    ) -> str:
        if checkpoint_number in self.checkpoints:
            raise ValueError(f"Checkpoint {checkpoint_number} already exists.")
        buffer_name = self._create_buffer(namespace, checkpoint_dim)
        self.lock_buffer_with_name(buffer_name)
        self.checkpoints[checkpoint_number] = buffer_name
        return buffer_name

    def get_checkpoint(self, checkpoint_number: int):
        if checkpoint_number not in self.checkpoints:
            raise ValueError(f"Checkpoint {checkpoint_number} does not exist.")
        return self.checkpoints[checkpoint_number]


class Node:
    def build(self) -> _maps.MapBase:
        """This method returns a MapBase for the Node."""
        pass


class MapNode(Node):
    def __init__(self, map: _maps.MapBase):
        self.map: _maps.MapBase = map

    def build(self):
        return self.map


class LinearNode(Node):
    """Linear Layer for a Neural Network"""

    def __init__(self, input_dim: int, output_dim: int):
        self.input_dim: int = input_dim
        self.output_dim: int = output_dim

    @staticmethod
    def dense(input_dim: int, output_dim: int) -> _maps.MapBase:
        k = 1 / input_dim
        A_0 = torch.nn.Parameter(
            (torch.rand(output_dim, input_dim, device=rdv.device()) * 2 - 1)
            * np.sqrt(k)
        )
        B_0 = torch.nn.Parameter(
            (2 * torch.rand(output_dim, device=rdv.device()) - 1) * np.sqrt(k)
        )
        return A_0 @ rdv.X + rdv.const[B_0]

    def build(self) -> _maps.MapBase:
        # Figure it out how to use the MatrixMultiplicationMap in here
        return LinearNode.dense(self.input_dim, self.output_dim)


def generate_checkpoint_indices(
    length: int, distance_between_checkpoints: int
) -> list[int]:
    res: list[int] = []
    for i in range(length):
        if (i + 1) % (distance_between_checkpoints + 1) == 0:
            res.append(i)
    return res


class Sequential(Node):
    def __init__(
        self,
        *layers: Node,
        forward_technique: ForwardTechnique = ForwardTechnique.technique_1,
        backward_technique: BackwardTechnique = BackwardTechnique.technique_1,
        distance_between_checkpoints: int = 2,
        checkpoint_indices: list[int] = None,
    ):
        self.layers: list[Node] = layers
        """The layers that are available inside the sequential"""
        self.maps: list[_maps.MapBase] = [l.build() for l in self.layers]
        """The maps of the sequential node"""
        self._cast_maps_dimensions()
        self._check_dimensions()
        self.forward_tech: ForwardTechnique = forward_technique
        """The type of forward code to create"""
        self.backward_tech: BackwardTechnique = backward_technique
        """The type of backward code to create"""
        self.distance_between_checkpoints: int = max(0, distance_between_checkpoints)
        self.checkpoint_indices: list[int] = (
            checkpoint_indices
            if checkpoint_indices
            else generate_checkpoint_indices(
                len(self.maps), self.distance_between_checkpoints
            )
        )
        """The list of indices in which to place checkpoints"""

    def _cast_maps_dimensions(self):
        for i in range(len(self.maps) - 1):
            actual_map = self.maps[i]
            next_map = self.maps[i + 1]
            if actual_map.is_generic_output and next_map.is_generic_input:
                raise Exception(
                    f"The map {actual_map._get_name()} has generic output and the map {next_map._get_name()} has generic input"
                )
            if actual_map.is_generic_output:
                print(
                    f"Casting output dimension of map {actual_map._get_name()} at index {i} to => {next_map.input_dim}"
                )
                self.maps[i] = actual_map.cast(output_dim=next_map.input_dim)
            elif next_map.is_generic_input:
                print(
                    f"Casting input dimension of map {next_map._get_name()} at index {i} to => {actual_map.output_dim}"
                )
                self.maps[i + 1] = next_map.cast(input_dim=actual_map.output_dim)

    def _check_dimensions(self):
        for i in range(len(self.maps)):
            if i == 0:
                continue
            if self.maps[i - 1].output_dim != self.maps[i].input_dim:
                print("=======================ERROR=======================")
                print(
                    f"Output dim of map '{self.maps[i-1]._get_name()}' at index {i-1}: {self.maps[i-1].output_dim}"
                )
                print(
                    f"Input dim of map '{self.maps[i]._get_name()}' at index {i}: {self.maps[i].input_dim}"
                )
                print("===================================================")
                raise ValueError(
                    f"Output dimension of layer {i-1} is not equal to input dimension of layer {i}"
                )

    def _build_forward_technique_1(self) -> str:
        forward_code: list[str] = []
        for i in range(len(self.maps) - 1):
            forward_code.append(f"float _intermediate_{i}[{self.maps[i].output_dim}];")
            forward_input = f"_input" if i == 0 else f"_intermediate_{i-1}"
            forward_code.append(
                f"forward(parameters.map_{i}, {forward_input}, _intermediate_{i});"
            )
        forward_code.append(
            f"forward(parameters.map_{len(self.maps) - 1}, _intermediate_{len(self.maps) - 2}, _output);"
        )
        return "\n".join(forward_code)

    def _build_forward_technique_2(self) -> str:
        forward_code: list[str] = []
        buffer_manager = BufferManager(code_list=forward_code)
        buffer_namespace = "_intermediate"
        for i in range(len(self.maps) - 1):
            output_dim: int = self.maps[i].output_dim
            buffer_input = (
                f"_input"
                if i == 0
                else buffer_manager.get_last_requested_buffer(
                    namespace=buffer_namespace
                )
            )
            buffer_manager.lock_buffer_with_name(buffer_input)
            buffer_output_name = buffer_manager.get_free_buffer_with_dimension(
                namespace=buffer_namespace, dimension=output_dim
            )
            buffer_manager.free_buffer_with_name(name=buffer_input)
            forward_code.append(
                f"forward(parameters.map_{i}, {buffer_input}, {buffer_output_name});"
            )
        forward_code.append(
            f"forward(parameters.map_{len(self.maps) - 1}, {buffer_manager.get_last_requested_buffer(namespace=buffer_namespace)}, _output);"
        )
        return "\n".join(forward_code)

    def _get_last_checkpoint_index(self, actual_index: Optional[int] = None) -> int:
        """Returns the previous checkpoint index that precedes the `actual_index`"""
        # checkpoint_step_size = self.distance_between_checkpoints + 1
        assert self.checkpoint_indices != None, "Checkpoint_indices is None"

        if actual_index == None:
            if len(self.checkpoint_indices) == 0:
                return len(self.maps) - 1
            return self.checkpoint_indices[-1]
            # return ((len(self.maps) - 1) // checkpoint_step_size) * checkpoint_step_size

        index = -1
        for i in self.checkpoint_indices:
            if i >= actual_index:
                break
            index = i
        return index

    def _at_checkpoint(self, actual_index: int) -> bool:
        """Returns True if the actual_index correspond to a checkpoint"""
        assert (
            self.checkpoint_indices != None
        ), "Checkpoint_indices is None or an Empty List"
        return actual_index in self.checkpoint_indices

    def _create_checkpoint_buffers_code_decl(
        self,
        buffer_manager: BufferManager,
    ):
        """This method declares the arrays for storing all checkpoints"""
        # checkpoints: list[str] = []
        for i in self.checkpoint_indices:
            buffer_manager.create_checkpoint(
                checkpoint_number=i,
                namespace="checkpoint",
                checkpoint_dim=self.maps[i].output_dim,
            )

    def _fill_checkpoints_code_1(
        self,
        code: list[str],
        buffer_manager: BufferManager,
    ):
        # checkpoint_step_size: int = self.distance_between_checkpoints + 1
        # last_checkpoint_index: int = (
        #     len(self.maps) // checkpoint_step_size
        # ) * checkpoint_step_size
        if len(self.checkpoint_indices) == 0:
            for i in range(len(self.maps) - 1):
                code.append(f"float _intermediate_{i}[{self.maps[i].output_dim}];")
            return
        last_checkpoint_index: int = self._get_last_checkpoint_index()
        for i in range(last_checkpoint_index + 1):
            from_where = (
                "_input"
                if i == 0
                else (
                    # f"checkpoint_{i-1}"
                    buffer_manager.get_checkpoint(i - 1)
                    # if (i) % checkpoint_step_size == 0
                    if self._at_checkpoint(i - 1)
                    else f"_intermediate_{i-1}"
                )
            )
            to_where = (
                # f"checkpoint_{i}"
                buffer_manager.get_checkpoint(i)
                # if (i + 1) % checkpoint_step_size == 0
                if self._at_checkpoint(i)
                else f"_intermediate_{i}"
                # else buffer_manager.get_free_buffer_with_dimension(namespace="_intermediate", dimension=self.maps[i].output_dim)
            )
            # buffer_manager.lock_buffer_with_name(to_where)
            if self._at_checkpoint(i):
                code.append(f"forward(parameters.map_{i}, {from_where}, {to_where});")
            else:
                # Declare array
                code.append(f"float _intermediate_{i}[{self.maps[i].output_dim}];")
                code.append(f"forward(parameters.map_{i}, {from_where}, {to_where});")

        for i in range(last_checkpoint_index, len(self.maps)):
            code.append(f"float _intermediate_{i}[{self.maps[i].output_dim}];")

    def generate_forwards_and_backwards_between_checkpoints_1(
        self,
        map_index: int,
        code: list[str],
        buffer_manager: BufferManager,
    ):
        if map_index < 0:
            return -1
        previous_checkpoint_index = self._get_last_checkpoint_index(map_index)
        # if (map_index) % checkpoint_step_size == 0:  # if we are at a checkpoint
        if self._at_checkpoint(map_index - 1):
            return previous_checkpoint_index
        length = len(self.maps)

        buff_namespace = "_intermediate"
        # Forward part
        for i in range(previous_checkpoint_index + 1, map_index):
            from_where = (
                "_input"
                if i == 0
                else (
                    # f"checkpoint_{i-1}"
                    buffer_manager.get_checkpoint(i - 1)
                    # if (i) % checkpoint_step_size == 0
                    if self._at_checkpoint(i - 1)
                    else f"_intermediate_{i-1}"
                    # else buffer_manager.get_last_requested_buffer(
                    #     namespace=buff_namespace
                    # )
                )
            )
            to_where = (
                # f"checkpoint_{i}"
                buffer_manager.get_checkpoint(i)
                # if (i + 1) % checkpoint_step_size == 0
                if self._at_checkpoint(i)
                else f"_intermediate_{i}"
                # else buffer_manager.get_free_buffer_with_dimension(
                #     namespace=buff_namespace, dimension=self.maps[i].output_dim
                # )
            )
            buffer_manager.lock_buffer_with_name(to_where)
            code.append(f"forward(parameters.map_{i}, {from_where}, {to_where});")

        # Now the backwards in reverse order
        for i in range(map_index, previous_checkpoint_index, -1):
            forward_input = (
                f"_input"
                if i == 0
                else (
                    # f"checkpoint_{i-1}"
                    buffer_manager.get_checkpoint(i - 1)
                    # if (i) % checkpoint_step_size == 0
                    if self._at_checkpoint(i - 1)
                    else f"_intermediate_{i-1}"
                )
            )
            output_grad = "_output_grad" if i == length - 1 else f"_backgrad_{i}"
            input_grad = "_input_grad" if i == 0 else f"_backgrad_{i-1}"
            if i > 0:
                out_dim = self.maps[i - 1].output_dim
                code.append(f"float _backgrad_{i-1}[{out_dim}];")
                code.append(
                    f"[[unroll]] for(int i=0; i<{out_dim}; i++) _backgrad_{i-1}[i] = 0.0;"
                    # f"for(int i=0; i<{out_dim}; i++) _backgrad_{i-1}[i] = 0.0;"
                )
            code.append(
                # f"backward(parameters.map_{i}, {forward_output}, {output_grad}, {input_grad});"
                f"backward(parameters.map_{i}, {forward_input}, {output_grad}, {input_grad});"
            )
        return previous_checkpoint_index

    def _build_backward_checkpoints_technique_1(self) -> str:
        """Technique 1 is without reusing buffers names and checkpoints"""
        code: list[str] = []
        length = len(self.maps)
        actual_index = length - 1
        checkpoint_step_size = self.distance_between_checkpoints + 1

        # checkpoints_decl = self._create_checkpoint_buffers_code_decl()
        code.append("\n//Defining the checkpoints\n")
        buffer_manager: BufferManager = BufferManager(code_list=code)
        self._create_checkpoint_buffers_code_decl(buffer_manager=buffer_manager)
        code.append("\n//Filling the checkpoints\n")
        self._fill_checkpoints_code_1(code=code, buffer_manager=buffer_manager)
        code.append(f"\n//The backward code\n")

        while actual_index >= 0:
            # if (actual_index) % checkpoint_step_size != 0:
            if not self._at_checkpoint(actual_index - 1):
                next_index = self.generate_forwards_and_backwards_between_checkpoints_1(
                    actual_index,
                    code=code,
                    buffer_manager=buffer_manager,
                )
                actual_index = next_index
                continue

            forward_output = (
                "_input"
                if actual_index == 0
                else (
                    # f"checkpoint_{actual_index-1}"
                    buffer_manager.get_checkpoint(actual_index - 1)
                    # if (actual_index) % checkpoint_step_size == 0
                    if self._at_checkpoint(actual_index - 1)
                    else f"_intermediate_{actual_index-1}"
                )
            )
            output_grad = (
                "_output_grad"
                if actual_index == length - 1
                else f"_backgrad_{actual_index}"
            )
            input_grad = (
                "_input_grad" if actual_index == 0 else f"_backgrad_{actual_index-1}"
            )
            out_dim = self.maps[actual_index - 1].output_dim
            if actual_index > 0:
                code.append(f"float _backgrad_{actual_index - 1}[{out_dim}];")
                code.append(
                    f"[[unroll]] for(int i=0; i<{out_dim}; i++) _backgrad_{actual_index-1}[i] = 0.0;"
                    # f"for(int i=0; i<{out_dim}; i++) _backgrad_{actual_index-1}[i] = 0.0;"
                )
            code.append(
                f"backward(parameters.map_{actual_index}, {forward_output}, {output_grad}, {input_grad});"
            )

            next_index = self.generate_forwards_and_backwards_between_checkpoints_1(
                actual_index - 1,
                code=code,
                buffer_manager=buffer_manager,
            )
            # code.append(forwards[0])
            actual_index = next_index
        return "\n".join(code)

    def _fill_checkpoints_code_2(self, code: list[str], buffer_manager: BufferManager):
        """This method fills the checkpoints and uses a buffer manager
        for reutilizing previously created buffers"""

        # checkpoint_step_size: int = self.distance_between_checkpoints + 1
        # last_checkpoint_index: int = (
        #     len(self.maps) // checkpoint_step_size
        # ) * checkpoint_step_size
        if len(self.checkpoint_indices) == 0:
            return

        last_checkpoint_index: int = self._get_last_checkpoint_index()

        buffer_namespace = "_intermediate"
        # checkpoint_namespace = "checkpoint"
        for i in range(last_checkpoint_index + 1):
            # from_checkpoint = i % checkpoint_step_size == 0
            from_checkpoint = self._at_checkpoint(i - 1)
            # to_checkpoint = (i + 1) % checkpoint_step_size == 0
            to_checkpoint = self._at_checkpoint(i)

            if to_checkpoint:
                code.append(f"// Creation of checkpoint")
            from_where = (
                "_input"
                if i == 0
                else (
                    # f"checkpoint_{i-1}"
                    buffer_manager.get_checkpoint(checkpoint_number=i - 1)
                    if from_checkpoint
                    else buffer_manager.get_last_requested_buffer(
                        namespace=buffer_namespace
                    )
                )
            )
            buffer_manager.lock_buffer_with_name(name=from_where)
            to_where = (
                buffer_manager.create_checkpoint(
                    checkpoint_number=i,
                    namespace=buffer_namespace,
                    checkpoint_dim=self.maps[i].output_dim,
                )
                if to_checkpoint
                else buffer_manager.get_free_buffer_with_dimension(
                    namespace=buffer_namespace,
                    dimension=self.maps[i].output_dim,
                )
            )
            if not from_checkpoint:
                buffer_manager.free_buffer_with_name(name=from_where)
            # if to_checkpoint:
            # buffer_manager.lock_buffer_with_name(name=to_where)
            # buffer_manager.set_last_requested_buffer(
            #     namespace=checkpoint_namespace, buffer_name=to_where
            # )
            code.append(f"forward(parameters.map_{i}, {from_where}, {to_where});")

        # return "\n".join(code)

    def generate_buffers(
        self,
        buffer_manager: BufferManager,
        namespace: str,
        start_index: int,
        end_index: int,
    ) -> list[str]:
        res: list[str] = []
        for i in range(start_index, end_index):
            actual_map = self.maps[i]
            buff = buffer_manager.get_free_buffer_with_dimension(
                namespace=namespace, dimension=actual_map.output_dim
            )
            buffer_manager.lock_buffer_with_name(buff)
            res.append(buff)
        return res

    def generate_forwards_and_backwards_between_checkpoints_2(
        self,
        actual_index: int,
        code: list[str],
        buffer_manager: BufferManager,
    ) -> int:
        if actual_index < 0:
            return -1
        checkpoint_step_size = self.distance_between_checkpoints + 1
        previous_checkpoint_index = actual_index - (actual_index % checkpoint_step_size)
        if (actual_index) % checkpoint_step_size == 0:  # If we are a checkpoint
            return previous_checkpoint_index
        length = len(self.maps)

        intermediate_namespace = "_intermediate"
        grad_namespace = "_grad"
        received_grad = buffer_manager.get_last_requested_buffer(
            namespace=grad_namespace
        )
        buffer_manager.lock_buffer_with_name(received_grad)

        intermediate_buffer_names: list[str] = self.generate_buffers(
            buffer_manager=buffer_manager,
            namespace="_intermediate",
            start_index=previous_checkpoint_index,
            end_index=actual_index,
        )

        # Forward part
        for index, map_index in enumerate(
            range(previous_checkpoint_index, actual_index)
        ):
            from_where = (
                "_input"
                if map_index == 0
                else (
                    buffer_manager.get_checkpoint(checkpoint_number=map_index - 1)
                    if (map_index) % checkpoint_step_size == 0
                    else intermediate_buffer_names[index - 1]
                )
            )
            assert (
                from_where != None
            ), "from_were is None in the `generate_forwards_and_backwards_between_checkpoints_2`"

            to_where = intermediate_buffer_names[index]

            code.append(
                f"forward(parameters.map_{map_index}, {from_where}, {to_where});"
            )

        # Now the backwards in reverse order
        for index, map_index in enumerate(
            range(actual_index, previous_checkpoint_index, -1)
        ):
            previous_map = self.maps[map_index - 1]
            forward_input = (
                f"_input"
                if map_index == 0
                else (
                    # f"checkpoint_{map_index-1}"
                    buffer_manager.get_checkpoint(checkpoint_number=map_index - 1)
                    if (map_index) % checkpoint_step_size == 0
                    else intermediate_buffer_names[-1 - index]
                )
            )
            output_grad = (
                "_output_grad"
                if map_index == length - 1
                else (
                    # buffer_manager.get_last_requested_buffer(namespace="_grad")
                    received_grad
                    if index == 0
                    # else gradient_buffer_names[-1 - (index - 1)]
                    else buffer_manager.get_last_requested_buffer(
                        namespace=grad_namespace
                    )
                )
            )
            buffer_manager.lock_buffer_with_name(output_grad)
            input_grad = (
                "_input_grad"
                if map_index == 0
                else buffer_manager.get_free_buffer_with_dimension(
                    namespace=intermediate_namespace,
                    dimension=previous_map.output_dim,
                    update_last_req_buff=False,
                )
            )
            buffer_manager.free_buffer_with_name(output_grad)
            buffer_manager.set_last_requested_buffer(
                namespace=grad_namespace, buffer_name=input_grad
            )
            if map_index > 0:
                out_dim = self.maps[map_index - 1].output_dim
                code.append(
                    # f"[[unroll]] for(int i=0; i<{out_dim}; i++) {gradient_buffer_names[-1-index]}[i] = 0.0;"
                    f"[[unroll]] for(int i=0; i<{out_dim}; i++) {input_grad}[i] = 0.0;"
                )
            code.append(
                # f"backward(parameters.map_{i}, {forward_output}, {output_grad}, {input_grad});"
                f"backward(parameters.map_{map_index}, {forward_input}, {output_grad}, {input_grad});"
            )
            buffer_manager.set_last_requested_buffer(
                namespace="_grad", buffer_name=input_grad
            )
            buffer_manager.free_buffer_with_name(forward_input)
            # buffer_manager.free_buffer_with_name()

        # for b_name in intermediate_buffer_names:
        #     buffer_manager.free_buffer_with_name(b_name)
        # for b_name in gradient_buffer_names:
        #     buffer_manager.free_buffer_with_name(b_name)
        return previous_checkpoint_index

    def _build_backward_checkpoints_technique_2(self) -> str:
        """Technique 2 is reusing buffer names when possible.
        This technique will be compared with the number 1 to see which one
        perform the better"""
        code: list[str] = []
        buffer_manager = BufferManager(code_list=code)

        # self._create_checkpoint_buffers_code_decl(code=code)
        code.append(f"\n// Now filling the checkpoints\n")
        self._fill_checkpoints_code_2(code=code, buffer_manager=buffer_manager)

        code.append(f"\n// Now the backward part\n")
        checkpoint_step_size = self.distance_between_checkpoints + 1

        actual_index: int = len(self.maps) - 1
        length = len(self.maps)

        intermediate_namespace = "_intermediate"
        # grad_namespace = "_grad"

        while actual_index >= 0:
            if actual_index % checkpoint_step_size != 0:
                next_index = self.generate_forwards_and_backwards_between_checkpoints_2(
                    actual_index=actual_index,
                    code=code,
                    buffer_manager=buffer_manager,
                )
                actual_index = next_index
                continue
            previous_map = self.maps[actual_index - 1]
            forward_input = (
                # f"_input" if actual_index == 0 else (f"checkpoint_{actual_index-1}")
                f"_input"
                if actual_index == 0
                else buffer_manager.get_checkpoint(actual_index - 1)
            )
            buffer_manager.lock_buffer_with_name(forward_input)

            output_grad = (
                "_output_grad"
                if actual_index == length - 1
                else (buffer_manager.get_last_requested_buffer(namespace="_grad"))
            )
            buffer_manager.lock_buffer_with_name(output_grad)
            input_grad = (
                "_input_grad"
                if actual_index == 0
                else buffer_manager.get_free_buffer_with_dimension(
                    # namespace="_grad",
                    namespace=intermediate_namespace,
                    dimension=previous_map.output_dim,
                )
            )
            buffer_manager.free_buffer_with_name(output_grad)
            if actual_index > 0:
                code.append(
                    # f"[[unroll]] for(int i=0; i<{previous_map.output_dim}; i++) {input_grad}[i] = 0.0;"
                    f"[[unroll]] for(int i=0; i<{previous_map.output_dim}; i++) {input_grad}[i] = 0.0;"
                )
            code.append(
                # f"backward(parameters.map_{i}, {forward_output}, {output_grad}, {input_grad});"
                f"backward(parameters.map_{actual_index}, {forward_input}, {output_grad}, {input_grad});"
            )

            buffer_manager.set_last_requested_buffer("_grad", input_grad)
            buffer_manager.free_buffer_with_name(name=forward_input)

            next_index = self.generate_forwards_and_backwards_between_checkpoints_2(
                actual_index=actual_index - 1,
                code=code,
                buffer_manager=buffer_manager,
            )
            actual_index = next_index

        return "\n".join(code)

    def _generate_forwards_for_3(
        self,
        start_index: int,
        end_index: int,
        code: list[str],
        buffer_manager: BufferManager,
    ) -> str:
        """
        This method generates the code for computing the forwards between 2 maps
        given by start_index and end_index (inclusive). Then returns the name of the buffer containing the output
        of the last forward
        """
        buffer_namespace = "_intermediate"
        checkpoint_step_size = self.distance_between_checkpoints + 1

        for i in range(start_index, end_index + 1):
            actual_map = self.maps[i]
            # from_checkpoint = (i) % checkpoint_step_size == 0
            from_checkpoint = self._at_checkpoint(i - 1)
            from_where = (
                "_input"
                if i == 0
                else (
                    buffer_manager.get_checkpoint(checkpoint_number=i - 1)
                    if from_checkpoint
                    # else intermediate_buffer_names[index - 1]
                    else buffer_manager.get_last_requested_buffer(
                        namespace=buffer_namespace
                    )
                )
            )
            buffer_manager.lock_buffer_with_name(from_where)
            assert (
                from_where != None
            ), "from_were is None in the `_generate_forwards_for_3`"

            to_where = buffer_manager.get_free_buffer_with_dimension(
                namespace=buffer_namespace, dimension=actual_map.output_dim
            )
            buffer_manager.set_last_requested_buffer(
                namespace=buffer_namespace, buffer_name=to_where
            )
            if not from_checkpoint:
                buffer_manager.free_buffer_with_name(from_where)

            code.append(f"forward(parameters.map_{i}, {from_where}, {to_where});")

        return buffer_manager.get_last_requested_buffer(namespace=buffer_namespace)

    def generate_forwards_and_backwards_between_checkpoints_3(
        self,
        actual_index: int,
        code: list[str],
        buffer_manager: BufferManager,
    ) -> int:
        if actual_index < 0:
            return -1
        checkpoint_step_size = self.distance_between_checkpoints + 1
        # previous_checkpoint_index = actual_index - (actual_index % checkpoint_step_size)
        previous_checkpoint_index = self._get_last_checkpoint_index(
            actual_index=actual_index
        )  # Could be -1 if there is no checkpoint, could be previous index

        length = len(self.maps)

        intermediate_namespace = "_intermediate"
        grad_namespace = "_grad"
        received_grad = buffer_manager.get_last_requested_buffer(
            namespace=grad_namespace
        )
        buffer_manager.lock_buffer_with_name(received_grad)

        # The backward code
        for index, map_index in enumerate(
            range(actual_index, previous_checkpoint_index, -1)
        ):
            previous_map = self.maps[map_index - 1]
            forward_input = (
                f"_input"
                if map_index == 0
                else (
                    buffer_manager.get_checkpoint(checkpoint_number=map_index - 1)
                    if self._at_checkpoint(map_index - 1)
                    else self._generate_forwards_for_3(
                        start_index=previous_checkpoint_index + 1,
                        end_index=map_index - 1,
                        code=code,
                        buffer_manager=buffer_manager,
                    )
                )
            )
            buffer_manager.lock_buffer_with_name(forward_input)
            output_grad = (
                "_output_grad"
                if map_index == length - 1
                else (
                    # buffer_manager.get_last_requested_buffer(namespace="_grad")
                    received_grad
                    if index == 0
                    # else gradient_buffer_names[-1 - (index - 1)]
                    else buffer_manager.get_last_requested_buffer(
                        namespace=grad_namespace
                    )
                )
            )
            buffer_manager.lock_buffer_with_name(output_grad)
            input_grad = (
                "_input_grad"
                if map_index == 0
                else buffer_manager.get_free_buffer_with_dimension(
                    namespace=intermediate_namespace,
                    dimension=previous_map.output_dim,
                    update_last_req_buff=False,
                )
            )
            buffer_manager.free_buffer_with_name(output_grad)
            buffer_manager.set_last_requested_buffer(
                namespace=grad_namespace, buffer_name=input_grad
            )
            if map_index > 0:
                out_dim = self.maps[map_index - 1].output_dim
                code.append(
                    # f"[[unroll]] for(int i=0; i<{out_dim}; i++) {gradient_buffer_names[-1-index]}[i] = 0.0;"
                    f"[[unroll]] for(int i=0; i<{out_dim}; i++) {input_grad}[i] = 0.0;"
                )
            code.append(
                # f"backward(parameters.map_{i}, {forward_output}, {output_grad}, {input_grad});"
                f"backward(parameters.map_{map_index}, {forward_input}, {output_grad}, {input_grad});"
            )
            buffer_manager.lock_buffer_with_name(name=input_grad)
            buffer_manager.set_last_requested_buffer(
                namespace="_grad", buffer_name=input_grad
            )
            buffer_manager.free_buffer_with_name(forward_input)

        return previous_checkpoint_index

    def _build_backward_checkpoints_technique_3(self) -> str:
        code: list[str] = []
        buffer_manager = BufferManager(code_list=code)

        code.append(f"\n// Now filling the checkpoints\n")
        self._fill_checkpoints_code_2(code=code, buffer_manager=buffer_manager)

        code.append(f"\n// Now the backward part\n")

        actual_index: int = len(self.maps) - 1

        while actual_index >= 0:
            next_index = self.generate_forwards_and_backwards_between_checkpoints_3(
                actual_index=actual_index,
                code=code,
                buffer_manager=buffer_manager,
            )
            actual_index = next_index
        return "\n".join(code)

    def get_forward_code(self) -> str:
        """Get the forward code"""
        match self.forward_tech:
            case ForwardTechnique.technique_1:
                return self._build_forward_technique_1()
            case ForwardTechnique.technique_2:
                return self._build_forward_technique_2()
            case _:
                raise Exception("No valid forward technique")

    def get_backward_code(self) -> str:
        """Get the backward code"""
        print(f"Backward technique is => {self.backward_tech.name}")
        match self.backward_tech:
            case BackwardTechnique.technique_1:
                return self._build_backward_checkpoints_technique_1()
            case BackwardTechnique.technique_2:
                return self._build_backward_checkpoints_technique_2()
            case BackwardTechnique.technique_3:
                return self._build_backward_checkpoints_technique_3()
            case _:
                raise Exception("No valid backward technique")

    def build(self):

        parameters = dict()
        for i, _ in enumerate(self.maps):
            parameters[f"map_{i}"] = _maps.MapBase

        forward_code: str = self.get_forward_code()
        backward_code: str = self.get_backward_code()

        code = f"""
FORWARD {{
    {forward_code}
}}

BACKWARD {{
    {backward_code}
}}
        """

        print("=============== Sequential Code =================")
        print(code)
        print("=================================================")

        class SequentialMap(_maps.MapBase):
            __extension_info__ = dict(
                code=code,
                parameters=parameters,
                bw_implementations=_maps.BACKWARD_IMPLEMENTATIONS.DEFAULT,
                # INPUT_DIM=self.maps[0].input_dim,
                # OUTPUT_DIM=self.maps[-1].output_dim,
            )

            def __init__(self, *maps):
                super().__init__(
                    INPUT_DIM=maps[0].input_dim, OUTPUT_DIM=list(maps)[-1].output_dim
                )
                for i in range(len(maps)):
                    setattr(self, f"map_{i}", maps[i])

        sequential_map = SequentialMap(*self.maps)
        return sequential_map
        # Crear el Map type
        # instantiate the map
