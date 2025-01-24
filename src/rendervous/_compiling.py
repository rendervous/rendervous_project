from . import _internal
from . import _maps


class Node:
    def build(self) -> _maps.MapBase:
        pass


class MapNode(Node):
    def __init__(self, map: _maps.MapBase):
        pass


class Sequential (Node):
    def __init__(self, *layers: Node):
        self.layers = layers

    def build(self):
        maps  = [l.build() for l in self.layers]
        parameters = dict()
        for i, m in maps:
            parameters[f'map_{i}'] = _maps.MapBase
        code = f"""
        FORWARD {{
            float _intermediate_0[{maps[0].output_dim}];
            forward(parameters.map_0, _input, _intermediate_0);
            //..
        }}
        """

        def map_init(_self, *maps):
            super(_self).__init__()
            for i in range(len(maps)):
                setattr(_self, f'map_{i}', maps[i])

        class SequentialMap(_maps.MapBase):
            __extension_info__ = dict(
                code = code,
                parameters=parameters,
                bw_implementations=_maps.BACKWARD_IMPLEMENTATIONS.NONE,
            )
            def __init__(self, *maps):
                super().__init__()
                for i in range(len(maps)):
                    setattr(self, f'map_{i}', maps[i])

        # map_type = MapMeta('custom_map_id', [], dict(
        #     __extension_info__ = dict(
        #         code = code,
        #         parameters=parameters,
        #         bw_implementations=BACKWARD_IMPLEMENTATIONS.NONE,
        #     ),
        #     __init__ = map_init
        # ))

        sequential_map = SequentialMap(*maps)
        return sequential_map
        # Crear el Map type
        # instantiate the map


# s = Sequential(MapNode(m1), MapNode(m2))
#
# map = s.build()

