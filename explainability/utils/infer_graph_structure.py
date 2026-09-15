import numpy as np

from tensorflow.keras import Model


def _infer_via_inbound_nodes(model: Model) -> np.ndarray:
    """Layer connectivity from Keras inbound nodes (works in eager mode)."""
    layers = list(model.layers)
    n = len(layers)
    by_id = {id(layer): i for i, layer in enumerate(layers)}
    by_name = {layer.name: i for i, layer in enumerate(layers)}
    neighbours = np.zeros((n, n), dtype=np.uint8)

    for j, layer in enumerate(layers):
        for node in getattr(layer, "_inbound_nodes", []) or []:
            inbound = getattr(node, "inbound_layers", None)
            if inbound is None:
                continue
            if not isinstance(inbound, (list, tuple)):
                inbound = [inbound]
            for src in inbound:
                if src is None:
                    continue
                i = by_id.get(id(src))
                if i is None:
                    i = by_name.get(getattr(src, "name", None))
                if i is not None and i != j:
                    neighbours[i, j] = 1
    return neighbours


def _tensor_key(tensor):
    """Stable identity; ``Tensor.name`` may be unavailable in eager mode."""
    try:
        name = tensor.name
    except (AttributeError, ValueError):
        return ("id", id(tensor))
    if name is None:
        return ("id", id(tensor))
    return ("name", name)


tensor_key = _tensor_key


def _is_neighbour(outputs, inputs) -> bool:
    outputs = outputs if isinstance(outputs, list) else [outputs]
    inputs = inputs if isinstance(inputs, list) else [inputs]
    out_keys = {_tensor_key(output) for output in outputs}
    in_keys = {_tensor_key(inp) for inp in inputs}
    return len(out_keys & in_keys) > 0


def _infer_via_tensor_names(model: Model) -> np.ndarray:
    inputs = [layer.input for layer in model.layers]
    outputs = [layer.output for layer in model.layers]
    return np.asarray(
        [
            [
                _is_neighbour(outputs[i], inputs[j]) if i != j else False
                for j in range(len(inputs))
            ]
            for i in range(len(inputs))
        ]
    ).astype(np.uint8)


def infer_graph_structure(model: Model) -> np.ndarray:
    """Adjacency matrix: neighbours[i, j] == 1 iff layer i feeds layer j.

    Prefer Keras inbound-node wiring. It remains valid when some layer inputs
    are eager ``Tensor``s without ``.name`` (e.g. constants in an Add used by
    SFCN ``restrict_range``), which breaks classic tensor-name matching.
    """
    via_nodes = _infer_via_inbound_nodes(model)
    if via_nodes.shape[0] <= 1 or int(via_nodes.sum()) > 0:
        return via_nodes

    # Rare fallback for exotic model types without inbound metadata.
    return _infer_via_tensor_names(model)
