"""Node implementations.

Add your own node classes to this package and export them here, so that
``pytest --node YourClass`` and ``sim_send_many --node YourClass`` can find
them by class name alone.
"""

import importlib

from .node_multi_leader import NodeMultiLeader
from .node_total_order_eager_broadcast import NodeTotalOrderEagerBroadcast
from .node_total_order_gossip import NodeTotalOrderGossip

__all__ = ["NodeMultiLeader", "NodeTotalOrderEagerBroadcast", "NodeTotalOrderGossip", "load_node_class"]


def load_node_class(spec: str) -> type:
    """Resolve a node class from a command-line spec.

    Accepted forms:

    - ``NodeMultiLeader``: a class exported from ``dslabs.nodes``
    - ``some.module:ClassName``: an explicit module and class
    - ``some.module.ClassName``: the same, with a dot
    """
    if ":" in spec:
        module_name, class_name = spec.split(":", 1)
    elif "." in spec:
        module_name, class_name = spec.rsplit(".", 1)
    else:
        module_name, class_name = __name__, spec
    module = importlib.import_module(module_name)
    try:
        return getattr(module, class_name)
    except AttributeError:
        raise ImportError(f"module {module_name!r} has no class {class_name!r}") from None
