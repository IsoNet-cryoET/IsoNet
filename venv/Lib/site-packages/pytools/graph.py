from __future__ import division, absolute_import, print_function

__copyright__ = """
Copyright (C) 2009-2013 Andreas Kloeckner
Copyright (C) 2020 Matt Wala
Copyright (C) 2020 James Stevens
"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""


__doc__ = """
Graph Algorithms
=========================

.. autofunction:: a_star
.. autofunction:: compute_sccs
.. autoclass:: CycleError
.. autofunction:: compute_topological_order
.. autofunction:: compute_transitive_closure
.. autofunction:: contains_cycle
.. autofunction:: compute_induced_subgraph
"""


# {{{ a_star

def a_star(  # pylint: disable=too-many-locals
        initial_state, goal_state, neighbor_map,
        estimate_remaining_cost=None,
        get_step_cost=lambda x, y: 1
        ):
    """
    With the default cost and heuristic, this amounts to Dijkstra's algorithm.
    """

    from heapq import heappop, heappush

    if estimate_remaining_cost is None:
        def estimate_remaining_cost(x):  # pylint: disable=function-redefined
            if x != goal_state:
                return 1
            else:
                return 0

    class AStarNode(object):
        __slots__ = ["state", "parent", "path_cost"]

        def __init__(self, state, parent, path_cost):
            self.state = state
            self.parent = parent
            self.path_cost = path_cost

    inf = float("inf")
    init_remcost = estimate_remaining_cost(initial_state)
    assert init_remcost != inf

    queue = [(init_remcost, AStarNode(initial_state, parent=None, path_cost=0))]
    visited_states = set()

    while queue:
        _, top = heappop(queue)
        visited_states.add(top.state)

        if top.state == goal_state:
            result = []
            it = top
            while it is not None:
                result.append(it.state)
                it = it.parent
            return result[::-1]

        for state in neighbor_map[top.state]:
            if state in visited_states:
                continue

            remaining_cost = estimate_remaining_cost(state)
            if remaining_cost == inf:
                continue
            step_cost = get_step_cost(top, state)

            estimated_path_cost = top.path_cost+step_cost+remaining_cost
            heappush(queue,
                (estimated_path_cost,
                    AStarNode(state, top, path_cost=top.path_cost + step_cost)))

    raise RuntimeError("no solution")

# }}}


# {{{ compute SCCs with Tarjan's algorithm

def compute_sccs(graph):
    to_search = set(graph.keys())
    visit_order = {}
    scc_root = {}
    sccs = []

    while to_search:
        top = next(iter(to_search))
        call_stack = [(top, iter(graph[top]), None)]
        visit_stack = []
        visiting = set()

        scc = []

        while call_stack:
            top, children, last_popped_child = call_stack.pop()

            if top not in visiting:
                # Unvisited: mark as visited, initialize SCC root.
                count = len(visit_order)
                visit_stack.append(top)
                visit_order[top] = count
                scc_root[top] = count
                visiting.add(top)
                to_search.discard(top)

            # Returned from a recursion, update SCC.
            if last_popped_child is not None:
                scc_root[top] = min(
                    scc_root[top],
                    scc_root[last_popped_child])

            for child in children:
                if child not in visit_order:
                    # Recurse.
                    call_stack.append((top, children, child))
                    call_stack.append((child, iter(graph[child]), None))
                    break
                if child in visiting:
                    scc_root[top] = min(
                        scc_root[top],
                        visit_order[child])
            else:
                if scc_root[top] == visit_order[top]:
                    scc = []
                    while visit_stack[-1] != top:
                        scc.append(visit_stack.pop())
                    scc.append(visit_stack.pop())
                    for item in scc:
                        visiting.remove(item)
                    sccs.append(scc)

    return sccs

# }}}


# {{{ compute topological order

class CycleError(Exception):
    """
    Raised when a topological ordering cannot be computed due to a cycle.

    :attr node: Node in a directed graph that is part of a cycle.
    """
    def __init__(self, node):
        self.node = node


def compute_topological_order(graph):
    """Compute a toplogical order of nodes in a directed graph.

    :arg graph: A :class:`collections.abc.Mapping` representing a directed
        graph. The dictionary contains one key representing each node in the
        graph, and this key maps to a :class:`collections.abc.Iterable` of
        nodes that are connected to the node by outgoing edges.

    :returns: A :class:`list` representing a valid topological ordering of the
        nodes in the directed graph.

    .. versionadded:: 2020.2
    """

    # find a valid ordering of graph nodes
    reverse_order = []
    visited = set()
    visiting = set()

    # go through each node
    for root in graph:

        if root in visited:
            # already encountered root as someone else's child
            # and processed it at that time
            continue

        stack = [(root, iter(graph[root]))]
        visiting.add(root)

        while stack:
            node, children = stack.pop()

            for child in children:
                # note: each iteration removes child from children
                if child in visiting:
                    raise CycleError(child)

                if child in visited:
                    continue

                visiting.add(child)

                # put (node, remaining children) back on stack
                stack.append((node, children))

                # put (child, grandchildren) on stack
                stack.append((child, iter(graph.get(child, ()))))
                break
            else:
                # loop did not break,
                # so either this is a leaf or all children have been visited
                visiting.remove(node)
                visited.add(node)
                reverse_order.append(node)

    return list(reversed(reverse_order))

# }}}


# {{{ compute transitive closure

def compute_transitive_closure(graph):
    """Compute the transitive closure of a directed graph using Warshall's
        algorithm.

    :arg graph: A :class:`collections.abc.Mapping` representing a directed
        graph. The dictionary contains one key representing each node in the
        graph, and this key maps to a :class:`collections.abc.MutableSet` of
        nodes that are connected to the node by outgoing edges. This graph may
        contain cycles. This object must be picklable. Every graph node must
        be included as a key in the graph.

    :returns: The transitive closure of the graph, represented using the same
        data type.

    .. versionadded:: 2020.2
    """
    # Warshall's algorithm

    from copy import deepcopy
    closure = deepcopy(graph)

    # (assumes all graph nodes are included in keys)
    for k in graph.keys():
        for n1 in graph.keys():
            for n2 in graph.keys():
                if k in closure[n1] and n2 in closure[k]:
                    closure[n1].add(n2)

    return closure

# }}}


# {{{ check for cycle

def contains_cycle(graph):
    """Determine whether a graph contains a cycle.

    :arg graph: A :class:`collections.abc.Mapping` representing a directed
        graph. The dictionary contains one key representing each node in the
        graph, and this key maps to a :class:`collections.abc.Iterable` of
        nodes that are connected to the node by outgoing edges.

    :returns: A :class:`bool` indicating whether the graph contains a cycle.

    .. versionadded:: 2020.2
    """

    try:
        compute_topological_order(graph)
        return False
    except CycleError:
        return True

# }}}


# {{{ compute induced subgraph

def compute_induced_subgraph(graph, subgraph_nodes):
    """Compute the induced subgraph formed by a subset of the vertices in a
        graph.

    :arg graph: A :class:`collections.abc.Mapping` representing a directed
        graph. The dictionary contains one key representing each node in the
        graph, and this key maps to a :class:`collections.abc.Set` of nodes
        that are connected to the node by outgoing edges.

    :arg subgraph_nodes: A :class:`collections.abc.Set` containing a subset of
        the graph nodes in the graph.

    :returns: A :class:`dict` representing the induced subgraph formed by
        the subset of the vertices included in `subgraph_nodes`.

    .. versionadded:: 2020.2
    """

    new_graph = {}
    for node, children in graph.items():
        if node in subgraph_nodes:
            new_graph[node] = children & subgraph_nodes
    return new_graph

# }}}

# vim: foldmethod=marker
