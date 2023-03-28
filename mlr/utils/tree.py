from abc import ABC, abstractmethod
from functools import reduce
import numpy as np

"""
Implementation of a tree used for searching neighborhoods of atoms in atomic graphs,

with the following constraints:
    1) All nodes in the tree are unique
    2) Nodes can have associated values or vectors

and the following use case:
    to return all atoms within X distance of a target atom
"""

"""Data are ultimately stored in arrays



"""

def hashable(obj): # https://stackoverflow.com/questions/66005005/how-to-judge-an-object-is-hashable-or-unhashable-in-python
    try:
        hash(obj)
        return True
    except Exception:
        return False
    
class UNode(ABC):

    def __init__(self, identity, value=None, parent=None):
        self._first_child = -1
        self._last_child = -1
        self._parent = -1
        self._next_neighbor = -1
        self.identity = identity
        self.value = value

        # allow initialization of root nodes with an empty init call
        if not parent is None:
            self.parent = parent

    @property
    def identity(self):
        return self._identity

    @identity.setter
    def identity(self, val):
        assert(hashable(val)), "Identity of node is not a hashable type"
        self._identity = val

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, idx):
        assert(isinstance(idx, int)), "Index not an integer"
        assert(idx > -1), "Parent not valid (index less than zero)"
        self._parent = idx

    @property
    def first_child(self):
        return self._first_child

    @first_child.setter
    def first_child(self, idx):
        """Return first child added to node"""
        assert(isinstance(idx, int)), "Index not an integer"
        assert(idx > 0), "Child not valid (index less than or equal to zero)"
        self.first_child = idx

    @property
    def last_child(self):
        return self._last_child

    @last_child.setter
    def last_child(self, idx):
        """Return last child added to node"""
        assert(isinstance(idx, int)), "Index not an integer"
        assert(idx > 0), "Child not valid (index less than or equal to zero)"
        self.last_child = idx

        if self.first_child == -1:
            self.first_child = idx

    @property
    def next_neighbor(self):
        return self._next_neighbor

    @next_neighbor.setter
    def next_neighbor(self, idx):
        """Return first child added to node"""
        assert(isinstance(idx, int)), "Index not an integer"
        assert(idx > 0), "Neighbor not valid (index less than or equal to zero)"
        self.next_neighbor = idx

    @property
    def is_leaf(self):
        if self.first_child == -1:
            return True
        else:
            return False

    @property
    def is_root(self):
        if self.parent == -1:
            return True
        else:
            return False

    @property
    def value(self):
        return self._value
    
    @value.setter
    def value(self, val):
        # No protections for the default setter
        self._value = val

    def __repr__(self):
        return f"UNode({self.identity}, {self.parent}, {self.next_neighbor}, {self.first_child}, {self.value})"

class UNodeArray(ABC):

    def __init__(self, init_size=64, realloc_size=64):
        self._arr = np.empty((init_size,), dtype=object)
        self._rsize = realloc_size

    def __getitem__(self, key):
        if key < 0:
            pass
        else:
            return self._arr[key]

    def __setitem__(self, key, value):
        if key < 0:
            pass
        else:
            if key >= self._arr.shape[0]:
                # increase array allocation size
                arr_copy = np.empty((self._arr.shape[0] + self._rsize,))
                arr_copy[:self._arr.shape[0]] = self._arr
                self._arr = arr_copy

            self._arr[key] = value

    # likely only need to protect setter for next neighbor, during construction of tree
    # getters aren't implemented, but maybe need to

    def set_parent(self, key, idx):
        # if key > -1:
        self[key].parent = idx

    def set_last_child(self, key, idx):
        # if key > -1:
        self[key].last_child = idx

    def set_next_neighbor(self, key, idx):
        if key > -1: # need to check in the instance that the parent has no children when a node is added
            self[key].next_neighbor = idx

    def set_value(self, key, val):
        # used only for clearing the root node when necessary
        self[key].value = val

class UTree(ABC):

    def __init__(self, root_node, init_size=64, realloc_size=64):
        self.node_type = root_node.__class__()
        self.node_filter = lambda x: True
        self._nodes = UNodeArray(init_size=init_size, realloc_size=realloc_size)

        # Initialize root node separately
        self._nodes[0] = root_node
        self._N = 1
        self._node_set = set([root_node.identity])

    def add_node(self, parent, new_node):
        assert(isinstance(new_node, self.node_type)), "Node is not the same type as root node"

        if self.node_exists(new_node):
            pass
        else:
            new_node.value = self.update_value(parent, new_node)

            if self.node_filter(new_node):
                self._node_set.update(new_node.identity)

                self.nodes[self.N] = new_node
                self.nodes.set_parent(self.N, parent)
                self.nodes.set_next_neighbor(self.nodes[parent].last_child, self.N)
                self.nodes.set_last_child(parent, self.N)

                self._N += 1

    def update_value(self, parent, new_node):
        return new_node.value

    def node_exists(self, new_node):
        return new_node.identity in self.node_set

    def node_valid(self, new_node):
        return self.node_filter(new_node)
    
    @property
    def nodes(self):
        return self._nodes
    
    @nodes.setter
    def nodes(self, val):
        assert(isinstance(val, UNodeArray)), "Node storage must be a UNodeArray"
        self._nodes = val

    @property
    def node_set(self):
        return self._node_set

    @property
    def N(self):
        """Number of nodes in the tree."""
        return self._N

    @property
    def node_type(self):
        return self._node_type

    @node_type.setter
    def node_type(self, node_class):
        assert(isinstance(node_class, UNode)), "UTrees must be constructed with UNode objects or subclasses"
        self._node_type = node_class
    
    @property
    def node_filter(self):
        """Filter function to check acceptance of nodes"""
        return self._node_filter

    @node_filter.setter
    def node_filter(self, f):
        assert(isinstance(f, callable)), "Filter function is not callable"
        assert(isinstance(f(self.node_type), bool)), "Filter function does not return bool for given node archetype"
        self._node_filter = f

    def __repr__(self):
        return reduce(lambda x, y: x+y, [r.__repr__() + "\n" for r in self.nodes[:self.N]])

class CNode(UNode):
    """Node with 4-length arrays as stored value, representing ||r|| and r_x, r_y, r_z,

       to be used for clustering
    """

    def __init__(self, identity, value, parent=None):
        super().__init_(identity, value, parent)

    @UNode.value.setter
    def value(self, val):
        assert(isinstance(val, np.ndarray)), "Value must be numpy nd.array"
        assert(val.shape == (3,)), "Input value must be array of shape (3,)"
        self._value = np.concatenate([np.linalg.norm(val)[None], val])

class CTree(UTree):

    def __init__(self, root_node, cluster_radius, init_size=64, realloc_size=64):
        super().__init__(root_node, init_size=init_size, realloc_size=realloc_size)
        self.cluster_radius = cluster_radius
        self.node_filter = lambda x: np.linalg.norm(x.value[1:]) <= cluster_radius

        # clear the distance and vector for the root node, if necessary
        self.nodes.set_value(0, np.zeros(3,))

    @property
    def cluster_radius(self):
        return self._cluster_radius
    
    @cluster_radius.setter
    def cluster_radius(self, r):
        assert(r > 0.0), "Radius must be positively valued"
        self._cluster_radius = r

    @UTree.node_type.setter
    def node_type(self, node_class):
        assert(isinstance(node_class, CNode)), "CTrees must be constructed with CNode objects or subclasses"
        self._node_type = node_class

    def update_value(self, parent, new_node):
        r_root_to_node = self.nodes[parent].value[1:] + new_node.value[1:] # calculate R3 vector to new leaf
        return np.concatenate([np.linalg.norm(r_root_to_node)[None], r_root_to_node])

def test():


    pass

if __name__ == "__main__":
    test()