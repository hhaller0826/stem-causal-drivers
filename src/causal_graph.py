import itertools

# From NCM Counterfactuals
class CausalGraph:
    def __init__(self, V, directed_edges=[], bidirected_edges=[]):
        self.de = directed_edges
        self.be = bidirected_edges

        self.v = list(V)
        self.set_v = set(V)
        self.pa = {v: set() for v in V}  # parents (directed edges)
        self.ch = {v: set() for v in V}  # children (directed edges)
        self.ne = {v: set() for v in V}  # neighbors (bidirected edges)
        self.bi = set(map(tuple, map(sorted, bidirected_edges)))  # bidirected edges

        for v1, v2 in directed_edges:
            self.pa[v2].add(v1)
            self.ch[v1].add(v2)

        for v1, v2 in bidirected_edges:
            self.ne[v1].add(v2)
            self.ne[v2].add(v1)
            self.bi.add(tuple(sorted((v1, v2))))

        self.pa = {v: sorted(self.pa[v]) for v in self.v}
        self.ch = {v: sorted(self.ch[v]) for v in self.v}
        self.ne = {v: sorted(self.ne[v]) for v in self.v}

        self._sort()
        self.v2i = {v: i for i, v in enumerate(self.v)}

        # self.cc = self._c_components()
        # self.v2cc = {v: next(c for c in self.cc if v in c) for v in self.v} # maps v to the associated c component
        
        self.c2 = self._maximal_cliques()
        self.v2c2 = {v: [c for c in self.c2 if v in c] for v in self.v}

    def __iter__(self):
        return iter(self.v)

    def _sort(self):  
        """Sort V topologically
        Taken from NCMCounterfactuals
        """
        L = []
        marks = {v: 0 for v in self.v}

        def visit(v):
            if marks[v] == 2:
                return
            if marks[v] == 1:
                raise ValueError('Not a DAG.')

            marks[v] = 1
            for c in self.ch[v]:
                visit(c)
            marks[v] = 2
            L.append(v)

        for v in marks:
            if marks[v] == 0:
                visit(v)
        self.v = L[::-1]

    def _maximal_cliques(self):
        """
        Finds all maximal cliques in an undirected graph.
        = All subsets of vertices with the two properties that each pair of vertices in one of the listed subsets is connected by an edge, and no listed subset can have any additional vertices added to it while preserving its complete connectivity
        Tryna find groups with bidirected edges between them

        Taken from NCMCounterfactuals
        """
        # find degeneracy ordering
        o = []
        p = set(self.v)
        while len(o) < len(self.v):
            v = min((len(set(self.ne[v]).difference(o)), v) for v in p)[1]
            o.append(v)
            p.remove(v)

        # brute-force bron_kerbosch algorithm
        c2 = set()

        def bron_kerbosch(r, p, x):
            if not p and not x:
                c2.add(tuple(sorted(r)))
            p = set(p)
            x = set(x)
            for v in list(p):
                bron_kerbosch(r.union({v}),
                              p.intersection(self.ne[v]),
                              x.intersection(self.ne[v]))
                p.remove(v)
                x.add(v)

        # apply brute-force bron_kerbosch with degeneracy ordering
        p = set(self.v)
        x = set()
        for v in o:
            bron_kerbosch({v},
                          p.intersection(self.ne[v]),
                          x.intersection(self.ne[v]))
            p.remove(v)
            x.add(v)

        return c2
    

    '''OMITTED:
    - self.pap -- it's literally never used
    - subgraph() -- could be useful but i don't think we need it yet so no use having it here
    - convert_set_to_sorted()
    - serialize()
    - read() -- will need to implement IF we decide to store graphs as strings
    - save()
    - graph_search()

    [] self._c_components -- will prob need to implement for L3 queries
    [] ancestors -- will prob need to implement
    '''

def create_expanded_sfm(X, Z_cols, W_cols, Y):
    v = [X,Y]
    de = [(X,Y),]
    be = []

    for Z in Z_cols:
        v.append(Z)
        be.append((X,Z))
        for W in W_cols:
            de.append((Z,W))
        de.append((Z,Y))

    for W in W_cols:
        v.append(W)
        de.append((X,W))
        de.append((W,Y))
    
    return CausalGraph(V=v, directed_edges=de, bidirected_edges=be)