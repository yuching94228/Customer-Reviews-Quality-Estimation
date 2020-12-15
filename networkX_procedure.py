import networkx as nx
from heapq import heappush, heappop
from itertools import count
import random
import numpy as np
from scipy.sparse import csr_matrix
import networkx as nx
from networkx.utils import *

def degree_centrality(G):
    """Compute the degree centrality for nodes.

    The degree centrality for a node v is the fraction of nodes it
    is connected to.

    Parameters
    ----------
    G : graph
      A networkx graph

    Returns
    -------
    nodes : dictionary
       Dictionary of nodes with degree centrality as the value.

    See Also
    --------
    betweenness_centrality, load_centrality, eigenvector_centrality

    Notes
    -----
    The degree centrality values are normalized by dividing by the maximum
    possible degree in a simple graph n-1 where n is the number of nodes in G.

    For multigraphs or graphs with self loops the maximum degree might
    be higher than n-1 and values of degree centrality greater than 1
    are possible.
    """
    centrality={}
    s=1.0/(len(G)-1.0)
    centrality=dict((n,d*s) for n,d in G.degree_iter())
    return centrality
def in_degree_centrality(G):
    """Compute the in-degree centrality for nodes.

    The in-degree centrality for a node v is the fraction of nodes its
    incoming edges are connected to.

    Parameters
    ----------
    G : graph
        A NetworkX graph

    Returns
    -------
    nodes : dictionary
        Dictionary of nodes with in-degree centrality as values.

    See Also
    --------
    degree_centrality, out_degree_centrality

    Notes
    -----
    The degree centrality values are normalized by dividing by the maximum
    possible degree in a simple graph n-1 where n is the number of nodes in G.

    For multigraphs or graphs with self loops the maximum degree might
    be higher than n-1 and values of degree centrality greater than 1
    are possible.
    """
    if not G.is_directed():
        raise nx.NetworkXError(\
            "in_degree_centrality() not defined for undirected graphs.")
    centrality={}
    s=1.0/(len(G)-1.0)
    centrality=dict((n,d*s) for n,d in G.in_degree_iter())
    return centrality
def out_degree_centrality(G):
    """Compute the out-degree centrality for nodes.

    The out-degree centrality for a node v is the fraction of nodes its
    outgoing edges are connected to.

    Parameters
    ----------
    G : graph
        A NetworkX graph

    Returns
    -------
    nodes : dictionary
        Dictionary of nodes with out-degree centrality as values.

    See Also
    --------
    degree_centrality, in_degree_centrality

    Notes
    -----
    The degree centrality values are normalized by dividing by the maximum
    possible degree in a simple graph n-1 where n is the number of nodes in G.

    For multigraphs or graphs with self loops the maximum degree might
    be higher than n-1 and values of degree centrality greater than 1
    are possible.
    """
    if not G.is_directed():
        raise nx.NetworkXError(\
            "out_degree_centrality() not defined for undirected graphs.")
    centrality={}
    s=1.0/(len(G)-1.0)
    centrality=dict((n,d*s) for n,d in G.out_degree_iter())
    return centrality
def closeness_centrality(G, u=None, distance=None, normalized=True):
    r"""Compute closeness centrality for nodes.

    Closeness centrality [1]_ of a node `u` is the reciprocal of the
    sum of the shortest path distances from `u` to all `n-1` other nodes.
    Since the sum of distances depends on the number of nodes in the
    graph, closeness is normalized by the sum of minimum possible
    distances `n-1`.

    .. math::

        C(u) = \frac{n - 1}{\sum_{v=1}^{n-1} d(v, u)},

    where `d(v, u)` is the shortest-path distance between `v` and `u`,
    and `n` is the number of nodes in the graph.

    Notice that higher values of closeness indicate higher centrality.

    Parameters
    ----------
    G : graph
      A NetworkX graph
    u : node, optional
      Return only the value for node u
    distance : edge attribute key, optional (default=None)
      Use the specified edge attribute as the edge distance in shortest
      path calculations
    normalized : bool, optional
      If True (default) normalize by the number of nodes in the connected
      part of the graph.

    Returns
    -------
    nodes : dictionary
      Dictionary of nodes with closeness centrality as the value.

    See Also
    --------
    betweenness_centrality, load_centrality, eigenvector_centrality,
    degree_centrality

    Notes
    -----
    The closeness centrality is normalized to `(n-1)/(|G|-1)` where
    `n` is the number of nodes in the connected part of graph
    containing the node.  If the graph is not completely connected,
    this algorithm computes the closeness centrality for each
    connected part separately.

    If the 'distance' keyword is set to an edge attribute key then the
    shortest-path length will be computed using Dijkstra's algorithm with
    that edge attribute as the edge weight.

    References
    ----------
    .. [1] Linton C. Freeman: Centrality in networks: I.
       Conceptual clarification. Social Networks 1:215-239, 1979.
       http://leonidzhukov.ru/hse/2013/socialnetworks/papers/freeman79-centrality.pdf
    """
    if distance is not None:
        # use Dijkstra's algorithm with specified attribute as edge weight
        path_length = functools.partial(nx.single_source_dijkstra_path_length,
                                        weight=distance)
    else:
        path_length = nx.single_source_shortest_path_length

    if u is None:
        nodes = G.nodes()
    else:
        nodes = [u]
    closeness_centrality = {}
    for n in nodes:
        sp = path_length(G,n)
        totsp = sum(sp.values())
        if totsp > 0.0 and len(G) > 1:
            closeness_centrality[n] = (len(sp)-1.0) / totsp
            # normalize to number of nodes-1 in connected part
            if normalized:
                s = (len(sp)-1.0) / ( len(G) - 1 )
                closeness_centrality[n] *= s
        else:
            closeness_centrality[n] = 0.0
    if u is not None:
        return closeness_centrality[u]
    else:
        return closeness_centrality
def pagerank(G, alpha=0.85, personalization=None,max_iter=100, tol=1.0e-6, nstart=None, weight='weight',dangling=None):
    """Return the PageRank of the nodes in the graph.

    PageRank computes a ranking of the nodes in the graph G based on
    the structure of the incoming links. It was originally designed as
    an algorithm to rank web pages.

    Parameters
    ----------
    G : graph
      A NetworkX graph.  Undirected graphs will be converted to a directed
      graph with two directed edges for each undirected edge.

    alpha : float, optional
      Damping parameter for PageRank, default=0.85.

    personalization: dict, optional
      The "personalization vector" consisting of a dictionary with a
      key for every graph node and nonzero personalization value for each node.
      By default, a uniform distribution is used.

    max_iter : integer, optional
      Maximum number of iterations in power method eigenvalue solver.

    tol : float, optional
      Error tolerance used to check convergence in power method solver.

    nstart : dictionary, optional
      Starting value of PageRank iteration for each node.

    weight : key, optional
      Edge data key to use as weight.  If None weights are set to 1.

    dangling: dict, optional
      The outedges to be assigned to any "dangling" nodes, i.e., nodes without
      any outedges. The dict key is the node the outedge points to and the dict
      value is the weight of that outedge. By default, dangling nodes are given
      outedges according to the personalization vector (uniform if not
      specified). This must be selected to result in an irreducible transition
      matrix (see notes under google_matrix). It may be common to have the
      dangling dict to be the same as the personalization dict.

    Returns
    -------
    pagerank : dictionary
       Dictionary of nodes with PageRank as value

    Examples
    --------
    >>> G = nx.DiGraph(nx.path_graph(4))
    >>> pr = nx.pagerank(G, alpha=0.9)

    Notes
    -----
    The eigenvector calculation is done by the power iteration method
    and has no guarantee of convergence.  The iteration will stop
    after max_iter iterations or an error tolerance of
    number_of_nodes(G)*tol has been reached.

    The PageRank algorithm was designed for directed graphs but this
    algorithm does not check if the input graph is directed and will
    execute on undirected graphs by converting each edge in the
    directed graph to two edges.

    See Also
    --------
    pagerank_numpy, pagerank_scipy, google_matrix

    References
    ----------
    .. [1] A. Langville and C. Meyer,
       "A survey of eigenvector methods of web information retrieval."
       http://citeseer.ist.psu.edu/713792.html
    .. [2] Page, Lawrence; Brin, Sergey; Motwani, Rajeev and Winograd, Terry,
       The PageRank citation ranking: Bringing order to the Web. 1999
       http://dbpubs.stanford.edu:8090/pub/showDoc.Fulltext?lang=en&doc=1999-66&format=pdf
    """
    if len(G) == 0:
        return {}

    if not G.is_directed():
        D = G.to_directed()
    else:
        D = G

    # Create a copy in (right) stochastic form
    W = nx.stochastic_graph(D, weight=weight)
    N = W.number_of_nodes()

    # Choose fixed starting vector if not given
    if nstart is None:
        x = dict.fromkeys(W, 1.0 / N)
    else:
        # Normalized nstart vector
        s = float(sum(nstart.values()))
        x = dict((k, v / s) for k, v in nstart.items())

    if personalization is None:
        # Assign uniform personalization vector if not given
        p = dict.fromkeys(W, 1.0 / N)
    else:
        missing = set(G) - set(personalization)
        if missing:
            raise NetworkXError('Personalization dictionary '
                                'must have a value for every node. '
                                'Missing nodes %s' % missing)
        s = float(sum(personalization.values()))
        p = dict((k, v / s) for k, v in personalization.items())

    if dangling is None:
        # Use personalization vector if dangling vector not specified
        dangling_weights = p
    else:
        missing = set(G) - set(dangling)
        if missing:
            raise NetworkXError('Dangling node dictionary '
                                'must have a value for every node. '
                                'Missing nodes %s' % missing)
        s = float(sum(dangling.values()))
        dangling_weights = dict((k, v/s) for k, v in dangling.items())
    dangling_nodes = [n for n in W if W.out_degree(n, weight=weight) == 0.0]

    # power iteration: make up to max_iter iterations
    for _ in range(max_iter):
        xlast = x
        x = dict.fromkeys(xlast.keys(), 0)
        danglesum = alpha * sum(xlast[n] for n in dangling_nodes)
        for n in x:
            # this matrix multiply looks odd because it is
            # doing a left multiply x^T=xlast^T*W
            for nbr in W[n]:
                x[nbr] += alpha * xlast[n] * W[n][nbr][weight]
            x[n] += danglesum * dangling_weights[n] + (1.0 - alpha) * p[n]
        # check convergence, l1 norm
        err = sum([abs(x[n] - xlast[n]) for n in x])
        if err < N*tol:
            return x
    raise NetworkXError('pagerank: power iteration failed to converge '
                        'in %d iterations.' % max_iter)
def betweenness_centrality(G, k=None, normalized=True, weight=None,endpoints=False,seed=None):
    r"""Compute the shortest-path betweenness centrality for nodes.

    Betweenness centrality of a node `v` is the sum of the
    fraction of all-pairs shortest paths that pass through `v`:

    .. math::

       c_B(v) =\sum_{s,t \in V} \frac{\sigma(s, t|v)}{\sigma(s, t)}

    where `V` is the set of nodes, `\sigma(s, t)` is the number of
    shortest `(s, t)`-paths,  and `\sigma(s, t|v)` is the number of those
    paths  passing through some  node `v` other than `s, t`.
    If `s = t`, `\sigma(s, t) = 1`, and if `v \in {s, t}`,
    `\sigma(s, t|v) = 0` [2]_.

    Parameters
    ----------
    G : graph
      A NetworkX graph

    k : int, optional (default=None)
      If k is not None use k node samples to estimate betweenness.
      The value of k <= n where n is the number of nodes in the graph.
      Higher values give better approximation.

    normalized : bool, optional
      If True the betweenness values are normalized by `2/((n-1)(n-2))`
      for graphs, and `1/((n-1)(n-2))` for directed graphs where `n`
      is the number of nodes in G.

    weight : None or string, optional
      If None, all edge weights are considered equal.
      Otherwise holds the name of the edge attribute used as weight.

    endpoints : bool, optional
      If True include the endpoints in the shortest path counts.

    Returns
    -------
    nodes : dictionary
       Dictionary of nodes with betweenness centrality as the value.

    See Also
    --------
    edge_betweenness_centrality
    load_centrality

    Notes
    -----
    The algorithm is from Ulrik Brandes [1]_.
    See [4]_ for the original first published version and [2]_ for details on
    algorithms for variations and related metrics.

    For approximate betweenness calculations set k=#samples to use
    k nodes ("pivots") to estimate the betweenness values. For an estimate
    of the number of pivots needed see [3]_.

    For weighted graphs the edge weights must be greater than zero.
    Zero edge weights can produce an infinite number of equal length
    paths between pairs of nodes.
    """
    betweenness = dict.fromkeys(G, 0.0)  # b[v]=0 for v in G
    if k is None:
        nodes = G
    else:
        random.seed(seed)
        nodes = random.sample(G.nodes(), k)
    for s in nodes:
        # single source shortest paths
        if weight is None:  # use BFS
            S, P, sigma = _single_source_shortest_path_basic(G, s)
        else:  # use Dijkstra's algorithm
            S, P, sigma = _single_source_dijkstra_path_basic(G, s, weight)
        # accumulation
        if endpoints:
            betweenness = _accumulate_endpoints(betweenness, S, P, sigma, s)
        else:
            betweenness = _accumulate_basic(betweenness, S, P, sigma, s)
    # rescaling
    betweenness = _rescale(betweenness, len(G),
                           normalized=normalized,
                           directed=G.is_directed(),
                           k=k)
    return betweenness
def eigenvector_centrality(G, max_iter=100, tol=1.0e-6, nstart=None,weight='weight'):
    from math import sqrt
    if type(G) == nx.MultiGraph or type(G) == nx.MultiDiGraph:
        raise nx.NetworkXException("Not defined for multigraphs.")

    if len(G) == 0:
        raise nx.NetworkXException("Empty graph.")

    if nstart is None:
        # choose starting vector with entries of 1/len(G)
        x = dict([(n,1.0/len(G)) for n in G])
    else:
        x = nstart
    # normalize starting vector
    s = 1.0/sum(x.values())
    for k in x:
        x[k] *= s
    nnodes = G.number_of_nodes()
    # make up to max_iter iterations
    for i in range(max_iter):
        xlast = x
        x = dict.fromkeys(xlast, 0)
        # do the multiplication y^T = x^T A
        for n in x:
            for nbr in G[n]:
                x[nbr] += xlast[n] * G[n][nbr].get(weight, 1)
        # normalize vector
        try:
            s = 1.0/sqrt(sum(v**2 for v in x.values()))
        # this should never be zero?
        except ZeroDivisionError:
            s = 1.0
        for n in x:
            x[n] *= s
        # check convergence
        err = sum([abs(x[n]-xlast[n]) for n in x])
        if err < nnodes*tol:
            return x

    raise nx.NetworkXError("""eigenvector_centrality():power iteration failed to converge in %d iterations."%(i+1))""")


def communicability_centrality(G):

    import numpy
    import numpy.linalg
    nodelist = G.nodes() # ordering of nodes in matrix
    A = nx.to_numpy_matrix(G,nodelist)
    # convert to 0-1 matrix
    A[A!=0.0] = 1
    w,v = numpy.linalg.eigh(A)
    vsquare = numpy.array(v)**2
    expw = numpy.exp(w)
    xg = numpy.dot(vsquare,expw)
    # convert vector dictionary keyed by node
    sc = dict(zip(nodelist,map(float,xg)))
    return sc

def katz_centrality(G, alpha=0.1, beta=1.0,max_iter=1000, tol=1.0e-6, nstart=None, normalized=True,weight = 'weight'):
    from math import sqrt

    if len(G) == 0:
        return {}

    nnodes = G.number_of_nodes()

    if nstart is None:
        # choose starting vector with entries of 0
        x = dict([(n,0) for n in G])
    else:
        x = nstart

    try:
        b = dict.fromkeys(G,float(beta))
    except (TypeError,ValueError,AttributeError):
        b = beta
        if set(beta) != set(G):
            raise nx.NetworkXError('beta dictionary '
                                   'must have a value for every node')

    # make up to max_iter iterations
    for i in range(max_iter):
        xlast = x
        x = dict.fromkeys(xlast, 0)
        # do the multiplication y^T = Alpha * x^T A - Beta
        for n in x:
            for nbr in G[n]:
                x[nbr] += xlast[n] * G[n][nbr].get(weight, 1)
        for n in x:
            x[n] = alpha*x[n] + b[n]

        # check convergence
        err = sum([abs(x[n]-xlast[n]) for n in x])
        if err < nnodes*tol:
            if normalized:
                # normalize vector
                try:
                    s = 1.0/sqrt(sum(v**2 for v in x.values()))
                # this should never be zero?
                except ZeroDivisionError:
                    s = 1.0
            else:
                s = 1
            for n in x:
                x[n] *= s
            return x

    raise nx.NetworkXError('Power iteration failed to converge in '
                           '%d iterations.' % max_iter)

def communicability_centrality_exp(G):
    # alternative implementation that calculates the matrix exponential
    import scipy.linalg
    nodelist = G.nodes()  # ordering of nodes in matrix
    A = nx.to_numpy_matrix(G,nodelist)
    # convert to 0-1 matrix
    A[A!=0.0] = 1
    expA = scipy.linalg.expm(A)
    # convert diagonal to dictionary keyed by node
    sc = dict(zip(nodelist,map(float,expA.diagonal())))
    return sc
def communicability_betweenness_centrality(G, normalized=True):
    import scipy
    import scipy.linalg
    nodelist = G.nodes() # ordering of nodes in matrix
    n = len(nodelist)
    A = nx.to_numpy_matrix(G,nodelist)
    # convert to 0-1 matrix
    A[A!=0.0] = 1
    expA = scipy.linalg.expm(A)
    mapping = dict(zip(nodelist,range(n)))
    sc = {}
    for v in G:
        # remove row and col of node v
        i = mapping[v]
        row = A[i,:].copy()
        col = A[:,i].copy()
        A[i,:] = 0
        A[:,i] = 0
        B = (expA - scipy.linalg.expm(A)) / expA
        # sum with row/col of node v and diag set to zero
        B[i,:] = 0
        B[:,i] = 0
        B -= scipy.diag(scipy.diag(B))
        sc[v] = float(B.sum())
        # put row and col back
        A[i,:] = row
        A[:,i] = col
    # rescaling
    sc = _rescale(sc,normalized=normalized)
    return sc
def _rescale(sc,normalized):
    # helper to rescale betweenness centrality
    if normalized is True:
        order=len(sc)
        if order <=2:
            scale=None
        else:
            scale=1.0/((order-1.0)**2-(order-1.0))
    if scale is not None:
        for v in sc:
            sc[v] *= scale
    return sc
def communicability(G):
    import numpy
    import scipy.linalg
    nodelist = G.nodes() # ordering of nodes in matrix
    A = nx.to_numpy_matrix(G,nodelist)
    # convert to 0-1 matrix
    A[A!=0.0] = 1
    w,vec = numpy.linalg.eigh(A)
    expw = numpy.exp(w)
    mapping = dict(zip(nodelist,range(len(nodelist))))
    sc={}
    # computing communicabilities
    for u in G:
        sc[u]={}
        for v in G:
            s = 0
            p = mapping[u]
            q = mapping[v]
            for j in range(len(nodelist)):
                s += vec[:,j][p,0]*vec[:,j][q,0]*expw[j]
            sc[u][v] = float(s)
    return sc
def communicability_exp(G):
    import scipy.linalg
    nodelist = G.nodes() # ordering of nodes in matrix
    A = nx.to_numpy_matrix(G,nodelist)
    # convert to 0-1 matrix
    A[A!=0.0] = 1
    # communicability matrix
    expA = scipy.linalg.expm(A)
    mapping = dict(zip(nodelist,range(len(nodelist))))
    sc = {}
    for u in G:
        sc[u]={}
        for v in G:
            sc[u][v] = float(expA[mapping[u],mapping[v]])
    return sc
def estrada_index(G):
    return sum(communicability_centrality(G).values())

# fixture for nose tests
def setup_module(module):
    from nose import SkipTest
    try:
        import scipy
    except:
        raise SkipTest("SciPy not available")
# obsolete name
def edge_betweenness(G, k=None, normalized=True, weight=None, seed=None):
    return edge_betweenness_centrality(G, k, normalized, weight, seed)
# helpers for betweenness centrality
def _single_source_shortest_path_basic(G, s):
    S = []
    P = {}
    for v in G:
        P[v] = []
    sigma = dict.fromkeys(G, 0.0)    # sigma[v]=0 for v in G
    D = {}
    sigma[s] = 1.0
    D[s] = 0
    Q = [s]
    while Q:   # use BFS to find shortest paths
        v = Q.pop(0)
        S.append(v)
        Dv = D[v]
        sigmav = sigma[v]
        for w in G[v]:
            if w not in D:
                Q.append(w)
                D[w] = Dv + 1
            if D[w] == Dv + 1:   # this is a shortest path, count paths
                sigma[w] += sigmav
                P[w].append(v)  # predecessors
    return S, P, sigma
def _single_source_dijkstra_path_basic(G, s, weight='weight'):
    # modified from Eppstein
    S = []
    P = {}
    for v in G:
        P[v] = []
    sigma = dict.fromkeys(G, 0.0)    # sigma[v]=0 for v in G
    D = {}
    sigma[s] = 1.0
    push = heappush
    pop = heappop
    seen = {s: 0}
    c = count()
    Q = []   # use Q as heap with (distance,node id) tuples
    push(Q, (0, next(c), s, s))
    while Q:
        (dist, _, pred, v) = pop(Q)
        if v in D:
            continue  # already searched this node.
        sigma[v] += sigma[pred]  # count paths
        S.append(v)
        D[v] = dist
        for w, edgedata in G[v].items():
            vw_dist = dist + edgedata.get(weight, 1)
            if w not in D and (w not in seen or vw_dist < seen[w]):
                seen[w] = vw_dist
                push(Q, (vw_dist, next(c), v, w))
                sigma[w] = 0.0
                P[w] = [v]
            elif vw_dist == seen[w]:  # handle equal paths
                sigma[w] += sigma[v]
                P[w].append(v)
    return S, P, sigma
def _accumulate_basic(betweenness, S, P, sigma, s):
    delta = dict.fromkeys(S, 0)
    while S:
        w = S.pop()
        coeff = (1.0 + delta[w]) / sigma[w]
        for v in P[w]:
            delta[v] += sigma[v] * coeff
        if w != s:
            betweenness[w] += delta[w]
    return betweenness
def _accumulate_endpoints(betweenness, S, P, sigma, s):
    betweenness[s] += len(S) - 1
    delta = dict.fromkeys(S, 0)
    while S:
        w = S.pop()
        coeff = (1.0 + delta[w]) / sigma[w]
        for v in P[w]:
            delta[v] += sigma[v] * coeff
        if w != s:
            betweenness[w] += delta[w] + 1
    return betweenness
def _accumulate_edges(betweenness, S, P, sigma, s):
    delta = dict.fromkeys(S, 0)
    while S:
        w = S.pop()
        coeff = (1.0 + delta[w]) / sigma[w]
        for v in P[w]:
            c = sigma[v] * coeff
            if (v, w) not in betweenness:
                betweenness[(w, v)] += c
            else:
                betweenness[(v, w)] += c
            delta[v] += c
        if w != s:
            betweenness[w] += delta[w]
    return betweenness
def _rescale(betweenness, n, normalized, directed=False, k=None):
    if normalized is True:
        if n <= 2:
            scale = None  # no normalization b=0 for all nodes
        else:
            scale = 1.0 / ((n - 1) * (n - 2))
    else:  # rescale by 2 for undirected graphs
        if not directed:
            scale = 1.0 / 2.0
        else:
            scale = None
    if scale is not None:
        if k is not None:
            scale = scale * n / k
        for v in betweenness:
            betweenness[v] *= scale
    return betweenness
def _rescale_e(betweenness, n, normalized, directed=False, k=None):
    if normalized is True:
        if n <= 1:
            scale = None  # no normalization b=0 for all nodes
        else:
            scale = 1.0 / (n * (n - 1))
    else:  # rescale by 2 for undirected graphs
        if not directed:
            scale = 1.0 / 2.0
        else:
            scale = None
    if scale is not None:
        if k is not None:
            scale = scale * n / k
        for v in betweenness:
            betweenness[v] *= scale
    return betweenness




dict_per={}
count_edge=0
count_per=0
with open('/Users/yuching/Desktop/test_data/id_trustnetwork.csv', 'r') as fnw:
    for line in fnw:
        count_edge += 1
        temp = line.split(' ')
        if temp[0] not in dict_per:
            dict_per.setdefault(temp[0].strip())
        if temp[1] not in dict_per:
            dict_per.setdefault(temp[1].strip())
fnw.close()
count_per=len(dict_per)
nr_persons = count_per
edge_num =count_edge

G_per = nx.DiGraph()
G_per.add_nodes_from(str(nr_persons))

with open('/Users/yuching/Desktop/test_data/id_trustnetwork.csv', 'r') as fnw:
    for line in fnw:
        temp = line.split(' ')
        G_per.add_edge(temp[0].strip(), temp[1].strip())
fnw.close()


# dict_deg_cen = {}
# dict_in_deg_cen = {}
# dict_out_deg_cen = {}
# dict_clo_cen = {}
# dict_PageR = {}
# dict_betw_cen = {}

dict_deg_cen = degree_centrality(G_per)
print("deg_cen done!")
dict_in_deg_cen = in_degree_centrality(G_per)
print("in_deg_cen done!")
dict_out_deg_cen = out_degree_centrality(G_per)
print("out_deg_cen done!")
dict_clo_cen = closeness_centrality(G_per)
print("clo_cen done!")
dict_PageR = pagerank(G_per)
print("PageR  done!")
dict_betw_cen = betweenness_centrality(G_per)
print("_betw_cen done!")
dict_eigen_cent = eigenvector_centrality(G_per)
print("eigenvector_centrality done!")



"""
Traceback (most recent call last):
  File "C:/Users/yuching/PycharmProjects/untitled/networkx_test.py", line 822, in <module>
    dict_commu_cent = communicability_centrality(G_per)
  File "C:/Users/yuching/PycharmProjects/untitled/networkx_test.py", line 470, in communicability_centrality
    A = nx.to_numpy_matrix(G,nodelist)
  File "C:\Python27\lib\site-packages\networkx-1.11-py2.7.egg\networkx\convert_matrix.py", line 369, in to_numpy_matrix
    M = np.zeros((nlen,nlen), dtype=dtype, order=order) + np.nan
MemoryError
"""
# dict_commun_cen =communicability_centrality(G_per)
# print("commun_cen!")

#etworkx.exception.NetworkXError: Power iteration failed to converge in 1000 iterations.
# #dict_katz_cen = katz_centrality(G_per)
#print("katz_centrality done!")

# dict_commu_cent = communicability_centrality(G_per)
# for item in dict_commu_cent.keys():
#     print item , dict_commu_cent[item]
count_dict = 0

filew = open('/Users/yuching/Desktop/result/0515_network_output.csv', 'w')
# filew.write("user,eigenvector_centrality,katz_centrality\n")
filew.write("user,dict_deg_cen,dict_in_deg_cen,dict_out_deg_cen,dict_clo_cen,dict_PageR,dict_betw_cen,dict_eigen_cent\n")
for i in range(1, G_per.number_of_nodes()+1, 1):
    count_dict+=1
    if(str(i) in dict_eigen_cent.keys()):
        # filew.write("%s,%s\n"
        #             % (str(i) ,dict_eigen_cent[str(i)],dict_katz_cen[str(i)]))
        filew.write("%s,%s,%s,%s,%s,%s,%s,%s\n"
                    % (str(i) , dict_deg_cen[str(i)] ,dict_in_deg_cen[str(i)],dict_out_deg_cen[str(i)],
                       dict_clo_cen[str(i)],dict_PageR[str(i)],dict_betw_cen[str(i)]
                       ,dict_eigen_cent[str(i)]))
        print str(i)
#, dict_deg_cen[str(i)] ,dict_in_deg_cen[str(i)],dict_out_deg_cen[str(i)],dict_clo_cen[str(i)],dict_PageR[str(i)],dict_betw_cen[str(i)]
filew.close()

print count_dict
print G_per.number_of_nodes()

