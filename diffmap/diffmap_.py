"""
This is a module to be used as a reference for building other modules
"""
import numpy as np
from scipy.sparse.linalg import eigs
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import kneighbors_graph
from sklearn.utils.validation import check_array, check_is_fitted
from scipy import sparse

def diffusion_mapping(X, n_components=2, n_neighbors=5, alpha=1.0, t=1,
                      gamma=0.5, metric='minkowski', p=2, metric_params=None,
                      n_jobs=1):
    knn = kneighbors_graph(X, n_neighbors, mode='distance', metric=metric,
                           metric_params=metric_params, p=p, n_jobs=n_jobs)

    K = sparse.csr_matrix(
        (np.exp(-gamma * knn.data ** 2),
         knn.indices, knn.indptr))

    mask = (K != 0).multiply(K.T != 0)
    L = K + K.T - K.multiply(mask)

    D = sparse.diags(np.asarray(L.sum(axis=0)).reshape(-1))


    L_a = D.power(-alpha) @ L @ D.power(-alpha)

    D_a = sparse.diags(np.asarray(L_a.sum(axis=1)).reshape(-1))

    m = D_a.power(-1) @ L_a

    w, v = eigs(m, n_components + 1)

    # eigs returns complex numbers, but for Markov matrices, all eigenvalues are
    # real and in [0, 1].
    return (m.dot(v[:, 1:]) * (w[1:] ** t)).real


class DiffusionMap(BaseEstimator, TransformerMixin):
    """ Diffusion mapping for non-linear dimensionality reduction.

    Creates a Markov matrix using the kernel matrix obtained with the given
    metric and nearest neighbor graph settings, and projects the sample data
    into a Euclidean space with n_components dimensions. The eigendecomposition
    of the Markov matrix gives the new embedding for each sample.

    Parameters
    ----------

    n_components: int, default 2
        The number of dimensions in the embedding.

    n_neighbors: int, default 5
        The number of nearest neighbors used when constructing the kernel
        matrix.

    alpha: float, default 1.0
        The parameter on the graph Laplacian used for normalizing the
        diffusion matrix.

    t: int, default 1
        The time the diffusion process will run. Larger values uncover more
        global structures.

    metric: str, default "minkowski"
        The metric used to calculate the nearest neighbor graph. The
        sklearn.neighbors.DistanceMetric class gives a list of available
        metrics.

    p: int, default 2
        The power parameter for the Minkowski metric. When p=1, this is
        equivalent to the L_{1} (Manhattan) norm. When p=2, this is the
        L_{2} (Euclidean) norm. Ignored if metric != "minkowski".

    metric_params: dict, optional
        Additional keyword arguments for the metric function.

    n_jobs: int, optional (default = 1)
        The number of parallel jobs to run for neighbors search and eigenvalue
        solving.
        If ``-1``, then the number of jobs is set to the number of CPU cores.

    Attributes
    ----------

    embedding_ : array, shape = (n_samples, n_components)
        Embedding of the training matrix in diffusion space.

    References
    ----------
    -   Geometric diffusions as a tool for harmonic analysis and structure
        definition of data: Diffusion maps, 2005
        R. R. Coifman, S. Lafon, A. B. Lee, M. Maggioni, F. Warner, S. Zucker
        http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.124.2918

    -   Diffusion maps, 2006
        R. Coifman, S. Lafon
        http://www.sciencedirect.com/science/article/pii/S1063520306000546
    """
    def __init__(self, n_components=2, n_neighbors=5, alpha=1.0, t=1, gamma=1.0,
                 metric='minkowski', p=2, metric_params=None, n_jobs=1):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.alpha = alpha
        self.t = t
        self.metric = metric
        self.p = p
        self.metric_params = metric_params
        self.n_jobs = n_jobs
        self.gamma = gamma

    def fit(self, X, y=None):
        """A reference implementation of a fitting function for a transformer.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            The training input samples, where n_samples is the number of samples
            and n_features is the number of features.
        y : Ignored

        Returns
        -------
        self : object
            Returns the instance itself.
        """

        X = check_array(X, accept_sparse=True, ensure_min_samples=3,
                        estimator=self)

        self.embedding_ = diffusion_mapping(X,
                                            n_components=self.n_components,
                                            n_neighbors=self.n_neighbors,
                                            alpha=self.alpha,
                                            t=self.t,
                                            gamma=self.gamma,
                                            metric=self.metric,
                                            p=self.p,
                                            metric_params=self.metric_params,
                                            n_jobs=self.n_jobs)

        # Return the transformer
        return self

    def fit_transform(self, X, y=None, **fit_params):
        """Fit the model from data in X and transform X.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples
            and n_features is the number of features.

        Y: Ignored

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        self.fit(X)
        return self.embedding_
