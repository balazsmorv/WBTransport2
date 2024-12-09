import numpy as np

from ot.utils import unif
from ot.da import SinkhornTransport
from msda.barycenters import sinkhorn_barycenter
import matplotlib.pyplot as plt

from msda.utils import ferradans_mapping
from msda.utils import barycentric_mapping
from msda.utils import bar_zeros_initializer
from msda.utils import bar_random_initializer
from msda.utils import bar_random_cls_initializer
from msda.mapping import MappingTransport

class MBTTransport:
    r"""Multi-source domain adaptation using Wasserstein barycenters. This class is intended
    to solve the domain adaptation problem when one has multiple sources s1, ..., sM. First,
    the Wasserstein barycenter of source measures :math:`\{\mu_{k}\}_{k=1}^{N}` is estimated,
    using the Fréchet mean,

    .. math::

        \mu^{*} = \underset{\mu}{min}\sum_{k=1}^{N}\omega_{k}W_{p}(\mu, \mu_{k}).

    Being :math:`\mu^{*}` the Barycenter (discrete) measure, it is expressed as,

    .. math::

        \mu^{*}(\mathbf{x}) = \sum_{i=1}^{N_{bar}}p_{i}\delta(\mathbf{x}-\mathbf{x}^{supp}_{i}),

    where,

        - :math:`p_{i}` are the weights for points in the support of :math:`\mu^{*}`.
        - :math:`\mathbf{x}^{supp}_{i}` are the points in the support of :math:`\mu^{*}`.

    We consider :math:`p_{i}` constant and uniform (hence, no optimization is done over this variable). The only
    variable being optimized in the Barycenter estimation, are the support points :math:`X_{bar} = \{x_{i}^{supp}\}`.
    The algorithm used for that end is Algorithm 2 of [1]_. After estimating :math:`X_{bar}`, the points of each
    domain are transported onto the Barycenter, using the Barycentric Mapping defined as,

    .. math::
        \hat{\mathbf{X}}_{s_{j}} = diag(a)^{-1}\gamma_{s_{j}}X_{bar}

    This yields a single source, :math:`\tilde{X}_{s}` for which we may apply standard single-source OT as in [2]_.

    References
    ----------
    [1] Cuturi, M., & Doucet, A. (2014). Fast computation of Wasserstein barycenters.
    [2] Flamary, R. (2016). Optimal transport for domain adaptation.
    """

    def __init__(self,
                 barycenter_initialization="zeros", weights=None, verbose=False, barycenter_solver=sinkhorn_barycenter,
                 mu=1.0, eta=1e-3, bias=True, class_reg=False, kernel = 'linear', sigma = 1.0, max_iter=100,
                 max_inner_iter = 1000, tol=1e-5, inner_tol=1e-6, log=True):
        self.barycenter_initialization = barycenter_initialization
        self.weights = None
        self.verbose = verbose
        self.barycenter_solver = barycenter_solver
        self.mu = mu
        self.eta = eta
        self.sigma = sigma
        self.max_iter = max_iter
        self.max_inner_iter = max_inner_iter
        self.class_reg = class_reg
        self.kernel = kernel
        self.bias = bias
        self.tol = tol
        self.log = log
        self.inner_tol = inner_tol

    def fit(self, Xs=None, Xt=None, ys=None, yt=None):
        r"""Estimates the coupling matrices between each source domain and the barycenter.
        After calculating the barycenter of sources, transport each source domain to the
        barycenter. Then, transport the sources onto the target.

        Parameters
        ----------
        Xs : list of K array-like objects, shape K x (n_samples_domain_k, n_features)
            A list containing the source samples of each domain.
        ys : list of K array-like objects, shape K x (n_samples_domain_k,)
            A list containing the labels of each sample on each source domain.
        Xt : array-like object, shape (n_samples_target, n_features)
            An array containing the target domain samples.
        yt : array-like object shape (n_samples_target,)
            An array containing the target domain labels. If given, semi-supervised
            loss is added to the cost matrix.
        """
        self.xs_ = Xs
        self.ys_ = ys
        self.xt_ = Xt

        # Barycenter variables
        mu_s = [unif(X.shape[0]) for X in Xs]

        if self.barycenter_initialization == "zeros":
            if self.verbose:
                print("[INFO] initializing barycenter position as zeros")
            self.Xbar, self.ybar = bar_zeros_initializer(np.concatenate(self.xs_, axis=0),
                                                         np.concatenate(self.ys_, axis=0))
        elif self.barycenter_initialization == "random":
            if self.verbose:
                print("[INFO] initializing barycenter at random positions")
            self.Xbar, self.ybar = bar_random_initializer(np.concatenate(self.xs_, axis=0),
                                                          np.concatenate(self.ys_, axis=0))
        elif self.barycenter_initialization == "random_cls":
            if self.verbose:
                print("[INFO] initializing barycenter at random positions using classes")
            self.Xbar, self.ybar = bar_random_cls_initializer(np.concatenate(self.xs_, axis=0),
                                                              np.concatenate(self.ys_, axis=0))
        else:
            raise ValueError("Expected 'barycenter_initialization' to be either"
                             "'zeros', 'random' or 'random_cls', but got {}".format(self.barycenter_initialization))

        if self.weights is None:
            self.weights = unif(len(self.xs_))

        if self.verbose:
            print("Estimating Barycenter")
            print("---------------------")

        # Barycenter estimation
        bary, couplings = self.barycenter_solver(mu_s=mu_s, Xs=self.xs_, Xbar=self.Xbar)

        couplings = [coupling.T for coupling in couplings]

        self.coupling_ = {
            'Barycenter Coupling {}'.format(i): coupling for i, coupling in enumerate(couplings)
        }
        self.Xbar = bary
        self.Xs = bary.copy()

        # Transport estimation
        self.Tbt = MappingTransport(mu=self.mu, eta=self.eta, bias=self.bias, kernel=self.kernel, sigma=self.sigma, max_iter=self.max_iter, tol=self.tol, max_inner_iter=self.max_inner_iter, inner_tol=self.inner_tol, log=self.log, verbose=self.verbose, class_reg=self.class_reg)
        if self.verbose:
            print("\n")
            print("Estimating transport Target => Barycenter")
            print("-----------------------------------------")
        self.Tbt.fit(Xs=Xt, ys=yt, Xt=self.Xs, yt=self.ybar)

        self.coupling_["Bar->Target Coupling"] = self.Tbt.coupling_
        self.mapping_ = self.Tbt.mapping_
        self.log_ = self.Tbt.log_


    def transform(self, Xs=None, ys=None, Xt=None, yt=None):
        self.txs_ = self.Tbt.transform(Xs=Xs)
        return self.txs_


    def plot_loss(self):
        plt.plot([l.cpu() for l in self.log_["loss"]])
        plt.show()