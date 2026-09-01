from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import optax
from jax import jit, random, vmap

# Numerical floor/ceiling for probabilities. Shared by the recalibrator so that
# training (``total_loss``) and inference (``predict_proba``) clip identically.
_PROB_EPS = 1e-6

# -------------------------
# Kernel and KLCE Functions
# -------------------------


def rbf_kernel(X: jnp.ndarray, Y: jnp.ndarray, gamma: float) -> jnp.ndarray:
    """
    Compute the RBF (Gaussian) kernel matrix between X and Y.

    This function computes the Gaussian (radial basis function) kernel
    elementwise between two datasets X and Y using kernel coefficient gamma.

    Parameters
    ----------
    X : jnp.ndarray
        First data array of shape (n_samples, n_features) or (n_samples,) for single feature.
    Y : jnp.ndarray
        Second data array of shape (m_samples, n_features) or (m_samples,). Must
        have the same number of features as X, but may have a different number of
        samples — rectangular kernels are supported (used e.g. by the LCB
        diagnostic to evaluate at query points).
    gamma : float
        Kernel coefficient, typically defined as 1 / (sigma^2).

    Returns
    -------
    jnp.ndarray
        Kernel matrix of shape (n_samples, m_samples).
    """
    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]
    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"X and Y must have the same number of features. Got {X.shape[1]} and {Y.shape[1]}."
        )
    squared_diff = (
        jnp.sum(X**2, axis=1)[:, None] + jnp.sum(Y**2, axis=1)[None, :] - 2 * jnp.dot(X, Y.T)
    )
    return jnp.exp(-gamma * squared_diff)


@jit
def create_kernel(
    X: jnp.ndarray, p: jnp.ndarray, prob_kernel_width: float, x_kernel_width: float
) -> jnp.ndarray:
    """
    Create combined kernel matrix for KLCE.

    This function computes the elementwise product of two RBF kernels:
    one over predicted probabilities p and one over feature matrix X.

    Parameters
    ----------
    X : jnp.ndarray
        Feature matrix of shape (n_samples, n_features).
    p : jnp.ndarray
        Predicted probabilities array of shape (n_samples,).
    prob_kernel_width : float
        Bandwidth parameter for the probability kernel.
    x_kernel_width : float
        Bandwidth parameter for the feature kernel.

    Returns
    -------
    jnp.ndarray
        Combined kernel matrix of shape (n_samples, n_samples).
    """
    if p.ndim != 1:
        raise ValueError(f"p must be a 1D array. Got ndim={p.ndim}.")
    if X.shape[0] != p.shape[0]:
        raise ValueError(
            f"Number of samples in X and p must match. Got {X.shape[0]} and {p.shape[0]}."
        )

    p_reshaped = p.reshape(-1, 1)
    gamma_p = 1.0 / (prob_kernel_width**2)
    gamma_x = 1.0 / (x_kernel_width**2)

    K_pp = rbf_kernel(p_reshaped, p_reshaped, gamma_p)
    K_xx = rbf_kernel(X, X, gamma_x)

    return K_pp * K_xx


@jit
def KLCE2_estimator(K: jnp.ndarray, err: jnp.ndarray) -> float:
    """
    Compute the KLCE2 estimator from kernel matrix K and error vector err.

    This function sums the off-diagonal elements of the elementwise product
    between K and the outer product of err with itself, normalized by n*(n-1).

    Parameters
    ----------
    K : jnp.ndarray
        Kernel matrix of shape (n_samples, n_samples).
    err : jnp.ndarray
        Error vector of shape (n_samples,).

    Returns
    -------
    float
        KLCE2 estimator value.
    """
    if K.ndim != 2:
        raise ValueError(f"K must be a 2D array. Got ndim={K.ndim}.")
    if err.ndim != 1:
        raise ValueError(f"err must be a 1D array. Got ndim={err.ndim}.")
    if K.shape[0] != K.shape[1] or K.shape[0] != err.shape[0]:
        raise ValueError(
            f"Shape mismatch: K must be square of size n and err length n. "
            f"Got K.shape={K.shape}, err.shape={err.shape}."
        )

    err_outer = jnp.outer(err, err)
    K_err = K * err_outer

    mask = jnp.ones_like(K_err) - jnp.eye(K_err.shape[0])
    K_err_off_diag = K_err * mask

    n = err.shape[0]
    return jnp.sum(K_err_off_diag) / (n * (n - 1))


def KLCE2_boosting(
    f: jnp.ndarray,
    X_cal: jnp.ndarray,
    y: jnp.ndarray,
    prob_kernel_width: float,
    x_kernel_width: float,
) -> float:
    """
    Compute the KLCE2 boosting loss for calibration.

    This function calculates the KLCE2 estimator using the base predictions f,
    calibration features X_cal, true labels y, and given kernel widths.

    Parameters
    ----------
    f : jnp.ndarray
        Base prediction probabilities of shape (n_samples,).
    X_cal : jnp.ndarray
        Calibration feature matrix of shape (n_samples, n_features).
    y : jnp.ndarray
        True labels of shape (n_samples,).
    prob_kernel_width : float
        Bandwidth for the probability kernel.
    x_kernel_width : float
        Bandwidth for the feature kernel.

    Returns
    -------
    float
        KLCE2 boosting loss.
    """
    p_err = y - f
    K = create_kernel(X_cal, f, prob_kernel_width, x_kernel_width)
    return KLCE2_estimator(K, p_err)


@jit
def KLCE2_null_estimator(err: jnp.ndarray, K: jnp.ndarray, key: jnp.ndarray) -> float:
    """
    Compute a single null sample for the KLCE2 test by permutation.

    This function permutes the error vector err and computes the KLCE2 estimator
    against the kernel matrix K for a single random key.

    Parameters
    ----------
    err : jnp.ndarray
        Error vector of shape (n_samples,).
    K : jnp.ndarray
        Kernel matrix of shape (n_samples, n_samples).
    key : jnp.ndarray
        PRNG key for permutation.

    Returns
    -------
    float
        KLCE2 estimator for permuted errors.
    """
    idx = random.permutation(key, len(err))
    return KLCE2_estimator(K, err[idx])


def compute_null_distribution(
    p_err: jnp.ndarray, K: jnp.ndarray, key: jnp.ndarray, iterations: int
) -> jnp.ndarray:
    """
    Compute the null distribution of KLCE2 estimators over multiple permutations.

    Parameters
    ----------
    p_err : jnp.ndarray
        Error vector of shape (n_samples,).
    K : jnp.ndarray
        Kernel matrix of shape (n_samples, n_samples).
    key : jnp.ndarray
        PRNG key for permutation splitting.
    iterations : int
        Number of null samples to generate.

    Returns
    -------
    jnp.ndarray
        Array of null KLCE2 estimates of length `iterations`.
    """
    vmapped_null = jit(vmap(KLCE2_null_estimator, (None, None, 0)))
    keys = random.split(key, iterations)
    return vmapped_null(p_err, K, keys)


@dataclass(frozen=True)
class KLCETestResult:
    """Result of :func:`KLCE_test`.

    Unpacks as ``(statistic, pvalue)`` for backward compatibility
    (``stat, p = KLCE_test(...)``), and also exposes attribute access
    (``.statistic``, ``.pvalue``, ``.null_distribution``) in the style of SciPy's
    hypothesis-test result objects.
    """

    statistic: float
    pvalue: float
    null_distribution: Optional[jnp.ndarray] = None

    def __iter__(self):
        yield self.statistic
        yield self.pvalue


def KLCE_test(
    X: jnp.ndarray,
    Y: jnp.ndarray,
    p: jnp.ndarray,
    prob_kernel_width: float,
    iterations: int,
    key,
    x_kernel_width: Optional[float] = None,
    add_one_correction: bool = True,
) -> tuple[float, float]:
    """
    Perform the KLCE hypothesis test comparing model predictions to true labels.

    This function computes the test statistic and p-value by comparing the observed
    KLCE2 estimator against a null distribution generated by permutations. The null
    hypothesis is that the model is locally calibrated (KLCE^2 = 0); a small p-value
    is evidence against it.

    Parameters
    ----------
    X : array_like
        Feature matrix of shape (n_samples, n_features). NumPy arrays are accepted.
    Y : array_like
        True label vector of shape (n_samples,).
    p : array_like
        Predicted probability vector of shape (n_samples,).
    prob_kernel_width : float
        Bandwidth for the probability kernel.
    iterations : int
        Number of permutations for null distribution.
    key : jax.Array or int
        PRNG key for random operations. An integer is accepted and converted with
        ``jax.random.PRNGKey``.
    x_kernel_width : float, optional
        Bandwidth for the feature kernel. If omitted, ``prob_kernel_width``
        is used for both kernels.
    add_one_correction : bool, optional
        If True (default), use the Monte-Carlo permutation p-value
        ``(1 + #{null >= observed}) / (1 + iterations)`` (Phipson & Smyth, 2010),
        which is never exactly zero and controls the Type-I error rate. If False,
        use the uncorrected ``#{null > observed} / iterations`` floored at
        ``1 / iterations``.

    Returns
    -------
    KLCETestResult
        A result object that unpacks as ``(statistic, pvalue)`` and also exposes
        ``.statistic``, ``.pvalue`` and ``.null_distribution`` (the permutation
        null samples, useful for plotting or a Type-I error check).
    """
    if x_kernel_width is None:
        x_kernel_width = prob_kernel_width
    X = jnp.asarray(X)
    Y = jnp.asarray(Y)
    p = jnp.asarray(p)
    if isinstance(key, int):
        key = random.PRNGKey(key)
    K = create_kernel(X, p, prob_kernel_width, x_kernel_width)
    p_err = Y - p
    test_value = KLCE2_estimator(K, p_err)
    test_null = compute_null_distribution(p_err, K, key, iterations)
    if add_one_correction:
        p_value = (1.0 + jnp.sum(test_null >= test_value)) / (1.0 + iterations)
    else:
        resolution = 1.0 / iterations
        p_value = jnp.maximum(resolution, resolution * jnp.sum(test_null > test_value))
    return KLCETestResult(statistic=test_value, pvalue=p_value, null_distribution=test_null)


# ------------------------------------
# Recalibration MLP Model using Optax
# ------------------------------------


def init_recalibrated_model_params(
    rng: jnp.ndarray, layer_sizes: Sequence[int], scale: float = 1e-1
) -> dict[str, jnp.ndarray]:
    """
    Initialize parameters for the recalibration MLP model.

    This function creates weight matrices and bias vectors for each layer
    based on `layer_sizes`, scaled by `scale` and initialized from a normal distribution.

    Parameters
    ----------
    rng : jnp.ndarray
        PRNG key for parameter initialization.
    layer_sizes : Sequence[int]
        Sizes of each layer including input and output dimensions.
    scale : float, optional
        Scaling factor for random initialization. Default is 1e-1.

    Returns
    -------
    Dict[str, jnp.ndarray]
        Dictionary mapping parameter names to initialized arrays.
    """
    keys = random.split(rng, 2 * (len(layer_sizes) - 1))
    params: dict[str, jnp.ndarray] = {}
    for i in range(len(layer_sizes) - 1):
        in_dim, out_dim = layer_sizes[i], layer_sizes[i + 1]
        W_key, b_key = keys[2 * i], keys[2 * i + 1]
        params[f"W{i}"] = scale * random.normal(W_key, (in_dim, out_dim))
        params[f"b{i}"] = scale * random.normal(b_key, (out_dim,))
    return params


def recalibrated_model_apply(params: dict[str, jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
    """
    Apply the recalibration MLP model to input features.

    This function performs a forward pass through the MLP layers using ReLU
    activations, producing a correction term for the base probabilities.

    Parameters
    ----------
    params : Dict[str, jnp.ndarray]
        Model parameters mapping layer names to weight and bias arrays.
    x : jnp.ndarray
        Input feature array of shape (n_samples, n_features).

    Returns
    -------
    jnp.ndarray
        Model output of shape (n_samples, 1).
    """
    num_layers = len(params) // 2
    h = x
    for i in range(num_layers):
        W = params[f"W{i}"]
        b = params[f"b{i}"]
        h = jnp.dot(h, W) + b
        if i < num_layers - 1:
            h = jax.nn.relu(h)
    return h


class recalibrated_model:
    """
    Recalibration model combining distillation loss and KLCE penalty.

    This class implements a simple MLP-based recalibration of base probabilities
    trained to minimize a combination of KL divergence and kernel-based calibration error.
    """

    def __init__(
        self,
        sigma_k: Optional[float] = None,
        sigma_l: Optional[float] = None,
        alpha: float = 0.02,
        beta: float = 1.0,
        num_steps: int = 1000,
        learning_rate: float = 0.01,
        hidden_layer_sizes: tuple[int, ...] = (64, 64),
        seed: int = 121,
        verbose: bool = False,
    ) -> None:
        """
        Initialize hyperparameters for the recalibration model.

        The defaults are chosen to reliably reduce local miscalibration while
        preserving ranking (see the ``experiments/`` sweep): a heavy KLCE penalty
        with light distillation, and kernel widths selected by the median heuristic.

        Parameters
        ----------
        sigma_k : float, optional
            Kernel width for the probability kernel. If None (default), it is set
            by the median heuristic on the base probabilities during ``fit``.
        sigma_l : float, optional
            Kernel width for the feature kernel. If None (default), it is set by
            the median heuristic on the features during ``fit``.
        alpha : float, optional
            Weight for the distillation loss term. Default is 0.02.
        beta : float, optional
            Weight for the KLCE penalty term. Default is 1.0.
        num_steps : int, optional
            Number of training steps. Default is 1000.
        learning_rate : float, optional
            Optimizer learning rate. Default is 0.01.
        hidden_layer_sizes : Tuple[int, ...], optional
            Sizes of hidden MLP layers. Default is (64, 64).
        seed : int, optional
            Random seed for initialization. Default is 121.
        verbose : bool, optional
            If True, print the training loss every 100 steps. Default is False.
        """
        self.sigma_k = sigma_k
        self.sigma_l = sigma_l
        self.alpha = alpha
        self.beta = beta
        self.num_steps = num_steps
        self.learning_rate = learning_rate
        self.hidden_layer_sizes = hidden_layer_sizes
        self.seed = seed
        self.verbose = verbose
        # Effective kernel widths, resolved in fit() (auto-selected when None).
        self._sigma_k: Optional[float] = None
        self._sigma_l: Optional[float] = None
        self.params: Optional[dict[str, jnp.ndarray]] = None
        self.loss_history: Optional[list[float]] = None

    def get_params(self, deep: bool = True) -> dict[str, object]:
        """Return the model hyperparameters (scikit-learn estimator API)."""
        return {
            "sigma_k": self.sigma_k,
            "sigma_l": self.sigma_l,
            "alpha": self.alpha,
            "beta": self.beta,
            "num_steps": self.num_steps,
            "learning_rate": self.learning_rate,
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "seed": self.seed,
            "verbose": self.verbose,
        }

    def set_params(self, **params) -> "recalibrated_model":
        """Set model hyperparameters (scikit-learn estimator API). Returns self."""
        valid = self.get_params()
        for name, value in params.items():
            if name not in valid:
                raise ValueError(
                    f"Invalid parameter {name!r} for recalibrated_model. "
                    f"Valid parameters are: {sorted(valid)}."
                )
            setattr(self, name, value)
        return self

    def total_loss(
        self,
        params: dict[str, jnp.ndarray],
        base_probs: jnp.ndarray,
        x: jnp.ndarray,
        y: jnp.ndarray,
    ) -> float:
        """
        Compute the total loss combining distillation and KLCE penalty.

        This function calculates the KL divergence between base_probs and
        recalibrated predictions, then adds the kernel-based calibration error.

        Parameters
        ----------
        params : Dict[str, jnp.ndarray]
            Recalibration model parameters.
        base_probs : jnp.ndarray
            Original predicted probabilities of shape (n_samples,).
        x : jnp.ndarray
            Calibration features of shape (n_samples, n_features).
        y : jnp.ndarray
            True labels of shape (n_samples,).

        Returns
        -------
        float
            Weighted sum of distillation loss and KLCE penalty.
        """
        n = base_probs.shape[0]
        features = jnp.column_stack([jnp.ones(n), base_probs, x])
        correction = recalibrated_model_apply(params, features).squeeze()
        f_recalibrated = base_probs + correction
        f_recalibrated = jnp.clip(f_recalibrated, _PROB_EPS, 1.0 - _PROB_EPS)
        base_probs_stable = jnp.clip(base_probs, _PROB_EPS, 1.0 - _PROB_EPS)
        log_ratio1 = jnp.log(jnp.maximum(base_probs_stable / f_recalibrated, 1e-10))
        log_ratio2 = jnp.log(jnp.maximum((1 - base_probs_stable) / (1 - f_recalibrated), 1e-10))
        kl_div = base_probs_stable * log_ratio1 + (1 - base_probs_stable) * log_ratio2
        distill_loss = jnp.mean(kl_div)
        x_2d = x[:, None] if x.ndim == 1 else x
        klce_loss = KLCE2_boosting(f_recalibrated, x_2d, y, self._sigma_k, self._sigma_l)
        return self.alpha * distill_loss + self.beta * klce_loss

    def fit(self, y_proba: jnp.ndarray, x_cal: jnp.ndarray, y: jnp.ndarray) -> "recalibrated_model":
        """
        Train the recalibration model on calibration data.

        This method optimizes model parameters to minimize the total loss
        over specified number of steps using the Adam optimizer.

        Parameters
        ----------
        y_proba : jnp.ndarray
            Base probability predictions of shape (n_samples,).
        x_cal : jnp.ndarray
            Calibration feature matrix of shape (n_samples, n_features).
        y : jnp.ndarray
            True labels of shape (n_samples,).

        Returns
        -------
        recalibrated_model
            The fitted estimator (``self``), to allow method chaining.
        """
        y_proba = jnp.asarray(y_proba)
        x_cal = jnp.asarray(x_cal)
        y = jnp.asarray(y)
        rng = random.PRNGKey(self.seed)
        if x_cal.ndim == 1:
            x_cal = x_cal[:, None]
        # Resolve kernel widths — median heuristic for any left as None.
        if self.sigma_k is None or self.sigma_l is None:
            from .bandwidth import select_bandwidths

            pw, xw = select_bandwidths(x_cal, y_proba)
        self._sigma_k = self.sigma_k if self.sigma_k is not None else pw
        self._sigma_l = self.sigma_l if self.sigma_l is not None else xw
        input_dim = 2 + x_cal.shape[1]
        layer_sizes = [input_dim] + list(self.hidden_layer_sizes) + [1]
        params = init_recalibrated_model_params(rng, layer_sizes)
        optimizer = optax.adam(self.learning_rate)
        opt_state = optimizer.init(params)

        @jit
        def step(
            params: dict[str, jnp.ndarray], opt_state: optax.OptState
        ) -> tuple[dict[str, jnp.ndarray], optax.OptState, float]:
            loss_val, grads = jax.value_and_grad(self.total_loss)(params, y_proba, x_cal, y)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss_val

        loss_history: list[float] = []
        for i in range(self.num_steps):
            params, opt_state, loss_val = step(params, opt_state)
            loss_history.append(float(loss_val))
            if self.verbose and i % 100 == 0:
                print(f"Step {i}: total loss = {loss_val:.6f}")
        self.params = params
        self.loss_history = loss_history
        return self

    def predict_proba(self, y_proba: jnp.ndarray, x_cal: jnp.ndarray) -> jnp.ndarray:
        """
        Generate recalibrated probability predictions.

        This method applies the trained recalibration model to new data,
        returning corrected probability estimates.

        Parameters
        ----------
        y_proba : jnp.ndarray
            Base probability predictions of shape (n_samples,).
        x_cal : jnp.ndarray
            Calibration features of shape (n_samples, n_features).

        Returns
        -------
        jnp.ndarray
            Recalibrated probability vector of shape (n_samples,).
        """
        if self.params is None:
            raise RuntimeError("Call fit() before predict_proba().")
        y_proba = jnp.asarray(y_proba)
        x_cal = jnp.asarray(x_cal)
        m = y_proba.shape[0]
        if x_cal.ndim == 1:
            x_cal = x_cal[:, None]
        features_new = jnp.column_stack([jnp.ones(m), y_proba, x_cal])
        correction = recalibrated_model_apply(self.params, features_new).squeeze()
        f_recalibrated_new = y_proba + correction
        return jnp.clip(f_recalibrated_new, _PROB_EPS, 1.0 - _PROB_EPS)

    def get_labels(self, y_proba: jnp.ndarray, threshold: float = 0.5) -> list[int]:
        """
        Convert probability predictions to binary labels.

        Parameters
        ----------
        y_proba : jnp.ndarray
            Probability vector of shape (n_samples,).
        threshold : float, optional
            Classification threshold. Default is 0.5.

        Returns
        -------
        List[int]
            Binary labels (0 or 1) for each sample.
        """
        return [1 if y > threshold else 0 for y in y_proba]

    def accuracy_score(self, y_pred: Sequence[int], y: Sequence[int]) -> float:
        """
        Compute the classification accuracy.

        Parameters
        ----------
        y_pred : Sequence[int]
            Predicted labels of shape (n_samples,).
        y : Sequence[int]
            True labels of shape (n_samples,).

        Returns
        -------
        float
            Proportion of correctly classified samples.
        """
        predictions = jnp.array(y_pred)
        actual_labels = jnp.array(y)
        return float(jnp.mean(predictions == actual_labels))
