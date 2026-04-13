import numpy as np


def mean_variance_weights(mu, cov, risk_aversion=1.0):
    mu = np.asarray(mu)
    cov = np.asarray(cov)

    if cov is None or cov.size == 0:
        raise ValueError("Invalid covariance matrix")

    inv_cov = np.linalg.pinv(cov)
    w = inv_cov @ mu

    if np.isclose(w.sum(), 0):
        return np.ones_like(w) / len(w)

    w = w / w.sum()
    return w


def cvar_constrained_weights(returns, alpha=0.05):
    """
    Simple CVaR-style weighting (diagnostic only).
    """
    returns = np.asarray(returns)

    if returns.ndim == 1:
        returns = returns.reshape(-1, 1)

    # Estimate downside risk
    var_threshold = np.percentile(returns, alpha * 100, axis=0)
    cvar = returns[returns <= var_threshold].mean(axis=0)

    weights = np.abs(cvar)

    if np.isclose(weights.sum(), 0):
        return np.ones_like(weights) / len(weights)

    weights = weights / weights.sum()
    return weights


def kelly_fraction(win_prob, win_return, loss_return):
    win_prob = float(win_prob)
    win_return = float(win_return)
    loss_return = float(loss_return)

    b = win_return / abs(loss_return)
    k = (win_prob * (b + 1) - 1) / b

    return float(np.clip(k, 0.0, 1.0))


def optimise_ensemble_weights(preds, y):
    preds = np.asarray(preds)
    y = np.asarray(y)

    if preds.ndim != 2:
        raise ValueError("preds must be (n_models, n_samples)")

    scores = []

    for i in range(preds.shape[0]):
        p = preds[i]
        score = np.corrcoef(p, y)[0, 1]

        if np.isnan(score):
            score = 0.0

        scores.append(abs(score))

    scores = np.array(scores)

    if np.isclose(scores.sum(), 0):
        return np.ones_like(scores) / len(scores)

    weights = scores / scores.sum()
    return weights