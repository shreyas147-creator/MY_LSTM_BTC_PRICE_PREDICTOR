"""
models/gp_model.py — Gaussian Process regression via GPyTorch.
Outputs a full predictive distribution (mean + uncertainty).
RTX 3060: GP kernel computations run on CUDA.
"""

import numpy as np
import torch
from pathlib import Path
from utils.logger import get_logger
from utils.gpu import get_device
from config import GP_LR, GP_TRAINING_ITER, MODELS_DIR, TORCH_DTYPE

logger = get_logger()
DEVICE = get_device()

try:
    import gpytorch
    GP_OK = True
except ImportError:
    GP_OK = False
    logger.warning("gpytorch not found.")


# ---------------------------------------------------------------------------
# GP model definition
# ---------------------------------------------------------------------------

class ExactGPModel(gpytorch.models.ExactGP if GP_OK else object):

    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module  = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
            + gpytorch.kernels.PeriodicKernel()
        )

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x),
            self.covar_module(x),
        )


# ---------------------------------------------------------------------------
# Trainer wrapper
# ---------------------------------------------------------------------------

class GPModel:

    def __init__(self, lr: float = GP_LR, n_iter: int = GP_TRAINING_ITER):
        if not GP_OK:
            raise ImportError("pip install gpytorch")
        self.lr     = lr
        self.n_iter = n_iter
        self.model  = None
        self.likelihood = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "GPModel":
        """
        Fit GP on training data.
        Note: GP scales as O(N^3) — use on a representative subset
        (e.g. last 2000 points) not the full dataset.
        """
        train_x = torch.tensor(X_train, dtype=TORCH_DTYPE).to(DEVICE)
        train_y = torch.tensor(y_train, dtype=TORCH_DTYPE).to(DEVICE)

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(DEVICE)
        self.model      = ExactGPModel(train_x, train_y, self.likelihood).to(DEVICE)

        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        mll       = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        for i in range(self.n_iter):
            optimizer.zero_grad()
            output = self.model(train_x)
            loss   = -mll(output, train_y)
            loss.backward()
            optimizer.step()
            if (i + 1) % 20 == 0:
                logger.debug(f"GP iter {i+1}/{self.n_iter} | loss={loss.item():.4f}")

        logger.info("GP training complete.")
        return self

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> dict:
        """
        Returns dict with:
            mean       : predicted mean (np.ndarray)
            std        : predictive std (uncertainty)
            lower_95   : 95% credible interval lower
            upper_95   : 95% credible interval upper
        """
        self.model.eval()
        self.likelihood.eval()

        test_x = torch.tensor(X, dtype=TORCH_DTYPE).to(DEVICE)
        with gpytorch.settings.fast_pred_var():
            preds = self.likelihood(self.model(test_x))

        mean  = preds.mean.cpu().numpy()
        std   = preds.stddev.cpu().numpy()
        lower, upper = preds.confidence_region()

        return {
            "mean":     mean,
            "std":      std,
            "lower_95": lower.cpu().numpy(),
            "upper_95": upper.cpu().numpy(),
        }

    def save(self, path: Path = None) -> None:
        path = path or MODELS_DIR / "gp_model.pt"
        torch.save({
            "model":      self.model.state_dict(),
            "likelihood": self.likelihood.state_dict(),
        }, path)
        logger.info(f"GP saved → {path}")

    def load(self, train_x: torch.Tensor, train_y: torch.Tensor,
             path: Path = None) -> "GPModel":
        path = path or MODELS_DIR / "gp_model.pt"
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(DEVICE)
        self.model      = ExactGPModel(train_x, train_y, self.likelihood).to(DEVICE)
        ckpt = torch.load(path, map_location=DEVICE)
        self.model.load_state_dict(ckpt["model"])
        self.likelihood.load_state_dict(ckpt["likelihood"])
        logger.info(f"GP loaded ← {path}")
        return self