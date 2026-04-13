"""
models/transformer.py — Transformer time-series model in PyTorch.
Aligned with LSTM interface and trainer expectations.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from utils.logger import get_logger
from utils.gpu import get_device
from config import (
    TRANSFORMER_D_MODEL,
    TRANSFORMER_NHEAD,
    TRANSFORMER_LAYERS,
    TRANSFORMER_DROPOUT,
    TRANSFORMER_FF,
    LSTM_SEQ_LEN,
    LSTM_BATCH,
    LSTM_LR,
    LSTM_EPOCHS,
    LSTM_PATIENCE,
    MODELS_DIR,
    PIN_MEMORY,
    NUM_WORKERS,
    TORCH_DTYPE,
)

from models.lstm import SequenceDataset

logger = get_logger()
DEVICE = get_device()


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-np.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Transformer Model
# ---------------------------------------------------------------------------

class TransformerModel(nn.Module):
    def __init__(
        self,
        input_size,
        d_model=TRANSFORMER_D_MODEL,
        nhead=TRANSFORMER_NHEAD,
        num_layers=TRANSFORMER_LAYERS,
        dim_feedforward=TRANSFORMER_FF,
        dropout=TRANSFORMER_DROPOUT,
        n_classes=3,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.shared = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.cls_head = nn.Linear(d_model // 2, n_classes)
        self.reg_head = nn.Linear(d_model // 2, 1)

    # ---------------- REQUIRED FOR TESTS ----------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns (batch,) predicted class index — matches LSTM
        """
        logits, _ = self._forward_full(x)
        return logits.argmax(dim=1).float()

    # ---------------- INTERNAL ----------------
    def _forward_full(self, x: torch.Tensor):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)

        h = x[:, -1, :]
        h = self.shared(self.dropout(self.norm(h)))

        logits = self.cls_head(h)
        ret = self.reg_head(h).squeeze(-1)

        return logits, ret


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class TransformerTrainer:
    def __init__(
            self,
            input_size,
            n_classes=3,
            task="classification",  # <-- REQUIRED
            lr=LSTM_LR,
            epochs=LSTM_EPOCHS,
            patience=LSTM_PATIENCE,
    ):
        self.task = task

        # --- MODEL ---
        self.model = TransformerModel(
            input_size=input_size,
            n_classes=n_classes
        ).to(DEVICE)

        # --- OPTIMIZER ---
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=1e-4
        )

        # --- SCHEDULER ---
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            patience=3,
            factor=0.5
        )

        # --- LOSS FUNCTION ---
        if task == "classification":
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.MSELoss()

        # --- TRAINING PARAMS ---
        self.epochs = epochs
        self.patience = patience

        # --- EARLY STOPPING ---
        self.best_val = float("inf")
        self.best_state = None

    def _loader(self, X, y, shuffle):
        ds = SequenceDataset(X, y)
        return DataLoader(
            ds,
            batch_size=LSTM_BATCH,
            shuffle=shuffle,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            persistent_workers=NUM_WORKERS > 0,
        )

    def fit(self, X_train, y_train, X_val, y_val):
        train_loader = self._loader(X_train, y_train, True)
        val_loader = self._loader(X_val, y_val, False)

        no_improve = 0

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_loss = 0.0

            for Xb, yb in train_loader:
                Xb = Xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)

                self.optimizer.zero_grad()

                logits, _ = self.model._forward_full(Xb)  # IMPORTANT
                loss = self.criterion(logits, yb)

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            val_loss, val_acc = self._evaluate(val_loader)
            self.scheduler.step(val_loss)

            logger.info(
                f"Transformer {epoch:03d} | train={train_loss:.4f} | "
                f"val={val_loss:.4f} | acc={val_acc:.3f}"
            )

            if val_loss < self.best_val:
                self.best_val = val_loss
                self.best_state = {
                    k: v.clone() for k, v in self.model.state_dict().items()
                }
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    logger.info(f"Transformer early stop at epoch {epoch}")
                    break

        if self.best_state:
            self.model.load_state_dict(self.best_state)

        return self

    @torch.no_grad()
    def _evaluate(self, loader):
        self.model.eval()
        loss, correct, total = 0.0, 0, 0

        for Xb, yb in loader:
            Xb = Xb.to(DEVICE)
            yb = yb.to(DEVICE)

            logits, _ = self.model._forward_full(Xb)

            loss += self.criterion(logits, yb).item()
            correct += (logits.argmax(1) == yb).sum().item()
            total += len(yb)

        return loss / len(loader), correct / total

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        loader = self._loader(X, np.zeros(len(X), dtype=np.int64), False)

        probs = []
        for Xb, _ in loader:
            logits, _ = self.model._forward_full(Xb.to(DEVICE))
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())

        return np.vstack(probs)

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def save(self, path: Path = None):
        path = path or MODELS_DIR / "transformer_weights.pt"
        torch.save(
            {"model_state": self.model.state_dict()},
            path,
        )
        logger.info(f"Transformer saved → {path}")

    def load(self, path: Path = None):
        path = path or MODELS_DIR / "transformer_weights.pt"
        ckpt = torch.load(path, map_location=DEVICE)
        self.model.load_state_dict(ckpt["model_state"])
        return self