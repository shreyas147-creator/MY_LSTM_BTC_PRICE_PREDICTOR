"""
models/lstm.py — LSTM time-series model in PyTorch.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from utils.logger import get_logger
from utils.gpu import get_device
from config import (
    LSTM_HIDDEN, LSTM_LAYERS, LSTM_DROPOUT,
    LSTM_SEQ_LEN, LSTM_BATCH, LSTM_LR, LSTM_EPOCHS, LSTM_PATIENCE,
    MODELS_DIR, PIN_MEMORY, NUM_WORKERS, TORCH_DTYPE,
)

logger = get_logger()
DEVICE = get_device()


class SequenceDataset(Dataset):
    def __init__(self, features, labels, seq_len=LSTM_SEQ_LEN, labels_reg=None):
        self.X       = torch.tensor(features, dtype=TORCH_DTYPE)
        self.y       = torch.tensor(labels, dtype=torch.long)
        self.y_reg   = (
            torch.tensor(labels_reg, dtype=torch.float32)
            if labels_reg is not None
            else torch.zeros(len(labels), dtype=torch.float32)
        )
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        return (
            self.X[idx : idx + self.seq_len],
            self.y[idx + self.seq_len],
        )


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=LSTM_HIDDEN,
                 num_layers=LSTM_LAYERS, dropout=LSTM_DROPOUT, n_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm     = nn.LayerNorm(hidden_size)
        self.dropout  = nn.Dropout(dropout)
        self.shared   = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Dropout(dropout),
        )
        self.cls_head = nn.Linear(hidden_size // 2, n_classes)
        self.reg_head = nn.Linear(hidden_size // 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (batch,) predicted class index — for test compatibility."""
        logits, _ = self._forward_full(x)
        return logits.argmax(dim=1).float()

    def _forward_full(self, x: torch.Tensor):
        """Returns (logits (B,3), ret_pred (B,)) — used internally for training."""
        out, _ = self.lstm(x)
        h = self.shared(self.dropout(self.norm(out[:, -1, :])))
        return self.cls_head(h), self.reg_head(h).squeeze(-1)


class LSTMTrainer:
    def __init__(self, input_size, n_classes=3, task="classification",
                 lr=LSTM_LR, epochs=LSTM_EPOCHS, patience=LSTM_PATIENCE,
                 cls_weight=0.7, reg_weight=0.3):
        self.task          = task
        self.model         = LSTMModel(input_size=input_size, n_classes=n_classes).to(DEVICE)
        self.optimizer     = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler     = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=3, factor=0.5)
        self.cls_criterion = nn.CrossEntropyLoss()
        self.reg_criterion = nn.HuberLoss(delta=0.01)
        self.cls_w         = cls_weight
        self.reg_w         = reg_weight
        self.epochs        = epochs
        self.patience      = patience
        self.best_val      = float("inf")
        self.best_state    = None
        self.history       = {"train_loss": [], "val_loss": [], "val_acc": []}

    def _loader(self, X, y, shuffle, y_reg=None):
        ds = SequenceDataset(X, y, labels_reg=y_reg)
        return DataLoader(ds, batch_size=LSTM_BATCH, shuffle=shuffle,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                          persistent_workers=NUM_WORKERS > 0)

    def fit(self, X_train, y_train, X_val, y_val,
            y_reg_train=None, y_reg_val=None):
        train_loader = self._loader(X_train, y_train, True,  y_reg_train)
        val_loader   = self._loader(X_val,   y_val,   False, y_reg_val)
        no_improve   = 0

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            t_loss = 0.0
            for Xb, yb in train_loader:
                Xb = Xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)
                self.optimizer.zero_grad()
                logits, ret_pred = self.model._forward_full(Xb)
                loss = self.cls_criterion(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                t_loss += loss.item()

            t_loss /= len(train_loader)
            v_loss, v_acc = self._evaluate(val_loader)
            self.scheduler.step(v_loss)
            self.history["train_loss"].append(t_loss)
            self.history["val_loss"].append(v_loss)
            self.history["val_acc"].append(v_acc)
            logger.info(f"LSTM {epoch:03d} | train={t_loss:.4f} | val={v_loss:.4f} | acc={v_acc:.3f}")

            if v_loss < self.best_val:
                self.best_val   = v_loss
                self.best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    logger.info(f"LSTM early stop at epoch {epoch}")
                    break

        if self.best_state:
            self.model.load_state_dict(self.best_state)
        return self

    @torch.no_grad()
    def _evaluate(self, loader):
        self.model.eval()
        loss, correct, total = 0.0, 0, 0
        for Xb, yb in loader:
            Xb = Xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            logits, _ = self.model._forward_full(Xb)
            loss    += self.cls_criterion(logits, yb).item()
            correct += (logits.argmax(1) == yb).sum().item()
            total   += len(yb)
        return loss / len(loader), correct / total

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        loader = self._loader(X, np.zeros(len(X), dtype=np.int64), False)
        probs  = []
        for Xb, _ in loader:
            logits, _ = self.model._forward_full(Xb.to(DEVICE))
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.vstack(probs)

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns (N,) class indices."""
        return self.predict_proba(X).argmax(axis=1)

    @torch.no_grad()
    def predict_ret(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        loader = self._loader(X, np.zeros(len(X), dtype=np.int64), False)
        rets   = []
        for Xb, _ in loader:
            _, ret = self.model._forward_full(Xb.to(DEVICE))
            rets.append(ret.cpu().numpy().reshape(-1))
        return np.concatenate(rets)

    def save(self, path: Path = None):
        path = path or MODELS_DIR / "lstm_weights.pt"
        torch.save({"model_state": self.model.state_dict(),
                    "history": self.history,
                    "input_size": self.model.lstm.input_size}, path)
        logger.info(f"LSTM saved → {path}")

    def load(self, path: Path = None):
        path = path or MODELS_DIR / "lstm_weights.pt"
        ckpt = torch.load(path, map_location=DEVICE)
        self.model.load_state_dict(ckpt["model_state"])
        self.history = ckpt.get("history", {})
        return self