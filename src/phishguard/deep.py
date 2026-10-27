"""Character-level deep models on URLs (Milestone 4): a char-CNN and a BiLSTM.

Both read the URL one character at a time, so they can learn patterns like
`paypa1`, `.top` or random strings without hand-made features. They are
wrapped as scikit-learn-style estimators (`fit` / `predict_proba`), so
`experiment.run()` tunes and tests them exactly like every other model.

    model = CharURLClassifier(arch="cnn").fit(train_urls, train_labels)
    model.predict_proba(test_urls)[:, 1]      # probability of phishing

The hybrid model (Milestone 5) adds our handcrafted URL features: pass a
DataFrame with a "url" column plus the feature columns.
"""

import copy
import os
import sys
import time

if sys.platform == "darwin":
    # On macOS, torch and xgboost each bring their own OpenMP library. Two OpenMP
    # runtimes in one process crash or hang once both start worker threads, so
    # each gets one OpenMP thread (training runs on the Apple GPU anyway).
    # Import this module before running any xgboost model.
    os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from phishguard.split import site_owner

PAD, UNK = 0, 1     # id 0 fills short URLs up to the batch length; id 1 = a character not seen in training


def pick_device() -> torch.device:
    """The Apple GPU (MPS) if there is one, else the CPU."""
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class CharVocab:
    """Maps each character to an integer id. Built on training URLs only.

    URLs are lowercased (like the n-gram model). Characters seen fewer than
    `min_count` times in training share the UNK id.
    """

    def __init__(self, min_count: int = 10):
        self.min_count = min_count

    def fit(self, urls: pd.Series) -> "CharVocab":
        counts = pd.Series(list("".join(urls.str.lower()))).value_counts()
        kept = sorted(counts[counts >= self.min_count].index)
        self.ids = {char: i for i, char in enumerate(kept, start=2)}
        return self

    def __len__(self) -> int:
        return len(self.ids) + 2

    def encode(self, urls: pd.Series, max_len: int) -> tuple[np.ndarray, np.ndarray]:
        """URLs → (ids padded to max_len, true lengths). Longer URLs are cut at max_len."""
        ids = np.full((len(urls), max_len), PAD, dtype=np.uint8)
        lengths = np.ones(len(urls), dtype=np.int64)
        for row, url in enumerate(urls.str.lower()):
            codes = [self.ids.get(char, UNK) for char in url[:max_len]] or [UNK]
            ids[row, :len(codes)] = codes
            lengths[row] = len(codes)
        return ids, lengths


class CharCNN(nn.Module):
    """Embedding → convolutions of width 3, 4, 5, 6 → max over positions → dense → logit.

    A filter of width 4 slides over the URL and fires on 4-character patterns
    (like a learned 4-gram). Taking the max over positions asks "does this
    pattern appear anywhere in the URL?".
    """

    def __init__(self, vocab_size: int, embed_dim: int = 32, filters: int = 128,
                 widths=(3, 4, 5, 6), hidden: int = 128, dropout: float = 0.5):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD)
        self.convs = nn.ModuleList(nn.Conv1d(embed_dim, filters, width) for width in widths)
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(filters * len(widths), hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, 1),
        )
        self.min_len = max(widths)

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(ids, lengths)).squeeze(1)

    def encode(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """One score per filter: how strongly its pattern appears anywhere in the URL."""
        x = self.embed(ids).transpose(1, 2)                 # (batch, embed_dim, characters)
        pooled = []
        for conv in self.convs:
            h = torch.relu(conv(x))                         # (batch, filters, positions)
            # Windows that start too late only cover padding: zero them so they can't win the max
            # (ReLU outputs are ≥ 0, so a 0 never beats a real window).
            last = (lengths - conv.kernel_size[0] + 1).clamp(min=1)
            valid = torch.arange(h.size(2), device=h.device)[None, :] < last[:, None]
            pooled.append(h.masked_fill(~valid[:, None, :], 0.0).max(dim=2).values)
        return torch.cat(pooled, dim=1)                     # (batch, filters × number of widths)


class BiLSTM(nn.Module):
    """Embedding → bidirectional LSTM → max over positions → dense → logit.

    The LSTM reads the URL left-to-right and right-to-left, so each position
    sees the whole URL around it. Packing makes it skip the padding.
    """

    min_len = 1

    def __init__(self, vocab_size: int, embed_dim: int = 32, hidden: int = 64, dropout: float = 0.5):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD)
        self.lstm = nn.LSTM(embed_dim, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(2 * hidden, 64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, 1),
        )

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(self.embed(ids), lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, padding_value=float("-inf"))
        return self.head(out.max(dim=1).values).squeeze(1)


class HybridNet(nn.Module):
    """Char-CNN pattern scores + our handcrafted URL features → dense → logit (Milestone 5).

    The CNN part reads the characters. The feature part gets the 40 numbers
    from `extract_features` (log-scaled and standardised by the classifier).
    Both are concatenated and trained together, so the network can learn how
    they complement each other.
    """

    def __init__(self, vocab_size: int, n_features: int, feature_hidden: int = 64, hidden: int = 128,
                 dropout: float = 0.5, **cnn_options):
        super().__init__()
        self.chars = CharCNN(vocab_size, dropout=dropout, **cnn_options)
        char_width = self.chars.head[1].in_features
        self.chars.head = nn.Identity()                     # only the CNN's pattern scores are used
        self.features = nn.Sequential(nn.Linear(n_features, feature_hidden), nn.ReLU())
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(char_width + feature_hidden, hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, 1),
        )
        self.min_len = self.chars.min_len

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        both = torch.cat([self.chars.encode(ids, lengths), self.features(features)], dim=1)
        return self.head(both).squeeze(1)


ARCHITECTURES = {"cnn": CharCNN, "bilstm": BiLSTM, "hybrid": HybridNet}


class CharURLClassifier:
    """Train a CharCNN, BiLSTM or HybridNet like a scikit-learn classifier (D12, D13).

    Input: a Series of URLs, or (for "hybrid") a DataFrame with a "url" column
    plus numeric feature columns.

    `fit` holds out 10% of the training *site owners* as an early-stopping set:
    after every epoch we measure its log-loss and keep the best epoch's weights.
    The real validation split is left for tuning and the threshold, as for
    every other model.
    """

    def __init__(self, arch: str = "cnn", max_len: int = 200, max_epochs: int = 15, patience: int = 2,
                 batch_size: int = 256, lr: float = 1e-3, seed: int = 0, device=None, verbose: bool = False,
                 **arch_options):
        self.arch, self.max_len, self.max_epochs, self.patience = arch, max_len, max_epochs, patience
        self.batch_size, self.lr, self.seed, self.verbose = batch_size, lr, seed, verbose
        self.device = torch.device(device) if device else pick_device()
        self.arch_options = arch_options

    def fit(self, X, y: pd.Series) -> "CharURLClassifier":
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        urls, features = _split_input(X)
        y = y.to_numpy(dtype=np.float32)

        holdout = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=self.seed)
        fit_rows, stop_rows = next(holdout.split(urls, groups=site_owner(urls)))
        self.vocab_ = CharVocab().fit(urls.iloc[fit_rows])
        ids, lengths = self.vocab_.encode(urls, self.max_len)
        options = dict(self.arch_options)
        if features is not None:
            # Counts like url_length are heavily skewed: log-scale, then standardise on training rows only.
            logged = np.log1p(features[fit_rows].astype(np.float64))
            self.feature_mean_, self.feature_sd_ = logged.mean(axis=0), logged.std(axis=0)
            # Constant features (e.g. no paths in a domain view): rounding leaves a tiny sd, don't divide by it.
            self.feature_sd_[self.feature_sd_ < 1e-6] = 1.0
            features = self._scale(features)
            options["n_features"] = features.shape[1]

        self.net_ = ARCHITECTURES[self.arch](len(self.vocab_), **options).to(self.device)
        optimiser = torch.optim.Adam(self.net_.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss()

        self.history_, best_loss, best_state, waited = [], np.inf, None, 0
        for epoch in range(1, self.max_epochs + 1):
            start = time.perf_counter()
            self.net_.train()
            total = 0.0
            for batch in _length_batches(lengths[fit_rows], self.batch_size, rng):
                rows = fit_rows[batch]
                optimiser.zero_grad()
                loss = loss_fn(self._forward(ids[rows], lengths[rows], _rows(features, rows)),
                               torch.from_numpy(y[rows]).to(self.device))
                loss.backward()
                optimiser.step()
                total += loss.item() * len(rows)

            stop_prob = self._predict(ids[stop_rows], lengths[stop_rows], _rows(features, stop_rows))
            stop_loss = _log_loss(y[stop_rows], stop_prob)
            self.history_.append({"epoch": epoch, "train_loss": total / len(fit_rows),
                                  "stop_loss": stop_loss, "seconds": round(time.perf_counter() - start, 1)})
            if self.verbose:
                print(self.history_[-1])
            if stop_loss < best_loss:
                best_loss, best_state, waited = stop_loss, copy.deepcopy(self.net_.state_dict()), 0
            else:
                waited += 1
                if waited >= self.patience:
                    break

        self.net_.load_state_dict(best_state)
        self.best_epoch_ = min(self.history_, key=lambda h: h["stop_loss"])["epoch"]
        return self

    def predict_proba(self, X) -> np.ndarray:
        prob = self.predict_encoded(*self.encode(X))
        return np.column_stack([1 - prob, prob])

    def encode(self, X) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Input → (character ids, lengths, scaled features or None), as the network sees it.

        Explanations (phishguard.explain) edit these, e.g. hide one character,
        and call predict_encoded() on the result.
        """
        urls, features = _split_input(X)
        ids, lengths = self.vocab_.encode(urls, self.max_len)
        return ids, lengths, None if features is None else self._scale(features)

    def predict_encoded(self, ids: np.ndarray, lengths: np.ndarray, features=None, logits: bool = False) -> np.ndarray:
        """Probability of phishing for already-encoded input, or the raw log-odds with logits=True.

        Explanations use the log-odds: for very confident URLs the probability
        rounds to exactly 1.0, so hiding a character could never change it.
        """
        return self._predict(ids, lengths, features, logits)

    def save(self, path) -> None:
        """Save everything needed to predict: settings, vocabulary, feature scaling, weights."""
        state = {
            "settings": {"arch": self.arch, "max_len": self.max_len, "max_epochs": self.max_epochs,
                         "patience": self.patience, "batch_size": self.batch_size, "lr": self.lr,
                         "seed": self.seed, **self.arch_options},
            "vocab": self.vocab_.ids, "min_count": self.vocab_.min_count,
            "weights": {k: v.cpu() for k, v in self.net_.state_dict().items()},
            "history": self.history_, "best_epoch": self.best_epoch_,
        }
        if hasattr(self, "feature_mean_"):
            state["feature_mean"] = torch.from_numpy(self.feature_mean_)
            state["feature_sd"] = torch.from_numpy(self.feature_sd_)
        torch.save(state, path)

    @classmethod
    def load(cls, path, device=None) -> "CharURLClassifier":
        state = torch.load(path, weights_only=True)         # weights_only: never runs code from the file
        model = cls(device=device, **state["settings"])
        model.vocab_ = CharVocab(state["min_count"])
        model.vocab_.ids = state["vocab"]
        options = dict(model.arch_options)
        if "feature_mean" in state:
            model.feature_mean_, model.feature_sd_ = state["feature_mean"].numpy(), state["feature_sd"].numpy()
            options["n_features"] = len(model.feature_mean_)
        model.net_ = ARCHITECTURES[model.arch](len(model.vocab_), **options).to(model.device)
        model.net_.load_state_dict(state["weights"])
        model.history_, model.best_epoch_ = state["history"], state["best_epoch"]
        return model

    def _scale(self, features: np.ndarray) -> np.ndarray:
        return ((np.log1p(features.astype(np.float64)) - self.feature_mean_) / self.feature_sd_).astype(np.float32)

    def _forward(self, ids: np.ndarray, lengths: np.ndarray, features=None) -> torch.Tensor:
        width = max(lengths.max(), self.net_.min_len)        # pad only up to the longest URL in the batch
        inputs = [torch.from_numpy(ids[:, :width].astype(np.int64)).to(self.device),
                  torch.from_numpy(lengths).to(self.device)]
        if features is not None:
            inputs.append(torch.from_numpy(features).to(self.device))
        return self.net_(*inputs)

    @torch.no_grad()
    def _predict(self, ids: np.ndarray, lengths: np.ndarray, features=None, logits: bool = False) -> np.ndarray:
        self.net_.eval()
        out = np.empty(len(lengths), dtype=np.float64)
        for batch in _length_batches(lengths, 1024):
            scores = self._forward(ids[batch], lengths[batch], _rows(features, batch))
            out[batch] = (scores if logits else torch.sigmoid(scores)).cpu().numpy()
        return out


def _split_input(X) -> tuple[pd.Series, np.ndarray | None]:
    """A Series of URLs → (urls, None); a DataFrame with a "url" column → (urls, other columns as floats)."""
    if isinstance(X, pd.Series):
        return X.reset_index(drop=True), None
    return X["url"].reset_index(drop=True), X.drop(columns="url").to_numpy(dtype=np.float32)


def _rows(features, rows):
    return None if features is None else features[rows]


def _length_batches(lengths: np.ndarray, batch_size: int, rng=None) -> list[np.ndarray]:
    """Row batches of similar-length URLs, so little time is spent on padding.

    With `rng` (training): shuffle, sort within chunks of 50 batches, shuffle
    the batch order. Without it (prediction): simply sorted by length.
    """
    if rng is None:
        order = np.argsort(lengths, kind="stable")
        return [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    order = rng.permutation(len(lengths))
    chunk = batch_size * 50
    batches = []
    for i in range(0, len(order), chunk):
        part = order[i:i + chunk]
        part = part[np.argsort(lengths[part], kind="stable")]
        batches += [part[j:j + batch_size] for j in range(0, len(part), batch_size)]
    return [batches[i] for i in rng.permutation(len(batches))]


def _log_loss(y: np.ndarray, prob: np.ndarray) -> float:
    prob = np.clip(prob, 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(prob) + (1 - y) * np.log(1 - prob)))
