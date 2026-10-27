import numpy as np
import pandas as pd
import pytest
import torch

from phishguard.deep import PAD, UNK, BiLSTM, CharCNN, CharURLClassifier, CharVocab, HybridNet, _length_batches


def test_vocab_is_built_from_training_urls_only():
    vocab = CharVocab(min_count=1).fit(pd.Series(["abc.com"]))
    ids, lengths = vocab.encode(pd.Series(["ABZ"]), max_len=5)
    assert ids[0, 0] == vocab.ids["a"]          # lowercased
    assert ids[0, 2] == UNK                     # 'z' never seen in training
    assert list(ids[0, 3:]) == [PAD, PAD]
    assert lengths[0] == 3


def test_vocab_cuts_long_urls():
    vocab = CharVocab(min_count=1).fit(pd.Series(["abcdef"]))
    _, lengths = vocab.encode(pd.Series(["abcdef", ""]), max_len=4)
    assert list(lengths) == [4, 1]              # an empty URL becomes one UNK


@pytest.mark.parametrize("arch", [CharCNN, BiLSTM])
def test_padding_does_not_change_the_prediction(arch):
    torch.manual_seed(0)
    net = arch(vocab_size=10).eval()
    short = torch.tensor([[2, 3, 4, 5, 6, 7, 8]])
    padded = torch.tensor([[2, 3, 4, 5, 6, 7, 8, PAD, PAD, PAD, PAD, PAD]])
    alone = net(short, torch.tensor([7]))
    in_batch = net(padded, torch.tensor([7]))
    assert torch.allclose(alone, in_batch, atol=1e-6)


def test_length_batches_cover_every_row_once():
    lengths = np.random.default_rng(0).integers(1, 50, size=1000)
    for rng in [None, np.random.default_rng(1)]:
        rows = np.concatenate(_length_batches(lengths, 64, rng))
        assert sorted(rows) == list(range(1000))


def test_classifier_learns_a_simple_pattern_and_keeps_row_order():
    rng = np.random.default_rng(0)
    names = ["".join(rng.choice(list("abcdefgh"), 8)) for _ in range(600)]
    urls = pd.Series([f"{n}.{'top' if i % 2 else 'org'}" for i, n in enumerate(names)], index=range(100, 700))
    y = pd.Series([i % 2 for i in range(600)], index=urls.index)
    model = CharURLClassifier(arch="cnn", max_epochs=5, batch_size=32, device="cpu").fit(urls, y)
    prob = model.predict_proba(urls)[:, 1]
    assert prob.shape == (600,)
    assert ((prob >= 0.5) == y.to_numpy()).mean() > 0.95
    assert model.best_epoch_ == min(model.history_, key=lambda h: h["stop_loss"])["epoch"]


def test_hybrid_padding_does_not_change_the_prediction():
    torch.manual_seed(0)
    net = HybridNet(vocab_size=10, n_features=3).eval()
    features = torch.tensor([[0.5, -1.0, 2.0]])
    alone = net(torch.tensor([[2, 3, 4, 5, 6, 7, 8]]), torch.tensor([7]), features)
    in_batch = net(torch.tensor([[2, 3, 4, 5, 6, 7, 8, PAD, PAD, PAD]]), torch.tensor([7]), features)
    assert torch.allclose(alone, in_batch, atol=1e-6)


def test_hybrid_uses_the_features():
    # Identical-looking URLs: only the feature column tells the classes apart.
    rng = np.random.default_rng(0)
    n = 600
    y = pd.Series(np.arange(n) % 2, index=range(1000, 1000 + n))
    X = pd.DataFrame({"url": [f"{''.join(rng.choice(list('abcdefgh'), 8))}.com" for _ in range(n)],
                      "signal": y * 5.0 + rng.random(n), "constant": 3.0}, index=y.index)
    model = CharURLClassifier(arch="hybrid", max_epochs=5, batch_size=32, device="cpu").fit(X, y)
    prob = model.predict_proba(X)[:, 1]
    assert ((prob >= 0.5) == y.to_numpy()).mean() > 0.95
    assert model.feature_sd_[1] == 1.0          # a constant feature is left unscaled, not divided by 0


def test_save_and_load_give_identical_predictions(tmp_path):
    rng = np.random.default_rng(0)
    n = 200
    y = pd.Series(np.arange(n) % 2)
    X = pd.DataFrame({"url": [f"{''.join(rng.choice(list('abcdefgh'), 6))}.{'top' if i % 2 else 'org'}" for i in range(n)],
                      "signal": rng.random(n)})
    model = CharURLClassifier(arch="hybrid", max_epochs=2, batch_size=32, device="cpu").fit(X, y)
    model.save(tmp_path / "model.pt")
    loaded = CharURLClassifier.load(tmp_path / "model.pt", device="cpu")
    assert np.allclose(model.predict_proba(X), loaded.predict_proba(X))
    assert loaded.arch_options == model.arch_options
