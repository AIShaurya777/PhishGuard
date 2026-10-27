# 5 · Hybrid model (Milestone 5)

Code: `src/phishguard/deep.py` (`HybridNet`), `src/phishguard/ensemble.py` (`LogitAverage`). Notebook: `09_hybrid.ipynb`. Decision: D13.

## The idea
Our URL models see different things:
- the **char-CNN** learns character patterns (`paypa1`, `.top`, random strings);
- our **40 handcrafted features** describe counts and structure (subdomains, IP address, entropy, hosting platform …).

If they make *different* mistakes, combining them could fix some of each other's errors. That is what the project title promises: a **hybrid ML + DL** model.

## Two ways to combine
| | Hybrid network (early / joint fusion) | Log-odds average (late fusion) |
|---|---|---|
| How | One network: CNN pattern scores **and** features → shared dense layers | Two separate models (CNN, XGBoost), predictions averaged |
| Trained | together, end to end | separately |
| Can learn interactions? | yes (e.g. "this pattern matters only if there's no subdomain") | no, each model votes on its own |
| Code | `CharURLClassifier(arch="hybrid")` | `LogitAverage({...})` |

### The hybrid network
```
URL text ─► embedding ─► 4 convolutions ─► max-pool ─► 512 pattern scores ─┐
                                                                          ├─► concat (576) ─► dense 128 ─► dropout ─► probability
40 features ─► log(1+x) ─► standardise ─► dense 64 ───────────────────────┘
```
```python
X = extract_features(urls).assign(url=urls)          # 40 feature columns + the URL text
model = CharURLClassifier(arch="hybrid", dropout=0.3).fit(X.loc[train_idx], y.loc[train_idx])
```
- **Why `log(1 + x)`?** Counts like `url_length` range from 14 to 6,000. Without log-scaling, a few huge URLs would dominate the weights. After the log, we **standardise** (subtract the mean, divide by the sd) using the **training rows only**.
- **Constant features** (in the domain-only view there is no path, so `path_length` is always 0) are left unscaled instead of being divided by ~0. A test caught this.

### The log-odds average
```python
make_average = lambda dropout: LogitAverage({
    "cnn": (lambda: CharURLClassifier(arch="cnn", dropout=dropout), "url"),
    "xgb": (make_xgb, feature_cols),
})
```
Each model gives a probability. We turn each into **log-odds**, log(p / (1 − p)), average them, and turn the result back into a probability. Log-odds are fairer than plain probabilities when one model is more confident than the other.

## Results
Test accuracy, mean ± sd over 5 splits (`notebooks/09_hybrid.ipynb`):

| View | n-gram | Char-CNN | **Hybrid** | Average CNN + XGBoost |
|---|---|---|---|---|
| raw URL | 99.69 ± 0.05% | 99.68 ± 0.03% | 99.68 ± 0.03% | 99.68 ± 0.03% |
| domain only | 86.27 ± 0.89% | 86.58 ± 0.92% | **86.72 ± 0.99%** | 85.70 ± 0.99% |
| registered domain | **82.49 ± 1.93%** | 80.85 ± 1.53% | 82.01 ± 0.85% | 80.97 ± 1.80% |

**What it means:**
- **Domain only:** the hybrid is our best URL model. It beats the n-gram model and the CNN on 4 of 5 splits. The gain is small (+0.45 and +0.14 points) but points the same way almost every time, and McNemar on seed 42 agrees (p < 0.0001 and p = 0.027).
- **Registered domain:** adding the features fixes most of the CNN's weakness (+1.16 points, better on 5/5 splits), but the simple n-gram model is still best.
- **Raw URL:** everything ties at ~99.7%, because of the `https://www.` shortcut.
- **Averaging is worse than the CNN alone.** It gives the weaker XGBoost an equal vote. The hybrid network *learns* how much to trust each input, which is why joint training wins.

**Does it use both inputs?** (permutation test, seed 42, domain only)

| What we shuffled | Accuracy |
|---|---|
| nothing | 88.1% |
| the 40 features (only the characters still match) | 77.9% |
| the URL text (only the features still match) | 65.4% |

Both inputs matter, and the characters matter more.

## Likely viva questions
**Q: Why is the averaged model worse than the CNN alone?**
A: Averaging gives both models an equal vote. On domains, XGBoost on our features only gets 81.4%, so it pulls good CNN predictions in the wrong direction. The hybrid network learns its own weighting during training.

**Q: The hybrid is only 0.45 points better than the n-gram model. Is that worth it?**
A: It's small, smaller than the variation between splits. But it is better on 4 of 5 splits and significant on the seed-42 test set, so we call it "slightly but consistently better", not a breakthrough. The n-gram model is much simpler and remains a strong baseline. We report both.

**Q: What's the difference between early and late fusion?**
A: Early (joint) fusion combines the inputs inside one model, which is trained once. It can learn how the inputs interact. Late fusion trains separate models and combines only their outputs. It is simpler and more robust, but it can't learn interactions.

**Q: How do you know the hybrid uses both inputs?**
A: A permutation test (notebook 09, section 6). We shuffle the features across test URLs and the accuracy drops; we shuffle the URL text instead and it drops too. So the model relies on both. No retraining is needed.

**Q: Why standardise features for the neural network but not for XGBoost?**
A: Trees split on thresholds, so the scale of a feature doesn't matter. A neural network multiplies inputs by weights, so a feature measured in thousands would swamp one measured between 0 and 1, and training becomes unstable.

**Q: Isn't adding features cheating, if the features contain the shortcut?**
A: The features are computed **from the same view** the model sees. In the domain-only view, `extract_features` only gets the domain, so it can't see `https://www.` or the path.
