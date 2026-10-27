# 4 · Deep learning on characters (Milestone 4)

Code: `src/phishguard/deep.py`. Notebook: `08_deep_models.ipynb`. Decision: D12.

## Why deep learning?
In Milestone 3 the char n-gram model beat XGBoost on every view, because the useful signal is in the **characters** of the URL (`.cf`, `paypa1`, `x7q9kz`), and our 40 count features throw that detail away. A character-level neural network reads the characters directly and **learns its own patterns** instead of using a fixed list of n-grams.

## Step 1: characters → numbers
```python
vocab = CharVocab().fit(train_urls)              # built on training URLs only
ids, lengths = vocab.encode(pd.Series(["ipfs.io"]), max_len=10)
# ids = [[i, p, f, s, ., i, o, PAD, PAD, PAD]] as integers
```
- Every character that appears at least 10 times in training gets its own id. Rarer characters share `UNK` (= 1).
- `PAD` (= 0) fills short URLs up to the length of the longest URL in the batch. The models are built so padding **never** changes a prediction, and a test checks this.
- URLs are lowercased and cut at 200 characters (99.5% are shorter).
- The first layer, the **embedding**, turns each id into 32 learned numbers. Characters that behave alike (say, digits) end up with similar vectors.

## Model A: Char-CNN (convolutional network)
```
URL → embedding (32 per char) → 4 convolutions (widths 3,4,5,6; 128 filters each)
    → max over positions → dense 128 → dropout → probability of phishing
```
- A **filter** of width 4 is a small set of weights that slides along the URL, 4 characters at a time, and gives a high score where it sees "its" pattern. It's a **learned 4-gram detector**.
- **Max over positions** keeps each filter's best match anywhere in the URL. So the CNN asks "does this pattern appear somewhere?", just like the n-gram model, except that the patterns are learned and similar patterns can share a filter (`paypa1`, `paypai`).
- 512 pattern scores go into a small dense layer, which outputs one number. A sigmoid turns that into a probability.

## Model B: BiLSTM (recurrent network)
```
URL → embedding → LSTM left→right + LSTM right→left (64 units each) → max over positions → dense → probability
```
- An **LSTM** reads one character at a time and keeps a **memory** (64 numbers) that it updates at each step. Its "gates" learn what to remember and what to forget.
- **Bidirectional:** a second LSTM reads the URL backwards, so every position knows what came before *and* after it.
- Unlike the CNN, it can in principle use **order and long-range context** (e.g., "a brand name followed later by a cheap TLD").

## How they're trained (same fair routine as every model)
```python
make_cnn = lambda dropout: CharURLClassifier(arch="cnn", dropout=dropout)
run("Char-CNN | domain only", make_cnn, param_grid(dropout=[0.3, 0.5]), urls, y, df,
    features="domain only", group="m4_deep")
```
`CharURLClassifier` has the same `fit` / `predict_proba` interface as scikit-learn, so `experiment.run()` works unchanged: 5 splits, tuning on validation, threshold on validation, test once.

| Choice | Value | Why |
|---|---|---|
| Loss | binary cross-entropy (log-loss) | the standard loss for a probability |
| Optimiser | Adam, learning rate 0.001 | adapts the step size per weight; works without much tuning |
| Batch | 256 URLs **of similar length** | less padding means faster training |
| Early stopping | 10% of training *site owners* held out; keep the best epoch, stop after 2 epochs without improvement (max 15) | stops before it memorises the training URLs, without touching the validation split |
| Tuned | dropout ∈ {0.3, 0.5} | dropout randomly switches off neurons during training, so the model can't rely on any single one (regularisation) |
| Hardware | Apple GPU (MPS) | about 3× faster than the CPU for the CNN on this laptop |

## Results
Test accuracy, mean ± sd over 5 splits (`notebooks/08_deep_models.ipynb`):

| View | Char n-gram LogReg | Char-CNN | BiLSTM |
|---|---|---|---|
| raw URL | 99.69 ± 0.05% | 99.68 ± 0.03% | 99.68 ± 0.04% |
| domain only | 86.27 ± 0.89% | **86.58 ± 0.92%** | 85.55 ± 1.35% |
| registered domain | **82.49 ± 1.93%** | 80.85 ± 1.53% | 80.95 ± 0.62% |

**Is the deep model better?** We compared each model with the n-gram model on the same test URLs (McNemar) and split by split:

| View | Verdict |
|---|---|
| raw URL | **No difference** (p = 0.15 for the CNN, 0.47 for the BiLSTM). Every model just reads the `https://www.` shortcut. |
| domain only | **CNN slightly better at the threshold:** higher accuracy on 4 of 5 splits; on seed 42, 1,074 URLs only the CNN got right vs 898 only the n-gram got right (p = 0.0001). But its ROC-AUC is the same (0.9165 vs 0.9162), so it doesn't rank URLs better. It just raises fewer false alarms (3.3% vs 4.3%) at the same recall (~73%). |
| registered domain | **n-gram better**, on every split. Only ~14 characters per input: the simpler model generalises better. |

The BiLSTM is never better than the CNN, and up to 3× slower. **So the CNN goes into the hybrid model (Milestone 5).**

Training took 10–24 s per epoch on the Apple GPU. The best epoch was 4–11, and the whole notebook ran in about 2 hours.

## Likely viva questions
**Q: Deep learning is supposed to be better. Why isn't it here?**
A: The patterns that separate the classes in a domain name are short and local: cheap TLDs, digits, hyphens, look-alike brands. A TF-IDF n-gram model already captures exactly those, with strong regularisation. The CNN learns similar detectors and ties or gains slightly. The BiLSTM's strength, long-range order, isn't needed for ~15-character inputs. Published "deep beats classical" results are usually on raw URLs, where (in this dataset) everything is ~99.7% because of the prefix shortcut.

**Q: The CNN is significant on seed 42 but you say it's only "slightly better". Why?**
A: McNemar uses one split. Across 5 splits its accuracy is higher on 4 of them, but its ROC-AUC is not consistently higher, so it doesn't rank URLs better overall. It only makes fewer false alarms at the chosen threshold. We report exactly that.

**Q: What is a convolution here?**
A: A filter of, say, 4 × 32 weights slides along the URL's character vectors. At each position it multiplies and sums, giving a score for "how much these 4 characters look like my pattern". It's a learned n-gram detector.

**Q: Why max-pooling?**
A: It keeps the highest score of each filter over the whole URL, so the answer doesn't depend on *where* a pattern appears, and URLs of any length give the same number of outputs.

**Q: What's the difference between the CNN and the BiLSTM?**
A: The CNN looks for local patterns (3–6 characters) anywhere in the URL. The BiLSTM reads the whole sequence in both directions with a memory, so it can use order and longer context. It is slower (up to 3× here) because it reads one character after another.

**Q: How do you stop the network from overfitting?**
A: Three ways: dropout (tuned on validation), early stopping on held-out training owners, and the site-owner split, so the test set only contains websites the model never saw.

**Q: Why not early-stop on the validation split?**
A: The validation split already chooses the dropout and the threshold. Using it for a third decision would make its score more optimistic, so early stopping uses its own held-out part of the training data, also grouped by site owner.

**Q: Why are the results not exactly reproducible?**
A: The weights start from a fixed seed, but GPU arithmetic can add numbers in a different order from run to run. That changes the last decimals, which is why we report mean ± sd over 5 splits rather than a single number.

**Q: What does `PAD` do, and can it leak information?**
A: It fills shorter URLs so a batch is a rectangle. The CNN zeroes every window that only covers padding, and the LSTM skips padding entirely (packed sequences). A test checks that adding padding doesn't change the output, so URL length can't leak through padding.
