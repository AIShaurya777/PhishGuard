# 1 · Data and split

Code: `src/phishguard/data.py`, `src/phishguard/split.py`.
Notebooks: `01_data_loading.ipynb`, `01b_train_test_split.ipynb`.

## Loading the data
Every notebook loads data the same way, so the whole team works on identical rows:

```python
from phishguard.data import load_clean, TARGET   # TARGET = "is_phishing"
df = load_clean()
```

`load_clean()` does three things, in this order.

### 1. Checks the file is complete
```python
EXPECTED_SHAPE = (235_795, 55)
if df.shape != EXPECTED_SHAPE:
    raise ValueError("... The file is probably truncated ...")
```
**Why:** our first notebook was given a Colab upload that had been cut off (196,403 rows instead of 235,795), and nothing complained. Now the code stops immediately if that happens.

### 2. Fixes the label direction
In the original data, `label = 1` means **legitimate**. Precision, recall and F1 are calculated for class 1, and we want them to describe how well we catch **phishing**. So we flip it:
```python
df["is_phishing"] = (df["label"] == 0).astype(int)   # 1 = phishing
df = df.drop(columns="label")                          # nobody can use the old one by mistake
```

### 3. Cleans the URLs
| Step | Why |
|---|---|
| Remove a trailing `/` | 36.5% of phishing URLs end with `/`, and **zero** legitimate URLs do. A model would learn "ends with / → phishing", which is about how the data was collected, not about phishing. |
| Drop URLs that appear with both labels | `https://www.education.gouv.fr` was labelled both ways. We can't know which is right, so both rows go. |
| Drop duplicates | 1,589 repeated URLs. Otherwise the same URL could be in training *and* testing. |

**Result:** 234,204 rows, 134,849 legitimate (57.6%) and 99,355 phishing (42.4%).

## The split: why not just random?
```python
from phishguard.split import get_splits
train_idx, val_idx, test_idx = get_splits(df)    # 70% / 15% / 15%
```

Attackers often create many phishing URLs on one domain: `a.evil.com`, `b.evil.com`, `c.evil.com`. With a **random** split, some land in training and some in testing. The model then just recognises `evil.com` and looks better than it really is.

We measured it: with a random split, **39% of phishing test URLs** have a "sibling" from the same owner in training.

So we split by **site owner**: all URLs of one owner go to the same split.

### What is a "site owner"?
| URL | Owner | Why |
|---|---|---|
| `http://a.evil.com/login` | `evil.com` | normal domain |
| `http://login.paypal.com.evil.co.uk/x` | `evil.co.uk` | `.co.uk` is one suffix |
| `https://foo.web.app` | `foo.web.app` | `web.app` is a hosting platform; every customer is a different person |
| `http://192.168.0.1/admin` | `192.168.0.1` | IP addresses are their own owner |
| `http://www.paypal.com:911@evil.top/fix` | `evil.top` | everything before `@` is a fake "username"; the browser really goes to `evil.top` |

This uses the **Public Suffix List** (through the `tldextract` library), which knows that `.co.uk` and `web.app` are suffixes.

```python
def parse_host(url):                      # urls.py — uses Python's own URL parser
    return split_url(url).hostname or ""  # handles user@host, ports, IPv6, missing "http://"

def site_owner(urls):
    return hostnames(urls).map(lambda h: TLD_EXTRACT(h).top_domain_under_public_suffix or h)
```

**A bug we found and fixed (D11).** Our first version found the host with a hand-written regex. It read `http://www.paypal.com:911@xn--80aaa….xn--p1ai/fix-account` as belonging to **`paypal.com`**, but the browser actually goes to the host after the `@`. That's a classic phishing trick, and 10 URLs in the dataset use it. We switched to Python's `urlsplit`, added tests for these cases, and re-made the split once, on purpose.

### How the split is made
1. Group URLs by owner.
2. Separately for phishing owners and legitimate owners, shuffle the owners (seed 42).
3. Fill test up to 15% of that class's rows, then validation up to 15%, and put the rest in train.

Because we do it per class, all three splits have the same phishing share (about 42.4%). Result (seed 42): **164,045 / 35,029 / 35,130** rows. Tests check that **no owner appears in two splits**, and that the split hasn't changed by accident (a fingerprint of the seed-42 split).

### Two facts about owners
- **105 owners (560 rows) have both labels**, e.g. `pinata.cloud`, `washington.edu`, `adobe.com`. Usually a real site with one abused page. Each owner goes to one split with its majority label, so a few test URLs can never all be classified correctly.
- **Some owners are huge.** `ipfs.io` alone has ~1,400 phishing URLs, and the top 5 owners are 15.6% of phishing test URLs. One split can therefore be luckier than another, which is why every model is evaluated on **5 splits** (seeds 42, 0, 1, 2, 3) and we report mean ± sd.

## Likely viva questions
**Q: Why did you flip the label?**
A: Metrics like precision and recall are computed for class 1. We want them to measure phishing detection, so phishing must be 1.

**Q: Why remove the trailing slash?**
A: Only phishing URLs had it (36.5% vs 0%). It's an artifact of how the data was collected, so a model would learn it as a shortcut.

**Q: Why not use `train_test_split`?**
A: A random split put 39% of phishing test URLs next to a training URL from the same owner. The model could recognise the owner instead of detecting phishing. We split by site owner so test URLs come from owners the model has never seen.

**Q: What's the difference between validation and test?**
A: We use validation to choose settings (hyperparameters, threshold). Test is used once at the end. If we chose settings using test, the test score would be too optimistic.

**Q: Why is the split stratified?**
A: So each split has the same share of phishing (42.4%). Otherwise one split could, by chance, be much easier or harder.

**Q: Any limitation in your split?**
A: Hosting platforms that aren't on the Public Suffix List (e.g. `weeblysite.com`, 3,079 URLs) are treated as one big owner. Also, a phishing *campaign* that uses several unrelated domains isn't grouped.

**Q: Why 5 splits instead of one?**
A: One split can be lucky. Our first domain-only result (88.6%) came from the most favourable of 6 splits we later tried. With 5 splits we report mean ± sd, so the professor sees how much the number moves.

**Q: What does `http://paypal.com@evil.top` do?**
A: Everything before `@` is a "username" that the browser ignores for navigation; it goes to `evil.top`. Attackers use it to make a URL look like PayPal. Our first parser got this wrong; we fixed it and added tests.
