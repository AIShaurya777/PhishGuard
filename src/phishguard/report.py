"""Build the README's results table from results/*.json, so the two can't drift apart.

Update the README with:  .venv/bin/python -m phishguard.report
tests/test_docs_numbers.py fails if the README is out of date.
"""

import re

from phishguard.data import REPO_ROOT
from phishguard.evaluate import load_results

README = REPO_ROOT / "README.md"
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"

# (results group, model name, short description of the input) in display order.
HEADLINE = [
    ("m2", "XGBoost (all columns)", "all provided columns"),
    ("m2", "Random Forest (all columns)", "all provided columns"),
    ("m2", "LogReg (all columns)", "all provided columns"),
    ("m2", "Average: n-gram + XGBoost (URL only)", "raw URL + URL features"),
    ("m2", "Char n-gram LogReg (raw URL)", "raw URL"),
    ("m2", "XGBoost (URL features)", "raw URL → 40 features"),
    ("m3_views", "Char n-gram LogReg | domain only", "**domain only** (shortcut-reduced)"),
    ("m3_views", "Char n-gram LogReg | registered domain", "**registered domain** (shortcut-reduced)"),
    ("m3_views", "XGBoost URL features | domain only", "**domain only** (shortcut-reduced)"),
]


def results_markdown() -> str:
    lines = [
        "Test accuracy, **mean ± sd over 5 site-owner splits**, and the seed-42 split with its 95% confidence interval.",
        "",
        "| Model | Input | Accuracy (5 splits) | Seed 42 [95% CI] | ROC-AUC |",
        "|---|---|---|---|---|",
    ]
    tables = {}
    for group, model, description in HEADLINE:
        if group not in tables:
            tables[group] = load_results(group).set_index("model")
        r = tables[group].loc[model]
        lines.append(
            f"| {model.split(' |')[0]} | {description} | {r['accuracy']:.2%} ± {r['accuracy_sd']:.2%} "
            f"| {r['ref_accuracy']:.2%} [{r['ref_ci_low']:.2%}, {r['ref_ci_high']:.2%}] | {r['roc_auc']:.4f} |"
        )
    lines += [
        "",
        "Every legitimate URL in PhiUSIIL is a bare `https://www.` homepage, so the near-perfect scores mostly "
        "reflect how the dataset was built ([shortcut analysis](docs/guide/03_shortcuts.md)). The "
        "shortcut-reduced rows show what a URL-only model can do from the domain name.",
    ]
    return "\n".join(lines)


def current_readme_block() -> str:
    text = README.read_text()
    match = re.search(re.escape(START) + r"\n(.*?)\n" + re.escape(END), text, flags=re.S)
    return match.group(1) if match else ""


def update_readme() -> None:
    text = README.read_text()
    block = f"{START}\n{results_markdown()}\n{END}"
    text = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: block, text, flags=re.S)
    README.write_text(text)


if __name__ == "__main__":
    update_readme()
    print("README results table updated.")
