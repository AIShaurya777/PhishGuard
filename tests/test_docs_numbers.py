import pytest

from phishguard.evaluate import RESULTS_DIR
from phishguard.report import current_readme_block, results_markdown


@pytest.mark.skipif(not (RESULTS_DIR / "m3_views").exists(), reason="results not generated yet")
def test_readme_results_match_saved_results():
    assert current_readme_block() == results_markdown(), (
        "README results are out of date: run  .venv/bin/python -m phishguard.report"
    )
