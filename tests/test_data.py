import pandas as pd
import pytest

from phishguard import data


def make_df(rows):
    return pd.DataFrame(rows, columns=["URL", "label"])


def test_add_target_maps_phishing_to_one():
    df = data.add_target(make_df([("https://a.com", 1), ("http://b.xyz/login", 0)]))
    assert df[data.TARGET].tolist() == [0, 1]
    assert "label" not in df.columns


def test_clean_strips_trailing_slash_and_dedupes():
    df = data.add_target(make_df([
        ("http://evil.com/x/", 0),
        ("http://evil.com/x", 0),
        ("https://good.com", 1),
    ]))
    out, report = data.clean(df)
    assert sorted(out["URL"]) == ["http://evil.com/x", "https://good.com"]
    assert report["duplicate_rows_dropped"] == 1


def test_clean_drops_every_copy_of_a_conflicting_url():
    df = data.add_target(make_df([
        ("https://site.fr/", 0),
        ("https://site.fr", 1),
        ("https://good.com", 1),
    ]))
    out, report = data.clean(df)
    assert out["URL"].tolist() == ["https://good.com"]
    assert report["conflicting_label_rows_dropped"] == 2


def test_load_raw_rejects_truncated_file(tmp_path):
    truncated = tmp_path / "truncated.csv"
    make_df([("https://a.com", 1)]).to_csv(truncated, index=False)
    with pytest.raises(ValueError, match="truncated"):
        data.load_raw(truncated)


@pytest.mark.skipif(not data.RAW_PATH.exists(), reason="raw dataset not present")
def test_real_dataset_loads_cleanly():
    df = data.load_clean()
    assert df["URL"].is_unique
    assert not df["URL"].str.endswith("/").any()
    assert set(df[data.TARGET].unique()) == {0, 1}
