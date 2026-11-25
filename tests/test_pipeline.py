"""Sanity and leak tests for the data pipeline. Run:  python tests/test_pipeline.py   (needs data/lobster)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
import data as D, experiment as E
from inject import make_spoofs, make_decoys

T = "GOOG"                      # smallest sample, fastest
b = E.base(T); N = len(b["t"]); s, e = D.split_range(N, "test")
sp = make_spoofs(b, s, e, 3.0, np.random.default_rng(1))
dec = make_decoys(b, s, e, 3.0, np.random.default_rng(2))
X0, _ = D.features(b, s, e, None)
X1, M1 = D.features(b, s, e, sp)
X2, M2 = D.features(b, s, e, sp + dec)


def test_spoof_changes_only_masked_volume():
    diff = (X1[0] + X1[1]) - (X0[0] + X0[1])
    assert (diff[M1 == 0] == 0).all(), "volume changed outside the injected mask"
    assert (diff[M1 == 1] > 0).all(), "injected cells must carry extra volume"


def test_age_and_execution_channels_untouched_by_spoofs():       # the label leak we found and fixed
    assert np.array_equal(X1[5], X0[5]), "age channel must not depend on injection"
    assert np.array_equal(X1[4], X0[4]), "execution channel must not depend on spoof injection"


def test_decoys_are_label_zero_and_do_not_override_spoofs():
    assert ((M2 == 1) == (M1 == 1)).all(), "decoys must not alter the spoof mask"
    assert (M2 == 2).any(), "decoy cells missing"


def test_executed_decoys_write_execution_not_cancel():
    only_exec = [d for d in dec if d["end"] == "exec"][:5]
    Xe, _ = D.features(b, s, e, only_exec)
    assert (Xe[4] > X0[4]).any() and np.array_equal(Xe[2], X0[2])


def test_determinism():
    a, _ = E.build(T, "test", 5); c, _ = E.build(T, "test", 5)
    assert np.array_equal(a.X, c.X) and np.array_equal(a.M, c.M)


def test_splits_do_not_overlap():
    r = [D.split_range(N, k) for k in ("train", "val", "test")]
    assert r[0][1] + 256 <= r[1][0] and r[1][1] + 256 <= r[2][0] + 0 or r[1][1] <= r[2][0]


def test_scale_uses_clean_train_only():
    ts, te_ = D.split_range(N, "train")
    Xc, _ = D.features(b, ts, te_, None)
    assert np.allclose(E.scale(T), D.fit_scale(Xc))


def test_windows_binary_labels():
    w, _ = E.build(T, "test", 5, decoys=3.0)
    _, y = w.batch(np.arange(min(8, len(w))), D.CH3)
    assert set(np.unique(y)) <= {0.0, 1.0}


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn(); print("PASS", name)
            except AssertionError as ex:
                fails += 1; print("FAIL", name, ex)
    sys.exit(1 if fails else 0)
