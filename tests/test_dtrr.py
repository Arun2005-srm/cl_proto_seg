"""DTRR and LoRA-audit tests.

    python -m pytest -q tests/test_dtrr.py                 # unit tests, seconds
    DTRR_SLOW=1 python -m pytest -q tests/test_dtrr.py     # + 2-task training runs on data/bddidd_small (minutes)
"""
import json
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from drift.dtrr import (DTRRError, DualController, SketchStore, classify_classes, hist_cdf, margin_edges,  # noqa: E402
                        margin_hist, margins_of, project_conflict, verify_reload)
from drift.models.components import FusionDecoder  # noqa: E402
from drift.models.drift import DRIFT  # noqa: E402
from drift.models.lora import lora_disabled, lora_layers  # noqa: E402

from test_review_fixes import batch, tiny_cfg  # noqa: E402

SLOW = pytest.mark.skipif(not os.environ.get("DTRR_SLOW"), reason="training run; set DTRR_SLOW=1")
SMALL = os.path.join(ROOT, "data", "bddidd_small")


def _entry(D=6, task=0, count=100, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"task": task, "mean": torch.randn(D, generator=g), "var": torch.rand(D, generator=g),
            "hist": torch.rand(8, generator=g), "count": count, "low_confidence": count < 20,
            "slot": torch.randn(2, 4, generator=g), "slot_count": torch.tensor([3.0, 0.0]),
            "stats_key": torch.randn(5, generator=g), "gn_mean": None, "gn_var": None,
            "err0": 0.1, "self_consistency": 0.9}


# ---------------------------------------------------------------- sketch store
def test_sketch_save_load_roundtrip(tmp_path):
    st = SketchStore([f"c{i}" for i in range(4)], margin_edges(8, 2.0))
    st.add(0, _entry(seed=0))
    st.add(2, _entry(seed=1, count=5))
    st.add(2, _entry(seed=2, task=1))
    st.task_protos[0] = torch.randn(4, 3)
    path = str(tmp_path / "mem.pt")
    st.save(path)
    fresh = verify_reload(st, path, seen=[0, 2])
    assert sorted(fresh.entries) == [0, 2] and len(fresh.entries[2]) == 2
    for c in (0, 2):
        assert fresh.digest(c) == st.digest(c)
    assert fresh.entries[2][0]["low_confidence"] is True
    assert torch.equal(fresh.entries[0][0]["mean"], st.entries[0][0]["mean"])
    with pytest.raises(DTRRError, match="seen class 'c1'"):
        verify_reload(st, path, seen=[0, 1])


def test_sketch_reload_rejects_nan(tmp_path):
    st = SketchStore(["a", "b"], margin_edges(8, 2.0))
    e = _entry()
    st.add(0, e)
    path = str(tmp_path / "mem.pt")
    st.save(path)
    st.entries[0][0] = dict(e, mean=e["mean"].clone())
    bad = SketchStore.load(path)
    bad.entries[0][0]["mean"][0] = float("nan")
    bad.save(path)
    with pytest.raises(DTRRError, match="NaN"):
        verify_reload(st, path, seen=[0])


def test_sketch_sample_matches_moments():
    e = _entry(D=3)
    x = SketchStore.sample(e, 20000, torch.Generator().manual_seed(0))
    assert torch.allclose(x.mean(0), e["mean"], atol=0.03)
    assert torch.allclose(x.var(0), e["var"], atol=0.03)


# ---------------------------------------------------------------- margins
def test_margin_hist_cdf():
    edges = margin_edges(4, 2.0)                        # bins [-2,-1), [-1,0), [0,1), [1,2)
    h = margin_hist(torch.tensor([-1.5, -0.5, 0.5, 0.5, 1.5, 5.0, -9.0]), edges)
    assert h.tolist() == [2, 1, 2, 2]                    # out-of-range values clamp into the end bins
    assert hist_cdf(h, edges, -2.0) == 0.0
    assert hist_cdf(h, edges, 0.0) == pytest.approx(3 / 7)
    assert hist_cdf(h, edges, 0.5) == pytest.approx(4 / 7)      # half of the [0,1) bin
    assert hist_cdf(h, edges, 2.0) == pytest.approx(1.0)
    assert hist_cdf(torch.zeros(4), edges, 1.0) == 0.0


def test_margins_of_respects_allowed_set():
    lg = torch.tensor([[3.0, 1.0, 5.0], [0.0, 2.0, 1.0]])
    y = torch.tensor([0, 1])
    allowed = torch.tensor([True, True, False])
    assert margins_of(lg, y, allowed).tolist() == [2.0, 2.0]
    assert margins_of(lg, y, torch.ones(3, dtype=torch.bool)).tolist() == [-2.0, 1.0]


# ---------------------------------------------------------------- controller
def test_dual_update_rises_clips_and_decays():
    c = DualController(3, alpha=0.5, beta=0.0, lam_max=1.0, rho=0.1)
    elig = torch.tensor([True, True, False])
    stale = torch.zeros(3)
    F = torch.tensor([0.9, 0.1, 0.9])
    for _ in range(10):
        c.update(F, elig, stale, used_eps=True)
    assert c.lam[0] == pytest.approx(1.0)                # clipped at lam_max
    assert c.lam[1] == pytest.approx(0.0)                # F == rho: no drift of lam
    assert c.lam[2] == 0.0                               # not eligible: no replay source
    for _ in range(3):
        c.update(torch.zeros(3), elig, stale, used_eps=True)
    assert c.lam[0] == pytest.approx(1.0 - 3 * 0.5 * 0.1)   # decays by alpha * rho per update
    for _ in range(100):
        c.update(torch.zeros(3), elig, stale, used_eps=True)
    assert c.lam.min() == 0.0                            # never negative
    assert c.updates == c.eps_updates == 113


def test_staleness_head_start_only_with_measured_forgetting():
    c = DualController(2, alpha=0.5, beta=0.05, lam_max=5.0, rho=0.02)
    elig = torch.ones(2, dtype=torch.bool)
    stale = torch.tensor([3.0, 3.0])
    c.update(torch.tensor([0.0, 0.03]), elig, stale, used_eps=True)
    assert c.lam[0] == 0.0
    assert c.lam[1] == pytest.approx(0.5 * 0.01 + 0.05)


def test_rho_by_share_gives_rare_classes_a_tighter_budget():
    c = DualController(3, rho=0.02)
    c.set_budget(torch.tensor([0.9, 0.1, 0.0]))
    assert c.rho_c[1] < c.rho_c[0]
    assert c.rho_c[2] == pytest.approx(0.02)


# ---------------------------------------------------------------- gradient surgery
def test_projection_removes_conflict():
    g_cur, g_mem = torch.tensor([1.0, -2.0, 0.5]), torch.tensor([0.0, 1.0, 1.0])
    g, dot, proj = project_conflict(g_cur, g_mem)
    assert dot < 0 and proj
    assert float(torch.dot(g, g_mem)) == pytest.approx(0.0, abs=1e-6)


def test_projection_leaves_aligned_gradients_alone():
    g_cur, g_mem = torch.tensor([1.0, 2.0]), torch.tensor([0.5, 0.1])
    g, dot, proj = project_conflict(g_cur, g_mem)
    assert dot > 0 and not proj and torch.equal(g, g_cur)


# ---------------------------------------------------------------- feature injection level
def test_forward_pixels_reproduces_dense_decoder():
    torch.manual_seed(0)
    ch = [8, 16, 32, 32]
    dec = FusionDecoder(ch, 32).eval()
    feats = [torch.randn(2, c, 20 // 2 ** i + (i == 3), 24 // 2 ** i) for i, c in enumerate(ch)]
    z = dec(feats)
    cap = {}
    h = dec.norm_module().register_forward_pre_hook(lambda m, i: cap.update(x=i[0]))
    dec(feats[:1] and [f[1:] for f in feats])           # GroupNorm statistics of image 1
    h.remove()
    xg = cap["x"].reshape(1, dec.norm_module().num_groups, -1)
    stats = (xg.mean((0, 2)), xg.var(2, unbiased=False).mean(0))
    yx = torch.stack(torch.meshgrid(torch.arange(20), torch.arange(24), indexing="ij"), -1).reshape(-1, 2)
    u = FusionDecoder.gather_inputs(feats, torch.ones(len(yx), dtype=torch.long), yx)
    zp = dec.forward_pixels(u, stats)
    assert torch.allclose(zp, z[1].flatten(1).t(), atol=1e-4)


def test_class_cells_keep_tiny_classes():
    from drift.dtrr import class_cells
    y = torch.zeros(1, 16, 16, dtype=torch.long)
    y[0, 5, 5] = 3                                       # one pixel, not at a stride-4 cell centre
    y[0, :, 15] = 255
    cells, lz = class_cells(y, 4, 4, 5, 255)
    assert 3 not in lz.unique().tolist()
    assert cells[3].tolist() == [[0, 1, 1]]
    assert len(cells[0]) == 16 and 255 not in cells


def test_classify_classes_minor_and_absent():
    pix = np.array([5000, 3000, 1000, 400, 300, 30, 0], dtype=float)
    info = classify_classes({"pixels": pix, "images": np.ones(7), "n_images": 1})
    assert info["absent"].tolist() == [False] * 6 + [True]
    assert info["minor"].tolist() == [False, False, False, False, True, True, False]
    assert info["q"][4] > 0.005                          # minor through the p25 rule, not the 0.5% floor


# ---------------------------------------------------------------- LoRA
def _lora_model():
    cfg = tiny_cfg(**{"model.encoder": {"name": "simple_cnn", "freeze": True, "channels": [8, 16, 32, 32]},
                      "model.lora.targets": ["query", "key", "value"]})
    torch.manual_seed(0)
    m = DRIFT(cfg).eval()
    with torch.no_grad():
        for l in lora_layers(m.encoder):
            l.lora_B.normal_(0, 0.1)
    return cfg, m


def test_lora_merge_and_reset_keeps_output_and_starts_from_zero():
    _, m = _lora_model()
    x, _ = batch()
    with torch.no_grad():
        before = m(x)["logits"]
        m.merge_and_reset_lora(seed=1)
        after = m(x)["logits"]
    assert torch.allclose(before, after, atol=1e-4)
    assert all(float(l.lora_B.abs().sum()) == 0 and l.lora_A.requires_grad and l.lora_B.requires_grad
               for l in lora_layers(m.encoder))
    assert m.lora_merges == 1


def test_checkpoint_after_merge_keeps_merged_base_weights():
    _, m = _lora_model()
    m.merge_and_reset_lora(seed=1)
    keys = m.checkpoint_state().keys()
    assert any(k.endswith("query.base.weight") for k in keys), "merged base weights must be saved"
    assert not any(k.startswith("encoder.stages.0.0") for k in keys), "untouched frozen weights stay out"


def test_lora_disabled_bypasses_adapters():
    _, m = _lora_model()
    x, _ = batch()
    with torch.no_grad():
        on = m(x)["logits"]
        with lora_disabled(m.encoder):
            off = m(x)["logits"]
        again = m(x)["logits"]
    assert (on - off).abs().mean() > 0
    assert torch.equal(on, again)


# ---------------------------------------------------------------- training runs
def _train(opts, name):
    from drift.config import load_config
    from drift.engine import Trainer
    from drift.utils import make_run_dir, seed_everything

    cfg = load_config(os.path.join(ROOT, "configs", "bdd_idd_smoke.yaml"),
                      [f"experiment.name={name}", "experiment.output_dir=outputs/_pytest"] + opts)
    seed_everything(cfg.experiment.seed, cfg.experiment.deterministic)
    tr = Trainer(cfg, make_run_dir(cfg))
    tr.fit()
    return tr


@SLOW
@pytest.mark.skipif(not os.path.isdir(SMALL), reason="needs data/bddidd_small")
def test_regression_disabled_matches_baseline():
    """dtrr.enabled=false (+ the read-only LoRA audit) reproduces the pre-DTRR code bit for bit."""
    os.chdir(ROOT)
    tr = _train(["dtrr.enabled=false"], "regress")
    with open(os.path.join(ROOT, "tests", "fixtures", "baseline_bdd_idd_smoke.json"), encoding="utf-8") as f:
        ref = json.load(f)
    assert np.asarray(tr.R).tolist() == ref["R"]
    assert [r["train_loss"] for r in tr.history] == ref["train_loss"]
    assert [r.get("mIoU") for r in tr.history] == ref["mIoU"]


@SLOW
@pytest.mark.skipif(not os.path.isdir(SMALL), reason="needs data/bddidd_small")
def test_smoke_two_tasks_dtrr_proof_of_life():
    os.chdir(ROOT)
    tr = _train(["dtrr.enabled=true", "dtrr.interval=10"], "smoke")
    t0, t1 = tr.dtrr.report["tasks"]
    assert t0["self_test"] and t1["self_test"]
    assert min(t0["eps_curve"]) > 0 and np.std(t1["eps_curve"]) > 0
    assert set(t0["sketched"]) <= set(t1["prev_classes_present"])
    assert all(v > 0 for v in t1["pushes"].values())
    assert tr.dtrr.ctrl.eps_updates > 0


@SLOW
@pytest.mark.skipif(not os.path.isdir(SMALL), reason="needs data/bddidd_small")
def test_alpha_insensitivity():
    """Final seen-task mIoU moves by less than 3 points over alpha in [0.1, 1]."""
    os.chdir(ROOT)
    res = {}
    for a in (0.1, 0.5, 1.0):
        tr = _train(["dtrr.enabled=true", "dtrr.interval=10", f"dtrr.controller.alpha={a}"], f"alpha{a}")
        res[a] = float(np.nanmean(tr.R[-1]))
    print("final avg mIoU by alpha:", {k: round(100 * v, 2) for k, v in res.items()})
    assert max(res.values()) - min(res.values()) < 0.03
