"""Unit tests, one or more per item of the risk register (IDs in the test names).

Run:  python -m pytest -q tests
All tests use the small CNN encoder, so they run offline on CPU in seconds.
"""
import copy
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from drift.config import Config, load_config  # noqa: E402
from drift.data.datasets import Sample, _balanced_groups, build_tasks, natural_key  # noqa: E402
from drift.engine import build_optimizer, load_model, shift_diagnostics  # noqa: E402
from drift.losses import DRIFTLoss, ce_loss  # noqa: E402
from drift.metrics import SegMetrics, bootstrap_miou, class_forgetting, miou_from_conf  # noqa: E402
from drift.models.components import FusionDecoder, PrototypeGenerator  # noqa: E402
from drift.models.drift import DRIFT, build_anchors  # noqa: E402
from drift.models.lora import LoRALinear, inject_lora  # noqa: E402

C = 5


def tiny_cfg(**over):
    cfg = load_config(os.path.join(ROOT, "configs", "defaults.yaml"))
    d = cfg.to_dict()
    d["dataset"]["classes"] = [f"c{i}" for i in range(C)]
    d["model"]["encoder"] = {"name": "simple_cnn", "freeze": False, "channels": [8, 16, 32, 32]}
    d["model"]["embed_dim"], d["model"]["shift_dim"] = 32, 8
    d["model"]["prototypes"]["anchors"]["source"] = "random"
    d["experiment"]["device"], d["experiment"]["amp"] = "cpu", False
    for k, v in over.items():
        node = d
        parts = k.split(".")
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = v
    return Config(d)


def batch(b=2, h=32, w=32):
    torch.manual_seed(0)
    return torch.randn(b, 3, h, w), torch.randint(0, C, (b, h, w))


# ---------------------------------------------------------------- memory (M1, M3, M4)
def test_M1_memory_is_count_weighted_running_mean_not_last_writer():
    pg = PrototypeGenerator(C, 4, 2, False, True, False, True, num_slots=1)
    a = torch.zeros(C, 4); a[:, 0] = 1
    b = torch.zeros(C, 4); b[:, 1] = 1
    pix = torch.zeros(C); pix[0] = 300
    pg.memory_write(a, pix, None)
    pix2 = torch.zeros(C); pix2[0] = 100
    pg.memory_write(b, pix2, None)
    m = pg.memory[0, 0]
    assert m[0] > m[1] > 0, "second (smaller) write must not overwrite the first"
    assert pg.mem_count[0, 0] == 400
    assert not pg.slot_filled[1:].any(), "absent classes stay untouched"


def test_M3_slot_read_attends_to_matching_domain_key():
    pg = PrototypeGenerator(C, 4, 2, False, True, False, True, num_slots=2, read_tau=0.05)
    pix = torch.full((C,), 10.0)
    va, vb = torch.zeros(C, 4), torch.zeros(C, 4)
    va[:, 0], vb[:, 1] = 1, 1
    pg.memory_write(va, pix, torch.tensor([1.0, 0.0]))
    pg.memory_write(vb, pix, torch.tensor([0.0, 1.0]))
    r = pg.memory_read(torch.tensor([[0.0, 1.0]]), 1)[0, 0]
    assert r[1] > 0.9 and r[0] < 0.1, "an image keyed like domain B should read domain B's slot"


def test_M4_never_seen_classes_get_no_shift():
    pg = PrototypeGenerator(C, 4, 2, True, False, True, False, anchors=torch.randn(C, 4))
    with torch.no_grad():
        pg.W_u.normal_()
    pg.class_seen[:2] = True
    s = torch.randn(3, 2)
    p_with = pg(s, 3)
    p_none = pg(None, 3)
    assert torch.allclose(p_with[:, 2:], p_none[:, 2:]), "unseen classes must ignore the shift"
    assert not torch.allclose(p_with[:, :2], p_none[:, :2])


def test_shift_term_starts_at_zero():
    pg = PrototypeGenerator(C, 4, 2, True, False, True, False, anchors=torch.randn(C, 4))
    pg.class_seen[:] = True
    assert torch.allclose(pg(torch.randn(2, 2), 2), pg(None, 2)), "W_u starts at 0: residual shift"


# ---------------------------------------------------------------- objective (T1, T2, T3, T5, N1)
def _forward_loss(cfg, hidden_all=True):
    torch.manual_seed(0)
    model = DRIFT(cfg)
    model.prototypes.class_seen[:] = True
    with torch.no_grad():               # trained-looking shift (W_u starts at 0)
        model.prototypes.W_u.normal_()
    feats = torch.randn(C, cfg.model.embed_dim)
    model.prototypes.memory_write(feats, torch.full((C,), 10.0), torch.zeros(cfg.model.shift_dim))
    crit = DRIFTLoss(cfg)
    x, y = batch()
    out = model(x)
    allowed = torch.ones(C, dtype=torch.bool)
    hidden = torch.zeros(C, dtype=torch.bool); hidden[1] = True
    return model, crit, out, y, allowed, hidden


def test_T1_hidden_loss_does_not_reach_encoder_lora_or_decoder():
    cfg = tiny_cfg(**{"loss.dice.enabled": False, "loss.shift_var.weight": 0.0, "loss.hidden.scope": "task"})
    model, crit, out, y, allowed, hidden = _forward_loss(cfg)
    loss, parts, _, _ = crit(model, out, y, allowed, model.memory_mask(), hidden)
    assert "hidden" in parts
    # isolate the hidden term
    z, st = out["z"].detach(), out["stats"].detach()
    ph = model.prototypes_from_stats(st, z.shape[0])
    hl = model.logits_from(z, ph, out_size=y.shape[-2:])
    tgt = torch.where(y == 1, y, torch.full_like(y, 255))
    torch.nn.functional.cross_entropy(hl, tgt, ignore_index=255).backward()
    for name, p in model.named_parameters():
        if name.startswith(("encoder", "decoder")):
            assert p.grad is None or p.grad.abs().sum() == 0, f"hidden loss leaked into {name}"
    assert model.prototypes.W_u.grad is not None and model.prototypes.W_u.grad.abs().sum() > 0
    assert model.shift_estimator.mlp[1].weight.grad.abs().sum() > 0


def test_T1_shift_estimator_input_is_detached_by_default():
    cfg = tiny_cfg()
    assert cfg.model.shift_estimator.detach_input is True
    model = DRIFT(cfg)
    x, _ = batch()
    out = model(x)
    out["s"].sum().backward()
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in model.encoder.parameters())


def test_T2_task_scope_keeps_one_hidden_set_for_the_whole_task():
    cfg = tiny_cfg(**{"loss.hidden.scope": "task"})
    model, crit, out, y, allowed, hidden = _forward_loss(cfg)
    for _ in range(5):
        _, _, _, h = crit(model, out, y, allowed, model.memory_mask(), hidden)
        assert torch.equal(h, hidden)


def test_T3_absent_classes_are_negatives_but_their_prototypes_get_no_gradient():
    cfg = tiny_cfg(**{"loss.dice.enabled": False, "loss.hidden.enabled": False, "loss.shift_var.weight": 0.0,
                      "model.modules.learnable_base": True, "model.modules.anchors": False})
    torch.manual_seed(0)
    model = DRIFT(cfg)
    crit = DRIFTLoss(cfg)
    x, y = batch()
    y[y == 4] = 0                                   # class 4 absent from this task
    allowed = torch.ones(C, dtype=torch.bool); allowed[4] = False
    out = model(x)
    loss, _, _, _ = crit(model, out, y, allowed, model.memory_mask())
    loss.backward()
    g = model.prototypes.base.grad
    assert g[4].abs().sum() == 0, "absent prototype must not be pushed"
    assert g[:4].abs().sum() > 0
    # ...but the absent class is still a negative: its logit enters the softmax
    assert crit.absent_mode() == "detached"


def test_T5_prototype_path_and_1d_params_have_no_weight_decay():
    cfg = tiny_cfg()
    opt = build_optimizer(DRIFT(cfg), cfg.train)
    for g in opt.param_groups:
        if g["name"].startswith("prototypes") or g["name"].endswith("_no_wd"):
            assert g["weight_decay"] == 0.0
    assert any(g["weight_decay"] > 0 for g in opt.param_groups)


def test_N1_ce_with_no_labelled_pixels_is_zero_and_finite():
    logits = torch.randn(1, C, 4, 4, requires_grad=True)
    y = torch.full((1, 4, 4), 255)
    l = ce_loss(logits, y, None, 255)
    l.backward()
    assert l.item() == 0 and torch.isfinite(logits.grad).all()


def test_ohem_with_ignore_pixels_does_not_index_out_of_range():
    torch.manual_seed(0)
    logits = torch.randn(2, C, 8, 8, requires_grad=True)
    y = torch.randint(0, C, (2, 8, 8))
    y[:, :3] = 255                                   # ignore pixels, as in every real mask
    l = ce_loss(logits, y, None, 255, ohem={"thresh": 0.7, "min_kept": 10})
    l.backward()
    assert torch.isfinite(l) and torch.isfinite(logits.grad).all()


def test_full_loss_with_ohem_config_and_ignore_pixels():
    cfg = tiny_cfg(**{"loss.ce.ohem": {"thresh": 0.7, "min_kept": 50}})
    model, crit, out, y, allowed, hidden = _forward_loss(cfg)
    y = y.clone()
    y[:, :5] = 255
    loss, _, _, _ = crit(model, out, y, allowed, model.memory_mask(), hidden)
    loss.backward()
    assert torch.isfinite(loss)


# ---------------------------------------------------------------- cross-dataset (multi-dataset) loading
def _write_ds(root, layout, split, seqs, size, ids):
    from PIL import Image
    for seq in seqs:
        for k in range(3):
            if layout == "cs":
                img = os.path.join(root, "leftImg8bit", split, seq, f"{seq}_{k:06d}_000019_leftImg8bit.png")
                lbl = os.path.join(root, "gtFine", split, seq, f"{seq}_{k:06d}_000019_gtFine_labelIds.png")
            else:
                img = os.path.join(root, "leftImg8bit", split, seq, f"{k:06d}_leftImg8bit.png")
                lbl = os.path.join(root, "masks", split, seq, f"{k:06d}.png")
            os.makedirs(os.path.dirname(img), exist_ok=True)
            os.makedirs(os.path.dirname(lbl), exist_ok=True)
            Image.fromarray(np.random.randint(0, 255, (*size, 3), dtype=np.uint8)).save(img)
            arr = np.full(size, ids[0], dtype=np.uint8)
            arr[: size[0] // 2] = ids[1]
            Image.fromarray(arr).save(lbl)


def test_cross_dataset_splits_label_maps_and_mixed_sizes(tmp_path):
    from functools import partial
    from torch.utils.data import DataLoader
    from drift.data.datasets import SegDataset, build_splits
    from drift.data.transforms import build_transforms
    from drift.engine import _collate

    cs, idd = str(tmp_path / "cs"), str(tmp_path / "idd")
    for sp in ("train", "val"):
        _write_ds(cs, "cs", sp, ["aachen", "bochum"], (16, 32), (7, 23))      # labelIds road, sky
        _write_ds(idd, "idd", sp, ["0", "10"], (18, 32), (0, 8))             # level3 road, autorickshaw
    cfg = load_config(os.path.join(ROOT, "configs", "cs_idd.yaml"),
                      [f"dataset.domains.cityscapes.root={cs}", f"dataset.domains.idd.root={idd}"])
    splits = build_splits(cfg, verbose=False)
    assert {s.domain for s in splits["train"]} == {"cityscapes", "idd"}
    assert {s.subdomain for s in splits["train"] if s.source == "idd"} == {"0", "10"}
    assert len(splits["test"]) == 12 and len(splits["val"]) == 4      # 1 per city/sequence carved
    ds = SegDataset(splits["test"], cfg.dataset, transform=build_transforms(cfg.transforms.test))
    by_src = {s.source: i for i, s in enumerate(ds.samples)}
    cs_lbl, idd_lbl = ds.read_label(by_src["cityscapes"]), ds.read_label(by_src["idd"])
    assert set(np.unique(cs_lbl)) == {0, 10}          # road, sky in Cityscapes train ids
    assert set(np.unique(idd_lbl)) == {0, 255}        # road; autorickshaw -> ignore
    dl = DataLoader(ds, batch_size=len(ds), collate_fn=partial(_collate, ignore_index=255))
    b = next(iter(dl))                                # different image sizes in one batch
    assert b["image"].shape[0] == len(ds) and b["label"].shape[-2:] == b["image"].shape[-2:]


# ---------------------------------------------------------------- architecture (A1, A3, LoRA)
def test_A1_decoder_uses_groupnorm_and_runs_with_batch_1():
    dec = FusionDecoder([8, 16], 32, 0.0, "gn")
    assert not any(isinstance(m, nn.BatchNorm2d) for m in dec.modules())
    dec.train()
    out = dec([torch.randn(1, 8, 8, 8), torch.randn(1, 16, 4, 4)])
    assert out.shape == (1, 32, 8, 8)


def test_A3_frozen_encoder_stays_in_eval_mode_during_training():
    cfg = tiny_cfg(**{"model.encoder.freeze": True})
    model = DRIFT(cfg).train()
    assert not model.encoder.training
    assert all(m.training for m in model.modules() if isinstance(m, LoRALinear))
    assert model.decoder.training


def test_lora_targets_match_renamed_attention_layers():
    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj, self.k_proj, self.v_proj, self.other = (nn.Linear(4, 4) for _ in range(4))

    m = nn.Sequential(Attn(), Attn())
    names = inject_lora(m, ["query", "key", "value", "q_proj", "k_proj", "v_proj"], 2, 4, 0.0)
    assert len(names) == 6 and all("other" not in n for n in names)


def test_lora_enabled_with_no_match_is_an_error():
    cfg = tiny_cfg(**{"model.lora.targets": ["does_not_exist"]})
    with pytest.raises(RuntimeError, match="LoRA is enabled"):
        DRIFT(cfg)


def test_N2_lora_merge_matches_unmerged_output():
    torch.manual_seed(0)
    lin = LoRALinear(nn.Linear(16, 8), rank=4, alpha=8)
    nn.init.normal_(lin.lora_B, std=0.5)
    x = torch.randn(3, 16)
    a = lin(x)
    lin.merge()
    b = lin(x)
    lin.unmerge()
    c = lin(x)
    assert torch.allclose(a, b, atol=1e-5) and torch.allclose(a, c, atol=1e-5)


def test_ablation_switches_build_nothing():
    cfg = tiny_cfg(**{"model.modules.shift_estimator": False, "model.modules.memory": False})
    m = DRIFT(cfg)
    assert m.shift_estimator is None and not m.prototypes.use_memory and not m.prototypes.use_shift
    assert not hasattr(m.prototypes, "memory") and not hasattr(m.prototypes, "W_u")


# ---------------------------------------------------------------- reproducibility (R1, R3, H3)
def test_R1_failed_anchor_source_is_an_error_unless_fallback_allowed():
    pc = Config({"anchors": {"source": "file", "file": "missing.pt", "allow_fallback": False}})
    with pytest.raises(RuntimeError, match="anchors"):
        build_anchors(pc, ["a", "b"], 8)
    pc.anchors.allow_fallback = True
    assert build_anchors(pc, ["a", "b"], 8).shape == (2, 8)


def test_R3_H3_checkpoint_round_trip_is_strict_and_omits_frozen_encoder(tmp_path):
    cfg = tiny_cfg(**{"model.encoder.freeze": True})
    m = DRIFT(cfg)
    sd = m.checkpoint_state()
    frozen = [n for n, p in m.encoder.named_parameters() if not p.requires_grad and "lora_" not in n]
    assert frozen and all(f"encoder.{n}" not in sd for n in frozen)
    path = tmp_path / "ck.pt"
    torch.save({"model": sd}, path)
    load_model(cfg, str(path), torch.device("cpu"))
    bad = dict(sd)
    bad["decoder.extra"] = torch.zeros(1)
    torch.save({"model": bad}, path)
    with pytest.raises(RuntimeError, match="does not match"):
        load_model(cfg, str(path), torch.device("cpu"))


# ---------------------------------------------------------------- tasks, splits, metrics (E2, E3, S1)
def test_tasks_natural_sort_and_balanced_groups():
    doms = sorted(["0", "1", "10", "2", "100"], key=natural_key)
    assert doms == ["0", "1", "2", "10", "100"]
    counts = {d: c for d, c in zip("abcdef", [100, 1, 1, 100, 50, 50])}
    g = _balanced_groups(list("abcdef"), counts, 3)
    assert len(g) == 3 and sum(len(x) for x in g) == 6


def test_min_images_per_task_merges_tiny_tasks():
    samples = [Sample("i", "l", d, f"{d}/{k}") for d, n in (("1", 1), ("2", 50), ("3", 2), ("4", 60))
               for k in range(n)]
    cfg = tiny_cfg(**{"continual.tasks": "auto", "continual.group_size": 1, "continual.min_images_per_task": 20})
    tasks = build_tasks(cfg, samples)
    sizes = [sum(1 for s in samples if s.domain in t) for t in tasks]
    assert min(sizes) >= 20, sizes


def test_metrics_on_known_confusion():
    m = SegMetrics(2, 255, compute_boundary=False, keep_per_image=True)
    pred = torch.tensor([[[0, 0, 1, 1]]])
    gt = torch.tensor([[[0, 1, 1, 1]]])
    m.update(pred, gt)
    r = m.compute()
    assert abs(r["IoU_per_class"][0] - 0.5) < 1e-6 and abs(r["IoU_per_class"][1] - 2 / 3) < 1e-6
    assert abs(r["mIoU"] - miou_from_conf(r["confusion"])) < 1e-9
    ci = bootstrap_miou(m.per_image, n=50)
    assert ci["n_images"] == 1


def test_E2_class_forgetting_uses_a_fixed_class_set():
    Rc = np.array([[0.8, 0.6, np.nan], [0.7, 0.6, 0.5]])
    f = class_forgetting(Rc)
    assert abs(f["class_forgetting"][0] - 0.1) < 1e-9 and f["class_forgetting"][2] is None


def test_S1_shift_diagnostics_detect_collapse_and_signal():
    dom = torch.arange(4).repeat_interleave(20)
    collapsed = shift_diagnostics(torch.ones(80, 8) + 1e-6 * torch.randn(80, 8), dom)
    informative = shift_diagnostics(torch.nn.functional.one_hot(dom, 8).float() + 0.05 * torch.randn(80, 8), dom)
    assert collapsed["s_std_mean"] < 1e-3
    assert informative["s_domain_acc"] > 0.9
