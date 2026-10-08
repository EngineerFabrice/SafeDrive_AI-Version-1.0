"""Transfer-learning MobileNetV3 on the prepared face crops (two stages).

    python -m training.train_mobilenet --stage head       # PRETRAINED MODEL DOWNLOAD + frozen-backbone training
    python -m training.train_mobilenet --stage finetune   # FINE-TUNING (starts from the best head checkpoint)
    python -m training.train_mobilenet --stage select     # compare both on validation (+ robustness), export

Model selection never looks at the test split. ``training/evaluate.py`` evaluates
the exported model on test exactly once.
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.impairment.alcohol_model import build_mobilenet_v3    # noqa: E402
from training import config                                       # noqa: E402
from training.data import FaceCropDataset, eval_transform, read_manifest, train_transform  # noqa: E402
from training.metrics import binary_metrics                       # noqa: E402

ARCH = "mobilenet_v3_small"   # ~1.5M parameters: fits CPU real time and a very small dataset
STAGES = {
    # frozen backbone: only the classifier head learns
    "head": {"epochs": 20, "lr": 1e-3, "unfreeze_blocks": 0, "patience": 6},
    # deeper layers unfrozen with a much smaller learning rate
    "finetune": {"epochs": 20, "lr": 1e-4, "unfreeze_blocks": 4, "patience": 6},
}


def seed_everything(seed=config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def set_trainable(net, unfreeze_blocks):
    """Freeze ``features`` except its last ``unfreeze_blocks`` modules; the classifier always trains."""
    for p in net.features.parameters():
        p.requires_grad = False
    if unfreeze_blocks:
        for block in list(net.features.children())[-unfreeze_blocks:]:
            for p in block.parameters():
                p.requires_grad = True
    for p in net.classifier.parameters():
        p.requires_grad = True


def frozen_bn_eval(net):
    """Keep BatchNorm statistics of frozen layers fixed (they come from ImageNet)."""
    for m in net.features.modules():
        if isinstance(m, nn.BatchNorm2d) and not any(p.requires_grad for p in m.parameters()):
            m.eval()


@torch.no_grad()
def predict_probs(net, loader):
    net.eval()
    probs, labels, loss_sum = [], [], 0.0
    crit = nn.CrossEntropyLoss(reduction="sum")
    for x, y in loader:
        logits = net(x)
        loss_sum += float(crit(logits, y))
        probs.append(torch.softmax(logits, 1)[:, 1].numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels), loss_sum / max(1, len(loader.dataset))


def loaders(batch_size=32):
    train_rows, val_rows = read_manifest(split="train"), read_manifest(split="val")
    g = torch.Generator().manual_seed(config.SEED)
    train = DataLoader(FaceCropDataset(train_rows, train_transform()), batch_size=batch_size, shuffle=True,
                       generator=g, num_workers=0)
    val = DataLoader(FaceCropDataset(val_rows, eval_transform()), batch_size=64, num_workers=0)
    return train, val, len(train_rows), len(val_rows)


def train_stage(stage, out_dir=config.RUNS_DIR):
    seed_everything()
    cfg = STAGES[stage]
    run_dir = os.path.join(out_dir, stage)
    os.makedirs(run_dir, exist_ok=True)

    if stage == "head":
        print("PRETRAINED MODEL DOWNLOAD STAGE: loading ImageNet MobileNetV3 weights (torchvision)")
        net = build_mobilenet_v3(ARCH, len(config.CLASSES), pretrained=True)
    else:
        print("FINE-TUNING STAGE: starting from the best frozen-backbone checkpoint")
        net = build_mobilenet_v3(ARCH, len(config.CLASSES), pretrained=False)
        net.load_state_dict(torch.load(os.path.join(out_dir, "head", "best.pt"), map_location="cpu"))
    set_trainable(net, cfg["unfreeze_blocks"])
    params = [p for p in net.parameters() if p.requires_grad]
    print(f"{stage}: trainable parameters {sum(p.numel() for p in params):,} / "
          f"{sum(p.numel() for p in net.parameters()):,}")

    train_loader, val_loader, n_train, n_val = loaders()
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    crit = nn.CrossEntropyLoss()

    history, best, bad = [], None, 0
    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        net.train()
        frozen_bn_eval(net)
        tr_loss, tr_correct = 0.0, 0
        for x, y in train_loader:
            opt.zero_grad()
            logits = net(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(y)
            tr_correct += int((logits.argmax(1) == y).sum())
        sched.step()
        probs, labels, val_loss = predict_probs(net, val_loader)
        m = binary_metrics(labels, probs)
        row = {"epoch": epoch, "train_loss": round(tr_loss / n_train, 4), "train_accuracy": round(tr_correct / n_train, 4),
               "val_loss": round(val_loss, 4), **{f"val_{k}": v for k, v in m.items() if k != "confusion_matrix"},
               "val_confusion_matrix": m["confusion_matrix"], "seconds": round(time.time() - t0, 1)}
        history.append(row)
        print(f"[{stage}] epoch {epoch:2d} train_loss {row['train_loss']:.4f} acc {row['train_accuracy']:.3f} | "
              f"val_loss {row['val_loss']:.4f} acc {m['accuracy']:.3f} P {m['precision']:.3f} "
              f"R {m['recall']:.3f} F1 {m['f1']:.3f} CM {m['confusion_matrix']}")
        key = (m["f1"], -val_loss)                      # best F1, ties broken by lower validation loss
        if best is None or key > best[0]:
            best, bad = (key, epoch), 0
            torch.save(net.state_dict(), os.path.join(run_dir, "best.pt"))
        else:
            bad += 1
            if bad >= cfg["patience"]:
                print(f"[{stage}] early stopping at epoch {epoch}")
                break

    summary = {"stage": stage, "arch": ARCH, "config": cfg, "best_epoch": best[1], "history": history,
               "best": next(h for h in history if h["epoch"] == best[1])}
    with open(os.path.join(run_dir, "history.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return summary


# ---------------------------------------------------------------- selection with robustness
def perturb(kind):
    def f(img):
        if kind == "darker":
            return ImageEnhance.Brightness(img).enhance(0.6)
        if kind == "brighter":
            return ImageEnhance.Brightness(img).enhance(1.4)
        if kind == "grayscale":
            return img.convert("L").convert("RGB")
        if kind == "blur":
            return img.filter(ImageFilter.GaussianBlur(1.5))
        if kind == "warm_cast":       # pushes non_alcoholic frames toward the alcoholic colour cast
            r, g, b = img.split()
            return Image.merge("RGB", (r.point(lambda v: min(255, int(v * 1.15))), g, b.point(lambda v: int(v * 0.85))))
        return img
    return f


def robustness(net, rows):
    from torchvision import transforms
    out = {}
    for kind in ("original", "darker", "brighter", "grayscale", "blur", "warm_cast"):
        t = transforms.Compose([transforms.Lambda(perturb(kind)), eval_transform()])
        probs, labels, _ = predict_probs(net, DataLoader(FaceCropDataset(rows, t), batch_size=64))
        out[kind] = binary_metrics(labels, probs)
    return out


def select_and_export(out_dir=config.RUNS_DIR, model_dir=config.MODEL_DIR):
    seed_everything()
    val_rows = read_manifest(split="val")
    report = {}
    for stage in ("head", "finetune"):
        path = os.path.join(out_dir, stage, "best.pt")
        if not os.path.isfile(path):
            continue
        net = build_mobilenet_v3(ARCH, len(config.CLASSES))
        net.load_state_dict(torch.load(path, map_location="cpu"))
        rob = robustness(net, val_rows)
        f1s = [m["f1"] for m in rob.values()]
        report[stage] = {"validation": rob["original"], "robustness": rob,
                         "mean_f1_under_perturbation": round(float(np.mean(f1s)), 4),
                         "worst_f1_under_perturbation": round(float(np.min(f1s)), 4)}
        print(f"{stage}: val F1 {rob['original']['f1']:.3f} | mean F1 under perturbation "
              f"{report[stage]['mean_f1_under_perturbation']:.3f} | worst {report[stage]['worst_f1_under_perturbation']:.3f}")

    # Robustness first (validation alone is saturated on this dataset), then plain validation F1.
    chosen = max(report, key=lambda s: (report[s]["mean_f1_under_perturbation"],
                                        report[s]["worst_f1_under_perturbation"], report[s]["validation"]["f1"]))
    print(f"selected: {chosen}")
    state = torch.load(os.path.join(out_dir, chosen, "best.pt"), map_location="cpu")
    meta = {
        "version": "1", "arch": ARCH, "classes": list(config.CLASSES), "positive_class": config.POSITIVE_CLASS,
        "input_size": 224, "mean": list(config.IMAGENET_MEAN), "std": list(config.IMAGENET_STD),
        "crop": "engine.preprocessing.align_face_crop (YuNet box + eye-line alignment)",
        "dataset": config.SOURCE_ID, "label_type": config.LABEL_TYPE, "selected_stage": chosen,
        "development_only": True,
        "limitations": ("Prototype trained on AlcoholDetectionDataset: one recording of one person per class, "
                        "undocumented labels, strong lighting/colour/glasses/background differences between "
                        "classes. Not validated on independent people; never a measurement of blood alcohol."),
        "selection": report,
    }
    os.makedirs(model_dir, exist_ok=True)
    torch.save({"state_dict": state, "metadata": meta}, os.path.join(model_dir, "model.pt"))
    with open(os.path.join(model_dir, "selection.json"), "w", encoding="utf-8") as fh:
        json.dump({k: v for k, v in meta.items()}, fh, indent=2)
    return chosen, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", choices=("head", "finetune", "select"), required=True)
    args = parser.parse_args()
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    if args.stage == "select":
        select_and_export()
    else:
        s = train_stage(args.stage)
        print(json.dumps({k: s["best"][k] for k in ("epoch", "val_loss", "val_accuracy", "val_precision",
                                                     "val_recall", "val_f1", "val_confusion_matrix")}))
