# coding: utf-8
"""Konfigurasi MTL dependency parsing (M3/M4), dibaca dari environment.

Pola yang sama dengan IS_CONCAT: tanpa MTL=1 seluruh jalur M1/M2 tidak tersentuh.

    MTL=1                 aktifkan tugas bantu dependency parsing
    MTL_LAMBDA=0.5        bobot L_dep pada loss gabungan  L = L_amr + lambda * L_dep
    MTL_LAMBDA_WARMUP=2000  jumlah step optimizer untuk menaikkan gradien L_dep yang MASUK
                          KE ENCODER secara linear 0 -> 1. Head sendiri selalu menerima
                          gradien penuh (lambda * dL_dep) sejak step 0.
    MTL_HEAD_LR=1e-4      learning rate parameter head (param group terpisah)
    MTL_ARC_DIM=500  MTL_REL_DIM=100  MTL_DROPOUT=0.33   ukuran head biaffine
    MTL_DEP_FILE          default: <folder train>/train.dep.jsonl
    MTL_DEP_LABELS        default: <folder train>/dep_labels.json
"""
import os
import json


def mtl_enabled():
    return os.environ.get("MTL", "0") == "1"


def get_mtl_config(train_file=None):
    if not mtl_enabled():
        return None
    base = os.path.dirname(train_file) if train_file else "."
    cfg = {
        "enabled": True,
        "lambda": float(os.environ.get("MTL_LAMBDA", "0.5")),
        "lambda_warmup": int(os.environ.get("MTL_LAMBDA_WARMUP", "2000")),
        "head_lr": float(os.environ.get("MTL_HEAD_LR", "1e-4")),
        "arc_dim": int(os.environ.get("MTL_ARC_DIM", "500")),
        "rel_dim": int(os.environ.get("MTL_REL_DIM", "100")),
        "dropout": float(os.environ.get("MTL_DROPOUT", "0.33")),
        "dep_file": os.environ.get("MTL_DEP_FILE", os.path.join(base, "train.dep.jsonl")),
        "label_file": os.environ.get("MTL_DEP_LABELS", os.path.join(base, "dep_labels.json")),
    }
    with open(cfg["label_file"], encoding="utf-8") as f:
        cfg["labels"] = json.load(f)["labels"]
    cfg["n_rels"] = len(cfg["labels"])
    return cfg
