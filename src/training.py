"""Training and evaluation helpers shared by the experiment notebook (03).

Same procedure as notebook 02: geometric + intensity augmentation, weighted CE + Dice, AdamW, exact mIoU on
validation and a fixed training subset after every epoch, learning-rate halving and early stopping on
validation mIoU.
"""
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from . import config as C, evaluation as E, model as M

AUTOTUNE = tf.data.AUTOTUNE
BRIGHTNESS_DB = 1.0
NOISE_DB = 0.3


# --- Data -------------------------------------------------------------------------------

def load_split(split, in_memory=True, smoke=False):
    X = np.load(C.PATCH_DIR / f"X_{split}.npy", mmap_mode=None if in_memory else "r")
    y = np.load(C.PATCH_DIR / f"y_{split}.npy", mmap_mode=None if in_memory else "r")
    meta = pd.read_csv(C.PATCH_DIR / f"meta_{split}.csv")
    if smoke:   # 64 random patches, so that all regions can appear
        idx = np.sort(np.random.default_rng(0).choice(len(X), min(64, len(X)), replace=False))
        return np.asarray(X[idx]), np.asarray(y[idx]), meta.iloc[idx].reset_index(drop=True)
    return X, y, meta


def augment_intensity(x, t):
    # brightness change (same dB offset on VV and VH, drawn separately for every stacked month: calibration
    # differs between acquisitions) and speckle-like noise, before normalisation
    x = tf.cast(x, tf.float32)
    n = x.shape[-1] // 3
    offset = tf.random.uniform([n], -BRIGHTNESS_DB, BRIGHTNESS_DB)
    x = x + tf.reshape(tf.stack([offset, offset, tf.zeros_like(offset)], -1), [-1])
    x = x + tf.random.normal(tf.shape(x), stddev=NOISE_DB)
    return x, t


def augment_geometry(x, t):
    k = tf.random.uniform([], 0, 4, tf.int32)
    x, t = tf.image.rot90(x, k), tf.image.rot90(t, k)
    flip = tf.random.uniform([]) < 0.5
    x = tf.cond(flip, lambda: tf.image.flip_left_right(x), lambda: x)
    t = tf.cond(flip, lambda: tf.image.flip_left_right(t), lambda: t)
    return x, t


def make_ds(X, y, stats, batch, training=False, index=None):
    index = np.arange(len(X)) if index is None else np.asarray(index)

    def gen():
        for i in (np.random.permutation(index) if training else index):
            yield X[i], y[i][..., None]

    ds = tf.data.Dataset.from_generator(gen, output_signature=(
        tf.TensorSpec((C.PATCH_PX, C.PATCH_PX, X.shape[-1]), tf.float16),
        tf.TensorSpec((C.PATCH_PX, C.PATCH_PX, 1), tf.uint8)))
    ds = ds.apply(tf.data.experimental.assert_cardinality(len(index)))
    if training:
        ds = ds.map(augment_intensity, num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda x, t: (M.normalise(x, stats), tf.cast(t, tf.int32)), num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.map(augment_geometry, num_parallel_calls=AUTOTUNE)
    return ds.batch(batch).prefetch(AUTOTUNE)


# --- Training ---------------------------------------------------------------------------

class EpochScores(keras.callbacks.Callback):
    """Exact mIoU / OA on validation and a fixed training subset, added to the logs for early stopping."""

    def __init__(self, val_ds, train_eval_ds):
        super().__init__()
        self.val_ds, self.train_eval_ds = val_ds, train_eval_ds

    def scores(self, ds):
        cm = np.zeros((M.NUM_CLASSES, M.NUM_CLASSES), np.int64)
        for x, t in ds:
            cm += E.confusion(t.numpy()[..., 0], np.argmax(self.model.predict_on_batch(x), -1))
        return E.scores(cm)

    def on_epoch_end(self, epoch, logs=None):
        v, tr = self.scores(self.val_ds), self.scores(self.train_eval_ds)
        logs.update(val_miou=v["mIoU"], val_oa=v["OA"], train_miou=tr["mIoU"], train_oa=tr["OA"])
        print(f"  mIoU train {tr['mIoU']:.3f} | val {v['mIoU']:.3f}   "
              f"lr {float(self.model.optimizer.learning_rate):.1e}")


def diagnose(history):
    best = history.val_miou.idxmax()
    tr, va = history.train_miou[best], history.val_miou[best]
    gap = tr - va
    if tr < 0.35:
        verdict = "underfitting: low mIoU even on training data"
    elif gap > 0.15:
        verdict = "overfitting: clearly better on training than on validation regions"
    else:
        verdict = "good fit: training and validation mIoU are close"
    return dict(best_epoch=int(best), train_mIoU=round(float(tr), 4), val_mIoU=round(float(va), 4),
                gap=round(float(gap), 4), verdict=verdict)


def train_model(name, data, out_dir, lr=3e-4, batch=16, max_epochs=50, patience=8, lr_patience=3, reuse=True):
    """Train M.MODELS[name] on data = dict(X_train, y_train, X_val, y_val, stats, class_weights).
    If final weights and log already exist (and reuse=True), they are loaded instead (resume after a crash)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    final, log, info_file = (out_dir / f"{name}{s}" for s in [".keras", "_log.csv", "_info.json"])
    if reuse and final.exists() and log.exists():
        model = keras.models.load_model(final, compile=False)
        history = pd.read_csv(log).set_index("epoch")
        history.index += 1
        info = json.loads(info_file.read_text()) if info_file.exists() else {}
        print(f"{name}: loaded from a previous run ({len(history)} epochs) - delete {final.name} to retrain")
        return model, history, dict(params=model.count_params(), epochs_run=len(history),
                                    minutes=info.get("minutes", float("nan")), batch=info.get("batch", batch),
                                    **diagnose(history))

    stats = data["stats"]
    train_ds = make_ds(data["X_train"], data["y_train"], stats, batch, training=True)
    val_ds = make_ds(data["X_val"], data["y_val"], stats, batch)
    eval_idx = np.random.default_rng(0).choice(len(data["X_train"]), min(1000, len(data["X_train"])), replace=False)
    train_eval_ds = make_ds(data["X_train"], data["y_train"], stats, batch, index=eval_idx)

    model = M.MODELS[name]()
    # jit_compile=False: XLA rejects float32 bilinear resizing inside a float16 (mixed precision) graph
    model.compile(optimizer=keras.optimizers.AdamW(lr, weight_decay=1e-4),
                  loss=M.make_loss(data["class_weights"]), jit_compile=False)
    ckpt = out_dir / f"{name}_ckpt.keras"
    callbacks = [
        EpochScores(val_ds, train_eval_ds),
        keras.callbacks.ReduceLROnPlateau(monitor="val_miou", mode="max", factor=0.5, patience=lr_patience,
                                          min_lr=1e-6, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_miou", mode="max", patience=patience, min_delta=0.002,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ModelCheckpoint(ckpt, monitor="val_miou", mode="max", save_best_only=True),
        keras.callbacks.CSVLogger(log),
    ]
    print(f"{name}: {model.count_params():,} parameters, batch {batch}, lr {lr}")
    t0 = time.time()
    hist = model.fit(train_ds, validation_data=val_ds, epochs=max_epochs, callbacks=callbacks,
                     shuffle=False, verbose=2)
    history = pd.DataFrame(hist.history)
    history.index += 1
    best = keras.models.load_model(ckpt, compile=False)
    best.save(final)
    ckpt.unlink()                                        # large file with optimizer state, no longer needed
    info = dict(params=model.count_params(), epochs_run=len(history), minutes=round((time.time() - t0) / 60, 1),
                batch=batch, **diagnose(history))
    info_file.write_text(json.dumps({"minutes": info["minutes"], "batch": batch}))
    return best, history, info


def plot_history(history, title):
    fig, ax = plt.subplots(1, 3, figsize=(18, 4))
    ax[0].plot(history.loss, label="train"); ax[0].plot(history.val_loss, label="validation")
    ax[0].set_title("Loss (weighted CE + Dice)")
    ax[1].plot(history.train_miou, label="train"); ax[1].plot(history.val_miou, label="validation")
    ax[1].axvline(history.val_miou.idxmax(), color="grey", ls=":", label="best epoch")
    ax[1].set_title("mIoU")
    ax[2].plot(history.learning_rate); ax[2].set_yscale("log"); ax[2].set_title("Learning rate")
    for a in ax:
        a.set_xlabel("epoch"); a.grid(alpha=0.3)
    ax[0].legend(); ax[1].legend()
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


# --- Evaluation -------------------------------------------------------------------------

def predict(model, X, stats, batch=16):
    out = np.empty(X.shape[:3], np.uint8)
    for i in range(0, len(X), batch):
        out[i:i + batch] = np.argmax(model.predict_on_batch(M.normalise(np.asarray(X[i:i + batch]), stats)), -1)
    return out


def evaluate(model, X, y, stats):
    pred = predict(model, X, stats)
    cm = E.confusion(y, pred)
    return dict(pred=pred, cm=cm, **E.scores(cm))
