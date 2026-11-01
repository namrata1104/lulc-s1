"""Segmentation models, loss and input normalisation (shared by training and the prototype)."""
import json

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from . import config as C

NUM_CLASSES = len(C.CLASSES)


# --- Normalisation ------------------------------------------------------------------

def load_stats(path=C.PATCH_DIR / "stats.json"):
    return json.loads(path.read_text())


def normalise(x, stats):
    """x: (..., 3) dB values -> clipped, z-scored float32. Works on numpy arrays and tensors."""
    lo, hi = np.array(stats["clip"], dtype=np.float32).T
    mean = np.array(stats["mean"], dtype=np.float32)
    std = np.array(stats["std"], dtype=np.float32)
    if isinstance(x, np.ndarray):
        return (np.clip(x.astype(np.float32), lo, hi) - mean) / std
    return (tf.clip_by_value(tf.cast(x, tf.float32), lo, hi) - mean) / std


# --- Building blocks ------------------------------------------------------------------

def conv_block(x, filters, name):
    for i in range(2):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv{i}")(x)
        x = layers.BatchNormalization(name=f"{name}_bn{i}")(x)
        x = layers.ReLU(name=f"{name}_relu{i}")(x)
    return x


def decoder_block(x, skip, filters, name):
    x = layers.UpSampling2D(2, interpolation="bilinear", name=f"{name}_up")(x)
    if skip is not None:
        x = layers.Concatenate(name=f"{name}_cat")([x, skip])
    return conv_block(x, filters, name)


def segmentation_head(x):
    x = layers.Conv2D(NUM_CLASSES, 1, name="logits")(x)
    return layers.Activation("softmax", dtype="float32", name="probs")(x)   # float32 under mixed precision


# --- Model A: U-Net from scratch -------------------------------------------------------------

def unet(input_shape=(C.PATCH_PX, C.PATCH_PX, 3), filters=(32, 64, 128, 256, 512), dropout=0.3):
    """Classic U-Net (Ronneberger et al. 2015) with batch norm."""
    inp = keras.Input(input_shape)
    x, skips = inp, []
    for i, f in enumerate(filters[:-1]):
        x = conv_block(x, f, f"enc{i}")
        skips.append(x)
        x = layers.MaxPooling2D(name=f"enc{i}_pool")(x)
    x = conv_block(x, filters[-1], "bottleneck")
    x = layers.SpatialDropout2D(dropout, name="bottleneck_drop")(x)
    for i, (f, skip) in enumerate(zip(reversed(filters[:-1]), reversed(skips))):
        x = decoder_block(x, skip, f, f"dec{i}")
    return keras.Model(inp, segmentation_head(x), name="unet_scratch")


# --- Model B: U-Net with ImageNet-pretrained ResNet50 encoder ------------------------------------

RESNET_SKIPS = ["conv1_relu", "conv2_block3_out", "conv3_block4_out", "conv4_block6_out"]  # 1/2 .. 1/16


def resnet50_unet(input_shape=(C.PATCH_PX, C.PATCH_PX, 3), decoder_filters=(256, 128, 64, 32, 16)):
    """U-Net decoder on a ResNet50 encoder. The three SAR channels are fed in place of RGB; the
    pretrained filters give a better starting point for edges and textures than random weights."""
    encoder = keras.applications.ResNet50(include_top=False, weights="imagenet", input_shape=input_shape)
    skips = [encoder.get_layer(n).output for n in reversed(RESNET_SKIPS)] + [None]
    x = encoder.output                                                    # 1/32
    for i, (f, skip) in enumerate(zip(decoder_filters, skips)):
        x = decoder_block(x, skip, f, f"dec{i}")
    return keras.Model(encoder.input, segmentation_head(x), name="unet_resnet50")


# --- Loss -------------------------------------------------------------------------------

def make_loss(class_weights, dice_weight=1.0):
    """Class-weighted cross-entropy + soft Dice. Pixels labelled IGNORE contribute nothing."""
    cw = tf.constant(class_weights, tf.float32)

    def loss(y_true, y_pred):
        y_true = tf.cast(tf.squeeze(y_true, -1), tf.int32)
        valid = tf.cast(tf.not_equal(y_true, C.IGNORE), tf.float32)
        y = tf.where(y_true == C.IGNORE, 0, y_true)
        y_pred = tf.cast(y_pred, tf.float32)

        ce = keras.losses.sparse_categorical_crossentropy(y, y_pred)
        w = tf.gather(cw, y) * valid
        ce = tf.reduce_sum(ce * w) / (tf.reduce_sum(w) + 1e-6)

        onehot = tf.one_hot(y, NUM_CLASSES) * valid[..., None]
        probs = y_pred * valid[..., None]
        inter = tf.reduce_sum(onehot * probs, axis=[0, 1, 2])
        denom = tf.reduce_sum(onehot + probs, axis=[0, 1, 2])
        dice = 1.0 - tf.reduce_mean((2.0 * inter + 1.0) / (denom + 1.0))
        return ce + dice_weight * dice

    return loss


def median_frequency_weights(pixel_counts, lo=0.2, hi=10.0):
    """w_c = median(f) / f_c (Eigen & Fergus 2015), clipped so rare classes don't dominate."""
    f = np.asarray(pixel_counts, dtype=np.float64)
    f = f / f.sum()
    w = np.median(f[f > 0]) / np.maximum(f, 1e-12)
    return np.clip(w, lo, hi).astype(np.float32)
