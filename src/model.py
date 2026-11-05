"""Segmentation models, loss and input normalisation (shared by training and the prototype).

Three models of increasing capacity, compared in notebook 02:
  baseline_cnn   - plain CNN, no down-sampling, only local context (~90 m)
  unet           - encoder-decoder with skip connections, trained from scratch
  deeplabv3plus  - ImageNet-pretrained ResNet50 + atrous spatial pyramid pooling (multi-scale context)
"""
import json

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from . import config as C

NUM_CLASSES = len(C.CLASSES)
INPUT_SHAPE = (C.PATCH_PX, C.PATCH_PX, len(C.S1_BANDS))


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

def conv_bn_relu(x, filters, kernel=3, dilation=1, name=None):
    x = layers.Conv2D(filters, kernel, padding="same", dilation_rate=dilation, use_bias=False,
                      name=f"{name}_conv")(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    return layers.ReLU(name=f"{name}_relu")(x)


def segmentation_head(x):
    x = layers.Conv2D(NUM_CLASSES, 1, name="logits")(x)
    return layers.Activation("softmax", dtype="float32", name="probs")(x)   # float32 under mixed precision


# --- 1. Baseline: plain CNN -----------------------------------------------------------------

def baseline_cnn(filters=(32, 64, 64, 32)):
    """Four 3x3 conv layers at full resolution. Receptive field 9x9 pixels (90 m): the model sees the
    backscatter and texture around a pixel, but no wider spatial context or object shapes."""
    inp = keras.Input(INPUT_SHAPE)
    x = inp
    for i, f in enumerate(filters):
        x = conv_bn_relu(x, f, name=f"block{i}")
    return keras.Model(inp, segmentation_head(x), name="baseline_cnn")


# --- 2. U-Net -----------------------------------------------------------------------------

def unet(filters=(32, 64, 128, 256, 512), dropout=0.3):
    """U-Net (Ronneberger et al. 2015) with batch norm. The encoder halves the resolution four times
    (receptive field > 2 km); skip connections bring the fine detail back for sharp boundaries."""
    inp = keras.Input(INPUT_SHAPE)
    x, skips = inp, []
    for i, f in enumerate(filters[:-1]):
        x = conv_bn_relu(x, f, name=f"enc{i}a")
        x = conv_bn_relu(x, f, name=f"enc{i}b")
        skips.append(x)
        x = layers.MaxPooling2D(name=f"enc{i}_pool")(x)
    x = conv_bn_relu(x, filters[-1], name="bottleneck_a")
    x = conv_bn_relu(x, filters[-1], name="bottleneck_b")
    x = layers.SpatialDropout2D(dropout, name="bottleneck_drop")(x)
    for i, (f, skip) in enumerate(zip(reversed(filters[:-1]), reversed(skips))):
        x = layers.UpSampling2D(2, interpolation="bilinear", name=f"dec{i}_up")(x)
        x = layers.Concatenate(name=f"dec{i}_cat")([x, skip])
        x = conv_bn_relu(x, f, name=f"dec{i}a")
        x = conv_bn_relu(x, f, name=f"dec{i}b")
    return keras.Model(inp, segmentation_head(x), name="unet")


# --- 3. DeepLabV3+ -------------------------------------------------------------------------

def aspp(x, filters=256, rates=(6, 12, 18)):
    """Atrous spatial pyramid pooling: parallel dilated convolutions see the same location at several
    scales at once (a field vs. a whole farm landscape), plus a global image-level branch."""
    h, w = x.shape[1], x.shape[2]
    pool = layers.GlobalAveragePooling2D(keepdims=True, name="aspp_pool")(x)
    pool = conv_bn_relu(pool, filters, kernel=1, name="aspp_pool")
    pool = layers.UpSampling2D((h, w), interpolation="bilinear", name="aspp_pool_up")(pool)
    branches = [conv_bn_relu(x, filters, kernel=1, name="aspp_1x1")]
    branches += [conv_bn_relu(x, filters, dilation=r, name=f"aspp_d{r}") for r in rates]
    x = layers.Concatenate(name="aspp_cat")(branches + [pool])
    return conv_bn_relu(x, filters, kernel=1, name="aspp_out")


def deeplabv3plus(dropout=0.3):
    """DeepLabV3+ (Chen et al. 2018) on an ImageNet-pretrained ResNet50. The three SAR channels are fed
    in place of RGB: the pretrained filters (edges, textures) are a better starting point than random
    weights and are fine-tuned to SAR."""
    encoder = keras.applications.ResNet50(include_top=False, weights="imagenet", input_shape=INPUT_SHAPE)
    high = encoder.get_layer("conv4_block6_2_relu").output     # 1/16 resolution, deep features
    low = encoder.get_layer("conv2_block3_2_relu").output      # 1/4 resolution, fine detail

    x = aspp(high)
    x = layers.Dropout(dropout, name="aspp_drop")(x)
    x = layers.UpSampling2D(4, interpolation="bilinear", name="dec_up4")(x)
    low = conv_bn_relu(low, 48, kernel=1, name="dec_low")
    x = layers.Concatenate(name="dec_cat")([x, low])
    x = conv_bn_relu(x, 256, name="dec_a")
    x = conv_bn_relu(x, 256, name="dec_b")
    x = layers.UpSampling2D(4, interpolation="bilinear", name="dec_up_final")(x)
    return keras.Model(encoder.input, segmentation_head(x), name="deeplabv3plus")


# --- 4. DeepLabV3+ / AE-DeepLabV3+ with exchangeable backbones (experiments, notebook 03) ---------------

# high: last layer at 1/16 resolution (end of the second-to-last stage); low: 1/4-resolution features
BACKBONES = {
    "resnet101": dict(app=keras.applications.ResNet101, high="conv4_block23_out", low="conv2_block3_out"),
    "resnet152": dict(app=keras.applications.ResNet152, high="conv4_block36_out", low="conv2_block3_out"),
    "xception": dict(app=keras.applications.Xception, high="block13_sepconv2_bn", low="block3_sepconv2_bn"),
}


def spatial(t):
    return t.shape[1], t.shape[2]


def atrous_encoder(encoder, high, rate=2):
    """Atrous version of a backbone: the last stage keeps 1/16 resolution. Layers that would down-sample
    from 1/16 to 1/32 get stride 1, and the convolutions after them get dilation `rate`, so they still see
    the same (pretrained) neighbourhood. Weights are copied unchanged."""
    s16 = spatial(encoder.get_layer(high).output)
    unstride, dilate = set(), set()
    for layer in encoder.layers:
        cfg = layer.get_config()
        if not hasattr(layer, "input") or isinstance(layer.input, list):
            continue
        size_in = spatial(layer.input)
        if cfg.get("strides") in [(2, 2), [2, 2]] and size_in == s16:
            unstride.add(layer.name)
        elif "dilation_rate" in cfg and max(cfg["kernel_size"]) > 1 and size_in[0] < s16[0]:
            dilate.add(layer.name)

    def clone(layer):
        cfg = layer.get_config()
        if layer.name in unstride:
            cfg["strides"] = (1, 1)
        if layer.name in dilate:
            cfg["dilation_rate"] = (rate, rate)
        return layer.__class__.from_config(cfg)

    atrous = keras.models.clone_model(encoder, clone_function=clone)
    atrous.set_weights(encoder.get_weights())
    return atrous


def deeplab(backbone, atrous=False, dropout=0.3):
    """DeepLabV3+ on an ImageNet-pretrained backbone.

    atrous=False  DeepLabV3+: backbone cut after the 1/16 stage (the last stage is not used).
    atrous=True   AE-DeepLabV3+ (atrous encoder): the full backbone; its last stage uses dilated instead of
                  strided convolutions, so features stay at 1/16 resolution but come from a deeper network
                  with a larger receptive field (output stride 16, as in Chen et al. 2017/2018).
    """
    spec = BACKBONES[backbone]
    encoder = spec["app"](include_top=False, weights="imagenet", input_shape=INPUT_SHAPE)
    if atrous:
        encoder = atrous_encoder(encoder, spec["high"])
        high = encoder.output
    else:
        high = encoder.get_layer(spec["high"]).output
    low = encoder.get_layer(spec["low"]).output

    x = aspp(high)
    x = layers.Dropout(dropout, name="aspp_drop")(x)
    x = layers.Resizing(*spatial(low), interpolation="bilinear", name="dec_up_low")(x)
    low = conv_bn_relu(low, 48, kernel=1, name="dec_low")
    x = layers.Concatenate(name="dec_cat")([x, low])
    x = conv_bn_relu(x, 256, name="dec_a")
    x = conv_bn_relu(x, 256, name="dec_b")
    x = layers.Resizing(*INPUT_SHAPE[:2], interpolation="bilinear", name="dec_up_final")(x)
    name = f"{'ae_' if atrous else ''}deeplabv3plus_{backbone}"
    return keras.Model(encoder.input, segmentation_head(x), name=name)


MODELS = {"baseline_cnn": baseline_cnn, "unet": unet, "deeplabv3plus": deeplabv3plus}
for _b in BACKBONES:
    MODELS[f"deeplabv3plus_{_b}"] = lambda b=_b: deeplab(b, atrous=False)
    MODELS[f"ae_deeplabv3plus_{_b}"] = lambda b=_b: deeplab(b, atrous=True)


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
