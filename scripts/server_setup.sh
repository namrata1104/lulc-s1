# Lab server setup. Only ~/vault survives a server restart, so the Python environment and the
# Earth Engine login are kept there. Run after every server start, from the repo folder:
#
#     source scripts/server_setup.sh
#
# Works with both server images:
#   * TensorFlow image (system Python has TensorFlow): a light venv on top of it (~/vault/.envs/venv-lulc)
#   * other images with conda: a self-contained environment incl. TensorFlow + CUDA (~/vault/.envs/conda-lulc)
# The environments live in a hidden folder in the vault, not in "Deep Learning project": paths with spaces
# break Python environments.
# The first run creates the environment (a few minutes, ~10 min for the conda one); later runs take seconds.

VAULT="$HOME/vault"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
deactivate 2> /dev/null

if /usr/bin/python3 -c "import tensorflow" 2> /dev/null; then
    ENV="$VAULT/.envs/venv-lulc"
    if [ ! -x "$ENV/bin/python" ] || ! "$ENV/bin/python" -c "import tensorflow" 2> /dev/null; then
        echo "Creating $ENV on top of the server's TensorFlow ..."
        rm -rf "$ENV"
        mkdir -p "$VAULT/.envs"
        /usr/bin/python3 -m venv --system-site-packages "$ENV"
        "$ENV/bin/python" -m pip install -q -r <(grep -v tensorflow "$REPO/requirements.txt")
    fi
    source "$ENV/bin/activate"
elif command -v conda > /dev/null; then
    ENV="$VAULT/.envs/conda-lulc"
    if [ ! -x "$ENV/bin/python" ]; then
        echo "No TensorFlow on this server image: creating $ENV with TensorFlow + CUDA (about 10 minutes) ..."
        conda create -y -q -p "$ENV" python=3.11 > /dev/null
        "$ENV/bin/python" -m pip install -q "tensorflow[and-cuda]==2.20.*" -r <(grep -v tensorflow "$REPO/requirements.txt")
    fi
    export PATH="$ENV/bin:$PATH"
else
    echo "ERROR: neither TensorFlow nor conda found on this server image."
    return 1 2> /dev/null || exit 1
fi

# Jupyter kernel "Python (lulc)" pointing to this environment
python -m ipykernel install --user --name lulc --display-name "Python (lulc)" > /dev/null

# Earth Engine login stored in the vault, linked to where the ee library looks for it
mkdir -p "$VAULT/.earthengine" "$HOME/.config"
if [ ! -L "$HOME/.config/earthengine" ]; then
    [ -d "$HOME/.config/earthengine" ] && cp -n "$HOME/.config/earthengine/"* "$VAULT/.earthengine/" 2> /dev/null
    rm -rf "$HOME/.config/earthengine"
    ln -s "$VAULT/.earthengine" "$HOME/.config/earthengine"
fi

# Check instead of assuming
python - << 'EOF'
import importlib.util, sys
missing = [m for m in ["tensorflow", "ipykernel", "ee", "rasterio"] if importlib.util.find_spec(m) is None]
if missing:
    sys.exit(f"NOT READY - missing packages: {missing}")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
print(f"Python {sys.version.split()[0]} | TensorFlow {tf.__version__} | GPUs: {len(gpus)}")
if not gpus:
    print("WARNING: no GPU visible - training would be very slow. Start the server with a GPU image.")
EOF
[ -f "$VAULT/.earthengine/credentials" ] && echo "Earth Engine login found." \
    || echo "Not logged in to Earth Engine yet. Run once: earthengine authenticate --auth_mode=notebook"
