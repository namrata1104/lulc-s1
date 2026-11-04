# Lab server setup. Only ~/vault survives a server restart, so the Python packages and the
# Earth Engine login are kept there. Run after every server start, from the repo folder:
#
#     source scripts/server_setup.sh
#
# The first run creates the environment (a few minutes); later runs take seconds.

VAULT="$HOME/vault"
VENV="$VAULT/venv-lulc"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Python environment on top of the server's TensorFlow (system site-packages)
if [ ! -x "$VENV/bin/python" ]; then
    echo "Creating $VENV ..."
    python3 -m venv --system-site-packages "$VENV"
    "$VENV/bin/pip" install -q -r <(grep -v tensorflow "$REPO/requirements.txt")
fi
source "$VENV/bin/activate"

# Jupyter kernel "Python (lulc)" pointing to this environment
python -m ipykernel install --user --name lulc --display-name "Python (lulc)" > /dev/null

# Earth Engine login stored in the vault, linked to where the ee library looks for it
mkdir -p "$VAULT/.earthengine" "$HOME/.config"
if [ ! -L "$HOME/.config/earthengine" ]; then
    [ -d "$HOME/.config/earthengine" ] && cp -n "$HOME/.config/earthengine/"* "$VAULT/.earthengine/" 2> /dev/null
    rm -rf "$HOME/.config/earthengine"
    ln -s "$VAULT/.earthengine" "$HOME/.config/earthengine"
fi

if [ -f "$VAULT/.earthengine/credentials" ]; then
    echo "Ready: environment active, kernel 'Python (lulc)' installed, Earth Engine login found."
else
    echo "Ready, but not logged in to Earth Engine yet. Run once:"
    echo "    earthengine authenticate --auth_mode=notebook"
fi
