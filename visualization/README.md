# Experiment visualization

`browser.py` is a marimo notebook that scans `experiments/outputs/` for
visualized experiments using the current output schema. Select an experiment
and one input to compare all available models. A shared frame slider steps
through temporal results while static results remain on their single image.

Extraction-only results and older output layouts are intentionally omitted.
After producing new visualizations, use **Refresh experiments** in the notebook.
The notebook has no custom zoom overlay, JavaScript, or CSS.

## Use in VS Code

1. Install the official **marimo** extension (`marimo-team.vscode-marimo`).
2. Open `visualization/browser.py` in VS Code.
3. If it opens as plain Python, run **marimo: Open as marimo notebook** from the
   Command Palette or use the marimo notebook icon in the editor title bar.
4. Select the repository's `.venv_vjepa2` Python environment when prompted.
5. Run the notebook cells.

The selected Python environment must contain marimo. The standard inference
environment includes `requirements/visualization.txt`; to add only the viewer
dependency to another environment, run:

```bash
python -m pip install -r requirements/visualization.txt
```

VS Code manages the notebook kernel and its local runtime, so no server command
or port selection is needed.

## Use in a browser

From the repository root, start an editable marimo session:

```bash
.venv_vjepa2/bin/marimo edit visualization/browser.py
```

Marimo prints the local URL to open. To serve the notebook as an application
without editing controls, run:

```bash
.venv_vjepa2/bin/marimo run visualization/browser.py --host 127.0.0.1 --port 43871
```

The server remains attached to that terminal; stop it with `Ctrl+C`. Binding to
`127.0.0.1` keeps it accessible only from the local machine. Choose another
port with `--port` if `43871` is already occupied.

## Data location

The notebook always reads the repository's `experiments/outputs/` directory.
Each browsable result must contain valid `visualization.png` and `metadata.json`
artifacts. Numbered `visualization_frame_*.png` files are shown in a manual
shared-frame presentation when present; `embeddings.h5` is not required for
browsing.
