# Embedding Visualization

`visualize_embedding.py` consumes model-independent token arrays and renders
PCA maps, clustering maps, and spatial-coherence summaries.

```bash
.venv_vjepa2/bin/python experiments/visualization/visualize_embedding.py \
  experiments/outputs/example_tokens.npy \
  --out-dir experiments/outputs/example_visualization
```

For video or volume tokens, provide the flattened token geometry:

```bash
.venv_vjepa2/bin/python experiments/visualization/visualize_embedding.py \
  experiments/outputs/video_tokens.npy \
  --grid-shape 8x24x24 \
  --slice-index 4 \
  --out-dir experiments/outputs/video_visualization
```

For video-model tokens, the existing static PNG can be supplemented with an
MP4 over every temporal slice:

```bash
.venv_vjepa2/bin/python experiments/visualization/visualize_embedding.py \
  experiments/outputs/video_tokens.npy \
  --grid-shape 8x24x24 \
  --slice-index 0 \
  --image experiments/outputs/input_frame.jpg \
  --reference-video experiments/inputs/example.mp4 \
  --animate \
  --tubelet-size 2 \
  --out-dir experiments/outputs/video_visualization
```

This writes both `embedding_visualization.png` and
`embedding_visualization.mp4`. The animation fits one PCA basis and one KMeans
model across all temporal and spatial tokens, keeping colors and cluster labels
stable across frames. For tubelet size 2, temporal slice `t` is shown with input
frame `2t`. When a video model receives a repeated still image, omit
`--reference-video`; `--image` is repeated in the input panel.
