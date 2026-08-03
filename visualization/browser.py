import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo

    root_dir = Path(__file__).resolve().parents[1]
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))

    from visualization.catalog import DEFAULT_OUTPUTS, discover_experiments

    return DEFAULT_OUTPUTS, discover_experiments, mo


@app.cell
def _(mo):
    refresh_experiments = mo.ui.run_button(
        label="Refresh experiments",
        kind="neutral",
    )
    refresh_experiments
    return (refresh_experiments,)


@app.cell
def _(DEFAULT_OUTPUTS, discover_experiments, refresh_experiments):
    # Reading the button value makes discovery rerun after every click.
    refresh_experiments.value
    experiments = discover_experiments(DEFAULT_OUTPUTS)
    return (experiments,)


@app.cell
def _(experiments, mo):
    mo.stop(
        not experiments,
        mo.md("No visualized experiments found.").callout(kind="warn"),
    )

    experiment_selector = mo.ui.dropdown(
        options={e["name"]: e["id"] for e in experiments},
        value=experiments[0]["name"],
        label="Experiment",
    )
    experiment_selector
    return (experiment_selector,)


@app.cell
def _(experiment_selector, experiments, mo):
    selected_experiment = next(
        e for e in experiments if e["id"] == experiment_selector.value
    )
    input_selector = mo.ui.dropdown(
        options=selected_experiment["rows"],
        value=selected_experiment["rows"][0],
        label="Input",
    )
    input_selector
    return input_selector, selected_experiment


@app.cell
def _(input_selector, selected_experiment):
    results_by_model = {
        r["model"]: r
        for r in selected_experiment["results"]
        if r["input"] == input_selector.value
    }
    max_frames = max(
        (len(r["frames"]) or 1 for r in results_by_model.values()), default=1
    )
    return max_frames, results_by_model


@app.cell
def _(max_frames, mo):
    # Defined here, but not displayed until the next cell so it renders
    # attached to the images instead of as its own separate block.
    frame_slider = mo.ui.slider(
        start=0,
        stop=max(max_frames - 1, 0),
        step=1,
        value=0,
        label="Frame",
        show_value=True,
        full_width=True,
    )
    return (frame_slider,)


@app.cell
def _(frame_slider, max_frames, mo, results_by_model, selected_experiment):
    frame_index = min(frame_slider.value, max_frames - 1)

    def model_card(model):
        result = results_by_model.get(model)
        if result is None:
            return mo.vstack([mo.md(f"**{model}**"), mo.md("—")])
        paths = result["frames"] or [result["image"]]
        path = paths[min(frame_index, len(paths) - 1)]
        return mo.vstack(
            [mo.md(f"**{model}**"), mo.image(src=str(path), width="100%")],
            gap=0.5,
        )

    mo.vstack(
        [
            frame_slider,
            mo.hstack(
                [model_card(model) for model in selected_experiment["columns"]],
                widths="equal",
                align="start",
                gap=1,
            ),
        ],
        gap=1,
    )


if __name__ == "__main__":
    app.run()
