(() => {
  "use strict";

  const data = window.REVIEW_DATA;
  if (!data) {
    document.body.textContent = "Review data could not be loaded.";
    return;
  }

  const viewLabels = {
    pca_rgb: "PCA RGB",
    pc1: "PC1",
    kmeans: "KMeans k=4",
    full: "Full figure",
  };
  const linkLabels = {
    full_figure: "Full figure",
    animation: "Temporal video",
    result: "Result JSON",
    summary: "Summary",
    visualization_manifest: "Viz manifest",
    embedding: "Tokens",
    extract_log: "Extract log",
    visualize_log: "Viz log",
  };

  const recordsByPair = new Map(
    data.records.map((record) => [`${record.input}/${record.model}`, record]),
  );
  const state = {
    mode: "matrix",
    context: data.inputs[0],
    view: "pca_rgb",
    tileSize: 240,
  };
  let detailItems = [];
  let detailIndex = 0;

  const elements = {
    title: document.querySelector("#app-title"),
    summary: document.querySelector("#run-summary"),
    modes: [...document.querySelectorAll("[data-mode]")],
    contextControl: document.querySelector("#context-control"),
    contextLabel: document.querySelector("#context-label"),
    contextSelect: document.querySelector("#context-select"),
    viewControl: document.querySelector("#view-control"),
    viewSelect: document.querySelector("#view-select"),
    tileSize: document.querySelector("#tile-size"),
    reference: document.querySelector("#reference-strip"),
    grid: document.querySelector("#review-grid"),
    dialog: document.querySelector("#detail-dialog"),
    detailContext: document.querySelector("#detail-context"),
    detailTitle: document.querySelector("#detail-title"),
    detailReference: document.querySelector("#detail-reference"),
    detailMap: document.querySelector("#detail-map"),
    detailViewLabel: document.querySelector("#detail-view-label"),
    detailMetadata: document.querySelector("#detail-metadata"),
    detailLinks: document.querySelector("#detail-links"),
    detailPrevious: document.querySelector("#detail-previous"),
    detailNext: document.querySelector("#detail-next"),
    detailPosition: document.querySelector("#detail-position"),
  };

  function label(value) {
    return String(value)
      .replaceAll("-", " ")
      .replaceAll("_", " ")
      .replace(/\b\w/g, (character) => character.toUpperCase());
  }

  function parseHash() {
    const params = new URLSearchParams(window.location.hash.slice(1));
    if (["input", "model", "matrix"].includes(params.get("mode"))) {
      state.mode = params.get("mode");
    }
    if (viewLabels[params.get("view")]) {
      state.view = params.get("view");
    }
    const context = params.get("context");
    if (context) {
      state.context = context;
    }
  }

  function writeHash() {
    const params = new URLSearchParams({
      mode: state.mode,
      view: state.view,
      context: state.context,
    });
    history.replaceState(null, "", `#${params.toString()}`);
  }

  function recordFor(input, model) {
    return recordsByPair.get(`${input}/${model}`);
  }

  function setContextOptions() {
    const values = state.mode === "model" ? data.models : data.inputs;
    if (!values.includes(state.context)) {
      state.context = values[0];
    }
    elements.contextLabel.textContent = state.mode === "model" ? "Model" : "Input";
    elements.contextSelect.replaceChildren(
      ...values.map((value) => {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = label(value);
        option.selected = value === state.context;
        return option;
      }),
    );
  }

  function renderReference() {
    elements.reference.replaceChildren();
    if (state.mode !== "input") {
      return;
    }
    const record = data.models
      .map((model) => recordFor(state.context, model))
      .find(Boolean);
    if (!record) {
      return;
    }

    const image = document.createElement("img");
    image.src = record.preview;
    image.alt = `${label(record.input)} canonical input`;

    const copy = document.createElement("div");
    copy.className = "reference-copy";
    const name = document.createElement("strong");
    name.textContent = label(record.input);
    const details = document.createElement("span");
    details.textContent = data.resolutionMode === "native"
      ? "Canonical crop resized to each model's native resolution"
      : `${record.inputResolution} x ${record.inputResolution} canonical reference`;
    const modality = document.createElement("span");
    modality.className = "modality";
    modality.textContent = record.inputModality;
    copy.append(name, details, modality);
    elements.reference.append(image, copy);
  }

  function createHeader(text, corner = false) {
    const header = document.createElement("div");
    header.className = `grid-header${corner ? " grid-corner" : ""}`;
    header.textContent = text;
    return header;
  }

  function createRowHeader(primary, secondary) {
    const header = document.createElement("div");
    header.className = "row-header";
    const strong = document.createElement("strong");
    strong.textContent = label(primary);
    const small = document.createElement("small");
    small.textContent = secondary;
    header.append(strong, small);
    return header;
  }

  function createTile(record, view, caption = "") {
    if (!record) {
      const missing = document.createElement("div");
      missing.className = "missing";
      missing.textContent = "Not run";
      return missing;
    }
    const button = document.createElement("button");
    button.type = "button";
    button.className = "tile";
    button.title = `Inspect ${label(record.model)} on ${label(record.input)}: ${viewLabels[view]}`;
    const image = document.createElement("img");
    image.loading = "lazy";
    image.src = record.views[view];
    image.alt = `${label(record.model)} on ${label(record.input)}, ${viewLabels[view]}`;
    button.append(image);
    if (caption) {
      const overlay = document.createElement("span");
      overlay.className = "tile-caption";
      overlay.textContent = caption;
      button.append(overlay);
    }
    button.addEventListener("click", () => openDetail(record, view));
    return button;
  }

  function renderInputOrModel() {
    const byInput = state.mode === "input";
    const rows = byInput ? data.models : data.inputs;
    const views = Object.keys(viewLabels);
    const table = document.createElement("div");
    table.className = "review-table four-views";
    table.style.setProperty("--column-count", String(views.length));
    table.append(createHeader(byInput ? "Model" : "Input", true));
    views.forEach((view) => table.append(createHeader(viewLabels[view])));

    detailItems = [];
    rows.forEach((row) => {
      const record = byInput
        ? recordFor(state.context, row)
        : recordFor(row, state.context);
      const secondary = record
        ? `${record.inputResolution}px / ${record.spatialGrid.join("x")} grid`
        : "missing";
      table.append(createRowHeader(row, secondary));
      views.forEach((view) => {
        table.append(createTile(record, view));
        if (record) {
          detailItems.push({ record, view });
        }
      });
    });
    elements.grid.replaceChildren(table);
  }

  function renderMatrix() {
    const table = document.createElement("div");
    table.className = "review-table matrix";
    table.style.setProperty("--column-count", String(data.inputs.length));
    table.append(createHeader("Model", true));
    data.inputs.forEach((input) => table.append(createHeader(label(input))));

    detailItems = [];
    data.models.forEach((model) => {
      const first = data.inputs.map((input) => recordFor(input, model)).find(Boolean);
      table.append(
        createRowHeader(
          model,
          first
            ? `${label(first.modelModality)} / ${first.inputResolution}px`
            : "missing",
        ),
      );
      data.inputs.forEach((input) => {
        const record = recordFor(input, model);
        const caption = record && record.adaptation !== "native_modality"
          ? label(record.adaptation)
          : "";
        table.append(createTile(record, state.view, caption));
        if (record) {
          detailItems.push({ record, view: state.view });
        }
      });
    });
    elements.grid.replaceChildren(table);
  }

  function render() {
    setContextOptions();
    elements.modes.forEach((button) => {
      const active = button.dataset.mode === state.mode;
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    elements.contextControl.hidden = state.mode === "matrix";
    elements.viewControl.hidden = state.mode !== "matrix";
    elements.viewSelect.value = state.view;
    document.documentElement.style.setProperty("--tile-size", `${state.tileSize}px`);
    renderReference();
    if (state.mode === "matrix") {
      renderMatrix();
    } else {
      renderInputOrModel();
    }
    writeHash();
  }

  function metadataEntries(record) {
    const metrics = record.metrics;
    const similarity = metrics.adjacent_cosine_similarity;
    const boundary = metrics.boundary_fraction;
    const entries = [
      ["Model modality", label(record.modelModality)],
      ["Input modality", label(record.inputModality)],
      ["Adaptation", label(record.adaptation)],
      ["Spatial grid", record.spatialGrid.join(" x ")],
      ["Temporal grid", record.temporalGrid ?? "none"],
      ["Embedding shape", record.embeddingShape.join(" x ")],
      ["Input resolution", `${record.inputResolution} x ${record.inputResolution}`],
      ["Native resolution", record.nativeResolution],
      ["Patch size", record.patchSize],
      ["Position geometry", record.positionalGeometry || "not recorded"],
    ];
    if (typeof similarity === "number") {
      entries.push(["Adjacent cosine similarity", similarity.toFixed(3)]);
    }
    if (typeof boundary === "number") {
      entries.push(["KMeans boundary fraction", boundary.toFixed(3)]);
    }
    return entries;
  }

  function showDetail() {
    const item = detailItems[detailIndex];
    const { record, view } = item;
    elements.detailContext.textContent = `${label(record.input)} / ${viewLabels[view]}`;
    elements.detailTitle.textContent = label(record.model);
    elements.detailReference.src = record.preview;
    elements.detailReference.alt = `${label(record.input)} canonical input`;
    elements.detailMap.src = record.views[view];
    elements.detailMap.alt = `${label(record.model)} ${viewLabels[view]} representation`;
    elements.detailViewLabel.textContent = viewLabels[view];
    elements.detailMetadata.replaceChildren();
    metadataEntries(record).forEach(([key, value]) => {
      const term = document.createElement("dt");
      term.textContent = key;
      const description = document.createElement("dd");
      description.textContent = value;
      elements.detailMetadata.append(term, description);
    });
    elements.detailLinks.replaceChildren();
    Object.entries(record.links).forEach(([key, url]) => {
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.textContent = linkLabels[key] || label(key);
      anchor.target = key === "embedding" ? "_self" : "_blank";
      elements.detailLinks.append(anchor);
    });
    if (record.inputSource) {
      const source = document.createElement("a");
      source.href = record.inputSource;
      source.textContent = "Input source";
      source.target = "_blank";
      source.rel = "noreferrer";
      elements.detailLinks.append(source);
    }
    elements.detailPosition.textContent = `${detailIndex + 1} / ${detailItems.length}`;
    elements.detailPrevious.disabled = detailIndex === 0;
    elements.detailNext.disabled = detailIndex === detailItems.length - 1;
  }

  function openDetail(record, view) {
    const index = detailItems.findIndex(
      (item) => item.record.name === record.name && item.view === view,
    );
    detailIndex = Math.max(index, 0);
    showDetail();
    elements.dialog.showModal();
  }

  elements.title.textContent = data.title;
  document.title = data.title;
  elements.summary.textContent =
    `${data.records.length} results / ${data.models.length} models / ${data.inputs.length} inputs`;

  elements.modes.forEach((button) => {
    button.addEventListener("click", () => {
      state.mode = button.dataset.mode;
      render();
    });
  });
  elements.contextSelect.addEventListener("change", (event) => {
    state.context = event.target.value;
    render();
  });
  elements.viewSelect.addEventListener("change", (event) => {
    state.view = event.target.value;
    render();
  });
  elements.tileSize.addEventListener("input", (event) => {
    state.tileSize = Number(event.target.value);
    document.documentElement.style.setProperty("--tile-size", `${state.tileSize}px`);
  });
  document.querySelector("#detail-close").addEventListener("click", () => {
    elements.dialog.close();
  });
  elements.detailPrevious.addEventListener("click", () => {
    detailIndex = Math.max(0, detailIndex - 1);
    showDetail();
  });
  elements.detailNext.addEventListener("click", () => {
    detailIndex = Math.min(detailItems.length - 1, detailIndex + 1);
    showDetail();
  });
  elements.dialog.addEventListener("click", (event) => {
    if (event.target === elements.dialog) {
      elements.dialog.close();
    }
  });
  document.addEventListener("keydown", (event) => {
    if (!elements.dialog.open) {
      return;
    }
    if (event.key === "ArrowLeft" && detailIndex > 0) {
      detailIndex -= 1;
      showDetail();
    }
    if (event.key === "ArrowRight" && detailIndex < detailItems.length - 1) {
      detailIndex += 1;
      showDetail();
    }
  });

  parseHash();
  render();
})();
