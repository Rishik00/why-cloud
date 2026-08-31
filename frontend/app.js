const $ = (selector) => document.querySelector(selector);

const state = {
  artifacts: [],
  file: null,
  lastUrl: "",
  view: 0,
  isAnimating: false,
  touchStartY: 0,
  touchStartedInList: false,
};

const form = $("#upload-form");
const fileInput = $("#pdf-file");
const titleInput = $("#artifact-title");
const slugInput = $("#artifact-slug");
const status = $("#form-status");
const list = $("#folio-list");
const listStatus = $("#library-status");
const search = $("#search");
const dropZone = $("#drop-zone");
const deck = $("#deck");
const uploadView = $("#upload-view");
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

function setView(nextView, moveFocus = false) {
  if (nextView === state.view || state.isAnimating) return;
  state.view = nextView;
  state.isAnimating = true;
  deck.dataset.view = String(nextView);
  if (nextView === 0) uploadView.scrollTop = 0;

  const unlock = (event) => {
    if (event && event.target !== deck) return;
    state.isAnimating = false;
    deck.removeEventListener("transitionend", unlock);
  };
  deck.addEventListener("transitionend", unlock);
  window.setTimeout(() => unlock(), reducedMotion.matches ? 50 : 820);

  if (moveFocus) {
    window.setTimeout(() => {
      (nextView === 1 ? $("#library") : $("#upload-view")).focus();
    }, reducedMotion.matches ? 0 : 700);
  }
}

function slugify(value) {
  return value.toLowerCase().trim()
    .normalize("NFKD").replace(/[\u0300-\u036f]/g, "")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 80);
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes === 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const amount = bytes / (1024 ** index);
  return `${amount.toFixed(index === 0 || amount >= 10 ? 0 : 1)} ${units[index]}`;
}

function formatDate(value) {
  const date = new Date(value);
  if (!Number.isFinite(date.getTime())) return "Unknown date";
  return new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric", year: "numeric" }).format(date);
}

function setStatus(message, isError = false) {
  status.textContent = message;
  status.classList.toggle("is-error", isError);
}

function setFile(file) {
  if (!file) return;
  if (file.type !== "application/pdf" && !file.name.toLowerCase().endsWith(".pdf")) {
    setStatus("Choose a PDF file.", true);
    return;
  }
  if (file.size > 95 * 1024 * 1024) {
    setStatus("PDFs must be smaller than 95 MB.", true);
    return;
  }
  if (file.size === 0) {
    setStatus("Choose a non-empty PDF file.", true);
    return;
  }

  state.file = file;
  $("#file-name").textContent = file.name;
  $("#file-note").textContent = `${formatBytes(file.size)} · ready to publish`;
  if (!titleInput.value) titleInput.value = file.name.replace(/\.pdf$/i, "").replace(/[-_]+/g, " ");
  if (!slugInput.value) slugInput.value = slugify(titleInput.value);
  setStatus("");
}

function updateTotals() {
  $("#artifact-count").textContent = String(state.artifacts.length).padStart(2, "0");
  $("#artifact-size").textContent = formatBytes(state.artifacts.reduce((total, item) => total + item.size, 0));
}

function createFolioRow(artifact, index) {
  const link = document.createElement("a");
  link.className = "folio-row";
  link.href = artifact.url;
  link.target = "_blank";
  link.rel = "noopener";

  const number = document.createElement("span");
  number.className = "folio-index";
  number.textContent = String(index + 1).padStart(2, "0");

  const name = document.createElement("div");
  name.className = "folio-name";
  const heading = document.createElement("h3");
  heading.textContent = artifact.title;
  const slug = document.createElement("p");
  slug.textContent = `/a/${artifact.slug}`;
  name.append(heading, slug);

  const meta = document.createElement("p");
  meta.className = "folio-meta";
  meta.textContent = `${formatDate(artifact.uploaded)}\n${formatBytes(artifact.size)}`;

  const open = document.createElement("span");
  open.className = "open-mark";
  const openLabel = document.createElement("span");
  openLabel.textContent = "Read";
  const arrow = document.createElement("span");
  arrow.textContent = "↗";
  open.append(openLabel, arrow);

  link.append(number, name, meta, open);
  return link;
}

function renderArtifacts() {
  const query = search.value.trim().toLowerCase();
  const visible = state.artifacts.filter((artifact) =>
    artifact.title.toLowerCase().includes(query) || artifact.slug.includes(query)
  );

  list.replaceChildren();
  if (!visible.length) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = query ? "No folio matches that search." : "The shelf is empty. Publish the first folio above.";
    list.append(empty);
    listStatus.textContent = query ? "No matching folios." : "The shelf is empty.";
    return;
  }

  visible.forEach((artifact, index) => list.append(createFolioRow(artifact, index)));
  listStatus.textContent = `${visible.length} ${visible.length === 1 ? "folio" : "folios"} shown.`;
}

async function loadArtifacts() {
  try {
    const response = await fetch("/api/artifacts", { cache: "no-store" });
    if (!response.ok) throw new Error("The shelf could not be read.");
    const data = await response.json();
    if (!Array.isArray(data.artifacts)) throw new Error("The shelf returned an invalid response.");
    state.artifacts = data.artifacts.filter((artifact) =>
      artifact
      && typeof artifact.slug === "string"
      && typeof artifact.title === "string"
      && typeof artifact.size === "number"
      && typeof artifact.uploaded === "string"
      && typeof artifact.url === "string"
      && /^\/a\/[a-z0-9]+(?:-[a-z0-9]+)*$/.test(artifact.url)
    );
    updateTotals();
    renderArtifacts();
  } catch (error) {
    list.replaceChildren();
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = error instanceof Error ? error.message : "The shelf could not be read.";
    list.append(empty);
  }
}

function showToast(message, url) {
  state.lastUrl = url;
  $("#toast-message").textContent = message;
  $("#copy-link").textContent = "Copy link";
  $("#toast").hidden = false;
}

fileInput.addEventListener("change", () => setFile(fileInput.files?.[0]));
titleInput.addEventListener("input", () => {
  if (slugInput.dataset.edited !== "true") slugInput.value = slugify(titleInput.value);
});
slugInput.addEventListener("input", () => {
  slugInput.dataset.edited = "true";
  slugInput.value = slugify(slugInput.value);
});
search.addEventListener("input", renderArtifacts);

$("#show-library").addEventListener("click", () => setView(1, true));
$("#show-upload").addEventListener("click", () => setView(0, true));
$(".skip-link").addEventListener("click", (event) => {
  event.preventDefault();
  setView(1, true);
});

window.addEventListener("wheel", (event) => {
  if (state.isAnimating || Math.abs(event.deltaY) < 12) return;

  if (state.view === 0 && event.deltaY > 0) {
    const uploadHasMore = uploadView.scrollHeight > uploadView.clientHeight
      && uploadView.scrollTop + uploadView.clientHeight < uploadView.scrollHeight - 1;
    if (uploadHasMore) return;
    event.preventDefault();
    setView(1);
    return;
  }

  if (state.view === 1 && event.deltaY < 0) {
    const insideList = event.target instanceof Element && event.target.closest("#folio-list");
    if (insideList && list.scrollTop > 0) return;
    event.preventDefault();
    setView(0);
  }
}, { passive: false });

window.addEventListener("keydown", (event) => {
  const interactive = event.target instanceof Element
    && event.target.matches("input, button, a, textarea, select");
  if (interactive) return;

  const downKeys = ["ArrowDown", "PageDown", "End", " "];
  const upKeys = ["ArrowUp", "PageUp", "Home"];
  if (state.view === 0 && downKeys.includes(event.key)) {
    event.preventDefault();
    setView(1);
  } else if (state.view === 1 && upKeys.includes(event.key) && list.scrollTop === 0) {
    event.preventDefault();
    setView(0);
  }
});

window.addEventListener("touchstart", (event) => {
  state.touchStartY = event.touches[0]?.clientY ?? 0;
  state.touchStartedInList = event.target instanceof Element && Boolean(event.target.closest("#folio-list"));
}, { passive: true });

window.addEventListener("touchend", (event) => {
  const endY = event.changedTouches[0]?.clientY ?? state.touchStartY;
  const delta = endY - state.touchStartY;
  if (Math.abs(delta) < 48 || state.isAnimating) return;
  if (state.view === 0 && delta < 0) setView(1);
  if (state.view === 1 && delta > 0 && (!state.touchStartedInList || list.scrollTop === 0)) setView(0);
}, { passive: true });

for (const eventName of ["dragenter", "dragover"]) {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("is-dragging");
  });
}
for (const eventName of ["dragleave", "drop"]) {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("is-dragging");
  });
}
dropZone.addEventListener("drop", (event) => setFile(event.dataTransfer?.files?.[0]));

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!state.file) {
    setStatus("Choose a PDF first.", true);
    return;
  }

  const slug = slugify(slugInput.value);
  const title = titleInput.value.trim();
  if (!slug) {
    setStatus("Enter a short link using letters or numbers.", true);
    return;
  }
  const button = form.querySelector("button[type='submit']");
  button.disabled = true;
  setStatus("Publishing…");

  try {
    const response = await fetch(`/api/artifacts/${encodeURIComponent(slug)}`, {
      method: "PUT",
      headers: {
        "content-type": "application/pdf",
        "x-artifact-title": encodeURIComponent(title),
      },
      body: state.file,
    });
    const isJson = response.headers.get("content-type")?.includes("application/json");
    const data = isJson ? await response.json() : null;
    if (!response.ok) throw new Error(data?.error || "The PDF could not be published.");
    if (!data?.artifact || typeof data.artifact.url !== "string") {
      throw new Error("The upload completed, but the server response was invalid.");
    }

    form.reset();
    state.file = null;
    slugInput.dataset.edited = "false";
    $("#file-name").textContent = "Choose a PDF";
    $("#file-note").textContent = "or drop one here · up to 95 MB";
    setStatus("Published.");
    showToast(`${title} is on the shelf.`, data.artifact.url);
    await loadArtifacts();
  } catch (error) {
    setStatus(error instanceof Error ? error.message : "The PDF could not be published.", true);
  } finally {
    button.disabled = false;
  }
});

$("#copy-link").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(state.lastUrl);
    $("#copy-link").textContent = "Copied";
  } catch {
    setStatus("The link could not be copied. Open it from the shelf instead.", true);
  }
});
$("#toast-close").addEventListener("click", () => { $("#toast").hidden = true; });

loadArtifacts();
