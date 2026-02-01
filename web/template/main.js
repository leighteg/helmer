import init from "./{{WASM_MODULE}}.js";
import { registerHelmerWorkerBridge } from "./worker_bridge.js";

const splashRoot = document.getElementById("helmer-splash");
const splashBar = document.getElementById("helmer-progress-bar");
const splashLabel = document.getElementById("helmer-progress-label");

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function setSplashProgress(value, text) {
  if (splashBar) {
    if (!Number.isFinite(value)) {
      splashBar.classList.add("is-indeterminate");
      splashBar.style.transform = "scaleX(0.2)";
    } else {
      const clamped = clamp(value, 0, 1);
      splashBar.classList.remove("is-indeterminate");
      splashBar.style.transform = `scaleX(${clamped})`;
    }
  }
  if (splashLabel && text) {
    splashLabel.textContent = text;
  }
}

function hideSplash() {
  if (!splashRoot) {
    return;
  }
  splashRoot.classList.add("splash--hidden");
  window.setTimeout(() => {
    splashRoot.remove();
  }, 700);
}

function waitForCanvasReady() {
  const mount = document.getElementById("{{MOUNT_ID}}");
  if (!mount) {
    hideSplash();
    return;
  }

  const hasCanvas = () => mount.querySelector("canvas");
  if (hasCanvas()) {
    hideSplash();
    return;
  }

  const observer = new MutationObserver(() => {
    if (hasCanvas()) {
      observer.disconnect();
      hideSplash();
    }
  });

  observer.observe(mount, { childList: true });
}

async function fetchWasmWithProgress(url, onProgress) {
  const response = await fetch(url, { credentials: "same-origin" });
  if (!response.ok) {
    throw new Error(`Failed to fetch wasm (${response.status} ${response.statusText})`);
  }

  const total = Number(response.headers.get("Content-Length")) || 0;
  if (!response.body || total <= 0) {
    onProgress?.(null, total);
    const buffer = await response.arrayBuffer();
    onProgress?.(1, buffer.byteLength);
    return new Uint8Array(buffer);
  }

  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  onProgress?.(0, total);
  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    if (value) {
      chunks.push(value);
      received += value.byteLength;
      onProgress?.(received / total, total);
    }
  }

  const buffer = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    buffer.set(chunk, offset);
    offset += chunk.byteLength;
  }
  onProgress?.(1, total);
  return buffer;
}

function normalizePath(path) {
  if (!path) {
    return "";
  }
  let normalized = path.replace(/\\/g, "/");
  while (normalized.startsWith("./")) {
    normalized = normalized.slice(2);
  }
  if (normalized.startsWith("/")) {
    normalized = normalized.slice(1);
  }
  return normalized;
}

function readEntryFile(entry) {
  return new Promise((resolve, reject) => {
    entry.file(resolve, reject);
  });
}

function readAllEntries(reader) {
  return new Promise((resolve, reject) => {
    const entries = [];
    const readNext = () => {
      reader.readEntries(
        (batch) => {
          if (!batch.length) {
            resolve(entries);
            return;
          }
          entries.push(...batch);
          readNext();
        },
        (err) => reject(err),
      );
    };
    readNext();
  });
}

async function walkEntry(entry, out) {
  if (entry.isFile) {
    const file = await readEntryFile(entry);
    const path = entry.fullPath || file.webkitRelativePath || file.name;
    out.push({ path, file });
    return;
  }
  if (entry.isDirectory) {
    const entries = await readAllEntries(entry.createReader());
    for (const child of entries) {
      await walkEntry(child, out);
    }
  }
}

async function collectDroppedFiles(event) {
  const out = [];
  const dataTransfer = event.dataTransfer;
  if (!dataTransfer) {
    return out;
  }
  const items = dataTransfer.items ? Array.from(dataTransfer.items) : [];
  const entries = [];
  for (const item of items) {
    if (item.kind !== "file") {
      continue;
    }
    const entry = item.webkitGetAsEntry ? item.webkitGetAsEntry() : null;
    if (entry) {
      entries.push(entry);
    } else {
      const file = item.getAsFile ? item.getAsFile() : null;
      if (file) {
        out.push({ path: file.webkitRelativePath || file.name, file });
      }
    }
  }
  if (entries.length) {
    await Promise.all(entries.map((entry) => walkEntry(entry, out)));
  } else if (dataTransfer.files && dataTransfer.files.length) {
    for (const file of Array.from(dataTransfer.files)) {
      out.push({ path: file.webkitRelativePath || file.name, file });
    }
  }
  return out;
}

function registerDropTarget(wasm) {
  if (!wasm || typeof wasm.helmer_store_virtual_asset !== "function") {
    console.warn("helmer_store_virtual_asset export missing; drop-to-import disabled.");
    return;
  }

  const target = document.getElementById("{{MOUNT_ID}}") || document.body;
  const swallow = (event) => {
    event.preventDefault();
    event.stopPropagation();
  };

  target.addEventListener("dragover", swallow);
  target.addEventListener("drop", async (event) => {
    swallow(event);
    const dropped = await collectDroppedFiles(event);
    for (const item of dropped) {
      const normalized = normalizePath(item.path);
      if (!normalized) {
        continue;
      }
      const bytes = new Uint8Array(await item.file.arrayBuffer());
      wasm.helmer_store_virtual_asset(normalized, bytes);
    }
  });
}

function registerUnfocusClear(wasm) {
  if (!wasm || typeof wasm.helmer_clear_input_state !== "function") {
    return;
  }

  const clearInput = () => {
    try {
      wasm.helmer_clear_input_state();
    } catch (err) {
      console.warn("Failed to clear input state on unfocus.", err);
    }
  };

  window.addEventListener("blur", clearInput);
  window.addEventListener("pagehide", clearInput);
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "hidden") {
      clearInput();
    }
  });
}

async function boot() {
  try {
    const wasmUrl = new URL("./{{WASM_MODULE}}_bg.wasm", import.meta.url);
    setSplashProgress(0, "Loading engine... 0%");
    const wasmBytes = await fetchWasmWithProgress(wasmUrl, (progress) => {
      if (!Number.isFinite(progress)) {
        setSplashProgress(null, "Loading engine...");
        return;
      }
      const percent = Math.round(progress * 100);
      setSplashProgress(progress, `Loading engine... ${percent}%`);
    });
    setSplashProgress(1, "Initializing runtime...");
    const initPromise = init({ module_or_path: wasmBytes });
    waitForCanvasReady();
    const wasm = await initPromise;
    registerHelmerWorkerBridge(wasm);
    registerDropTarget(wasm);
    registerUnfocusClear(wasm);
    hideSplash();
  } catch (err) {
    const message = err && err.message ? err.message : String(err);
    if (message.includes("Using exceptions for control flow")) {
      console.debug(err);
      return;
    }
    console.error(err);
    const root = document.getElementById("{{MOUNT_ID}}");
    if (root) {
      root.innerHTML = "<pre style=\"margin:16px;font:13px/1.4 'Space Mono',monospace;color:#f5f5f5;white-space:pre-wrap;\">" +
        "Web boot failed. Open the console for details.\n\n" +
        String(err) +
        "</pre>";
    }
    throw err;
  }
}

boot();
