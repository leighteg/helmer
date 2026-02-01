import init from "./{{WASM_MODULE}}.js";
import { registerHelmerWorkerBridge } from "./worker_bridge.js";

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

async function boot() {
  try {
    const wasm = await init();
    registerHelmerWorkerBridge(wasm);
    registerDropTarget(wasm);
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
