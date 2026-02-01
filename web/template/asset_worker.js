import init, {
  handle_worker_request,
  helmer_worker_set_opfs_enabled,
  helmer_worker_store_virtual_asset,
  helmer_worker_release_scene_buffers,
} from "./{{WASM_WORKER_MODULE}}.js";

let ready = null;

async function ensureReady() {
  if (!ready) {
    ready = init();
  }
  await ready;
}

async function dispatchAssetRequest(data) {
  await ensureReady();
  const payload = data instanceof Uint8Array ? data : new Uint8Array(data);
  const response = await handle_worker_request(payload);
  const bytes = response instanceof Uint8Array ? response : new Uint8Array(response);
  self.postMessage(bytes, [bytes.buffer]);
}

self.onmessage = async (event) => {
  const data = event.data;
  if (data && typeof data === "object" && data.type) {
    await ensureReady();
    switch (data.type) {
      case "config": {
        const values = data.values && typeof data.values === "object" ? data.values : {};
        for (const [key, value] of Object.entries(values)) {
          self[key] = value;
        }
        return;
      }
      case "opfs":
        helmer_worker_set_opfs_enabled(Boolean(data.enabled));
        return;
      case "virtual-asset": {
        const path = String(data.path || "");
        const bytes = data.bytes instanceof Uint8Array ? data.bytes : new Uint8Array(data.bytes || []);
        helmer_worker_store_virtual_asset(path, bytes);
        return;
      }
      case "release-scene":
        helmer_worker_release_scene_buffers(Number(data.sceneId));
        return;
      default:
        break;
    }
  }

  try {
    await dispatchAssetRequest(data);
  } catch (err) {
    console.error("Asset worker request failed:", err);
  }
};
