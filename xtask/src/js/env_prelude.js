let wasm_exports = null;
const env_allocations = new Map();
const ENV_ALLOC_ALIGN = 16;

function to_u32(value) {
  if (typeof value === "bigint") {
    return Number(value & 0xffffffffn) >>> 0;
  }
  const numeric = Number(value);
  return Number.isFinite(numeric) ? (numeric >>> 0) : 0;
}

function __set_wasm_exports(exports) {
  wasm_exports = exports;
}

function env_malloc(size) {
  if (!wasm_exports || typeof wasm_exports.__wbindgen_malloc !== "function") {
    return 0;
  }
  const len = to_u32(size);
  const alloc_len = len === 0 ? 1 : len;
  const ptr = wasm_exports.__wbindgen_malloc(alloc_len, ENV_ALLOC_ALIGN) >>> 0;
  env_allocations.set(ptr, alloc_len);
  return ptr;
}

function env_free(ptr) {
  if (!wasm_exports || typeof wasm_exports.__wbindgen_free !== "function") {
    return;
  }
  const addr = to_u32(ptr);
  if (addr === 0) {
    return;
  }
  const len = env_allocations.get(addr);
  if (len === undefined) {
    return;
  }
  env_allocations.delete(addr);
  wasm_exports.__wbindgen_free(addr, len, ENV_ALLOC_ALIGN);
}

function env_realloc(ptr, size) {
  if (!wasm_exports || typeof wasm_exports.__wbindgen_realloc !== "function") {
    return 0;
  }
  const addr = to_u32(ptr);
  const new_len = to_u32(size);
  if (new_len === 0) {
    env_free(addr);
    return 0;
  }
  if (addr === 0) {
    return env_malloc(new_len);
  }
  const old_len = env_allocations.get(addr);
  if (old_len === undefined) {
    return 0;
  }
  const next = wasm_exports.__wbindgen_realloc(addr, old_len, new_len, ENV_ALLOC_ALIGN) >>> 0;
  env_allocations.delete(addr);
  env_allocations.set(next, new_len);
  return next;
}

function env_abort(..._args) {
  throw new Error("abort called from wasm env import");
}

function env_assert_fail(..._args) {
  throw new Error("__assert_fail called from wasm env import");
}

function env_printf(..._args) {
  return 0;
}

function env_fputs(..._args) {
  return 0;
}

function env_fwrite(_ptr, _size, nmemb, _stream) {
  return to_u32(nmemb);
}

function basisu_encoder_init() {
  return 0;
}

const env = {
  "__set_wasm_exports": __set_wasm_exports,
