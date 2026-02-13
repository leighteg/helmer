# helmer_build

`helmer_build` is a standalone project build CLI that dynamically loads a compiled
`helmer_build_runtime` library at runtime.

This allows build execution in environments that do **not** have this repository
or its crates available, as long as precompiled build/runtime binaries are
shipped with the tooling.

## Build artifacts in this repo

- Runtime library crate: `helmer_build_runtime` (`cdylib` + `rlib`)
- CLI crate: `helmer_build` (no direct dependency on editor/runtime source crates)
- Runtime executable crate: `helmer_player` (runs built projects)
- Shared modular runtime logic crate: `helmer_editor_runtime`

## Usage

```bash
helmer_build build \
  --project /path/to/project \
  --output /path/to/build/game.hpk \
  --runtime-lib /path/to/libhelmer_build_runtime.so \
  --player-exe /path/to/helmer_player \
  --key "hex:00112233445566778899aabbccddeeff"
```

### Pack sharding

The builder automatically emits multiple packs when a pack would exceed limits.
Defaults:

- `max_pack_bytes = 536870912` (512 MiB)
- `max_assets_per_pack = 50000`

Override with:

```bash
--max-pack-bytes 268435456 --max-assets-per-pack 20000
```

### Outputs

For output base `/build/game.hpk` and project name `My Game`:

- single pack: `/build/game.hpk`
- sharded: `/build/game_0000.hpk`, `/build/game_0001.hpk`, ...
- manifest (always): `/build/game.packs.json`
- runnable executable: `/build/My_Game` (or `/build/My_Game.exe`)
- runtime launch manifest: `/build/My_Game.launch.json`

Platform runtime library names:

- Linux: `libhelmer_build_runtime.so`
- macOS: `libhelmer_build_runtime.dylib`
- Windows: `helmer_build_runtime.dll`

Platform player executable names:

- Linux/macOS: `helmer_player`
- Windows: `helmer_player.exe`

`helmer_build` resolves `--player-exe` in this order:

1. explicit `--player-exe`
2. `HELMER_BUILD_PLAYER_EXE`
3. executable next to `helmer_build` (`helmer_player(.exe)`)

`helmer_player` resolves launch manifest in this order:

1. explicit `--manifest`
2. `<executable_stem>.launch.json` next to the executable

## How to use the packs

Use `helmer_build_runtime::PackSetReader` with the manifest + key.

```rust
use helmer_build_runtime::PackSetReader;

let reader = PackSetReader::open("/build/game.packs.json", "hex:001122...")?;
let bytes = reader.read_asset("materials/default.ron")?;
```

This gives the original asset bytes back in memory (deobfuscated + decompressed).
Integrate it into runtime asset IO by replacing direct `fs::read` calls with:

1. `PackSetReader::contains_asset(path)`
2. `PackSetReader::read_asset(path)`
3. fallback to filesystem when not found

## Running a built project

The build output executable is directly runnable:

```bash
/path/to/build/My_Game
```

Optional overrides:

```bash
/path/to/build/My_Game --manifest /path/to/build/My_Game.launch.json --key "hex:..."
```

## Pack format highlights

- Binary packs with fixed header + indexed TOC.
- Per-asset compression (deflate).
- Keyed binary transform for payload and path blobs (no plaintext/original asset
  bytes stored in pack).
- Chunk deduplication inside each pack.
- Per-entry checksums and content hashes for integrity/diagnostics.
