## build editor project
`helmer_player` & libs should be compiled for target platform
- `cargo run -p helmer_view --release`
- `cargo build -p helmer_build_runtime --release`
- `cargo build -p helmer_editor_runtime --release`
- `cargo run -p helmer_build --release build --project <PROJ_DIR> --output dist/pack.hpck --key 01234567`

## web build example [`helmer_view`]
- install wasm-bindgen-cli: `cargo install wasm-bindgen-cli`
- build helmer_view dist: `cargo run -p xtask -- web -p helmer_view --bin helmer_view --assets test_game/assets`
- output: `dist/helmer_view` (serve with `python3 -m http.server --directory dist/helmer_view`)
