## build editor project
`helmer_player` & libs should be compiled for target platform
- `cargo build -p helmer_player -p helmer_build_runtime -p helmer_editor_runtime --release`
- `cargo run -p helmer_build --release build --project <PROJ_DIR> --output dist/pack.hpck --key 01234567`

## web build example [`helmer_view`]
- add wasm target: `rustup target add wasm32-unknown-unknown`
- install `clang`/`clang++` (required for `basis-universal-sys` C++ sources on wasm)
- install matching wasm-bindgen-cli: `cargo install wasm-bindgen-cli --locked --version 0.2.111`
- build helmer_view dist: `cargo run -p xtask -- web -p helmer_view --bin helmer_view`
- *or* build becs_bench dist: `cargo run -p xtask -- web -p becs_bench --bin becs_bench --assets test_game/assets`
- output: `dist/helmer_view` (serve with `python3 -m http.server --directory dist/helmer_view`)
