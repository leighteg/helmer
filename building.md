## web builds
- install wasm-bindgen-cli: `cargo install wasm-bindgen-cli`
- build helmer_view dist: `cargo run -p xtask -- web -p helmer_view --bin helmer_view --assets test_game/assets`
- output: `dist/helmer_view` (serve with `python3 -m http.server --directory dist/helmer_view`)
