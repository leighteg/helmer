use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{Context, Result, bail};
use wasmparser::{MemoryType, Payload, TableType, TypeRef, ValType};

const DEFAULT_MOUNT_ID: &str = "helmer-root";
const DEFAULT_CANVAS_ID: &str = "helmer-canvas";
const TEMPLATE_DIR: &str = "web/template";
const WORKER_PACKAGE: &str = "helmer";
const WORKER_BIN: &str = "helmer_asset_worker";
const WORKER_FEATURE: &str = "asset-worker";

fn main() -> Result<()> {
    let mut args = env::args().skip(1);
    let Some(cmd) = args.next() else {
        print_usage();
        return Ok(());
    };

    match cmd.as_str() {
        "web" => cmd_web(args.collect()),
        "-h" | "--help" | "help" => {
            print_usage();
            Ok(())
        }
        _ => {
            print_usage();
            bail!("unknown command: {}", cmd);
        }
    }
}

fn cmd_web(args: Vec<String>) -> Result<()> {
    let mut package: Option<String> = None;
    let mut bin: Option<String> = None;
    let mut out: Option<PathBuf> = None;
    let mut assets: Vec<PathBuf> = Vec::new();
    let mut profile: BuildProfile = BuildProfile::Release;
    let mut features: Option<String> = None;
    let mut no_default_features = false;
    let mut title: Option<String> = None;

    let mut iter = args.into_iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "-p" | "--package" => {
                package = Some(next_value(&mut iter, &arg)?);
            }
            "--bin" => {
                bin = Some(next_value(&mut iter, &arg)?);
            }
            "--out" => {
                out = Some(PathBuf::from(next_value(&mut iter, &arg)?));
            }
            "--assets" => {
                assets.push(PathBuf::from(next_value(&mut iter, &arg)?));
            }
            "--profile" => {
                let value = next_value(&mut iter, &arg)?;
                profile = BuildProfile::Custom(value);
            }
            "--release" => profile = BuildProfile::Release,
            "--dev" => profile = BuildProfile::Debug,
            "--features" => {
                features = Some(next_value(&mut iter, &arg)?);
            }
            "--no-default-features" => no_default_features = true,
            "--title" => {
                title = Some(next_value(&mut iter, &arg)?);
            }
            "-h" | "--help" => {
                print_web_usage();
                return Ok(());
            }
            _ => bail!("unknown arg: {}", arg),
        }
    }

    let Some(package) = package else {
        print_web_usage();
        bail!("--package is required");
    };

    let bin = bin.unwrap_or_else(|| package.clone());

    let root = repo_root()?;
    let out_dir = match out {
        Some(path) if path.is_absolute() => path,
        Some(path) => root.join(path),
        None => root.join("dist").join(&bin),
    };

    if out_dir.exists() {
        fs::remove_dir_all(&out_dir)
            .with_context(|| format!("failed to clear {}", out_dir.display()))?;
    }
    fs::create_dir_all(&out_dir)
        .with_context(|| format!("failed to create {}", out_dir.display()))?;

    build_wasm(
        &root,
        &package,
        &bin,
        &profile,
        features.clone(),
        no_default_features,
    )?;
    let worker_features = append_feature(features.clone(), WORKER_FEATURE);
    build_wasm(
        &root,
        WORKER_PACKAGE,
        WORKER_BIN,
        &profile,
        worker_features,
        no_default_features,
    )?;

    let wasm_path = wasm_output_path(&root, &bin, &profile);
    if !wasm_path.exists() {
        bail!("wasm output missing: {}", wasm_path.display());
    }

    let worker_wasm_path = wasm_output_path(&root, WORKER_BIN, &profile);
    if !worker_wasm_path.exists() {
        bail!("wasm worker output missing: {}", worker_wasm_path.display());
    }

    run_wasm_bindgen(&root, &wasm_path, &out_dir, &bin, profile.is_debug())?;
    run_wasm_bindgen(
        &root,
        &worker_wasm_path,
        &out_dir,
        WORKER_BIN,
        profile.is_debug(),
    )?;
    patch_env_imports(&out_dir, &bin)?;
    patch_env_imports(&out_dir, WORKER_BIN)?;
    patch_worker_bridge_registration(&out_dir, &bin)?;
    write_web_template(
        &root,
        &out_dir,
        title.as_deref().unwrap_or(&bin),
        &bin,
        WORKER_BIN,
    )?;

    for asset in assets {
        copy_path(&asset, &out_dir)?;
    }

    println!("web dist ready: {}", out_dir.display());
    Ok(())
}

#[derive(Clone, Debug)]
enum BuildProfile {
    Release,
    Debug,
    Custom(String),
}

impl BuildProfile {
    fn is_debug(&self) -> bool {
        matches!(self, BuildProfile::Debug)
    }

    fn dir_name(&self) -> &str {
        match self {
            BuildProfile::Release => "release",
            BuildProfile::Debug => "debug",
            BuildProfile::Custom(name) => name.as_str(),
        }
    }
}

fn build_wasm(
    root: &Path,
    package: &str,
    bin: &str,
    profile: &BuildProfile,
    features: Option<String>,
    no_default_features: bool,
) -> Result<()> {
    let mut cmd = Command::new("cargo");
    cmd.current_dir(root);
    cmd.arg("build")
        .arg("-p")
        .arg(package)
        .arg("--bin")
        .arg(bin)
        .arg("--target")
        .arg("wasm32-unknown-unknown");

    match profile {
        BuildProfile::Release => {
            cmd.arg("--release");
        }
        BuildProfile::Debug => {}
        BuildProfile::Custom(name) => {
            cmd.arg("--profile").arg(name);
        }
    }

    if let Some(features) = features {
        cmd.arg("--features").arg(features);
    }
    if no_default_features {
        cmd.arg("--no-default-features");
    }

    let status = cmd.status().context("failed to spawn cargo build")?;
    if !status.success() {
        bail!("cargo build failed with status {}", status);
    }
    Ok(())
}

fn wasm_output_path(root: &Path, bin: &str, profile: &BuildProfile) -> PathBuf {
    root.join("target")
        .join("wasm32-unknown-unknown")
        .join(profile.dir_name())
        .join(format!("{}.wasm", bin))
}

fn run_wasm_bindgen(
    root: &Path,
    wasm_path: &Path,
    out_dir: &Path,
    out_name: &str,
    debug: bool,
) -> Result<()> {
    let mut cmd = Command::new("wasm-bindgen");
    cmd.current_dir(root);
    cmd.arg("--target")
        .arg("web")
        .arg("--no-typescript")
        .arg("--out-dir")
        .arg(out_dir)
        .arg("--out-name")
        .arg(out_name)
        .arg(wasm_path);

    if debug {
        cmd.arg("--debug");
    }

    cmd.stdout(Stdio::inherit());
    cmd.stderr(Stdio::inherit());

    let status = cmd.status().context("failed to spawn wasm-bindgen")?;
    if !status.success() {
        bail!("wasm-bindgen failed with status {}", status);
    }
    Ok(())
}

fn patch_env_imports(out_dir: &Path, module: &str) -> Result<()> {
    let js_path = out_dir.join(format!("{}.js", module));
    let wasm_path = out_dir.join(format!("{}_bg.wasm", module));
    if !js_path.exists() || !wasm_path.exists() {
        return Ok(());
    }

    let js_source = fs::read_to_string(&js_path)
        .with_context(|| format!("failed to read {}", js_path.display()))?;
    let (patched_js, needs_env) = rewrite_env_import(&js_source);

    if needs_env {
        fs::write(&js_path, patched_js)
            .with_context(|| format!("failed to write {}", js_path.display()))?;
        let env_imports = collect_env_imports(&wasm_path)?;
        write_env_module(out_dir, &env_imports)?;
    }

    Ok(())
}

fn patch_worker_bridge_registration(out_dir: &Path, module: &str) -> Result<()> {
    let js_path = out_dir.join(format!("{}.js", module));
    if !js_path.exists() {
        return Ok(());
    }

    let source = fs::read_to_string(&js_path)
        .with_context(|| format!("failed to read {}", js_path.display()))?;
    let mut patched = source.clone();
    let mut updated = false;

    if !patched.contains("registerHelmerWorkerBridge") {
        if let Some(new_source) = inject_worker_bridge_import(&patched) {
            patched = new_source;
            updated = true;
        }
    }

    let has_register = patched.contains("registerHelmerWorkerBridge");
    if has_register
        && patched.contains("wasm.__wbindgen_start();")
        && !patched.contains("registerHelmerWorkerBridge(wasm);")
    {
        patched = patched.replacen(
            "wasm.__wbindgen_start();",
            "wasm.__wbindgen_start();\n    registerHelmerWorkerBridge(wasm);",
            1,
        );
        updated = true;
    }

    if updated {
        fs::write(&js_path, patched)
            .with_context(|| format!("failed to write {}", js_path.display()))?;
    }

    Ok(())
}

fn inject_worker_bridge_import(source: &str) -> Option<String> {
    for line in source.lines() {
        let trimmed = line.trim_start();
        if !trimmed.starts_with("import {") || !line.contains("worker_bridge.js") {
            continue;
        }
        if line.contains("registerHelmerWorkerBridge") {
            return None;
        }
        let start = line.find('{')?;
        let end = line.find('}')?;
        let prefix = &line[..start + 1];
        let suffix = &line[end..];
        let list = &line[start + 1..end];
        let mut items: Vec<&str> = list
            .split(',')
            .map(|item| item.trim())
            .filter(|item| !item.is_empty())
            .collect();
        items.push("registerHelmerWorkerBridge");
        let new_list = items.join(", ");
        let new_line = format!("{} {} {}", prefix, new_list, suffix);
        let updated = source.replacen(line, &new_line, 1);
        return Some(updated);
    }
    None
}

fn rewrite_env_import(source: &str) -> (String, bool) {
    let mut needs_env = false;
    let mut output = String::with_capacity(source.len());

    for line in source.lines() {
        let trimmed = line.trim_start();
        let indent_len = line.len() - trimmed.len();
        let indent = &line[..indent_len];
        let mut replaced = false;

        if let Some(rest) = trimmed.strip_prefix("import * as ") {
            if let Some(from_idx) = rest.find(" from ") {
                let alias = rest[..from_idx].trim();
                let mut module = rest[from_idx + " from ".len()..].trim();
                module = module.trim_end_matches(';').trim();

                if module == "\"env\"" || module == "'env'" {
                    let replacement = format!("{}import {} from \"./env.js\";", indent, alias);
                    output.push_str(&replacement);
                    output.push('\n');
                    needs_env = true;
                    replaced = true;
                }
            }
        }

        if !replaced {
            output.push_str(line);
            output.push('\n');
        }
    }

    if !source.ends_with('\n') && output.ends_with('\n') {
        output.pop();
    }

    (output, needs_env)
}

#[derive(Clone, Debug)]
enum EnvImportKind {
    Func,
    Table(TableType),
    Memory(MemoryType),
    Global(ValType, bool),
}

#[derive(Clone, Debug)]
struct EnvImport {
    name: String,
    kind: EnvImportKind,
}

fn collect_env_imports(wasm_path: &Path) -> Result<Vec<EnvImport>> {
    let bytes =
        fs::read(wasm_path).with_context(|| format!("failed to read {}", wasm_path.display()))?;
    let mut imports = Vec::new();

    for payload in wasmparser::Parser::new(0).parse_all(&bytes) {
        match payload? {
            Payload::ImportSection(section) => {
                for import in section {
                    let import = import?;
                    if import.module != "env" {
                        continue;
                    }
                    let kind = match import.ty {
                        TypeRef::Func(_) => EnvImportKind::Func,
                        TypeRef::Table(table) => EnvImportKind::Table(table),
                        TypeRef::Memory(mem) => EnvImportKind::Memory(mem),
                        TypeRef::Global(global) => {
                            EnvImportKind::Global(global.content_type, global.mutable)
                        }
                        TypeRef::Tag(_) => continue,
                    };
                    imports.push(EnvImport {
                        name: import.name.to_string(),
                        kind,
                    });
                }
            }
            Payload::End(_) => break,
            _ => {}
        }
    }

    Ok(imports)
}

fn write_env_module(out_dir: &Path, imports: &[EnvImport]) -> Result<()> {
    let env_path = out_dir.join("env.js");
    let mut output = String::from("const env = {\n");

    for import in imports {
        let key = js_string(&import.name);
        let value = match &import.kind {
            EnvImportKind::Func => "(..._args) => 0".to_string(),
            EnvImportKind::Global(val_type, mutable) => {
                let (value_type, init) = js_global_init(*val_type)?;
                format!(
                    "new WebAssembly.Global({{ value: \"{}\", mutable: {} }}, {})",
                    value_type, mutable, init
                )
            }
            EnvImportKind::Memory(mem) => js_memory_init(mem)?,
            EnvImportKind::Table(table) => js_table_init(table)?,
        };
        output.push_str(&format!("  \"{}\": {},\n", key, value));
    }

    output.push_str("};\nexport default env;\n");
    fs::write(&env_path, output)
        .with_context(|| format!("failed to write {}", env_path.display()))?;
    Ok(())
}

fn js_string(input: &str) -> String {
    input.replace('\\', "\\\\").replace('"', "\\\"")
}

fn js_global_init(val_type: ValType) -> Result<(&'static str, &'static str)> {
    match val_type {
        ValType::I32 => Ok(("i32", "0")),
        ValType::I64 => Ok(("i64", "0n")),
        ValType::F32 => Ok(("f32", "0")),
        ValType::F64 => Ok(("f64", "0")),
        ValType::V128 => bail!("env import uses v128 global, which is unsupported"),
        ValType::Ref(ref_type) => {
            if ref_type.is_func_ref() {
                Ok(("anyfunc", "null"))
            } else {
                Ok(("externref", "null"))
            }
        }
    }
}

fn js_memory_init(mem: &MemoryType) -> Result<String> {
    if mem.memory64 {
        bail!("env import uses memory64, which is unsupported");
    }
    let mut args = format!("{{ initial: {}", mem.initial);
    if let Some(max) = mem.maximum {
        args.push_str(&format!(", maximum: {}", max));
    }
    if mem.shared {
        args.push_str(", shared: true");
    }
    args.push_str(" }");
    Ok(format!("new WebAssembly.Memory({})", args))
}

fn js_table_init(table: &TableType) -> Result<String> {
    let element = if table.element_type.is_func_ref() {
        "anyfunc"
    } else {
        "externref"
    };
    let mut args = format!("{{ element: \"{}\", initial: {}", element, table.initial);
    if let Some(max) = table.maximum {
        args.push_str(&format!(", maximum: {}", max));
    }
    args.push_str(" }");
    Ok(format!("new WebAssembly.Table({})", args))
}

fn write_web_template(
    root: &Path,
    out_dir: &Path,
    title: &str,
    module: &str,
    worker_module: &str,
) -> Result<()> {
    let template_dir = root.join(TEMPLATE_DIR);
    let index_template = fs::read_to_string(template_dir.join("index.html"))
        .with_context(|| format!("missing {}/index.html", TEMPLATE_DIR))?;
    let main_template = fs::read_to_string(template_dir.join("main.js"))
        .with_context(|| format!("missing {}/main.js", TEMPLATE_DIR))?;
    let worker_bridge_template = fs::read_to_string(template_dir.join("worker_bridge.js"))
        .with_context(|| format!("missing {}/worker_bridge.js", TEMPLATE_DIR))?;
    let worker_template = fs::read_to_string(template_dir.join("asset_worker.js"))
        .with_context(|| format!("missing {}/asset_worker.js", TEMPLATE_DIR))?;

    let index_output = render_template(&index_template, title, module, worker_module);
    let main_output = render_template(&main_template, title, module, worker_module);
    let worker_bridge_output =
        render_template(&worker_bridge_template, title, module, worker_module);
    let worker_output = render_template(&worker_template, title, module, worker_module);

    fs::write(out_dir.join("index.html"), index_output)
        .with_context(|| "failed to write index.html")?;
    fs::write(out_dir.join("main.js"), main_output).with_context(|| "failed to write main.js")?;
    fs::write(out_dir.join("worker_bridge.js"), worker_bridge_output)
        .with_context(|| "failed to write worker_bridge.js")?;
    fs::write(out_dir.join("asset_worker.js"), worker_output)
        .with_context(|| "failed to write asset_worker.js")?;

    Ok(())
}

fn render_template(template: &str, title: &str, module: &str, worker_module: &str) -> String {
    template
        .replace("{{TITLE}}", title)
        .replace("{{WASM_MODULE}}", module)
        .replace("{{WASM_WORKER_MODULE}}", worker_module)
        .replace("{{MOUNT_ID}}", DEFAULT_MOUNT_ID)
        .replace("{{CANVAS_ID}}", DEFAULT_CANVAS_ID)
}

fn copy_path(src: &Path, out_dir: &Path) -> Result<()> {
    let name = src
        .file_name()
        .ok_or_else(|| anyhow::anyhow!("invalid asset path: {}", src.display()))?;
    let dest = out_dir.join(name);

    if dest.exists() {
        if dest.is_dir() {
            fs::remove_dir_all(&dest)
                .with_context(|| format!("failed to remove {}", dest.display()))?;
        } else {
            fs::remove_file(&dest)
                .with_context(|| format!("failed to remove {}", dest.display()))?;
        }
    }

    if src.is_dir() {
        copy_dir_all(src, &dest)?;
    } else {
        fs::copy(src, &dest).with_context(|| format!("failed to copy {}", src.display()))?;
    }

    Ok(())
}

fn copy_dir_all(src: &Path, dst: &Path) -> Result<()> {
    fs::create_dir_all(dst).with_context(|| format!("failed to create {}", dst.display()))?;

    for entry in fs::read_dir(src).with_context(|| format!("failed to read {}", src.display()))? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let dest_path = dst.join(entry.file_name());

        if file_type.is_dir() {
            copy_dir_all(&entry.path(), &dest_path)?;
        } else {
            fs::copy(entry.path(), &dest_path)
                .with_context(|| format!("failed to copy {}", entry.path().display()))?;
        }
    }

    Ok(())
}

fn repo_root() -> Result<PathBuf> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = manifest_dir
        .parent()
        .context("xtask must live inside the workspace")?;
    Ok(root.to_path_buf())
}

fn next_value<I>(iter: &mut I, flag: &str) -> Result<String>
where
    I: Iterator<Item = String>,
{
    iter.next()
        .with_context(|| format!("missing value for {}", flag))
}

fn append_feature(features: Option<String>, extra: &str) -> Option<String> {
    let extra = extra.trim();
    if extra.is_empty() {
        return features;
    }
    match features {
        Some(existing) => {
            let has_extra = existing
                .split(',')
                .map(|entry| entry.trim())
                .any(|entry| entry == extra);
            if has_extra {
                Some(existing)
            } else if existing.trim().is_empty() {
                Some(extra.to_string())
            } else {
                Some(format!("{},{}", existing.trim(), extra))
            }
        }
        None => Some(extra.to_string()),
    }
}

fn print_usage() {
    println!("xtask <command> [options]\n");
    println!("commands:\n  web        build wasm + dist output\n");
    println!("run 'xtask web --help' for details");
}

fn print_web_usage() {
    println!("xtask web -p <package> [--bin <name>] [--out <dir>] [--assets <path>] [options]\n");
    println!("options:");
    println!("  --release                 build release (default)");
    println!("  --dev                     build debug");
    println!("  --profile <name>           build with custom profile");
    println!("  --features <list>          enable cargo features");
    println!("  --no-default-features      disable default features");
    println!("  --title <title>            override HTML title");
    println!("  --assets <path>            copy path into dist root (repeatable)");
}
