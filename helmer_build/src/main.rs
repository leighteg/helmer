use std::{
    env,
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
    process,
};

use libloading::{Library, Symbol};
use serde::{Deserialize, Serialize};

const REQUIRED_ABI_VERSION: u32 = 1;
const BUILD_LAUNCH_MANIFEST_VERSION: u32 = 1;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct HelmerBuildBuffer {
    ptr: *mut u8,
    len: usize,
    cap: usize,
}

type AbiVersionFn = unsafe extern "C" fn() -> u32;
type PackJsonFn = unsafe extern "C" fn(*const u8, usize, *mut HelmerBuildBuffer) -> i32;
type FreeBufferFn = unsafe extern "C" fn(HelmerBuildBuffer);

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BuildRequest {
    project_root: String,
    output_pack: String,
    key: String,
    compression_level: Option<u32>,
    include_hidden: Option<bool>,
    max_threads: Option<usize>,
    max_pack_bytes: Option<u64>,
    max_assets_per_pack: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BuildPackOutput {
    pack_path: String,
    asset_count: usize,
    chunk_count: usize,
    deduped_assets: usize,
    bytes_written: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BuildProjectConfig {
    name: String,
    version: u32,
    #[serde(default)]
    startup_scene: Option<String>,
    vscode_config_dir: String,
    assets_dir: String,
    models_dir: String,
    textures_dir: String,
    materials_dir: String,
    scenes_dir: String,
    scripts_dir: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BuildResponse {
    pack_path: String,
    pack_paths: Vec<String>,
    manifest_path: String,
    pack_count: usize,
    packs: Vec<BuildPackOutput>,
    project_root: String,
    project_name: String,
    project_config: BuildProjectConfig,
    startup_scene: String,
    created_unix_ms: u64,
    bytes_written: u64,
    asset_count: usize,
    chunk_count: usize,
    deduped_assets: usize,
    elapsed_ms: u128,
    key_fingerprint: String,
    warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BuildResultEnvelope {
    ok: bool,
    response: Option<BuildResponse>,
    error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BuildLaunchManifest {
    version: u32,
    created_unix_ms: u64,
    project_name: String,
    project_config: BuildProjectConfig,
    startup_scene: String,
    pack_manifest: String,
    pack_key: String,
    key_fingerprint: String,
}

#[derive(Debug)]
struct BuildCliArgs {
    project_root: PathBuf,
    output_pack: PathBuf,
    runtime_lib: PathBuf,
    key: String,
    compression_level: Option<u32>,
    include_hidden: bool,
    max_threads: Option<usize>,
    max_pack_bytes: Option<u64>,
    max_assets_per_pack: Option<usize>,
    player_exe: PathBuf,
    executable_name: Option<String>,
    launch_manifest: Option<PathBuf>,
}

#[derive(Debug)]
struct RuntimeBundleOutput {
    executable_path: PathBuf,
    launch_manifest_path: PathBuf,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("helmer_build: {err}");
        process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut args = env::args_os();
    let _exe = args.next();
    let Some(cmd) = args.next() else {
        print_usage();
        return Err("missing command".to_string());
    };

    if cmd != OsString::from("build") {
        print_usage();
        return Err(format!("unsupported command '{}'", cmd.to_string_lossy()));
    }

    let parsed = parse_build_args(args.collect())?;
    let request = BuildRequest {
        project_root: parsed.project_root.to_string_lossy().into_owned(),
        output_pack: parsed.output_pack.to_string_lossy().into_owned(),
        key: parsed.key.clone(),
        compression_level: parsed.compression_level,
        include_hidden: Some(parsed.include_hidden),
        max_threads: parsed.max_threads,
        max_pack_bytes: parsed.max_pack_bytes,
        max_assets_per_pack: parsed.max_assets_per_pack,
    };

    let response = invoke_runtime(&parsed.runtime_lib, &request)?;
    let bundle = emit_runtime_bundle(&parsed, &response)?;

    println!("Pack manifest: {}", response.manifest_path);
    println!("Pack count: {}", response.pack_count);
    for (index, pack) in response.packs.iter().enumerate() {
        println!(
            "Pack[{index}] {} assets={} chunks={} deduped={} bytes={}",
            pack.pack_path,
            pack.asset_count,
            pack.chunk_count,
            pack.deduped_assets,
            pack.bytes_written
        );
    }
    println!("Startup scene: {}", response.startup_scene);
    println!("Executable: {}", bundle.executable_path.to_string_lossy());
    println!(
        "Launch manifest: {}",
        bundle.launch_manifest_path.to_string_lossy()
    );
    println!("Total assets: {}", response.asset_count);
    println!("Total chunks: {}", response.chunk_count);
    println!("Total deduped assets: {}", response.deduped_assets);
    println!("Total bytes: {}", response.bytes_written);
    println!("Elapsed ms: {}", response.elapsed_ms);
    println!("Key fingerprint: {}", response.key_fingerprint);
    if !response.warnings.is_empty() {
        for warning in response.warnings {
            println!("Warning: {warning}");
        }
    }

    Ok(())
}

fn emit_runtime_bundle(
    parsed: &BuildCliArgs,
    response: &BuildResponse,
) -> Result<RuntimeBundleOutput, String> {
    if !parsed.player_exe.exists() {
        return Err(format!(
            "player executable template was not found: {}",
            parsed.player_exe.to_string_lossy()
        ));
    }

    let pack_manifest_path = PathBuf::from(&response.manifest_path);
    let output_dir = pack_manifest_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    fs::create_dir_all(&output_dir).map_err(|err| {
        format!(
            "failed to create output directory {}: {}",
            output_dir.to_string_lossy(),
            err
        )
    })?;

    let executable_name = runtime_executable_name(
        &parsed.player_exe,
        parsed.executable_name.as_deref(),
        &response.project_name,
    );
    let executable_path = output_dir.join(executable_name);
    copy_executable_template(&parsed.player_exe, &executable_path)?;

    let launch_manifest_path = parsed
        .launch_manifest
        .clone()
        .unwrap_or_else(|| default_launch_manifest_path(&executable_path));

    let pack_manifest_ref = relative_or_absolute_path(
        launch_manifest_path
            .parent()
            .unwrap_or_else(|| Path::new(".")),
        &pack_manifest_path,
    );
    let manifest = BuildLaunchManifest {
        version: BUILD_LAUNCH_MANIFEST_VERSION,
        created_unix_ms: response.created_unix_ms,
        project_name: response.project_name.clone(),
        project_config: response.project_config.clone(),
        startup_scene: response.startup_scene.clone(),
        pack_manifest: pack_manifest_ref,
        pack_key: parsed.key.clone(),
        key_fingerprint: response.key_fingerprint.clone(),
    };
    write_launch_manifest(&launch_manifest_path, &manifest)?;

    Ok(RuntimeBundleOutput {
        executable_path,
        launch_manifest_path,
    })
}

fn copy_executable_template(source: &Path, destination: &Path) -> Result<(), String> {
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            format!(
                "failed to create executable output directory {}: {}",
                parent.to_string_lossy(),
                err
            )
        })?;
    }

    fs::copy(source, destination).map_err(|err| {
        format!(
            "failed to copy runtime executable from {} to {}: {}",
            source.to_string_lossy(),
            destination.to_string_lossy(),
            err
        )
    })?;

    let permissions = fs::metadata(source)
        .map(|meta| meta.permissions())
        .map_err(|err| {
            format!(
                "failed to read executable permissions from {}: {}",
                source.to_string_lossy(),
                err
            )
        })?;
    fs::set_permissions(destination, permissions).map_err(|err| {
        format!(
            "failed to apply executable permissions to {}: {}",
            destination.to_string_lossy(),
            err
        )
    })?;

    Ok(())
}

fn runtime_executable_name(
    template_path: &Path,
    explicit_name: Option<&str>,
    project_name: &str,
) -> String {
    let template_ext = template_path
        .extension()
        .and_then(|ext| ext.to_str())
        .filter(|ext| !ext.is_empty());

    let base_name = explicit_name
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| sanitize_executable_stem(project_name));

    let has_extension = Path::new(&base_name).extension().is_some();
    if has_extension {
        base_name
    } else if let Some(ext) = template_ext {
        format!("{base_name}.{ext}")
    } else {
        base_name
    }
}

fn sanitize_executable_stem(project_name: &str) -> String {
    let mut output = String::with_capacity(project_name.len());
    for ch in project_name.trim().chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
            output.push(ch);
        } else if ch.is_whitespace() {
            output.push('_');
        } else {
            output.push('_');
        }
    }
    let output = output.trim_matches('_');
    if output.is_empty() {
        "helmer_project".to_string()
    } else {
        output.to_string()
    }
}

fn default_launch_manifest_path(executable_path: &Path) -> PathBuf {
    let stem = executable_path
        .file_stem()
        .and_then(|name| name.to_str())
        .unwrap_or("helmer_project");
    executable_path.with_file_name(format!("{stem}.launch.json"))
}

fn write_launch_manifest(path: &Path, manifest: &BuildLaunchManifest) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            format!(
                "failed to create launch-manifest directory {}: {}",
                parent.to_string_lossy(),
                err
            )
        })?;
    }

    let payload = serde_json::to_vec_pretty(manifest)
        .map_err(|err| format!("failed to encode launch manifest JSON: {err}"))?;
    fs::write(path, payload).map_err(|err| {
        format!(
            "failed to write launch manifest {}: {}",
            path.to_string_lossy(),
            err
        )
    })
}

fn relative_or_absolute_path(base_dir: &Path, path: &Path) -> String {
    if let Ok(relative) = path.strip_prefix(base_dir) {
        normalize_path(relative)
    } else {
        normalize_path(path)
    }
}

fn normalize_path(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

fn invoke_runtime(runtime_lib: &PathBuf, request: &BuildRequest) -> Result<BuildResponse, String> {
    // SAFETY: loading a dynamic library is inherently unsafe; symbol usage is validated below.
    let library = unsafe {
        Library::new(runtime_lib).map_err(|err| {
            format!(
                "failed to load runtime library {}: {}",
                runtime_lib.to_string_lossy(),
                err
            )
        })?
    };

    // SAFETY: symbol signatures are expected to match the exported C ABI in helmer_build_runtime.
    let abi_version: Symbol<'_, AbiVersionFn> = unsafe { library.get(b"helmer_build_abi_version") }
        .map_err(|err| format!("missing symbol helmer_build_abi_version: {err}"))?;

    // SAFETY: symbol signatures are expected to match the exported C ABI in helmer_build_runtime.
    let pack_json: Symbol<'_, PackJsonFn> = unsafe { library.get(b"helmer_build_pack_json") }
        .map_err(|err| format!("missing symbol helmer_build_pack_json: {err}"))?;

    // SAFETY: symbol signatures are expected to match the exported C ABI in helmer_build_runtime.
    let free_buffer: Symbol<'_, FreeBufferFn> = unsafe { library.get(b"helmer_build_free_buffer") }
        .map_err(|err| format!("missing symbol helmer_build_free_buffer: {err}"))?;

    // SAFETY: function pointer is loaded from validated symbol above.
    let abi = unsafe { abi_version() };
    if abi != REQUIRED_ABI_VERSION {
        return Err(format!(
            "runtime ABI mismatch: expected {}, found {}",
            REQUIRED_ABI_VERSION, abi
        ));
    }

    let payload = serde_json::to_vec(request)
        .map_err(|err| format!("failed to encode request JSON: {err}"))?;

    let mut out = HelmerBuildBuffer {
        ptr: std::ptr::null_mut(),
        len: 0,
        cap: 0,
    };

    // SAFETY: pointers and lengths are valid for the request payload and out-parameter.
    let code = unsafe {
        pack_json(
            payload.as_ptr(),
            payload.len(),
            &mut out as *mut HelmerBuildBuffer,
        )
    };

    if out.ptr.is_null() {
        return Err(format!(
            "runtime returned null response buffer (code {code})"
        ));
    }

    // SAFETY: buffer is allocated and owned by runtime; we only read len bytes.
    let result_slice = unsafe { std::slice::from_raw_parts(out.ptr as *const u8, out.len) };
    let envelope: BuildResultEnvelope = serde_json::from_slice(result_slice)
        .map_err(|err| format!("failed to decode runtime response JSON: {err}"))?;

    // SAFETY: buffer originates from runtime and must be released by its matching free function.
    unsafe {
        free_buffer(out);
    }

    if !envelope.ok {
        return Err(envelope
            .error
            .unwrap_or_else(|| format!("runtime build failed (code {code})")));
    }

    envelope
        .response
        .ok_or_else(|| "runtime did not return a response payload".to_string())
}

fn parse_build_args(args: Vec<OsString>) -> Result<BuildCliArgs, String> {
    let mut project_root: Option<PathBuf> = None;
    let mut output_pack: Option<PathBuf> = None;
    let mut runtime_lib: Option<PathBuf> = None;
    let mut key: Option<String> = None;
    let mut compression_level: Option<u32> = None;
    let mut include_hidden = false;
    let mut max_threads: Option<usize> = None;
    let mut max_pack_bytes: Option<u64> = None;
    let mut max_assets_per_pack: Option<usize> = None;
    let mut player_exe: Option<PathBuf> = None;
    let mut executable_name: Option<String> = None;
    let mut launch_manifest: Option<PathBuf> = None;

    let mut idx = 0usize;
    while idx < args.len() {
        let arg = args[idx].to_string_lossy().to_string();
        match arg.as_str() {
            "--project" => {
                idx += 1;
                let value = args
                    .get(idx)
                    .ok_or_else(|| "--project expects a path".to_string())?;
                project_root = Some(PathBuf::from(value));
            }
            "--output" => {
                idx += 1;
                let value = args
                    .get(idx)
                    .ok_or_else(|| "--output expects a path".to_string())?;
                output_pack = Some(PathBuf::from(value));
            }
            "--runtime-lib" => {
                idx += 1;
                let value = args
                    .get(idx)
                    .ok_or_else(|| "--runtime-lib expects a path".to_string())?;
                runtime_lib = Some(PathBuf::from(value));
            }
            "--player-exe" => {
                idx += 1;
                let value = args
                    .get(idx)
                    .ok_or_else(|| "--player-exe expects a path".to_string())?;
                player_exe = Some(PathBuf::from(value));
            }
            "--executable-name" => {
                idx += 1;
                let value = args
                    .get(idx)
                    .ok_or_else(|| "--executable-name expects a value".to_string())?;
                executable_name = Some(value.to_string_lossy().into_owned());
            }
            "--launch-manifest" => {
                idx += 1;
                let value = args
                    .get(idx)
                    .ok_or_else(|| "--launch-manifest expects a path".to_string())?;
                launch_manifest = Some(PathBuf::from(value));
            }
            "--key" => {
                idx += 1;
                let value = args
                    .get(idx)
                    .ok_or_else(|| "--key expects a value".to_string())?;
                key = Some(value.to_string_lossy().into_owned());
            }
            "--compression" => {
                idx += 1;
                let value = args
                    .get(idx)
                    .ok_or_else(|| "--compression expects a value".to_string())?;
                let parsed = value
                    .to_string_lossy()
                    .parse::<u32>()
                    .map_err(|err| format!("invalid --compression value: {err}"))?;
                if parsed > 9 {
                    return Err("--compression must be between 0 and 9".to_string());
                }
                compression_level = Some(parsed);
            }
            "--threads" => {
                idx += 1;
                let value = args
                    .get(idx)
                    .ok_or_else(|| "--threads expects a value".to_string())?;
                let parsed = value
                    .to_string_lossy()
                    .parse::<usize>()
                    .map_err(|err| format!("invalid --threads value: {err}"))?;
                if parsed == 0 {
                    return Err("--threads must be greater than zero".to_string());
                }
                max_threads = Some(parsed);
            }
            "--max-pack-bytes" => {
                idx += 1;
                let value = args
                    .get(idx)
                    .ok_or_else(|| "--max-pack-bytes expects a value".to_string())?;
                let parsed = value
                    .to_string_lossy()
                    .parse::<u64>()
                    .map_err(|err| format!("invalid --max-pack-bytes value: {err}"))?;
                if parsed == 0 {
                    return Err("--max-pack-bytes must be greater than zero".to_string());
                }
                max_pack_bytes = Some(parsed);
            }
            "--max-assets-per-pack" => {
                idx += 1;
                let value = args
                    .get(idx)
                    .ok_or_else(|| "--max-assets-per-pack expects a value".to_string())?;
                let parsed = value
                    .to_string_lossy()
                    .parse::<usize>()
                    .map_err(|err| format!("invalid --max-assets-per-pack value: {err}"))?;
                if parsed == 0 {
                    return Err("--max-assets-per-pack must be greater than zero".to_string());
                }
                max_assets_per_pack = Some(parsed);
            }
            "--include-hidden" => {
                include_hidden = true;
            }
            "-h" | "--help" => {
                print_usage();
                process::exit(0);
            }
            other => {
                return Err(format!("unknown argument '{other}'"));
            }
        }
        idx += 1;
    }

    let project_root = project_root.ok_or_else(|| "missing --project".to_string())?;
    let output_pack = output_pack.ok_or_else(|| "missing --output".to_string())?;

    let runtime_lib = runtime_lib
        .or_else(|| env::var_os("HELMER_BUILD_RUNTIME_LIB").map(PathBuf::from))
        .or_else(default_runtime_library_path)
        .ok_or_else(|| {
            "missing --runtime-lib and HELMER_BUILD_RUNTIME_LIB is not set".to_string()
        })?;

    let key = key
        .or_else(|| env::var("HELMER_BUILD_KEY").ok())
        .ok_or_else(|| "missing --key and HELMER_BUILD_KEY is not set".to_string())?;

    let player_exe = player_exe
        .or_else(|| env::var_os("HELMER_BUILD_PLAYER_EXE").map(PathBuf::from))
        .or_else(default_player_executable_path)
        .ok_or_else(|| "missing --player-exe and HELMER_BUILD_PLAYER_EXE is not set".to_string())?;

    Ok(BuildCliArgs {
        project_root,
        output_pack,
        runtime_lib,
        key,
        compression_level,
        include_hidden,
        max_threads,
        max_pack_bytes,
        max_assets_per_pack,
        player_exe,
        executable_name,
        launch_manifest,
    })
}

fn default_runtime_library_path() -> Option<PathBuf> {
    let exe = env::current_exe().ok()?;
    let parent = exe.parent()?;
    Some(parent.join(default_runtime_library_file_name()))
}

fn default_runtime_library_file_name() -> &'static str {
    #[cfg(target_os = "windows")]
    {
        "helmer_build_runtime.dll"
    }
    #[cfg(target_os = "macos")]
    {
        "libhelmer_build_runtime.dylib"
    }
    #[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
    {
        "libhelmer_build_runtime.so"
    }
}

fn default_player_executable_path() -> Option<PathBuf> {
    let exe = env::current_exe().ok()?;
    let parent = exe.parent()?;
    Some(parent.join(default_player_executable_file_name()))
}

fn default_player_executable_file_name() -> &'static str {
    #[cfg(target_os = "windows")]
    {
        "helmer_player.exe"
    }
    #[cfg(not(target_os = "windows"))]
    {
        "helmer_player"
    }
}

fn print_usage() {
    eprintln!(
        "Usage:\n  helmer_build build --project <path> --output <pack_file> --runtime-lib <compiled_runtime_lib> --key <secret> [--player-exe <compiled_player_exe>] [--executable-name <name>] [--launch-manifest <path>] [--compression 0..9] [--threads N] [--max-pack-bytes N] [--max-assets-per-pack N] [--include-hidden]"
    );
}
