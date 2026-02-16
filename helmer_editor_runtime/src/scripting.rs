use std::{
    fs,
    path::{Path, PathBuf},
};

pub const VISUAL_SCRIPT_EXTENSION: &str = "hvs";
pub const RUST_PREBUILT_PLUGIN_ROOT_DIR: &str = ".helmer/rust_scripts";
pub const RUST_PREBUILT_PLUGIN_FILE_STEM: &str = "script_plugin";

fn has_ascii_case_insensitive_suffix(text: &str, suffix: &str) -> bool {
    let text_bytes = text.as_bytes();
    let suffix_bytes = suffix.as_bytes();
    if text_bytes.len() < suffix_bytes.len() {
        return false;
    }
    text_bytes[text_bytes.len() - suffix_bytes.len()..]
        .iter()
        .zip(suffix_bytes.iter())
        .all(|(lhs, rhs)| lhs.eq_ignore_ascii_case(rhs))
}

pub fn is_script_path(path: &Path) -> bool {
    is_lua_script_path(path) || is_visual_script_path(path) || is_rust_script_path(path)
}

pub fn is_lua_script_path(path: &Path) -> bool {
    let Some(ext) = path.extension().and_then(|ext| ext.to_str()) else {
        return false;
    };
    if !ext.eq_ignore_ascii_case("lua") && !ext.eq_ignore_ascii_case("luau") {
        return false;
    }

    let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
        return true;
    };

    !has_ascii_case_insensitive_suffix(file_name, ".d.lua")
        && !has_ascii_case_insensitive_suffix(file_name, ".d.luau")
}

pub fn is_visual_script_path(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case(VISUAL_SCRIPT_EXTENSION))
        .unwrap_or(false)
}

pub fn is_rust_script_path(path: &Path) -> bool {
    resolve_rust_script_manifest(path).is_some()
}

pub fn resolve_rust_script_manifest(path: &Path) -> Option<PathBuf> {
    if path.is_dir() {
        let manifest = path.join("Cargo.toml");
        if manifest.exists() {
            return Some(manifest);
        }
        return None;
    }

    if path
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.eq_ignore_ascii_case("cargo.toml"))
    {
        return Some(path.to_path_buf());
    }

    if path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("rs"))
        .unwrap_or(false)
    {
        let mut current = path.parent();
        while let Some(dir) = current {
            let manifest = dir.join("Cargo.toml");
            if manifest.exists() {
                return Some(manifest);
            }
            current = dir.parent();
        }
    }

    None
}

pub fn script_registry_key_for_path(path: &Path) -> Option<PathBuf> {
    if is_lua_script_path(path) || is_visual_script_path(path) {
        return Some(path.to_path_buf());
    }
    resolve_rust_script_manifest(path)
}

pub fn script_language_from_path(path: &Path) -> String {
    if is_lua_script_path(path) {
        return if path
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("lua"))
        {
            "lua".to_string()
        } else {
            "luau".to_string()
        };
    }

    if is_visual_script_path(path) {
        return "visual".to_string();
    }

    if resolve_rust_script_manifest(path).is_some() {
        return "rust".to_string();
    }

    "text".to_string()
}

pub fn rust_target_library_path(manifest_path: &Path) -> Result<PathBuf, String> {
    let (_, lib_name) = rust_manifest_names(manifest_path)?;
    let Some(root) = manifest_path.parent() else {
        return Err("Rust script manifest has no parent directory".to_string());
    };

    let file_name = format!(
        "{}{}.{}",
        rust_dynamic_library_prefix(),
        lib_name,
        rust_dynamic_library_extension()
    );
    Ok(root.join("target").join("debug").join(file_name))
}

pub fn rust_manifest_names(manifest_path: &Path) -> Result<(String, String), String> {
    let manifest = fs::read_to_string(manifest_path).map_err(|err| {
        format!(
            "Failed to read Rust script manifest {}: {}",
            manifest_path.to_string_lossy(),
            err
        )
    })?;

    let mut section = String::new();
    let mut package_name: Option<String> = None;
    let mut lib_name: Option<String> = None;

    for line in manifest.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if trimmed.starts_with('[') && trimmed.ends_with(']') {
            section = trimmed
                .trim_start_matches('[')
                .trim_end_matches(']')
                .trim()
                .to_ascii_lowercase();
            continue;
        }

        let Some((key, raw_value)) = trimmed.split_once('=') else {
            continue;
        };
        if key.trim() != "name" {
            continue;
        }

        let raw_value = raw_value.split('#').next().unwrap_or("").trim();
        let Some(value) = parse_manifest_string_literal(raw_value) else {
            continue;
        };

        if section == "package" && package_name.is_none() {
            package_name = Some(value);
        } else if section == "lib" && lib_name.is_none() {
            lib_name = Some(value);
        }
    }

    let package_name = package_name.ok_or_else(|| {
        format!(
            "Rust script manifest missing [package].name: {}",
            manifest_path.to_string_lossy()
        )
    })?;
    let lib_name = lib_name
        .unwrap_or_else(|| package_name.replace('-', "_"))
        .replace('-', "_");
    Ok((package_name, lib_name))
}

pub fn rust_dynamic_library_prefix() -> &'static str {
    #[cfg(target_os = "windows")]
    {
        ""
    }
    #[cfg(not(target_os = "windows"))]
    {
        "lib"
    }
}

pub fn rust_dynamic_library_extension() -> &'static str {
    #[cfg(target_os = "windows")]
    {
        "dll"
    }
    #[cfg(target_os = "macos")]
    {
        "dylib"
    }
    #[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
    {
        "so"
    }
}

pub fn rust_prebuilt_plugin_relative_path(manifest_asset_relative: &Path) -> Option<PathBuf> {
    let file_name = manifest_asset_relative
        .file_name()
        .and_then(|name| name.to_str())?;
    if !file_name.eq_ignore_ascii_case("cargo.toml") {
        return None;
    }

    let mut relative = PathBuf::from(RUST_PREBUILT_PLUGIN_ROOT_DIR);
    if let Some(parent) = manifest_asset_relative.parent() {
        for component in parent.components() {
            match component {
                std::path::Component::Normal(segment) => relative.push(segment),
                std::path::Component::CurDir => {}
                std::path::Component::ParentDir
                | std::path::Component::RootDir
                | std::path::Component::Prefix(_) => return None,
            }
        }
    }
    relative.push(format!(
        "{}.{}",
        RUST_PREBUILT_PLUGIN_FILE_STEM,
        rust_dynamic_library_extension()
    ));
    Some(relative)
}

fn parse_manifest_string_literal(value: &str) -> Option<String> {
    let mut chars = value.chars();
    let quote = chars.next()?;
    if quote != '"' && quote != '\'' {
        return None;
    }
    let rest = chars.as_str();
    let end = rest.find(quote)?;
    Some(rest[..end].to_string())
}
