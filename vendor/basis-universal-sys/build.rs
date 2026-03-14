use std::process::{Command, Stdio};

// args from the basis cmake file
fn build_with_common_settings() -> cc::Build {
    let mut build = cc::Build::new();
    build
        .flag_if_supported("-fvisibility=hidden")
        .flag_if_supported("-fno-strict-aliasing")
        .flag_if_supported("-Wall")
        .flag_if_supported("-Wextra")
        .flag_if_supported("-Wno-unused-local-typedefs")
        .flag_if_supported("-Wno-unused-value")
        .flag_if_supported("-Wno-unused-parameter")
        .flag_if_supported("-Wno-unused-variable");

    build
}

fn main() {
    let target = std::env::var("TARGET").unwrap_or_default();
    let is_wasm = target.starts_with("wasm32");

    let mut build = build_with_common_settings();
    build
        .cpp(true)
        .define("BASISD_SUPPORT_KTX2_ZSTD", "0")
        //.define("BASISU_SUPPORT_SSE", "1") TODO: expose this in a futher release
        .flag_if_supported("--std=c++11");

    if is_wasm {
        build
            .flag_if_supported("-stdlib=libc++")
            .define("_LIBCPP_HAS_NO_THREADS", "1");

        let mut include_paths = detect_host_cxx_include_paths(true);
        if include_paths.is_empty() {
            include_paths = detect_host_cxx_include_paths(false);
        }
        for include in include_paths {
            build.flag("-isystem");
            build.flag(&include);
        }

        build
            .cpp_link_stdlib(None)
            .include("wasm-libcxx")
            .define("BASISU_NO_STD_STRING", "1")
            .flag_if_supported("-fno-exceptions")
            .flag_if_supported("-fno-rtti");

        build
            .file("vendor/basis_universal/transcoder/basisu_transcoder.cpp")
            .file("vendor/transcoding_wrapper.cpp");
    } else {
        build
            .file("vendor/basis_universal/encoder/pvpngreader.cpp")
            .file("vendor/basis_universal/encoder/jpgd.cpp")
            .file("vendor/basis_universal/encoder/basisu_uastc_enc.cpp")
            .file("vendor/basis_universal/encoder/basisu_ssim.cpp")
            .file("vendor/basis_universal/encoder/basisu_resampler.cpp")
            .file("vendor/basis_universal/encoder/basisu_resample_filters.cpp")
            .file("vendor/basis_universal/encoder/basisu_pvrtc1_4.cpp")
            .file("vendor/basis_universal/encoder/basisu_opencl.cpp")
            .file("vendor/basis_universal/encoder/basisu_kernels_sse.cpp")
            .file("vendor/basis_universal/encoder/basisu_gpu_texture.cpp")
            .file("vendor/basis_universal/encoder/basisu_frontend.cpp")
            .file("vendor/basis_universal/encoder/basisu_etc.cpp")
            .file("vendor/basis_universal/encoder/basisu_enc.cpp")
            .file("vendor/basis_universal/encoder/basisu_comp.cpp")
            .file("vendor/basis_universal/encoder/basisu_bc7enc.cpp")
            .file("vendor/basis_universal/encoder/basisu_basis_file.cpp")
            .file("vendor/basis_universal/encoder/basisu_backend.cpp")
            .file("vendor/basis_universal/transcoder/basisu_transcoder.cpp")
            .file("vendor/transcoding_wrapper.cpp")
            .file("vendor/encoding_wrapper.cpp");
    }

    build.compile("basisuniversal");

    // We regenerate binding code and check it in. (See generate_bindings.sh)
}

fn detect_host_cxx_include_paths(prefer_libcpp: bool) -> Vec<String> {
    let compiler = std::env::var("CXX_wasm32_unknown_unknown")
        .or_else(|_| std::env::var("CXX_wasm32-unknown-unknown"))
        .or_else(|_| std::env::var("CXX"))
        .unwrap_or_else(|_| "clang++".to_string());

    let mut command = Command::new(&compiler);
    if prefer_libcpp {
        command.arg("-stdlib=libc++");
    }
    let output = command
        .args(["-E", "-x", "c++", "-", "-v"])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .output();

    let output = match output {
        Ok(output) => output,
        Err(err) => {
            println!(
                "cargo:warning=failed to probe C++ include paths with {}: {}",
                compiler, err
            );
            return Vec::new();
        }
    };

    if !output.status.success() {
        println!(
            "cargo:warning=failed to probe C++ include paths with {} (status: {})",
            compiler, output.status
        );
        return Vec::new();
    }

    let stderr = String::from_utf8_lossy(&output.stderr);
    let mut include_paths = Vec::new();
    let mut in_block = false;

    for line in stderr.lines() {
        let trimmed = line.trim();
        if trimmed == "#include <...> search starts here:" {
            in_block = true;
            continue;
        }
        if !in_block {
            continue;
        }
        if trimmed == "End of search list." {
            break;
        }
        if trimmed.is_empty() {
            continue;
        }

        let candidate = trimmed
            .split(" (framework directory)")
            .next()
            .unwrap_or(trimmed)
            .trim();
        if !candidate.is_empty() {
            include_paths.push(candidate.to_string());
        }
    }

    include_paths
}
