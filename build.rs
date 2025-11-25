use std::env;

fn main() {
    // Set PKG_CONFIG_PATH to include all necessary directories for macOS
    if cfg!(target_os = "macos") {
        let homebrew_prefix = "/opt/homebrew";
        let pkg_config_paths = vec![
            format!("{}/opt/libxml2/lib/pkgconfig", homebrew_prefix),
            format!("{}/lib/pkgconfig", homebrew_prefix),
            format!("{}/share/pkgconfig", homebrew_prefix),
        ];

        let existing_path = env::var("PKG_CONFIG_PATH").unwrap_or_default();
        let new_path = if existing_path.is_empty() {
            pkg_config_paths.join(":")
        } else {
            format!("{}:{}", pkg_config_paths.join(":"), existing_path)
        };

        println!("cargo:warning=Setting PKG_CONFIG_PATH to: {}", new_path);
        env::set_var("PKG_CONFIG_PATH", &new_path);
        println!("cargo:rerun-if-env-changed=PKG_CONFIG_PATH");
    }

    // Set PLUGIN_VARIANT for plugin/element name suffixing
    let plugin_variant = env::var("PLUGIN_VARIANT").unwrap_or_else(|_| "".to_string());
    println!("cargo:rustc-env=PLUGIN_VARIANT={}", plugin_variant);

    // Generate the full plugin name
    // GStreamer plugin names should use hyphens, not underscores
    let pkg_name = env::var("CARGO_PKG_NAME").expect("CARGO_PKG_NAME should be set by Cargo");
    let plugin_name = if plugin_variant.is_empty() {
        pkg_name
    } else {
        format!("{}-{}", pkg_name, plugin_variant)
    };
    println!("cargo:rustc-env=PLUGIN_NAME={}", plugin_name);

    // Generate the plugin identifier for gst::plugin_define!
    // GStreamer derives the registration function name from the library filename, not the plugin name
    // For libgstedgeimpulse_banana.dylib, it expects gst_plugin_edgeimpulse_banana_register
    // So we strip "libgst" and ".dylib" from the library name to get the identifier
    let lib_name = if plugin_variant.is_empty() {
        "edgeimpulse".to_string()
    } else {
        // Convert variant to a valid identifier (replace any non-alphanumeric with underscore)
        let variant_ident = plugin_variant
            .chars()
            .map(|c| {
                if c.is_alphanumeric() || c == '_' {
                    c
                } else {
                    '_'
                }
            })
            .collect::<String>();
        format!("edgeimpulse_{}", variant_ident)
    };
    let plugin_identifier = lib_name;

    // Generate type names for GObject types to make them unique per variant
    // This prevents type registration conflicts when multiple variants are loaded
    let type_suffix = if plugin_variant.is_empty() {
        "".to_string()
    } else {
        // Convert variant to a valid identifier for type names (capitalize first letter)
        let variant_capitalized = plugin_variant
            .chars()
            .enumerate()
            .map(|(i, c)| {
                if i == 0 {
                    c.to_uppercase().collect::<String>()
                } else if c.is_alphanumeric() || c == '_' {
                    c.to_string()
                } else {
                    "_".to_string()
                }
            })
            .collect::<String>();
        variant_capitalized
    };

    // Generate the plugin definition code with the correct identifier
    // This allows the registration function name to match what GStreamer expects
    let out_dir = env::var("OUT_DIR").unwrap();
    let plugin_def_path = std::path::Path::new(&out_dir).join("plugin_define.rs");
    let plugin_def_code = format!(
        r#"gst::plugin_define!(
    {},
    env!("CARGO_PKG_DESCRIPTION"),
    plugin_init,
    env!("CARGO_PKG_VERSION"),
    "MIT/X11",
    env!("PLUGIN_NAME"),
    env!("PLUGIN_NAME"),
    "https://github.com/edgeimpulse/gst-plugins-edgeimpulse",
    env!("BUILD_REL_DATE")
);
"#,
        plugin_identifier
    );
    std::fs::write(&plugin_def_path, plugin_def_code)
        .expect("Failed to write plugin definition file");

    // Generate type name constants for each element type
    // This prevents type registration conflicts when multiple variants are loaded
    let type_names_path = std::path::Path::new(&out_dir).join("type_names.rs");
    let type_names_code = format!(
        r#"// Auto-generated type names for variant: {}
#[allow(dead_code)]
pub const VIDEO_INFER_TYPE_NAME: &str = "EdgeImpulseVideoInfer{}";
#[allow(dead_code)]
pub const AUDIO_INFER_TYPE_NAME: &str = "EdgeImpulseAudioInfer{}";
#[allow(dead_code)]
pub const OVERLAY_TYPE_NAME: &str = "EdgeImpulseOverlay{}";
#[allow(dead_code)]
pub const SINK_TYPE_NAME: &str = "GstEdgeImpulseSink{}";
"#,
        plugin_variant, type_suffix, type_suffix, type_suffix, type_suffix
    );
    std::fs::write(&type_names_path, type_names_code).expect("Failed to write type names file");

    if !plugin_variant.is_empty() {
        println!("cargo:warning=PLUGIN_VARIANT is set to: {}", plugin_variant);
        println!("cargo:warning=Plugin name: {}", plugin_name);
        println!("cargo:warning=Plugin identifier: {}", plugin_identifier);
        println!(
            "cargo:warning=After build completes, run: PLUGIN_VARIANT={} ./rename-library.sh",
            plugin_variant
        );
    }

    gst_plugin_version_helper::info()
}
