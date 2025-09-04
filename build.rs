fn main() {
    // Set PKG_CONFIG_PATH to include all necessary directories for macOS
    if cfg!(target_os = "macos") {
        let homebrew_prefix = "/opt/homebrew";
        let pkg_config_paths = vec![
            format!("{}/opt/libxml2/lib/pkgconfig", homebrew_prefix),
            format!("{}/lib/pkgconfig", homebrew_prefix),
            format!("{}/share/pkgconfig", homebrew_prefix),
        ];

        let existing_path = std::env::var("PKG_CONFIG_PATH").unwrap_or_default();
        let new_path = if existing_path.is_empty() {
            pkg_config_paths.join(":")
        } else {
            format!("{}:{}", pkg_config_paths.join(":"), existing_path)
        };

        println!("cargo:warning=Setting PKG_CONFIG_PATH to: {}", new_path);
        std::env::set_var("PKG_CONFIG_PATH", &new_path);
        println!("cargo:rerun-if-env-changed=PKG_CONFIG_PATH");
    }

    gst_plugin_version_helper::info()
}
