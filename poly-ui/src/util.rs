use std::path::{Path, PathBuf};

static RESOURCES_DIR: &str = "data";

fn resources_path() -> PathBuf {
	if std::env::var_os("CARGO").is_some() {
		PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap())
			.parent()
			.unwrap()
			.join("poly-ui")
			.join(RESOURCES_DIR)
	} else {
		#[cfg(target_os = "macos")]
		{
			let bundle = core_foundation::bundle::CFBundle::main_bundle();
			let bundle_path = bundle.path().unwrap();
			let resources_path = bundle.resources_path().unwrap();
			bundle_path.join(resources_path).join(RESOURCES_DIR)
		}
		#[cfg(not(any(target_os = "macos")))]
		PathBuf::from(RESOURCES_DIR)
	}
}

/// Return the path to the resource with the specified file name
/// The resource must be in RESOURCES_DIR relative to the crate root
pub fn resource_path(resource: &str) -> PathBuf {
	let rp = resources_path();
	let path = Path::new(&rp);
	let p = path.join(resource);
	tracing::debug!("PATH={p:?}");
	p
}
