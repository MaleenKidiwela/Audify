use std::net::TcpStream;
use std::process::{Child, Command};
use std::sync::Mutex;
use std::time::Duration;

use tauri::{Manager, RunEvent};

const ENGINE_ADDR: &str = "127.0.0.1:8765";

struct Engine(Mutex<Option<Child>>);

fn reclaim_port() {
    // previous instances / dev engines may still hold the port
    let _ = Command::new("pkill")
        .args(["-f", "spike/streaming_server.py"])
        .status();
    let _ = Command::new("pkill").args(["-x", "audify-engine"]).status();
    std::thread::sleep(Duration::from_millis(500));
}

/// Dev builds run the engine from the repo venv; release builds run the
/// PyInstaller-frozen engine bundled inside the .app resources.
fn spawn_engine(app: &tauri::App) -> Result<Child, Box<dyn std::error::Error>> {
    reclaim_port();

    #[cfg(debug_assertions)]
    {
        let _ = app; // unused in dev
        let repo_root = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
        Ok(Command::new(format!("{repo_root}/.venv/bin/python"))
            .arg(format!("{repo_root}/spike/streaming_server.py"))
            .spawn()?)
    }

    #[cfg(not(debug_assertions))]
    {
        use std::os::unix::fs::PermissionsExt;
        let bin = app
            .path()
            .resource_dir()?
            .join("engine")
            .join("audify-engine");
        // the bundler may not preserve the executable bit
        if let Ok(meta) = std::fs::metadata(&bin) {
            let mut perms = meta.permissions();
            if perms.mode() & 0o111 == 0 {
                perms.set_mode(0o755);
                let _ = std::fs::set_permissions(&bin, perms);
            }
        }
        Ok(Command::new(bin).spawn()?)
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            let child = spawn_engine(app)?;
            app.manage(Engine(Mutex::new(Some(child))));

            // The static splash is served from tauri://localhost, a secure
            // context, and WKWebView refuses mixed-content fetches to
            // http://127.0.0.1 -- so once the engine port accepts
            // connections we navigate the webview to the engine's own UI,
            // making everything same-origin http.
            let handle = app.handle().clone();
            std::thread::spawn(move || {
                for _ in 0..240 {
                    if TcpStream::connect(ENGINE_ADDR).is_ok() {
                        if let Some(win) = handle.get_webview_window("main") {
                            let _ = win.eval(&format!(
                                "location.replace('http://{ENGINE_ADDR}/')"
                            ));
                        }
                        return;
                    }
                    std::thread::sleep(Duration::from_millis(500));
                }
            });
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app, event| {
            if let RunEvent::Exit = event {
                if let Some(engine) = app.try_state::<Engine>() {
                    if let Some(mut child) = engine.0.lock().unwrap().take() {
                        let _ = child.kill();
                    }
                }
            }
        });
}
