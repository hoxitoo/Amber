# Packaging Amber

Two ways to get a "just launch it" experience. The launcher is recommended.

## Option A — one-click launcher (recommended, reliable)

No build step. Ship the repo folder and double-click:

- **Windows:** `Amber.bat`
- **macOS:** `launch.command` (first time: right-click → Open to bypass Gatekeeper)
- **Linux:** `./launch.command`

On first run it creates a `.venv`, installs dependencies, and opens the dashboard
at `http://localhost:8501`. Later runs just launch it. This is the safest path and
gives the same "double-click to open" result as an exe.

## Option B — standalone executable (advanced, build on your OS)

A true single distributable via PyInstaller. **Must be built on the target OS**
(a Windows `.exe` builds on Windows, a macOS app on macOS). Bundling Streamlit is
finicky and the spec below may need tweaks for your environment — the launcher
above avoids all of this.

```bash
pip install -r requirements.txt -r requirements-dashboard.txt pyinstaller
pyinstaller packaging/amber.spec
# result: dist/Amber/Amber(.exe)  — run it, the dashboard opens in the browser
```

Notes / known gotchas:
- Build on the same OS/arch you want to ship to. There is no reliable
  Linux→Windows cross-build.
- If Streamlit fails to find static assets at runtime, ensure `collect_all`
  picked up `streamlit` data files (it should via the spec).
- The bundle writes `data/` next to where it runs; launch it from a folder you
  can write to.
- For a true installer (`.msi`), wrap the PyInstaller `dist/Amber` folder with a
  tool like Inno Setup (Windows) — that step is environment-specific and left to
  the user.

## Why not an exe by default?

Streamlit runs its own local web server and loads many data files; PyInstaller
single-file builds of it are fragile and hard to verify across machines. The
launcher scripts deliver the same one-click experience without that risk.
