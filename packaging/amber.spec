# PyInstaller spec for Amber. Build ON the target OS:
#   pip install pyinstaller
#   pyinstaller packaging/amber.spec
# Produces dist/Amber(.exe). See packaging/README.md for caveats.

from PyInstaller.utils.hooks import collect_all, copy_metadata

datas = []
binaries = []
hiddenimports = []

# Streamlit ships data files and needs its metadata at runtime.
for pkg in ("streamlit", "plotly", "pandas", "lightgbm", "sklearn", "pydantic"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

for pkg in ("streamlit", "plotly", "pandas", "numpy", "lightgbm", "scikit-learn", "pydantic"):
    try:
        datas += copy_metadata(pkg)
    except Exception:
        pass

# Bundle the app source, config and dashboard so the frozen launcher can run it.
datas += [
    ("../amber", "amber"),
    ("../config", "config"),
]

block_cipher = None

a = Analysis(
    ["amber_launcher.py"],
    pathex=[".."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Amber",
    console=True,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    name="Amber",
)
