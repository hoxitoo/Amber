"""Launch the Amber dashboard.

Usage:
    python scripts/run_dashboard.py
(equivalent to: streamlit run amber/dashboard/app.py)
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    try:
        from streamlit.web import cli as stcli
    except Exception:  # pragma: no cover
        sys.exit(
            "streamlit is not installed. Run: pip install -r requirements-dashboard.txt"
        )
    app = ROOT / "amber" / "dashboard" / "app.py"
    sys.argv = ["streamlit", "run", str(app)]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
