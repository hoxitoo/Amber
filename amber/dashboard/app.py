"""Amber dashboard — a read-only monitoring UI.

Run from the project root:
    streamlit run amber/dashboard/app.py
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from amber.dashboard import data as D

st.set_page_config(page_title="Amber", page_icon="🟡", layout="wide")

st.markdown(
    """
    <style>
      .amber-banner {padding:18px 22px;border-radius:12px;color:#fff;font-weight:700;
                     font-size:20px;margin-bottom:6px;}
      .amber-sub {opacity:.85;font-weight:500;font-size:14px;}
      div[data-testid="stMetricValue"] {font-size:26px;}
      .pill {padding:2px 10px;border-radius:999px;font-weight:600;font-size:12px;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=15)
def _load_everything(root_str: str) -> dict:
    from pathlib import Path

    root = Path(root_str)
    config = D.load_config(root)
    storage = D.storage_paths(config, root)
    monitoring = config.get("monitoring", {})
    report, err = D.safe_system_report(
        storage,
        model_eval_fresh_sec=int(monitoring.get("model_eval_fresh_sec", 21600)),
        require_model_eval=bool(monitoring.get("require_model_eval_for_overall_ok", True)),
    )
    symbols = config.get("exchange", {}).get("bybit", {}).get("symbols", [])
    model = D.load_latest_model(storage["models_dir"])
    return {
        "config": config,
        "storage": storage,
        "symbols": symbols,
        "report": report,
        "report_error": err,
        "model": model,
        "dataset": D.dataset_info(storage["datasets_dir"]),
        "signals": D.load_signals(storage["logs_dir"], limit=100),
        "candle_stats": D.candle_stats(storage["raw_dir"], symbols),
        "drift": D.drift_report(storage["features_dir"], model, symbols),
    }


def _fmt_age(minutes: float | None) -> str:
    if minutes is None:
        return "—"
    if minutes < 90:
        return f"{minutes:.0f} мин"
    if minutes < 60 * 48:
        return f"{minutes / 60:.1f} ч"
    return f"{minutes / 1440:.1f} дн"


def _num(value, fmt="{:.3f}", dash="—"):
    try:
        if value is None:
            return dash
        return fmt.format(float(value))
    except (TypeError, ValueError):
        return dash


root = D.find_project_root()

with st.sidebar:
    st.title("🟡 Amber")
    st.caption("ML-сканер событий Bybit")
    st.write(f"**Проект:** `{root}`")
    if st.button("🔄 Обновить", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.caption("Данные кэшируются на 15 сек. Нажми «Обновить» для пересчёта.")

state = _load_everything(str(root))
report = state["report"]

# --- Top status banner -------------------------------------------------------
if report is None:
    st.markdown(
        f"<div class='amber-banner' style='background:#6b7280'>НЕТ ДАННЫХ ДЛЯ ОТЧЁТА"
        f"<div class='amber-sub'>{state['report_error'] or 'запусти сбор и обучение'}</div></div>",
        unsafe_allow_html=True,
    )
else:
    ok = bool(report.get("overall_ok"))
    color = "#16a34a" if ok else "#dc2626"
    label = "СИСТЕМА ГОТОВА" if ok else "ТРЕБУЕТ ВНИМАНИЯ"
    reason = report.get("overall_reason", "")
    st.markdown(
        f"<div class='amber-banner' style='background:{color}'>{label}"
        f"<div class='amber-sub'>причина: {reason}</div></div>",
        unsafe_allow_html=True,
    )

# --- Tiles -------------------------------------------------------------------
checks = (report or {}).get("health", {}).get("checks", {})
raw_age = checks.get("raw_age_sec")
model_status = (report or {}).get("model_eval_status", "missing")
ds = state["dataset"]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Свежесть данных", _fmt_age(None if raw_age is None else raw_age / 60))
c2.metric("Модель (eval)", {"fresh": "свежая", "stale": "устарела", "missing": "нет"}.get(model_status, model_status))
c3.metric("Символов", len(state["symbols"]))
c4.metric("Строк в датасете", _num(ds.get("rows") if ds else None, "{:.0f}", "0"))
c5.metric("Сигналов (лог)", len(state["signals"]))

tab_overview, tab_model, tab_data, tab_symbol = st.tabs(
    ["📊 Обзор", "🧠 Модель", "🗂 Данные и дрифт", "📈 По символу"]
)

# --- Overview: live signals --------------------------------------------------
with tab_overview:
    st.subheader("Живые сигналы")
    signals = state["signals"]
    if not signals:
        st.info("Сигналов пока нет. Запусти `python scripts/run_scanner.py` после обучения модели.")
    else:
        rows = []
        for s in signals:
            rows.append(
                {
                    "время": str(s.get("event_ts", ""))[:19].replace("T", " "),
                    "символ": s.get("symbol", ""),
                    "направление": "🟢 pump" if D.signal_direction(s) == "pump" else "🔴 dump",
                    "P(pump)": round(float(s.get("prob_up_calibrated", 0) or 0), 3),
                    "P(dump)": round(float(s.get("prob_down_calibrated", 0) or 0), 3),
                    "spread_bps": round(float(s.get("market_context", {}).get("spread_bps", 0) or 0), 2),
                    "драйверы": D.signal_top_drivers(s),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# --- Model quality -----------------------------------------------------------
with tab_model:
    model = state["model"]
    if model is None:
        st.info("Модель ещё не обучена. Запусти `python scripts/train_model.py`.")
    else:
        st.subheader("Модель")
        cv = model.get("cv", {})
        labeling = model.get("labeling", {})
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Тип", model.get("model_type", "—"))
        m2.metric("Строк обучения", _num(model.get("train_rows"), "{:.0f}"))
        m3.metric("CV Brier", _num(cv.get("brier_mean")))
        m4.metric("CV AUC", _num(cv.get("auc_mean")))
        st.caption(
            f"Сплит: {model.get('split_mode', '?')} · "
            f"горизонт {labeling.get('horizon_steps', '?')} свечей · "
            f"target ≈ {_num(labeling.get('avg_up_pct'), '{:.3%}')}"
        )

    if report is not None:
        st.subheader("Качество на реальных исходах")
        q = report.get("quality", {})
        ev = report.get("model_eval", {})
        bt = report.get("backtest", {})
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Rolling AUC (подтв.)", _num(q.get("rolling_auc")))
        q2.metric("Подтв. исходов", _num(q.get("auc_confirmed_outcomes"), "{:.0f}", "0"))
        q3.metric("Bias (pump−dump)", _num(q.get("prediction_bias")))
        q4.metric("PSI дрифт", str(q.get("psi", {}).get("level", "—")))
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Precision↑@thr", _num(ev.get("model_precision_up_at_threshold")))
        e2.metric("Precision↓@thr", _num(ev.get("model_precision_down_at_threshold")))
        e3.metric("Brier↑", _num(ev.get("model_brier_up_cal")))
        e4.metric("Brier↓", _num(ev.get("model_brier_down_cal")))

        st.subheader("Бэктест (promotion gate)")
        if bt.get("error"):
            st.warning(f"Бэктест недоступен: {bt['error']}")
        else:
            b1, b2, b3, b4, b5 = st.columns(5)
            b1.metric("Сделок", _num(bt.get("signals"), "{:.0f}", "0"))
            b2.metric("Win rate", _num(bt.get("win_rate")))
            b3.metric("Sharpe", _num(bt.get("sharpe")))
            b4.metric("Profit factor", _num(bt.get("profit_factor"), "{:.2f}"))
            b5.metric("Max drawdown", _num(bt.get("max_drawdown"), "{:.4f}"))
            st.caption(f"режим: {bt.get('mode', '—')} · сегмент: {bt.get('segment', '—')}")

# --- Data & drift ------------------------------------------------------------
with tab_data:
    st.subheader("Собранные данные")
    cs = state["candle_stats"]
    if not any(r["candles"] for r in cs):
        st.info("Нормализованных свечей пока нет. Запусти сбор: `run_ws_collector.py` → `run_normalize.py`.")
    else:
        df = pd.DataFrame(cs)
        df["synthetic_pct"] = df["synthetic_pct"].map(lambda v: f"{v:.1f}%")
        df["last_update_min"] = df["last_update_min"].map(_fmt_age)
        df = df.rename(
            columns={
                "symbol": "символ",
                "candles": "свечей",
                "synthetic_pct": "синтетика",
                "last_update_min": "обновлено",
            }
        )
        st.dataframe(df, use_container_width=True, hide_index=True)

    st.subheader("Дрифт признаков (PSI vs обучающая выборка)")
    drift = state["drift"]
    ddf = pd.DataFrame(drift)
    if not ddf.empty:
        ddf["max_psi"] = ddf["max_psi"].map(lambda v: _num(v, "{:.3f}"))
        ddf = ddf.rename(
            columns={
                "symbol": "символ",
                "level": "уровень",
                "max_psi": "max PSI",
                "drifting_features": "поплывшие фичи",
                "reference": "эталон",
            }
        )
        st.dataframe(ddf, use_container_width=True, hide_index=True)
        st.caption("Уровни PSI: low < 0.1 · medium 0.1–0.2 · high > 0.2 (модель, вероятно, устарела).")

# --- Per-symbol chart --------------------------------------------------------
with tab_symbol:
    symbols = state["symbols"]
    if not symbols:
        st.info("В config/amber.yaml не заданы символы.")
    else:
        symbol = st.selectbox("Символ", symbols)
        candles = D.load_candles(state["storage"]["raw_dir"], symbol, limit=500)
        if not candles:
            st.info(f"Нет нормализованных свечей для {symbol}.")
        else:
            import plotly.graph_objects as go

            def _dt(ms: int) -> datetime:
                return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)

            xs = [_dt(int(c.get("ts", 0) or 0)) for c in candles]
            fig = go.Figure(
                data=[
                    go.Candlestick(
                        x=xs,
                        open=[c.get("open") for c in candles],
                        high=[c.get("high") for c in candles],
                        low=[c.get("low") for c in candles],
                        close=[c.get("close") for c in candles],
                        name=symbol,
                    )
                ]
            )
            # overlay signals for this symbol
            sig_x, sig_y, sig_txt = [], [], []
            for s in state["signals"]:
                if s.get("symbol") != symbol:
                    continue
                try:
                    ts = datetime.fromisoformat(str(s.get("event_ts")).replace("Z", "+00:00"))
                except ValueError:
                    continue
                sig_x.append(ts)
                sig_y.append(float(s.get("market_context", {}).get("mid_price", 0) or 0))
                sig_txt.append(D.signal_direction(s))
            if sig_x:
                fig.add_trace(
                    go.Scatter(
                        x=sig_x,
                        y=sig_y,
                        mode="markers",
                        marker={"size": 11, "symbol": "triangle-up", "color": "#f59e0b"},
                        name="сигналы",
                        text=sig_txt,
                    )
                )
            fig.update_layout(height=520, xaxis_rangeslider_visible=False, margin={"t": 30, "b": 10})
            st.plotly_chart(fig, use_container_width=True)
