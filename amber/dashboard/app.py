"""Amber dashboard — monitoring and control UI.

Run from the project root:
    streamlit run amber/dashboard/app.py
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from amber.dashboard import control as C
from amber.dashboard import data as D

st.set_page_config(page_title="Amber", page_icon="🟡", layout="wide")

# Visual system: amber accent, theme-agnostic translucent cards, tabular
# numerals for metrics. Works on both Streamlit light and dark base themes.
st.markdown(
    """
    <style>
      :root{
        --amb-accent:#E8A33D; --amb-ok:#2E9E6B; --amb-bad:#D64545; --amb-warn:#D97706;
        --amb-card:rgba(128,128,128,.08); --amb-line:rgba(128,128,128,.22);
      }
      .block-container{padding-top:2.2rem; max-width:1200px;}

      /* status band */
      .amb-band{display:flex; align-items:center; gap:14px; padding:16px 20px;
                border:1px solid var(--amb-line); border-left:6px solid var(--amb-accent);
                border-radius:14px; background:var(--amb-card); margin-bottom:14px;}
      .amb-band.ok{border-left-color:var(--amb-ok);} .amb-band.bad{border-left-color:var(--amb-bad);}
      .amb-dot{width:14px; height:14px; border-radius:50%; flex:none;}
      .amb-band.ok .amb-dot{background:var(--amb-ok); box-shadow:0 0 10px var(--amb-ok);}
      .amb-band.bad .amb-dot{background:var(--amb-bad); box-shadow:0 0 10px var(--amb-bad);}
      .amb-band.none .amb-dot{background:#9aa0a6;}
      .amb-band h1{font-size:19px; font-weight:700; margin:0; letter-spacing:.01em;}
      .amb-band .sub{opacity:.72; font-size:13px; margin-top:2px;}
      .amb-band .when{margin-left:auto; font-size:12px; opacity:.6; font-variant-numeric:tabular-nums;}

      /* KPI cards */
      .amb-kpis{display:grid; grid-template-columns:repeat(5,1fr); gap:10px; margin-bottom:6px;}
      @media (max-width:900px){ .amb-kpis{grid-template-columns:repeat(2,1fr);} }
      .amb-kpi{border:1px solid var(--amb-line); border-radius:12px; background:var(--amb-card);
               padding:12px 14px;}
      .amb-kpi .k{font-size:11px; text-transform:uppercase; letter-spacing:.09em; opacity:.62;}
      .amb-kpi .v{font-size:24px; font-weight:700; margin-top:3px; font-variant-numeric:tabular-nums;}
      .amb-kpi .s{font-size:12px; opacity:.65; margin-top:1px;}
      .amb-kpi.good .v{color:var(--amb-ok);} .amb-kpi.bad .v{color:var(--amb-bad);}
      .amb-kpi.accent .v{color:var(--amb-accent);}

      /* tabs */
      button[data-baseweb="tab"]{font-weight:600;}
      button[data-baseweb="tab"][aria-selected="true"]{color:var(--amb-accent) !important;}
      div[data-baseweb="tab-highlight"]{background-color:var(--amb-accent) !important;}

      /* native metrics -> cards */
      div[data-testid="stMetric"]{border:1px solid var(--amb-line); border-radius:12px;
        background:var(--amb-card); padding:10px 14px;}
      div[data-testid="stMetricValue"]{font-size:24px; font-variant-numeric:tabular-nums;}
      div[data-testid="stMetricLabel"] p{font-size:12px; opacity:.75;}

      .amb-h{display:flex; align-items:baseline; gap:10px; margin:18px 0 8px;}
      .amb-h .t{font-size:17px; font-weight:700;}
      .amb-h .c{font-size:12.5px; opacity:.6;}
      .amb-chip{display:inline-block; padding:2px 10px; border-radius:999px;
                font-size:12px; font-weight:600;}
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


def _kpi(label: str, value: str, sub: str = "", tone: str = "") -> str:
    return (
        f"<div class='amb-kpi {tone}'><div class='k'>{label}</div>"
        f"<div class='v'>{value}</div><div class='s'>{sub}</div></div>"
    )


def _section(title: str, caption: str = "") -> None:
    cap = f"<span class='c'>{caption}</span>" if caption else ""
    st.markdown(f"<div class='amb-h'><span class='t'>{title}</span>{cap}</div>", unsafe_allow_html=True)


root = D.find_project_root()

with st.sidebar:
    st.markdown("## 🟡 Amber")
    st.caption("ML-сканер событий Bybit · pump/dump вероятности")
    st.markdown(f"<span style='font-size:12px;opacity:.7'>Проект: <code>{root}</code></span>", unsafe_allow_html=True)
    if st.button("🔄 Обновить данные", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.caption("Кэш 15 сек — «Обновить» пересчитывает всё сразу.")

state = _load_everything(str(root))
report = state["report"]

# --- Status band -------------------------------------------------------------
now_str = datetime.now(timezone.utc).strftime("%H:%M UTC")
if report is None:
    st.markdown(
        f"<div class='amb-band none'><div class='amb-dot'></div><div>"
        f"<h1>Нет данных для отчёта</h1>"
        f"<div class='sub'>{state['report_error'] or 'запусти сбор и обучение на вкладке «Управление»'}</div>"
        f"</div><div class='when'>{now_str}</div></div>",
        unsafe_allow_html=True,
    )
else:
    ok = bool(report.get("overall_ok"))
    cls, label = ("ok", "Система готова") if ok else ("bad", "Требует внимания")
    reason = report.get("overall_reason", "")
    failed = ", ".join(report.get("readiness_failed_components", []))
    sub = "все проверки пройдены" if ok else f"проблемные компоненты: {failed or reason}"
    st.markdown(
        f"<div class='amb-band {cls}'><div class='amb-dot'></div><div>"
        f"<h1>{label}</h1><div class='sub'>{sub}</div>"
        f"</div><div class='when'>{now_str}</div></div>",
        unsafe_allow_html=True,
    )

# --- KPI strip ---------------------------------------------------------------
checks = (report or {}).get("health", {}).get("checks", {})
raw_age = checks.get("raw_age_sec")
model_status = (report or {}).get("model_eval_status", "missing")
ds = state["dataset"]
ev = (report or {}).get("model_eval", {})

age_txt = _fmt_age(None if raw_age is None else raw_age / 60)
age_tone = "good" if checks.get("raw_fresh") else ("bad" if raw_age is not None else "")
model_txt = {"fresh": "свежая", "stale": "устарела", "missing": "нет"}.get(model_status, model_status)
model_tone = {"fresh": "good", "stale": "bad", "missing": ""}.get(model_status, "")
pr_up = ev.get("model_pr_auc_up_cal")
pr_lift = ev.get("model_pr_auc_up_lift")
pr_sub = f"lift ×{_num(pr_lift, '{:.2f}')}" if pr_lift else "нужно обучение"

st.markdown(
    "<div class='amb-kpis'>"
    + _kpi("Свежесть данных", age_txt, "последняя свеча", age_tone)
    + _kpi("Модель (eval)", model_txt, "статус метрик", model_tone)
    + _kpi("Символов", str(len(state["symbols"])), "в конфиге")
    + _kpi("Датасет", _num(ds.get("rows") if ds else None, "{:,.0f}", "0").replace(",", " "), "строк для обучения")
    + _kpi("PR-AUC↑", _num(pr_up), pr_sub, "accent")
    + "</div>",
    unsafe_allow_html=True,
)

tab_overview, tab_model, tab_data, tab_symbol, tab_control = st.tabs(
    ["📊 Обзор", "🧠 Модель", "🗂 Данные и дрифт", "📈 По символу", "⚙️ Управление"]
)

# --- Overview: live signals --------------------------------------------------
with tab_overview:
    _section("Живые сигналы", "последние 100, свежие сверху")
    signals = state["signals"]
    if not signals:
        st.info("Сигналов пока нет. Обучи модель и запусти сканер на вкладке «Управление».")
    else:
        rows = []
        for s in signals:
            rows.append(
                {
                    "время": str(s.get("event_ts", ""))[:19].replace("T", " "),
                    "символ": s.get("symbol", ""),
                    "направление": "🟢 pump" if D.signal_direction(s) == "pump" else "🔴 dump",
                    "P(pump)": float(s.get("prob_up_calibrated", 0) or 0),
                    "P(dump)": float(s.get("prob_down_calibrated", 0) or 0),
                    "spread": round(float(s.get("market_context", {}).get("spread_bps", 0) or 0), 2),
                    "драйверы": D.signal_top_drivers(s),
                }
            )
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
            column_config={
                "P(pump)": st.column_config.ProgressColumn("P(pump)", min_value=0.0, max_value=1.0, format="%.3f"),
                "P(dump)": st.column_config.ProgressColumn("P(dump)", min_value=0.0, max_value=1.0, format="%.3f"),
                "spread": st.column_config.NumberColumn("spread, bps"),
            },
        )

# --- Model quality -----------------------------------------------------------
with tab_model:
    model = state["model"]
    if model is None:
        st.info("Модель ещё не обучена — вкладка «Управление» → «Обучение».")
    else:
        cv = model.get("cv", {})
        labeling = model.get("labeling", {})
        _section(
            "Модель",
            f"сплит {model.get('split_mode', '?')} · горизонт {labeling.get('horizon_steps', '?')} свечей · "
            f"target ≈ {_num(labeling.get('avg_up_pct'), '{:.3%}')}",
        )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Тип", model.get("model_type", "—"))
        m2.metric("Строк обучения", _num(model.get("train_rows"), "{:.0f}"))
        m3.metric("CV Brier", _num(cv.get("brier_mean")))
        m4.metric("CV AUC", _num(cv.get("auc_mean")))

    if report is not None:
        q = report.get("quality", {})
        bt = report.get("backtest", {})
        _section("Качество на реальных исходах", "только подтверждённые события")
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Rolling AUC", _num(q.get("rolling_auc")))
        q2.metric("Подтв. исходов", _num(q.get("auc_confirmed_outcomes"), "{:.0f}", "0"))
        q3.metric("Bias (pump−dump)", _num(q.get("prediction_bias")))
        q4.metric("PSI дрифт", str(q.get("psi", {}).get("level", "—")))

        _section("Out-of-sample метрики", "test-сегмент; PR-AUC — главная метрика для редких событий")
        e1, e2, e3, e4 = st.columns(4)
        e1.metric(
            "PR-AUC↑",
            _num(ev.get("model_pr_auc_up_cal")),
            delta=(f"lift ×{_num(ev.get('model_pr_auc_up_lift'), '{:.2f}')}" if ev.get("model_pr_auc_up_lift") else None),
        )
        e2.metric(
            "PR-AUC↓",
            _num(ev.get("model_pr_auc_down_cal")),
            delta=(f"lift ×{_num(ev.get('model_pr_auc_down_lift'), '{:.2f}')}" if ev.get("model_pr_auc_down_lift") else None),
        )
        e3.metric("Precision↑@thr", _num(ev.get("model_precision_up_at_threshold")))
        e4.metric("Brier↑", _num(ev.get("model_brier_up_cal")))

        _section("Бэктест", "promotion gate · вход с лагом 1 бар · test-сегмент")
        if bt.get("error"):
            st.warning(f"Бэктест недоступен: {bt['error']}")
        else:
            b1, b2, b3, b4, b5 = st.columns(5)
            b1.metric("Сделок", _num(bt.get("signals"), "{:.0f}", "0"))
            b2.metric("Win rate", _num(bt.get("win_rate")))
            b3.metric("Sharpe", _num(bt.get("sharpe")))
            b4.metric("Profit factor", _num(bt.get("profit_factor"), "{:.2f}"))
            b5.metric("Max drawdown", _num(bt.get("max_drawdown"), "{:.4f}"))
            st.caption(
                f"режим: {bt.get('mode', '—')} · сегмент: {bt.get('segment', '—')} · "
                f"лаг входа: {bt.get('entry_lag_bars', 0)} бар"
            )

# --- Data & drift ------------------------------------------------------------
with tab_data:
    _section("Собранные данные", "нормализованные свечи по символам")
    cs = state["candle_stats"]
    if not any(r["candles"] for r in cs):
        st.info("Нормализованных свечей пока нет. Вкладка «Управление» → запусти WS-коллектор и «Нормализацию».")
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

    _section("Дрифт признаков", "PSI против обучающей выборки · low < 0.1 · medium 0.1–0.2 · high > 0.2")
    drift = state["drift"]
    ddf = pd.DataFrame(drift)
    if not ddf.empty:
        level_icon = {"low": "🟢 low", "medium": "🟡 medium", "high": "🔴 high"}
        ddf["level"] = ddf["level"].map(lambda v: level_icon.get(v, v))
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
                        increasing_line_color="#2E9E6B",
                        decreasing_line_color="#D64545",
                        name=symbol,
                    )
                ]
            )
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
                        marker={"size": 12, "symbol": "triangle-up", "color": "#E8A33D",
                                "line": {"width": 1, "color": "#00000055"}},
                        name="сигналы",
                        text=sig_txt,
                    )
                )
            fig.update_layout(
                height=520,
                xaxis_rangeslider_visible=False,
                margin={"t": 24, "b": 10, "l": 10, "r": 10},
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(128,128,128,0.05)",
                legend={"orientation": "h", "y": 1.06},
            )
            st.plotly_chart(fig, use_container_width=True)

# --- Control panel -----------------------------------------------------------
with tab_control:
    from pathlib import Path

    pm = C.ProcessManager(Path(state["storage"]["state_dir"]) / "procs", root)

    _section("Символы для сбора", "по одному на строку; сохранение делает бэкап конфига")
    current = "\n".join(state["symbols"])
    edited = st.text_area("Символы", value=current, height=140, label_visibility="collapsed")
    if st.button("💾 Сохранить символы"):
        try:
            saved = C.set_symbols(root, edited.splitlines())
            st.success(f"Сохранено {len(saved)} символов (бэкап: config/amber.yaml.bak). Перезапусти коллектор.")
            st.cache_data.clear()
        except Exception as exc:
            st.error(f"Не удалось сохранить: {exc}")

    _section("Живые процессы", "работают в фоне и переживают перезапуск дашборда")
    for name, svc in C.SERVICES.items():
        stt = pm.status(name)
        col_lbl, col_state, col_btn = st.columns([2, 2, 1])
        col_lbl.markdown(f"**{svc['label']}**")
        if stt["running"]:
            up = stt["uptime_sec"] or 0
            col_state.markdown(
                f"<span class='amb-chip' style='background:rgba(46,158,107,.15);color:#2E9E6B'>"
                f"● работает · PID {stt['pid']} · {up/60:.0f} мин</span>",
                unsafe_allow_html=True,
            )
            if col_btn.button("⏹ Стоп", key=f"stop_{name}"):
                pm.stop(name)
                st.rerun()
        else:
            col_state.markdown(
                "<span class='amb-chip' style='background:rgba(128,128,128,.15);opacity:.8'>○ остановлен</span>",
                unsafe_allow_html=True,
            )
            if col_btn.button("▶ Старт", key=f"start_{name}"):
                pm.start(name, svc["argv"])
                st.rerun()
        log = pm.tail_log(name, lines=15)
        if log:
            with st.expander(f"Лог: {svc['label']}"):
                st.code(log)

    _section("Конвейер по шагам", "backfill/normalize → features → dataset → train → backtest")
    cols = st.columns(3)
    for i, step in enumerate(C.PIPELINE_STEPS):
        if cols[i % 3].button(step["label"], key=f"step_{step['key']}", use_container_width=True):
            with st.spinner(f"{step['label']}…"):
                rc, out = pm.run_once(step["argv"])
            st.session_state["last_step_out"] = (step["label"], rc, out)
            st.cache_data.clear()

    st.divider()
    if st.button("🚀 Полный цикл: backfill → normalize → features → dataset → train → backtest", type="primary"):
        results = []
        progress = st.progress(0.0)
        for i, step in enumerate(C.FULL_CYCLE):
            with st.spinner(f"{step['label']}…"):
                rc, out = pm.run_once(step["argv"])
            results.append((step["label"], rc, out))
            progress.progress((i + 1) / len(C.FULL_CYCLE))
            if rc == C.NOT_ENOUGH_DATA_CODE:
                st.warning(f"Пока недостаточно данных для обучения — {step['label']} отложен. Детали ниже.")
                break
            if rc != 0:
                st.error(f"Шаг «{step['label']}» завершился с ошибкой — цикл остановлен. Детали ниже.")
                break
        st.session_state["full_cycle_out"] = results
        st.cache_data.clear()

    if "last_step_out" in st.session_state:
        label, rc, out = st.session_state["last_step_out"]
        (st.success if rc == 0 else st.error)(f"{label}: код возврата {rc}")
        with st.expander("Вывод последнего шага", expanded=True):
            st.code(out[-4000:] or "(пусто)")

    if "full_cycle_out" in st.session_state:
        with st.expander("Вывод полного цикла", expanded=True):
            for label, rc, out in st.session_state["full_cycle_out"]:
                st.markdown(f"**{label}** — код {rc}")
                st.code(out[-1500:] or "(пусто)")
