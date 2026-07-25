"""Every second counts — a live QAOA demo that splits simultaneous emergency calls
between two ambulance crews on a real map, and shows the quantum optimization
converge in real time."""
import os
import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pydeck as pdk
import streamlit as st
from dotenv import load_dotenv
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_bloch_multivector

from src.classical_solver import greedy_solve
from src.graph_problem import PLEASANTON_CA, build_graph_from_points, generate_calls
from src.impact import evaluate_split, survival_odds_gain
from src.llm_intake import intake_calls
from src.qaoa_solver import brute_force_optimum, cut_value, run_qaoa

load_dotenv()

DEMO_CACHE_PATH = Path(__file__).resolve().parent.parent / "demo_cache.pkl"

CREW_COLOR = {0: [226, 83, 75], 1: [63, 138, 224]}
CREW_NAME = {0: "Crew A", 1: "Crew B"}


def play_collapse_animation(result, placeholder, frame_delay: float) -> None:
    n = len(result.partition)
    all_bits = [format(x, f"0{n}b") for x in range(2**n)]
    top_bits = sorted(all_bits, key=lambda b: -result.prob_snapshots[-1].get(b, 0.0))[:12]

    for snap in result.prob_snapshots:
        fig, ax = plt.subplots(figsize=(5, 3))
        fig.patch.set_facecolor("#0b1120")
        ax.set_facecolor("#0b1120")
        probs = [snap.get(b, 0.0) for b in top_bits]
        best_bit = max(snap, key=snap.get)
        bar_colors = ["#7f77dd" if b == best_bit else "#3a4258" for b in top_bits]
        ax.bar(range(len(top_bits)), probs, color=bar_colors)
        ax.set_xticks([])
        ax.set_ylim(0, max(0.35, max(snap.values()) * 1.1))
        ax.tick_params(colors="#7d8aa3")
        for spine in ax.spines.values():
            spine.set_color("#232d45")
        ax.set_ylabel("probability", color="#7d8aa3")
        placeholder.pyplot(fig)
        plt.close(fig)
        if frame_delay:
            time.sleep(frame_delay)

st.set_page_config(page_title="Every second counts", layout="wide")

st.title("Every second counts")
st.caption("QAOA dispatch optimizer, running live on a Qiskit Aer simulator")

with st.sidebar:
    st.header("AI call intake")
    call_text = st.text_area(
        "Describe simultaneous calls, one per line",
        placeholder="Elderly woman, chest pain, at Stoneridge Mall Pleasanton CA.\nTwo-car collision on Bernal Ave, Pleasanton CA.",
        height=100,
    )
    parse_clicked = st.button("Parse with AI")
    if parse_clicked:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            st.error("No GROQ_API_KEY found in .env")
        elif not call_text.strip():
            st.error("Describe at least one call first.")
        else:
            with st.spinner("LLM extracting calls, then geocoding each one..."):
                try:
                    points = intake_calls(call_text, api_key)
                except Exception as exc:
                    points = []
                    st.error(f"Intake failed: {exc}")
            if points:
                st.session_state["ai_points"] = points
                st.success(f"Parsed and geocoded {len(points)} call(s).")
            elif call_text.strip():
                st.warning("Could not geocode any calls — try more specific addresses.")

    if ("ai_points" in st.session_state or "cached_graph" in st.session_state) and st.button(
        "Clear, use random scenario"
    ):
        st.session_state.pop("ai_points", None)
        st.session_state.pop("cached_graph", None)
        st.session_state.pop("result", None)

    st.header("Scenario")
    locked = "ai_points" in st.session_state or "cached_graph" in st.session_state
    n_calls = st.slider("Simultaneous calls", 6, 10, 8, disabled=locked)
    seed = st.number_input("Map seed", value=1, step=1, disabled=locked)
    st.header("QAOA settings")
    reps = st.slider("Circuit depth (p)", 1, 3, 2)
    maxiter = st.slider("Optimizer iterations", 30, 150, 80, step=10)
    run_clicked = st.button("Run quantum optimization", type="primary")

    st.header("Pitch mode")
    st.caption("Pre-bake a run once, then replay it instantly with zero live API or compute risk.")
    load_demo_clicked = st.button("Load pitch demo (instant)", disabled=not DEMO_CACHE_PATH.exists())
    if not DEMO_CACHE_PATH.exists():
        st.caption("No saved demo yet — run a scenario below, then save it.")

if load_demo_clicked:
    with open(DEMO_CACHE_PATH, "rb") as f:
        cached = pickle.load(f)
    st.session_state["cached_graph"] = cached["graph"]
    st.session_state["result"] = cached["result"]
    st.session_state["graph_key"] = "cached"
    st.session_state.pop("ai_points", None)

if "cached_graph" in st.session_state:
    graph = st.session_state["cached_graph"]
    graph_key = "cached"
elif "ai_points" in st.session_state:
    ai_points = st.session_state["ai_points"]
    graph = build_graph_from_points(ai_points)
    graph_key = ("ai", tuple(p["label"] for p in ai_points))
else:
    graph = generate_calls(n_calls=n_calls, seed=int(seed))
    graph_key = ("random", n_calls, seed)

map_col, chart_col = st.columns([1.1, 1])

with map_col:
    st.subheader("Active calls — Pleasanton, CA")
    if "result" in st.session_state and st.session_state.get("graph_key") == graph_key:
        partition = st.session_state["result"].partition
        colors = [CREW_COLOR[partition[n]] for n in graph.nodes]
    else:
        colors = [[150, 150, 150] for _ in graph.nodes]

    call_data = [
        {"lat": d["lat"], "lon": d["lon"], "color": colors[i]} for i, d in enumerate(dict(graph.nodes(data=True)).values())
    ]
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=call_data,
        get_position="[lon, lat]",
        get_fill_color="color",
        get_radius=90,
        pickable=True,
    )
    view = pdk.ViewState(latitude=PLEASANTON_CA[0], longitude=PLEASANTON_CA[1], zoom=12)
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, map_style="dark"))

with chart_col:
    st.subheader("Superposition collapsing to an answer")
    chart_placeholder = st.empty()
    if "result" not in st.session_state:
        chart_placeholder.info("Run the optimizer to watch this fill in.")

if run_clicked:
    with st.spinner("Building the QAOA circuit and optimizing on the simulator..."):
        result = run_qaoa(graph, reps=reps, maxiter=maxiter, seed=42)
    st.session_state["result"] = result
    st.session_state["graph_key"] = graph_key
    play_collapse_animation(result, chart_placeholder, frame_delay=0.35)
    st.rerun()

if load_demo_clicked:
    play_collapse_animation(st.session_state["result"], chart_placeholder, frame_delay=0.05)
    st.rerun()

if "result" in st.session_state and st.session_state.get("graph_key") == graph_key:
    result = st.session_state["result"]

    st.subheader(f"Quantum circuit — QAOA, p={result.reps}")
    fig = result.ansatz.decompose(reps=3).draw(output="mpl", style="iqp", fold=18)
    fig.set_size_inches(fig.get_size_inches() * 1.3)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Qubit states — each call, undecided (sphere) to locked-in (arrow)")
    sv = Statevector.from_instruction(result.ansatz.assign_parameters(result.optimal_params).remove_final_measurements(inplace=False))
    fig = plot_bloch_multivector(sv)
    fig.set_facecolor("#0b1120")
    fig.set_size_inches(fig.get_size_inches() * 1.8)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Classical vs quantum")
    _, greedy_val = greedy_solve(graph)
    greedy_bits, _ = greedy_solve(graph)
    greedy_partition = {n: int(greedy_bits[n]) for n in graph.nodes}

    qaoa_impact = evaluate_split(graph, result.partition)
    greedy_impact = evaluate_split(graph, greedy_partition)
    minutes_saved = round(greedy_impact.time_to_clear_minutes - qaoa_impact.time_to_clear_minutes, 2)
    odds_gain = survival_odds_gain(minutes_saved)

    _, optimal_cut = brute_force_optimum(graph) if graph.number_of_nodes() <= 12 else (None, None)

    c1, c2, c3 = st.columns(3)
    c1.metric("Classical greedy", f"{greedy_impact.time_to_clear_minutes} min")
    c2.metric("QAOA result", f"{qaoa_impact.time_to_clear_minutes} min", delta=f"{-minutes_saved:.2f} min")
    c3.metric(
        "Modeled survival gain",
        f"+{odds_gain}%",
        help=(
            "Illustrative estimate: minutes saved x ~8%/min, the middle of AHA's studied "
            "cardiac-arrest response-time range, capped at 25% so this never overstates "
            "what that research actually supports."
        ),
    )

    if optimal_cut is not None:
        st.caption(
            f"Cut value — QAOA: {result.best_cut_value:.2f} · greedy: {greedy_val:.2f} · "
            f"true optimum (brute force): {optimal_cut:.2f}"
        )

    if graph_key != "cached" and st.button("Save this run as pitch demo"):
        with open(DEMO_CACHE_PATH, "wb") as f:
            pickle.dump({"graph": graph, "result": result}, f)
        st.success("Saved. Use \"Load pitch demo (instant)\" in the sidebar any time before you go on stage.")
