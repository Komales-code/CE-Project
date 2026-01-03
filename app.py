import streamlit as st
import matplotlib.pyplot as plt

from src.preprocessing import load_and_preprocess
from src.de_algorithm import differential_evolution

st.set_page_config(
    page_title="Used Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)

st.markdown("<h1 style='text-align:center;'>🚗 Used Car Price Prediction</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;'>Differential Evolution + SVR</h4>", unsafe_allow_html=True)

st.sidebar.header("⚙️ Optimization Settings")
pop_size = st.sidebar.slider("Population Size", 10, 30, 15)
generations = st.sidebar.slider("Generations", 10, 40, 20)
F = st.sidebar.slider("Mutation Factor (F)", 0.4, 1.0, 0.8)
CR = st.sidebar.slider("Crossover Rate (CR)", 0.4, 1.0, 0.9)

(X_train, X_test, y_train, y_test), df = load_and_preprocess(
    "data/used_cars.csv"
)

tab1, tab2, tab3 = st.tabs(["📂 Dataset", "⚡ Optimization", "📊 Results"])

with tab1:
    st.dataframe(df.head(10), use_container_width=True)

with tab2:
    if st.button("🚀 Run Optimization"):
        progress = st.progress(0)
        status = st.empty()

        def update_progress(g):
            progress.progress(int(g / generations * 100))

        status.text("Optimizing SVR parameters...")
        best_params, best_rmse, history = differential_evolution(
            X_train, X_test, y_train, y_test,
            pop_size=pop_size,
            generations=generations,
            F=F,
            CR=CR,
            progress_callback=update_progress
        )

        st.session_state["best_params"] = best_params
        st.session_state["best_rmse"] = best_rmse
        st.session_state["history"] = history

        status.success("Optimization Completed ✅")

with tab3:
    if "best_params" in st.session_state:
        c1, c2, c3 = st.columns(3)
        c1.metric("Best RMSE", f"{st.session_state['best_rmse']:.2f}")
        c2.metric("Best C", f"{st.session_state['best_params'][0]:.4f}")
        c3.metric("Best Gamma", f"{st.session_state['best_params'][2]:.5f}")

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(st.session_state["history"], linewidth=2)
        ax.set_xlabel("Generation")
        ax.set_ylabel("RMSE")
        ax.set_title("DE Convergence Curve")
        ax.grid(True)
        st.pyplot(fig)
    else:
        st.info("Run optimization to see results.")
