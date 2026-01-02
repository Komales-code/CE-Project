import streamlit as st
import matplotlib.pyplot as plt

from src.preprocessing import load_and_preprocess
from src.de_algorithm import differential_evolution

st.set_page_config(
    page_title="Used Car Price Prediction (DE + SVR)",
    layout="wide"
)

st.title("🚗 Used Car Price Prediction using Differential Evolution")
st.markdown("**JIE42903 – Evolutionary Computing Project**")

st.sidebar.header("⚙️ Differential Evolution Parameters")

pop_size = st.sidebar.slider("Population Size", 10, 60, 30)
generations = st.sidebar.slider("Generations", 10, 100, 50)
F = st.sidebar.slider("Mutation Factor (F)", 0.1, 1.0, 0.8)
CR = st.sidebar.slider("Crossover Rate (CR)", 0.1, 1.0, 0.9)

(X_train, X_test, y_train, y_test), df = load_and_preprocess(
    "data/used_cars.csv"
)

st.subheader("📂 Dataset Preview")
st.dataframe(df.head())

if st.button("🚀 Run Differential Evolution"):
    with st.spinner("Optimizing SVR parameters..."):
        best_params, best_rmse, history = differential_evolution(
            X_train, X_test, y_train, y_test,
            pop_size, generations, F, CR
        )

    st.success("Optimization Completed")

    st.subheader("🏆 Best Parameters")
    st.write(f"C: {best_params[0]:.4f}")
    st.write(f"Epsilon: {best_params[1]:.4f}")
    st.write(f"Gamma: {best_params[2]:.4f}")

    st.metric("Best RMSE", f"{best_rmse:.2f}")

    st.subheader("📈 Convergence Curve")
    fig, ax = plt.subplots()
    ax.plot(history)
    ax.set_xlabel("Generation")
    ax.set_ylabel("RMSE")
    ax.set_title("DE Convergence")
    st.pyplot(fig)

