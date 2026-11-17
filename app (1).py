import json
import joblib
import pandas as pd
import streamlit as st
import os

st.set_page_config(
    page_title="Late Delivery Risk Predictor",
    page_icon="📦",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Use modern caching with validation to avoid endless reloading loops
@st.cache_resource(show_spinner=False)
def load_assets():
    model_path = "late_delivery_model.joblib"
    meta_path = "model_meta.json"
    
    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        st.error("⚠️ Crucial model assets are missing from your repository path! Make sure both 'late_delivery_model.joblib' and 'model_meta.json' are placed in the same folder as app.py.")
        st.stop()
        
    model = joblib.load(model_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta

# User-friendly initialization spinner
with st.spinner("Initializing Predictive Intelligence Engine... (This takes a few seconds on first startup)"):
    model, meta = load_assets()

threshold = meta["threshold"]
cats = meta["categories"]
rng = meta["ranges"]

st.title("📦 Late Delivery Risk Predictor")
st.markdown("Enter details known at the **order checkout** to assess the risk of delivery delay.")

col1, col2 = st.columns(2)
with col1:
    mode = st.selectbox("Shipping mode", cats["shipping_mode"])
    default_sched = int(meta["default_scheduled_days"].get(mode, 4))
    sched = st.number_input(
        "Scheduled shipping days", 
        min_value=0, 
        max_value=10, 
        value=default_sched,
        key=f"sched_{mode}"
    )
    market = st.selectbox("Market", cats["market"])
    region = st.selectbox("Order region", cats["order_region"])
    order_date = st.date_input("Order date")

with col2:
    dept = st.selectbox("Department", cats["department_name"])
    segment = st.selectbox("Customer segment", cats["customer_segment"])
    qty = st.number_input("Item quantity", min_value=1, max_value=rng["quantity_max"], value=1)
    price = st.number_input("Product price", min_value=0.0, max_value=float(rng["price_max"]), value=100.0)
    discount = st.slider("Discount rate", 0.0, float(rng["discount_max"]), 0.0, 0.01)

st.markdown("--- ")

if st.button("Run Predictor", type="primary", use_container_width=True):
    # Reconstruct exact features template
    row = pd.DataFrame([{
        "days_for_shipment_scheduled": sched,
        "shipping_mode": mode,
        "market": market,
        "order_region": region,
        "department_name": dept,
        "customer_segment": segment,
        "order_item_quantity": qty,
        "order_item_product_price": price,
        "order_item_discount_rate": discount,
        "order_month": order_date.month,
        "order_dayofweek": order_date.weekday(),
    }])[meta["features"]]

    # Generate probability prediction
    prob = float(model.predict_proba(row)[0, 1])
    
    st.markdown("### Prediction Outcome")
    st.metric(label="Estimated Probability of Late Delivery", value=f"{prob:.1%}")
    st.progress(min(max(prob, 0.0), 1.0))

    if prob >= threshold:
        st.error(f"⚠️ **Likely LATE** (Probability is at or above the alert threshold of **{threshold:.2f}**).\n\nAction recommended: Proactively coordinate with carrier or notify client.")
    else:
        st.success(f"✅ **Likely On-Time** (Probability is below the alert threshold of **{threshold:.2f}**).")

    st.info(f"Historical base lateness rate for *{mode}*: {meta['late_rate_by_mode'].get(mode, 0.0):.0%}")

with st.expander("🔍 Operational Validation Details"):
    m = meta["test_metrics"]
    st.markdown(f"""
    * **Model Engine**: {meta['model']}
    * **Alert Threshold**: {threshold:.2f} *(chosen to protect precision floor at {meta['precision_floor']:.0%})*
    * **Reliability Metrics on Unseen Test Split**:
        * **Recall**: {m['Recall']:.1%}
        * **Precision**: {m['Precision']:.1%}
        * **AUC Score**: {m['AUC']:.3f}
    """)
