import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Late Delivery Risk", page_icon="📦", layout="centered")


@st.cache_resource
def load_assets():
    model = joblib.load("late_delivery_model.joblib")
    with open("model_meta.json") as f:
        meta = json.load(f)
    return model, meta


model, meta = load_assets()
threshold = meta["threshold"]
cats = meta["categories"]
rng = meta["ranges"]

st.title("📦 Late Delivery Risk Predictor")
st.caption("Enter order details known at order time. The model estimates the chance the order will be delivered late.")

col1, col2 = st.columns(2)
with col1:
    mode = st.selectbox("Shipping mode", cats["shipping_mode"])
    sched = st.number_input("Scheduled shipping days", min_value=0, max_value=10,
                            value=int(meta["default_scheduled_days"].get(mode, 4)), key=f"sched_{mode}")
    market = st.selectbox("Market", cats["market"])
    region = st.selectbox("Order region", cats["order_region"])
    order_date = st.date_input("Order date")
with col2:
    dept = st.selectbox("Department", cats["department_name"])
    segment = st.selectbox("Customer segment", cats["customer_segment"])
    qty = st.number_input("Item quantity", min_value=1, max_value=rng["quantity_max"], value=1)
    price = st.number_input("Product price", min_value=0.0, max_value=float(rng["price_max"]), value=100.0)
    discount = st.slider("Discount rate", 0.0, float(rng["discount_max"]), 0.0, 0.01)

if st.button("Predict", type="primary"):
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

    prob = float(model.predict_proba(row)[0, 1])
    st.metric("Probability of late delivery", f"{prob:.1%}")
    st.progress(min(max(prob, 0.0), 1.0))

    if prob >= threshold:
        st.error(f"⚠️ Likely LATE (probability is at or above the alert threshold of {threshold:.2f}). "
                 "Consider early customer notification or monitoring.")
    else:
        st.success(f"✅ Likely on time (probability is below the alert threshold of {threshold:.2f}).")

    st.caption(f"Historical late rate for {mode}: {meta['late_rate_by_mode'].get(mode, float('nan')):.0%}")

with st.expander("How reliable is this?"):
    m = meta["test_metrics"]
    st.write(f"Model: **{meta['model']}** | Alert threshold: **{threshold:.2f}** "
             f"(chosen to maximise recall while keeping precision at least {meta['precision_floor']:.0%}).")
    st.write(f"Test results: recall **{m['Recall']:.1%}**, precision **{m['Precision']:.1%}**, AUC **{m['AUC']:.3f}**.")
    st.write("About 4 in 10 late orders are still missed, because shipping mode and schedule carry most of the "
             "available signal. Treat the flag as an early-warning aid, not a guarantee.")
