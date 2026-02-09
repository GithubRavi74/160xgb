# -------------------------------------------------
# FORECAST
# -------------------------------------------------
if st.button("📈 Forecast Next 7 Days"):

    batch_df_full = df[
        (df["farm_id"] == farm_id) &
        (df["batch_id"] == batch_id)
    ].sort_values("day_number")

    if batch_df_full.empty:
        st.error("❌ No data for selected batch")
        st.stop()

    max_day = int(batch_df_full["day_number"].max())

    # -----------------------------------
    # CUTOFF DAY (Simulate missing future)
    # -----------------------------------
    cutoff_day = st.slider(
        "Select Cutoff Day (simulate prediction from this day)",
        min_value=5,
        max_value=max_day - 1,
        value=max_day - 7
    )

    batch_df = batch_df_full[
        batch_df_full["day_number"] <= cutoff_day
    ]

    future_actual = batch_df_full[
        (batch_df_full["day_number"] > cutoff_day) &
        (batch_df_full["day_number"] <= cutoff_day + 7)
    ]

    if batch_df.empty:
        st.error("❌ No data before cutoff")
        st.stop()

    # -----------------------------
    # LATEST STATE (from cutoff)
    # -----------------------------
    last_row = batch_df.iloc[-1]

    age_today = int(last_row["day_number"])
    birds_alive = int(last_row["birds_alive"])
    feed_today = float(last_row["feed_today_kg"])
    mortality_today = float(last_row["mortality_today"])

    temp = float(last_row["temp"])
    rh   = float(last_row["rh"])
    co   = float(last_row["co"])
    nh   = float(last_row["nh"])

    # -----------------------------
    # ROLLING FEATURES
    # -----------------------------
    recent = batch_df.tail(7)

    if len(recent) >= 3:
        rolling_feed = recent["feed_today_kg"].mean()
        rolling_gain = recent["daily_weight_gain_kg"].mean()
        rolling_note = "✅ Rolling features computed from batch data"
    else:
        rolling_feed = feed_today
        rolling_gain = 0.045
        rolling_note = "⚠️ Insufficient continuity — using safe defaults"

    st.info(rolling_note)

    # -----------------------------
    # BUILD FUTURE FEATURES
    # -----------------------------
    horizon = 7
    days = np.arange(age_today + 1, age_today + horizon + 1)

    rows = []

    for d in days:

        rows.append({
            "day_number": d,
            "birds_alive": birds_alive,
            "feed_today_kg": feed_today,
            "feed_per_bird": feed_today / birds_alive,
            "mortality_today": mortality_today,
            "mortality_rate": mortality_today / birds_alive,
            "rolling_7d_feed": rolling_feed,
            "rolling_7d_gain": rolling_gain,
            "temp": temp,
            "rh": rh,
            "co": co,
            "nh": nh
        })

    X = pd.DataFrame(rows)

    # -----------------------------
    # PREDICTIONS
    # -----------------------------
    pred_weight = weight_model.predict(X)
    pred_mort   = mort_model.predict(X)
    pred_fcr    = fcr_model.predict(X)

    # -----------------------------
    # OUTPUT TABLE
    # -----------------------------
    out = pd.DataFrame({
        "Day": days,
        "Predicted Weight (kg)": np.round(pred_weight, 3),
        "Predicted Mortality": np.maximum(0, np.round(pred_mort).astype(int)),
        "Predicted Feed / Bird": np.round(pred_fcr, 3)
    })

    # -----------------------------------
    # MERGE WITH ACTUAL (IF EXISTS)
    # -----------------------------------
    if not future_actual.empty:
        compare = future_actual[[
            "day_number",
            "sample_weight_kg",
            "mortality_today",
            "feed_per_bird"
        ]].copy()

        compare.columns = [
            "Day",
            "Actual Weight (kg)",
            "Actual Mortality",
            "Actual Feed / Bird"
        ]

        final_compare = out.merge(compare, on="Day", how="left")

        st.subheader("📊 Prediction vs Actual")
        st.dataframe(final_compare, use_container_width=True)

        # Calculate MAE
        mae = np.mean(
            np.abs(final_compare["Predicted Weight (kg)"] -
                   final_compare["Actual Weight (kg)"])
        )

        st.metric("📉 Weight MAE (7-day)", round(mae, 4))

    else:
        st.subheader("📊 Forecast Results")
        st.dataframe(out, use_container_width=True)
