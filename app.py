import streamlit as st
import pandas as pd
import pulp

# Page configuration
st.set_page_config(
    page_title="NutriSync — Automated Macro & Grocery Optimizer",
    page_icon="🥗",
    layout="wide",
)

# Header
st.title("🥗 NutriSync — Automated Macro & Grocery Optimizer")
st.caption(
    "SIMATS Engineering Capstone Project | Python Programming (CSA0829) | "
    "Linear Programming Meal & Grocery Optimization"
)

# Default Staple Ingredient Database (nutrition & cost per 100g / 1 standard serving)
DEFAULT_STAPLES = [
    {"Ingredient": "Rolled Oats", "Aisle": "Grains & Cereals", "Calories": 389, "Protein": 16.9, "Cost": 18, "Max_Daily_Units": 1.5, "Veg": True},
    {"Ingredient": "Chicken Breast", "Aisle": "Lean Proteins", "Calories": 165, "Protein": 31.0, "Cost": 32, "Max_Daily_Units": 2.5, "Veg": False},
    {"Ingredient": "Soya Chunks", "Aisle": "Lean Proteins", "Calories": 345, "Protein": 52.0, "Cost": 15, "Max_Daily_Units": 1.2, "Veg": True},
    {"Ingredient": "Whole Eggs (2 units / ~100g)", "Aisle": "Dairy & Eggs", "Calories": 143, "Protein": 12.6, "Cost": 14, "Max_Daily_Units": 2.0, "Veg": False},
    {"Ingredient": "Paneer (Low Fat)", "Aisle": "Dairy & Eggs", "Calories": 265, "Protein": 18.3, "Cost": 38, "Max_Daily_Units": 1.5, "Veg": True},
    {"Ingredient": "Brown Rice", "Aisle": "Grains & Pulses", "Calories": 350, "Protein": 7.9, "Cost": 11, "Max_Daily_Units": 2.0, "Veg": True},
    {"Ingredient": "Toor / Moong Dal", "Aisle": "Grains & Pulses", "Calories": 343, "Protein": 22.3, "Cost": 14, "Max_Daily_Units": 1.5, "Veg": True},
    {"Ingredient": "Greek Curd / Hung Yogurt", "Aisle": "Dairy & Eggs", "Calories": 97, "Protein": 9.0, "Cost": 16, "Max_Daily_Units": 2.0, "Veg": True},
    {"Ingredient": "Spinach & Broccoli", "Aisle": "Fresh Produce", "Calories": 34, "Protein": 2.8, "Cost": 10, "Max_Daily_Units": 2.5, "Veg": True},
    {"Ingredient": "Banana", "Aisle": "Fresh Produce", "Calories": 89, "Protein": 1.1, "Cost": 6, "Max_Daily_Units": 2.0, "Veg": True},
    {"Ingredient": "Peanut Butter", "Aisle": "Pantry & Spreads", "Calories": 588, "Protein": 25.0, "Cost": 28, "Max_Daily_Units": 0.5, "Veg": True},
]

# Sidebar Controls
st.sidebar.header("🎯 Daily Nutritional Targets")
target_calories = st.sidebar.slider("Daily Calorie Target (kcal)", min_value=1400, max_value=3500, value=2200, step=50)
target_protein = st.sidebar.slider("Minimum Daily Protein (g)", min_value=60, max_value=240, value=140, step=5)
calorie_tolerance = st.sidebar.slider("Calorie Tolerance (±%)", min_value=1, max_value=10, value=3, step=1)
diet_mode = st.sidebar.radio("Dietary Preference", ["All Staples", "Vegetarian Only"])
days = st.sidebar.selectbox("Planning Horizon (Days)", [7, 14, 30], index=0)

# Filter staples based on diet mode
df_staples = pd.DataFrame(DEFAULT_STAPLES)
if diet_mode == "Vegetarian Only":
    df_staples = df_staples[df_staples["Veg"]].reset_index(drop=True)

st.subheader("1. Fixed Staple Ingredient Matrix (Per 100g Serving)")
edited_df = st.data_editor(
    df_staples[["Ingredient", "Aisle", "Calories", "Protein", "Cost", "Max_Daily_Units"]],
    use_container_width=True,
    num_rows="dynamic",
    key="staples_editor",
)

def solve_meal_plan(df: pd.DataFrame, cal_target: float, prot_target: float, tol_pct: float, horizon_days: int):
    """Solve linear program minimizing weekly cost subject to daily macro constraints."""
    model = pulp.LpProblem("NutriSync_Macro_Optimizer", pulp.LpMinimize)

    # Decision variables: daily servings (1 unit = 100g) for each ingredient
    qty = {
        idx: pulp.LpVariable(
            f"qty_{idx}",
            lowBound=0.0,
            upBound=float(df.loc[idx, "Max_Daily_Units"]),
            cat="Continuous",
        )
        for idx in df.index
    }

    # Objective: Minimize total cost over the planning horizon
    model += pulp.lpSum(df.loc[idx, "Cost"] * qty[idx] * horizon_days for idx in df.index), "Total_Cost"

    # Constraints: Daily calorie band and minimum protein target
    min_cal = cal_target * (1 - tol_pct / 100.0)
    max_cal = cal_target * (1 + tol_pct / 100.0)

    model += pulp.lpSum(df.loc[idx, "Calories"] * qty[idx] for idx in df.index) >= min_cal, "Min_Calories"
    model += pulp.lpSum(df.loc[idx, "Calories"] * qty[idx] for idx in df.index) <= max_cal, "Max_Calories"
    model += pulp.lpSum(df.loc[idx, "Protein"] * qty[idx] for idx in df.index) >= prot_target, "Min_Protein"

    # Encourage variety: require at least 1.0 unit of Fresh Produce per day if available
    produce_indices = df[df["Aisle"] == "Fresh Produce"].index.tolist()
    if produce_indices:
        model += pulp.lpSum(qty[idx] for idx in produce_indices) >= 1.0, "Min_Produce"

    status = model.solve(pulp.PULP_CBC_CMD(msg=False))
    return pulp.LpStatus[status], qty

# Run Optimization
status_label, decision_vars = solve_meal_plan(
    edited_df, target_calories, target_protein, calorie_tolerance, days
)

st.divider()
st.subheader(f"2. Optimal {days}-Day Plan & Auto-Sorted Grocery List")

if status_label != "Optimal":
    st.error(
        f"⚠️ Solver Status: **{status_label}**. Your target ({target_calories} kcal, {target_protein}g protein) "
        "cannot be met within the current daily portion caps. Try increasing Max Daily Units or relaxing the tolerance."
    )
else:
    results = []
    for idx, var in decision_vars.items():
        daily_units = var.varValue or 0.0
        if daily_units > 0.01:
            daily_grams = round(daily_units * 100, 1)
            weekly_grams = round(daily_grams * days, 1)
            daily_cal = round(daily_units * edited_df.loc[idx, "Calories"], 1)
            daily_prot = round(daily_units * edited_df.loc[idx, "Protein"], 1)
            total_cost = round(daily_units * edited_df.loc[idx, "Cost"] * days, 2)
            results.append({
                "Aisle": edited_df.loc[idx, "Aisle"],
                "Ingredient": edited_df.loc[idx, "Ingredient"],
                "Daily Portion (g)": daily_grams,
                f"{days}-Day Grocery Qty (g)": weekly_grams,
                "Daily kcal": daily_cal,
                "Daily Protein (g)": daily_prot,
                f"{days}-Day Cost (₹)": total_cost,
            })

    res_df = pd.DataFrame(results).sort_values(by=["Aisle", f"{days}-Day Cost (₹)"], ascending=[True, False])

    # Summary KPI Metrics
    achieved_cal = res_df["Daily kcal"].sum()
    achieved_prot = res_df["Daily Protein (g)"].sum()
    total_budget = res_df[f"{days}-Day Cost (₹)"].sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Solver Status", "Optimal ✅")
    col2.metric("Achieved Daily Calories", f"{achieved_cal:.0f} kcal", f"{achieved_cal - target_calories:+.0f} kcal")
    col3.metric("Achieved Daily Protein", f"{achieved_prot:.1f} g", f"{achieved_prot - target_protein:+.1f} g")
    col4.metric(f"Total {days}-Day Grocery Cost", f"₹ {total_budget:,.2f}")

    # Display Aisle-Sorted Grocery Table
    st.dataframe(res_df, use_container_width=True, hide_index=True)

    # Aisle Breakdown Chart
    st.subheader("3. Grocery Spend & Protein Breakdown by Store Aisle")
    aisle_summary = (
        res_df.groupby("Aisle", as_index=False)
        .agg({f"{days}-Day Grocery Qty (g)": "sum", "Daily Protein (g)": "sum", f"{days}-Day Cost (₹)": "sum"})
    )
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.bar_chart(aisle_summary, x="Aisle", y=f"{days}-Day Cost (₹)", color="#0F766E")
    with chart_col2:
        st.bar_chart(aisle_summary, x="Aisle", y="Daily Protein (g)", color="#EA580C")

    # One-Click CSV Download
    csv_data = res_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"📥 Download {days}-Day Auto-Sorted Grocery Checklist (.CSV)",
        data=csv_data,
        file_name=f"nutrisync_grocery_list_{days}days.csv",
        mime="text/csv",
    )
