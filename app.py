import streamlit as st
import numpy as np
import pandas as pd
import pydeck as pdk
import plotly.express as px
from ortools.linear_solver import pywraplp

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Chennai Dark Store Location Optimizer",
    page_icon="📦",
    layout="wide",
)

st.title("📦 Quick Commerce: Dark Store Location Optimization (Chennai)")
st.markdown(
    """
    **Capacitated Facility Location Problem (MILP)** powered by **Google OR-Tools (SCIP Solver)**.  
    Enhances the base model with a **Maximum Service Distance Constraint ($D_{\\max}$)**, **Profit Maximization Objective**, 
    interactive **Chennai Map**, and **Multi-Store Sensitivity Analysis** ($K \\in \\{2, 5, 10, 20, 25\\}$).
    """
)

# =============================================================================
# 1. SYNTHETIC CHENNAI GEOGRAPHY & DATA GENERATOR
# =============================================================================
# Major residential/commercial hubs in Chennai (West of the Bay of Bengal coastline)
CHENNAI_HUBS = [
    ("T. Nagar", 13.0418, 80.2341),
    ("Anna Nagar", 13.0850, 80.2101),
    ("Adyar", 13.0012, 80.2565),
    ("Velachery", 12.9759, 80.2206),
    ("Tambaram", 12.9249, 80.1000),
    ("Porur", 13.0382, 80.1565),
    ("Ambattur", 13.1143, 80.1548),
    ("OMR / Sholinganallur", 12.9010, 80.2279),
    ("Mylapore", 13.0368, 80.2676),
    ("Guindy", 13.0067, 80.2206),
    ("Chromepet", 12.9516, 80.1462),
    ("Perambur", 13.1185, 80.2333),
    ("Vadapalani", 13.0500, 80.2121),
    ("Medavakkam", 12.9171, 80.1923),
    ("Madipakkam", 12.9647, 80.1961),
]


@st.cache_data
def generate_chennai_data(num_customers: int, num_stores: int, seed: int = 42):
    """
    Generates realistic customer demand points and candidate dark store locations
    within the Chennai metropolitan area (avoiding the ocean).
    """
    rng = np.random.default_rng(seed)
    hub_coords = np.array([[h[1], h[2]] for h in CHENNAI_HUBS])
    num_hubs = len(CHENNAI_HUBS)

    # --- Generate Candidate Dark Stores ---
    store_hub_idx = rng.integers(0, num_hubs, size=num_stores)
    store_lats = hub_coords[store_hub_idx, 0] + rng.normal(0, 0.018, size=num_stores)
    store_lons = hub_coords[store_hub_idx, 1] + rng.normal(0, 0.018, size=num_stores)
    # Clip longitude to stay safely inland from the Chennai Marina/Elliot's coastline
    store_lons = np.clip(store_lons, 80.08, 80.265)
    store_lats = np.clip(store_lats, 12.88, 13.16)

    # Align with lecture slide stats:
    # Capacity: 1500 - 2000 orders/day
    # Fixed Cost: $1,400 - $1,800/day
    store_capacities = rng.integers(1500, 2001, size=num_stores)
    store_fixed_costs = rng.integers(1400, 1801, size=num_stores)

    stores_df = pd.DataFrame({
        "store_id": np.arange(num_stores),
        "name": [f"DS-{j+1:02d} ({CHENNAI_HUBS[store_hub_idx[j]][0]})" for j in range(num_stores)],
        "lat": store_lats,
        "lon": store_lons,
        "capacity": store_capacities,
        "fixed_cost": store_fixed_costs,
    })

    # --- Generate Customer Locations ---
    cust_hub_idx = rng.integers(0, num_hubs, size=num_customers)
    cust_lats = hub_coords[cust_hub_idx, 0] + rng.normal(0, 0.025, size=num_customers)
    cust_lons = hub_coords[cust_hub_idx, 1] + rng.normal(0, 0.025, size=num_customers)
    cust_lons = np.clip(cust_lons, 80.07, 80.270)
    cust_lats = np.clip(cust_lats, 12.87, 13.17)

    # Daily demand per customer node (scaled so total demand aligns with store capacities)
    # For 10,000 customers, 1 to 6 orders/day gives ~35,000 total daily orders
    cust_demands = rng.integers(1, 7, size=num_customers)

    customers_df = pd.DataFrame({
        "customer_id": np.arange(num_customers),
        "lat": cust_lats,
        "lon": cust_lons,
        "demand": cust_demands,
    })

    return customers_df, stores_df


def compute_haversine_matrix(cust_lats, cust_lons, store_lats, store_lons):
    """
    Vectorized Haversine distance matrix (in km) of shape (num_customers, num_stores).
    """
    R = 6371.0  # Earth radius in km
    lat1 = np.radians(cust_lats)[:, np.newaxis]
    lon1 = np.radians(cust_lons)[:, np.newaxis]
    lat2 = np.radians(store_lats)[np.newaxis, :]
    lon2 = np.radians(store_lons)[np.newaxis, :]

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return R * c


# =============================================================================
# 2. MILP OPTIMIZATION ENGINE (GOOGLE OR-TOOLS + SCIP)
# =============================================================================
def solve_dark_store_milp(
    customers_df: pd.DataFrame,
    stores_df: pd.DataFrame,
    max_dist_km: float,
    revenue_per_order: float = 4.0,
    cost_per_km: float = 1.50,
    max_open_stores: int = None,
    time_limit_sec: int = 15,
):
    """
    Solves the Enhanced Capacitated Dark Store Location MILP:
      Maximize: sum_{i,j} (Revenue - d_ij * cost_per_km) * x_ij - sum_j FixedCost_j * y_j
      Subject to:
        1. Demand Satisfaction (Partial allowed per Slide 28): sum_j x_ij <= D_i
        2. Facility Capacity: sum_i x_ij <= C_j * y_j
        3. Maximum Distance Constraint: x_ij = 0 if dist_ij > max_dist_km
        4. Optional Max Stores Constraint: sum_j y_j <= K
    """
    num_customers = len(customers_df)
    num_stores = len(stores_df)

    # Compute distance matrix (num_customers x num_stores)
    dist_matrix = compute_haversine_matrix(
        customers_df["lat"].values,
        customers_df["lon"].values,
        stores_df["lat"].values,
        stores_df["lon"].values,
    )

    # Create SCIP Solver
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if not solver:
        raise RuntimeError("SCIP solver is not available in OR-Tools.")

    solver.SetTimeLimit(time_limit_sec * 1000)
    # Set 0.5% relative optimality gap for fast convergence on 10,000 customers
    solver.SetSolverSpecificParametersAsString("limits/gap = 0.005")

    demands = customers_df["demand"].values.astype(float)
    capacities = stores_df["capacity"].values.astype(float)
    fixed_costs = stores_df["fixed_cost"].values.astype(float)

    # Decision Variables: y_j in {0, 1}
    y = [solver.BoolVar(f"y_{j}") for j in range(num_stores)]

    # Decision Variables: x_ij >= 0 ONLY for (i, j) pairs within max_dist_km
    # This enforces Constraint 3 (d_ij <= max_dist_km) and keeps the model sparse!
    valid_i, valid_j = np.where(dist_matrix <= max_dist_km)

    # Group valid connections by customer and by store for fast constraint building
    cust_to_vars = [[] for _ in range(num_customers)]
    store_to_vars = [[] for _ in range(num_stores)]
    x_vars = {}

    objective = solver.Objective()
    for j in range(num_stores):
        objective.SetCoefficient(y[j], -float(fixed_costs[j]))

    for idx in range(len(valid_i)):
        i = int(valid_i[idx])
        j = int(valid_j[idx])
        dist_ij = float(dist_matrix[i, j])
        deliv_cost_ij = dist_ij * cost_per_km
        net_margin_ij = revenue_per_order - deliv_cost_ij

        # Only create variable if it is within max_dist_km
        var = solver.NumVar(0.0, demands[i], f"x_{i}_{j}")
        x_vars[(i, j)] = var
        cust_to_vars[i].append(var)
        store_to_vars[j].append(var)
        objective.SetCoefficient(var, net_margin_ij)

    objective.SetMaximization()

    # Constraint 1: Demand Limit (sum_j x_ij <= Demand_i) - Slide 28
    for i in range(num_customers):
        if len(cust_to_vars[i]) > 1:
            solver.Add(solver.Sum(cust_to_vars[i]) <= demands[i])
        # Note: if len == 1, the variable upper bound `demands[i]` already enforces x_ij <= D_i!

    # Constraint 2: Facility Capacity (sum_i x_ij <= Capacity_j * y_j) - Slide 27
    for j in range(num_stores):
        if store_to_vars[j]:
            solver.Add(solver.Sum(store_to_vars[j]) <= capacities[j] * y[j])
        else:
            # If no customer is within range of store j, force it closed
            solver.Add(y[j] == 0)

    # Constraint 4: Optional Maximum Number of Open Dark Stores (sum_j y_j <= K)
    if max_open_stores is not None:
        solver.Add(solver.Sum(y) <= int(max_open_stores))

    status = solver.Solve()

    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        return None

    # Extract Results
    open_flags = np.array([1 if y[j].solution_value() > 0.5 else 0 for j in range(num_stores)])
    store_orders_served = np.zeros(num_stores)
    cust_orders_served = np.zeros(num_customers)
    cust_primary_store = np.full(num_customers, -1, dtype=int)
    cust_max_flow = np.zeros(num_customers)

    total_revenue = 0.0
    total_delivery_cost = 0.0
    total_weighted_dist = 0.0
    flows = []

    for (i, j), var in x_vars.items():
        val = var.solution_value()
        if val > 1e-4:
            dist_ij = dist_matrix[i, j]
            del_cost = dist_ij * cost_per_km * val
            rev = revenue_per_order * val

            store_orders_served[j] += val
            cust_orders_served[i] += val
            total_revenue += rev
            total_delivery_cost += del_cost
            total_weighted_dist += dist_ij * val

            if val > cust_max_flow[i]:
                cust_max_flow[i] = val
                cust_primary_store[i] = j

            flows.append({
                "customer_id": i,
                "store_id": j,
                "orders": val,
                "distance_km": dist_ij,
            })

    total_fixed_cost = float(np.sum(fixed_costs * open_flags))
    total_profit = total_revenue - total_delivery_cost - total_fixed_cost
    total_demand = float(np.sum(demands))
    total_served = float(np.sum(cust_orders_served))
    total_open_capacity = float(np.sum(capacities * open_flags))

    avg_distance = (total_weighted_dist / total_served) if total_served > 0 else 0.0
    pct_demand_met = (total_served / total_demand * 100.0) if total_demand > 0 else 0.0
    pct_cap_utilized = (total_served / total_open_capacity * 100.0) if total_open_capacity > 0 else 0.0
    num_cust_covered = int(np.sum(cust_orders_served > 1e-4))

    # Build Updated DataFrames for Map & Tables
    stores_res = stores_df.copy()
    stores_res["status"] = np.where(open_flags == 1, "OPEN", "CLOSED")
    stores_res["orders_served"] = np.round(store_orders_served, 1)
    stores_res["utilization_pct"] = np.where(
        open_flags == 1,
        np.round(store_orders_served / capacities * 100.0, 1),
        0.0,
    )

    cust_res = customers_df.copy()
    cust_res["orders_served"] = np.round(cust_orders_served, 1)
    cust_res["assigned_store"] = cust_primary_store
    cust_res["is_served"] = cust_orders_served > 1e-4

    kpis = {
        "Total Daily Profit ($)": total_profit,
        "Total Daily Revenue ($)": total_revenue,
        "Total Fixed Store Cost ($)": total_fixed_cost,
        "Total Delivery Cost ($)": total_delivery_cost,
        "Dark Stores Opened": f"{int(np.sum(open_flags))} / {num_stores}",
        "Total Orders Served / Day": f"{total_served:,.0f} / {total_demand:,.0f}",
        "Demand Fulfillment (%)": f"{pct_demand_met:.2f}%",
        "Customers Served": f"{num_cust_covered:,} / {num_customers:,}",
        "Open Store Capacity Utilization (%)": f"{pct_cap_utilized:.2f}%",
        "Avg Delivery Distance (km)": f"{avg_distance:.2f} km",
    }

    return {
        "kpis": kpis,
        "profit": total_profit,
        "revenue": total_revenue,
        "fixed_cost": total_fixed_cost,
        "delivery_cost": total_delivery_cost,
        "stores_Opened_count": int(np.sum(open_flags)),
        "demand_met_pct": pct_demand_met,
        "stores_df": stores_res,
        "customers_df": cust_res,
        "flows": flows,
    }


# =============================================================================
# 3. SIDEBAR INPUT CONTROLS
# =============================================================================
st.sidebar.header("⚙️ Model Parameters")

num_stores_input = st.sidebar.number_input(
    "Candidate Dark Stores Count (|J|)",
    min_value=4,
    max_value=100,
    value=50,
    step=1,
)

num_customers_input = st.sidebar.number_input(
    "Number of Customers (|I|)",
    min_value=10,
    max_value=15000,
    value=2000,
    step=500,
    help="Set to 10,000 for the full scale test, or use Tab 2 for the automated 10,000-customer benchmark.",
)

max_dist_input = st.sidebar.slider(
    "Max Distance: Dark Store to Customer (km)",
    min_value=0.5,
    max_value=10.0,
    value=2.0,
    step=0.1,
    help="Quick commerce service radius (typically 1.7 - 2.0 km per Slide 25).",
)

use_max_store_limit = st.sidebar.checkbox("Limit Maximum Open Dark Stores?", value=False)
max_open_input = None
if use_max_store_limit:
    max_open_input = st.sidebar.slider(
        "Max Dark Stores Allowed to Open (K)",
        min_value=1,
        max_value=int(num_stores_input),
        value=min(10, int(num_stores_input)),
    )

st.sidebar.markdown("---")
st.sidebar.subheader("💰 Economic Assumptions (Daily)")
revenue_per_order = st.sidebar.number_input("Revenue (Margin) per Order ($)", value=4.0, min_value=0.5, step=0.5)
cost_per_km = st.sidebar.number_input("Delivery Cost per km ($/order)", value=1.50, min_value=0.1, step=0.1)
random_seed = st.sidebar.number_input("Random Seed (Geography)", value=42, step=1)

# =============================================================================
# 4. MAIN TABS: (1) INTERACTIVE OPTIMIZER & MAP | (2) 10,000-CUSTOMER BENCHMARK
# =============================================================================
tab1, tab2 = st.tabs([
    "🗺️ Interactive Chennai Optimizer & Map",
    "📈 Sensitivity Benchmark (10,000 Customers & 50 Stores)",
])

# -----------------------------------------------------------------------------
# TAB 1: INTERACTIVE MODEL, KPI TABLE & CHENNAI MAP
# -----------------------------------------------------------------------------
with tab1:
    with st.spinner("Generating Chennai geography and solving MILP via SCIP..."):
        cust_df, store_df = generate_chennai_data(
            int(num_customers_input), int(num_stores_input), int(random_seed)
        )
        result = solve_dark_store_milp(
            customers_df=cust_df,
            stores_df=store_df,
            max_dist_km=float(max_dist_input),
            revenue_per_order=float(revenue_per_order),
            cost_per_km=float(cost_per_km),
            max_open_stores=max_open_input,
        )

    if result is None:
        st.error("Solver could not find a feasible solution.")
    else:
        # Top Metric Cards
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Net Daily Profit", f"${result['profit']:,.2f}")
        m2.metric("Dark Stores Opened", result["kpis"]["Dark Stores Opened"])
        m3.metric("Demand Fulfilled", result["kpis"]["Demand Fulfillment (%)"])
        m4.metric("Avg Delivery Distance", result["kpis"]["Avg Delivery Distance (km)"])

        col_kpi, col_map = st.columns([1, 1.65])

        with col_kpi:
            st.subheader("1️⃣ Optimization KPIs Table")
            kpi_table = pd.DataFrame([
                {"KPI Metric": k, "Value": f"${v:,.2f}" if isinstance(v, float) else str(v)}
                for k, v in result["kpis"].items()
            ])
            st.dataframe(kpi_table, use_container_width=True, hide_index=True)

            st.subheader("🏪 Candidate Dark Stores Status")
            st.dataframe(
                result["stores_df"][["name", "status", "orders_served", "capacity", "utilization_pct", "fixed_cost"]],
                use_container_width=True,
                hide_index=True,
                height=310,
            )

        with col_map:
            st.subheader("2️⃣ Detailed Solution on Chennai Map")
            show_radii = st.checkbox("Show Max Distance Service Radius Circles", value=True)
            show_links = st.checkbox("Show Store-to-Customer Fulfillment Lines (Sampled for clarity)", value=True)

            stores_res = result["stores_df"]
            cust_res = result["customers_df"]

            open_stores = stores_res[stores_res["status"] == "OPEN"].copy()
            closed_stores = stores_res[stores_res["status"] == "CLOSED"].copy()
            served_cust = cust_res[cust_res["is_served"]].copy()
            unserved_cust = cust_res[~cust_res["is_served"]].copy()

            layers = []

            # Layer 1: Service Radius Polygons around OPEN stores
            if show_radii and not open_stores.empty:
                layers.append(
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=open_stores,
                        get_position=["lon", "lat"],
                        get_radius=float(max_dist_input) * 1000.0,  # meters
                        get_fill_color=[22, 163, 74, 30],
                        get_line_color=[22, 163, 74, 160],
                        stroked=True,
                        filled=True,
                        line_width_min_pixels=2,
                        pickable=False,
                    )
                )

            # Layer 2: Unserved Customers (Gray dots)
            if not unserved_cust.empty:
                layers.append(
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=unserved_cust,
                        get_position=["lon", "lat"],
                        get_radius=65,
                        get_fill_color=[148, 163, 184, 140],
                        pickable=True,
                    )
                )

            # Layer 3: Served Customers (Blue dots)
            if not served_cust.empty:
                layers.append(
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=served_cust,
                        get_position=["lon", "lat"],
                        get_radius=80,
                        get_fill_color=[37, 99, 235, 200],
                        pickable=True,
                    )
                )

            # Layer 4: Sampled Fulfillment Flow Lines
            if show_links and result["flows"]:
                flows_df = pd.DataFrame(result["flows"])
                # Cap drawn lines at 1,500 so browser WebGL stays fast
                if len(flows_df) > 1500:
                    flows_df = flows_df.sample(n=1500, random_state=42)
                flows_df["store_lat"] = stores_res.loc[flows_df["store_id"], "lat"].values
                flows_df["store_lon"] = stores_res.loc[flows_df["store_id"], "lon"].values
                flows_df["cust_lat"] = cust_res.loc[flows_df["customer_id"], "lat"].values
                flows_df["cust_lon"] = cust_res.loc[flows_df["customer_id"], "lon"].values

                layers.append(
                    pdk.Layer(
                        "LineLayer",
                        data=flows_df,
                        get_source_position=["store_lon", "store_lat"],
                        get_target_position=["cust_lon", "cust_lat"],
                        get_color=[37, 99, 235, 90],
                        get_width=1.5,
                    )
                )

            # Layer 5: CLOSED Dark Stores (Red markers)
            if not closed_stores.empty:
                layers.append(
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=closed_stores,
                        get_position=["lon", "lat"],
                        get_radius=260,
                        get_fill_color=[220, 38, 38, 210],
                        get_line_color=[15, 23, 42, 255],
                        stroked=True,
                        line_width_min_pixels=2,
                        pickable=True,
                    )
                )

            # Layer 6: OPEN Dark Stores (Bright Green markers)
            if not open_stores.empty:
                layers.append(
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=open_stores,
                        get_position=["lon", "lat"],
                        get_radius=340,
                        get_fill_color=[22, 163, 74, 245],
                        get_line_color=[255, 255, 255, 255],
                        stroked=True,
                        line_width_min_pixels=2,
                        pickable=True,
                    )
                )

            view_state = pdk.ViewState(
                latitude=13.025,
                longitude=80.200,
                zoom=10.6,
                pitch=0,
            )

            st.pydeck_chart(
                pdk.Deck(
                    map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
                    initial_view_state=view_state,
                    layers=layers,
                    tooltip={"text": "{name}\nStatus: {status}\nServed: {orders_served} orders"},
                )
            )
            st.caption(
                "🟢 **Green Large Dots:** Opened Dark Stores (with service radius) | "
                "🔴 **Red Medium Dots:** Closed Candidate Dark Stores | "
                "🔵 **Blue Small Dots:** Served Customers | "
                "⚪ **Gray Small Dots:** Unserved Customers (outside $D_{\\max}$ or capacity limit)"
            )

# -----------------------------------------------------------------------------
# TAB 2: 10,000 CUSTOMERS & 50 DARK STORES SENSITIVITY PLOT
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("📈 Profit vs. Maximum Allowed Dark Stores (10,000 Customers & 50 Candidate Stores)")
    st.markdown(
        f"""
        Evaluates the optimization model on **10,000 customers** and **50 candidate dark stores** in Chennai 
        with a maximum service distance of **{max_dist_input:.1f} km** when at most **$K \\in \\{{2, 5, 10, 20, 25\\}}$** 
        dark stores are allowed to open ($\sum_{{j \\in J}} y_j \\le K$).
        """
    )

    run_benchmark = st.button("🚀 Run 10,000-Customer Sensitivity Analysis (K = 2, 5, 10, 20, 25)", type="primary")

    if run_benchmark:
        k_values = [2, 5, 10, 20, 25]
        benchmark_rows = []
        progress_bar = st.progress(0, text="Generating 10,000 Chennai customers and 50 candidate stores...")

        cust_10k, stores_50 = generate_chennai_data(10000, 50, int(random_seed))

        for idx, k in enumerate(k_values):
            progress_bar.progress(
                int((idx / len(k_values)) * 100),
                text=f"Solving MILP for K <= {k} dark stores (10,000 customers, 50 candidates)...",
            )
            res_k = solve_dark_store_milp(
                customers_df=cust_10k,
                stores_df=stores_50,
                max_dist_km=float(max_dist_input),
                revenue_per_order=float(revenue_per_order),
                cost_per_km=float(cost_per_km),
                max_open_stores=k,
                time_limit_sec=10,
            )
            if res_k:
                benchmark_rows.append({
                    "Max Allowed Stores (K)": k,
                    "Actual Stores Opened": res_k["stores_Opened_count"],
                    "Total Profit ($)": round(res_k["profit"], 2),
                    "Total Revenue ($)": round(res_k["revenue"], 2),
                    "Fixed Cost ($)": round(res_k["fixed_cost"], 2),
                    "Delivery Cost ($)": round(res_k["delivery_cost"], 2),
                    "Demand Met (%)": round(res_k["demand_met_pct"], 2),
                })

        progress_bar.progress(100, text="Sensitivity analysis complete!")
        bench_df = pd.DataFrame(benchmark_rows)
        st.session_state["bench_df"] = bench_df

    if "bench_df" in st.session_state:
        bench_df = st.session_state["bench_df"]

        col_chart, col_data = st.columns([1.4, 1])

        with col_chart:
            fig = px.line(
                bench_df,
                x="Max Allowed Stores (K)",
                y="Total Profit ($)",
                markers=True,
                text="Total Profit ($)",
                title=f"Daily Profit vs. Max Allowed Dark Stores (10,000 Customers, 50 Candidates, D_max={max_dist_input} km)",
            )
            fig.update_traces(
                texttemplate="$%{text:,.0f}",
                textposition="top center",
                line=dict(width=3, color="#2563eb"),
                marker=dict(size=10, color="#1d4ed8"),
            )
            fig.update_layout(
                xaxis=dict(tickmode="array", tickvals=[2, 5, 10, 20, 25]),
                yaxis_title="Total Daily Profit ($)",
                xaxis_title="Max Allowed Dark Stores to Open (K)",
                height=450,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_data:
            st.markdown("#### Sensitivity Summary Table")
            st.dataframe(bench_df, use_container_width=True, hide_index=True)
            st.info(
                "**Observation:** As $K$ increases from 2 to 20, total daily profit rises sharply as high-density "
                "clusters in Chennai are captured. Beyond a threshold (e.g., $K=20$ to $25$), diminishing returns "
                "set in: the solver only opens a new store if the incremental margin from uncovered customers "
                "exceeds the store's fixed daily opening cost."
            )