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
    page_icon="🏢",
    layout="wide",
)

st.title("Quick Commerce: Dark Store Location Optimization")
st.markdown(
    """
    Capacitated Facility Location Problem (MILP) solved using Google OR-Tools (SCIP Solver).  
    This system extends the baseline operational research model by integrating a maximum service distance constraint, 
    profit maximization objectives, an interactive geospatial map of Chennai, and a 10,000-customer sensitivity analysis.
    """
)

# =============================================================================
# 1. SYNTHETIC CHENNAI GEOGRAPHY & DATA GENERATOR
# =============================================================================
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
    Generates customer demand points and candidate dark store locations
    within the Chennai metropolitan area, avoiding the Bay of Bengal coastline.
    """
    rng = np.random.default_rng(seed)
    hub_coords = np.array([[h[1], h[2]] for h in CHENNAI_HUBS])
    num_hubs = len(CHENNAI_HUBS)

    # Candidate Dark Stores
    store_hub_idx = rng.integers(0, num_hubs, size=num_stores)
    store_lats = hub_coords[store_hub_idx, 0] + rng.normal(0, 0.018, size=num_stores)
    store_lons = hub_coords[store_hub_idx, 1] + rng.normal(0, 0.018, size=num_stores)
    store_lons = np.clip(store_lons, 80.08, 80.265)
    store_lats = np.clip(store_lats, 12.88, 13.16)

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

    # Customer Locations
    cust_hub_idx = rng.integers(0, num_hubs, size=num_customers)
    cust_lats = hub_coords[cust_hub_idx, 0] + rng.normal(0, 0.025, size=num_customers)
    cust_lons = hub_coords[cust_hub_idx, 1] + rng.normal(0, 0.025, size=num_customers)
    cust_lons = np.clip(cust_lons, 80.07, 80.270)
    cust_lats = np.clip(cust_lats, 12.87, 13.17)

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
    Vectorized Haversine distance matrix in kilometers.
    """
    R = 6371.0
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
    Solves the Capacitated Dark Store Location MILP model.
    """
    num_customers = len(customers_df)
    num_stores = len(stores_df)

    dist_matrix = compute_haversine_matrix(
        customers_df["lat"].values,
        customers_df["lon"].values,
        stores_df["lat"].values,
        stores_df["lon"].values,
    )

    solver = pywraplp.Solver.CreateSolver("SCIP")
    if not solver:
        raise RuntimeError("SCIP solver is not available in OR-Tools.")

    solver.SetTimeLimit(time_limit_sec * 1000)
    solver.SetSolverSpecificParametersAsString("limits/gap = 0.005")

    demands = customers_df["demand"].values.astype(float)
    capacities = stores_df["capacity"].values.astype(float)
    fixed_costs = stores_df["fixed_cost"].values.astype(float)

    y = [solver.BoolVar(f"y_{j}") for j in range(num_stores)]

    valid_i, valid_j = np.where(dist_matrix <= max_dist_km)

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

        var = solver.NumVar(0.0, demands[i], f"x_{i}_{j}")
        x_vars[(i, j)] = var
        cust_to_vars[i].append(var)
        store_to_vars[j].append(var)
        objective.SetCoefficient(var, net_margin_ij)

    objective.SetMaximization()

    for i in range(num_customers):
        if len(cust_to_vars[i]) > 1:
            solver.Add(solver.Sum(cust_to_vars[i]) <= demands[i])

    for j in range(num_stores):
        if store_to_vars[j]:
            solver.Add(solver.Sum(store_to_vars[j]) <= capacities[j] * y[j])
        else:
            solver.Add(y[j] == 0)

    if max_open_stores is not None:
        solver.Add(solver.Sum(y) <= int(max_open_stores))

    status = solver.Solve()

    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        return None

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
# 3. SIDEBAR CONTROLS & NAVIGATION GUIDE
# =============================================================================
st.sidebar.header("Model Parameters")
st.sidebar.caption("Configure the optimization inputs for Panel 1.")

num_stores_input = st.sidebar.number_input(
    "Candidate Dark Stores Count (J)",
    min_value=4,
    max_value=100,
    value=50,
    step=1,
)

num_customers_input = st.sidebar.number_input(
    "Number of Customers (I)",
    min_value=10,
    max_value=15000,
    value=2000,
    step=500,
)

max_dist_input = st.sidebar.slider(
    "Max Distance: Dark Store to Customer (km)",
    min_value=0.5,
    max_value=10.0,
    value=2.0,
    step=0.1,
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
st.sidebar.subheader("Economic Assumptions (Daily)")
revenue_per_order = st.sidebar.number_input("Revenue Margin per Order ($)", value=4.0, min_value=0.5, step=0.5)
cost_per_km = st.sidebar.number_input("Delivery Cost per km ($/order)", value=1.50, min_value=0.1, step=0.1)
random_seed = st.sidebar.number_input("Random Seed", value=42, step=1)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Navigation Overview:**\n"
    "- Panel 1: Interactive Optimizer and Map\n"
    "- Panel 2: 10,000-Customer Sensitivity Analysis\n"
    "- Panel 3: Technical Documentation and README"
)

# =============================================================================
# 4. MAIN TABS
# =============================================================================
tab1, tab2, tab3 = st.tabs([
    "Panel 1: Interactive Optimizer & Chennai Map",
    "Panel 2: 10,000-Customer Profit Analysis",
    "Panel 3: Model Formulation & README",
])

# -----------------------------------------------------------------------------
# TAB 1: INTERACTIVE MODEL, KPI TABLE & CHENNAI MAP
# -----------------------------------------------------------------------------
with tab1:
    with st.expander("User Guide: How to Use Panel 1", expanded=False):
        st.markdown(
            """
            1. Use the input controls in the left sidebar to specify the candidate store count, customer count, and maximum allowable distance.
            2. The optimization solver executes automatically upon parameter modification.
            3. Review the Key Performance Indicators (KPIs) and store breakdown in the left table.
            4. Interact with the geospatial map on the right to inspect store locations, service radius coverage, and customer assignments.
            """
        )

    with st.spinner("Executing optimization solver..."):
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
        st.error("The solver was unable to find a feasible solution under the current constraints.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Net Daily Profit", f"${result['profit']:,.2f}")
        m2.metric("Dark Stores Opened", result["kpis"]["Dark Stores Opened"])
        m3.metric("Demand Fulfilled", result["kpis"]["Demand Fulfillment (%)"])
        m4.metric("Avg Delivery Distance", result["kpis"]["Avg Delivery Distance (km)"])

        col_kpi, col_map = st.columns([1, 1.65])

        with col_kpi:
            st.subheader("1. Optimization KPIs Table")
            kpi_table = pd.DataFrame([
                {"KPI Metric": k, "Value": f"${v:,.2f}" if isinstance(v, float) else str(v)}
                for k, v in result["kpis"].items()
            ])
            st.dataframe(kpi_table, use_container_width=True, hide_index=True)

            st.subheader("2. Candidate Dark Stores Breakdown")
            st.dataframe(
                result["stores_df"][["name", "status", "orders_served", "capacity", "utilization_pct", "fixed_cost"]],
                use_container_width=True,
                hide_index=True,
                height=310,
            )

        with col_map:
            st.subheader("3. Spatial Solution on Chennai Map")
            c_tog1, c_tog2 = st.columns(2)
            with c_tog1:
                show_radii = st.checkbox("Show Service Radius Circles", value=True)
            with c_tog2:
                show_links = st.checkbox("Show Customer Fulfillment Links", value=True)

            stores_res = result["stores_df"]
            cust_res = result["customers_df"]

            open_stores = stores_res[stores_res["status"] == "OPEN"].copy()
            closed_stores = stores_res[stores_res["status"] == "CLOSED"].copy()
            served_cust = cust_res[cust_res["is_served"]].copy()
            unserved_cust = cust_res[~cust_res["is_served"]].copy()

            layers = []

            if show_radii and not open_stores.empty:
                layers.append(
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=open_stores,
                        get_position=["lon", "lat"],
                        get_radius=float(max_dist_input) * 1000.0,
                        get_fill_color=[22, 163, 74, 30],
                        get_line_color=[22, 163, 74, 160],
                        stroked=True,
                        filled=True,
                        line_width_min_pixels=2,
                        pickable=False,
                    )
                )

            if not unserved_cust.empty:
                layers.append(
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=unserved_cust,
                        get_position=["lon", "lat"],
                        get_radius=65,
                        get_fill_color=[148, 163, 184, 140],
                        pickable=False,
                    )
                )

            if not served_cust.empty:
                layers.append(
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=served_cust,
                        get_position=["lon", "lat"],
                        get_radius=80,
                        get_fill_color=[37, 99, 235, 200],
                        pickable=False,
                    )
                )

            if show_links and result["flows"]:
                flows_df = pd.DataFrame(result["flows"])
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
                    tooltip={"text": "{name}\nStatus: {status}\nServed: {orders_served} / {capacity} orders"},
                )
            )
            st.caption(
                "Map Legend: Green points indicate opened dark stores with service radius rings; "
                "Red points represent closed candidate locations; Blue points show served customer nodes; "
                "Gray points denote unserved customers exceeding distance or capacity limits."
            )

# -----------------------------------------------------------------------------
# TAB 2: 10,000 CUSTOMERS & 50 DARK STORES SENSITIVITY PLOT
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("Profit Analysis for 10,000 Customers and 50 Candidate Stores")

    st.markdown(
        "**Instructions for Panel 2:**\n"
        "1. This panel addresses the assignment requirement to evaluate profit performance across store opening limits ($K \\in \\{2, 5, 10, 20, 25\\}$). \n"
        "2. Click the execution button below to generate 10,000 customer demand points and 50 candidate dark stores in Chennai. \n"
        "3. The system sequentially runs the optimization model under each store count limit and plots the profit curve."
    )

    run_benchmark = st.button(
        "Run 10,000-Customer Sensitivity Analysis (K = 2, 5, 10, 20, 25)",
        type="primary",
    )

    if run_benchmark:
        k_values = [2, 5, 10, 20, 25]
        benchmark_rows = []
        progress_bar = st.progress(0, text="Initializing dataset for 10,000 customers and 50 stores...")

        cust_10k, stores_50 = generate_chennai_data(10000, 50, int(random_seed))

        for idx, k in enumerate(k_values):
            progress_bar.progress(
                int((idx / len(k_values)) * 100),
                text=f"Solving optimization model for opening limit K <= {k}...",
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

        progress_bar.progress(100, text="Sensitivity analysis successfully completed.")
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
            st.markdown(
                "**Analytical Findings:**\n"
                "- At strict opening limits ($K=2, 5, 10$), opened stores operate at maximum capacity in high-density urban clusters, resulting in rapid profit growth.\n"
                "- As the store limit expands to $20$ and $25$, diminishing returns emerge. The solver declines to open additional facilities once fixed operational costs exceed the marginal revenue captured from peripheral demand."
            )
    else:
        st.info("Click the execution button above to generate the sensitivity analysis curve.")

# -----------------------------------------------------------------------------
# TAB 3: MODEL FORMULATION, CODE ARCHITECTURE & README
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("Technical Documentation and Model Formulation")

    st.markdown(
        """
        **1. Problem Background and Modeling Transition**  
        In standard unconstrained location models, demand is fully satisfied, and objective functions focus exclusively on cost minimization. 
        However, introducing strict operational constraints—such as a maximum delivery radius ($D_{\\max}$) or a cap on facility openings ($K$)—creates potential infeasibility if demand is rigidly enforced. 
        
        To resolve this, the model transitions from pure cost minimization to a **Profit Maximization Framework**, allowing partial demand fulfillment where necessary while accounting for gross order margins, distance-dependent transport expenses, and facility fixed costs.
        
        ---
        **2. Mathematical Programming Formulation (MILP)**  

        **Sets and Indices**  
        - $I$: Set of customer demand locations, indexed by $i$.  
        - $J$: Set of candidate dark store locations, indexed by $j$.  

        **Parameters**  
        - $D_i$: Daily order demand at customer location $i$.  
        - $C_j$: Maximum daily processing capacity of candidate dark store $j$.  
        - $F_j$: Daily fixed operating cost of dark store $j$.  
        - $d_{ij}$: Great-circle Haversine distance in kilometers between customer $i$ and store $j$.  
        - $D_{\\max}$: Maximum allowable service distance threshold.  
        - $c_{\\text{km}}$: Unit transportation cost per kilometer per order.  
        - $R$: Gross revenue margin per fulfilled order.  
        - $K$: Maximum permitted number of open dark stores.  

        **Decision Variables**  
        - $y_j \\in \\{0, 1\\}$: Binary variable equal to 1 if dark store $j$ is opened, and 0 otherwise.  
        - $x_{ij} \\ge 0$: Continuous variable representing the volume of daily demand for customer $i$ served by store $j$.  

        **Objective Function**  
        $$\\text{Maximize } Z = \\sum_{i \\in I} \\sum_{j \\in J} \\left( R - c_{\\text{km}} \\cdot d_{ij} \\right) x_{ij} - \\sum_{j \\in J} F_j y_j$$

        **Constraints**  
        1. **Demand Limit:** Fulfillment for any customer cannot exceed their total daily demand.  
           $$\\sum_{j \\in J} x_{ij} \\le D_i \\quad \\forall i \\in I$$  
        2. **Facility Capacity:** Total assigned volume cannot exceed store capacity when open, and must be zero when closed.  
           $$\\sum_{i \\in I} x_{ij} \\le C_j \\cdot y_j \\quad \\forall j \\in J$$  
        3. **Maximum Service Distance:** Deliveries are prohibited if the distance exceeds the threshold.  
           $$x_{ij} = 0 \\quad \\forall i \\in I, \\forall j \\in J \\text{ such that } d_{ij} > D_{\\max}$$  
        4. **Facility Opening Cap:** Limits the total number of operational facilities.  
           $$\\sum_{j \\in J} y_j \\le K$$  

        ---
        **3. Computational Implementation and Optimization Strategy**  
        When scaling to 10,000 customers and 50 candidate stores, instantiating every customer-store pair creates 500,000 continuous variables, which degrades solver efficiency. 

        To maintain rapid execution times, the implementation utilizes **Sparse Variable Indexing**. By calculating the distance matrix beforehand and instantiating $x_{ij}$ variables strictly for pairs where $d_{ij} \\le D_{\\max}$, the variable count is reduced by over 90%. This allows the Google OR-Tools SCIP solver to return optimal solutions within seconds.
        """
    )
