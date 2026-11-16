# 10-Minute Presentation Outline – P8 Warehouse Re-Slotting from Predicted Demand

**Total Time**: 10 Minutes (10 Slides, ~1 Minute per Slide)  
**Target Audience**: Course Instructor & Teaching Assistants

---

## Slide 1: Title & Project Overview (0:00 – 0:45)
- **Title**: P8 – Warehouse Re-Slotting from Predicted Demand (Search + CSP + Neural Network)
- **One-Sentence Summary**: Every week, our system predicts how many units each product will sell, builds a safe initial shelf layout obeying warehouse rules, and refines that layout with up to 6 pairwise swaps so human pickers walk less when fulfilling orders.
- **Real-World Impact**: Order picking accounts for over 50% of warehouse operating labor; placing high-demand and co-purchased items closer to the packing depot directly reduces walking distance.

## Slide 2: Dataset & Preprocessing (0:45 – 1:45)
- **Dataset**: UCI Online Retail II (`data/online_retail_II.xlsx`), containing **1,067,371** real UK retail transactions from 1 Dec 2009 to 9 Dec 2011 across two Excel sheets.
- **Cleaning Pipeline (`data/make_data.py`)**:
  - Removed cancelled invoices (starting with `'C'`), non-positive quantities/prices, and postage/accounting codes (`POST`, `DOT`, `M`, `BANK CHARGES`), leaving **1,037,361** valid transactions.
  - Selected the **top 120 products** by total volume (**173,404** transactions) across **105** weeks.
  - **Chronological Split**: First 8 weeks = lag history; next **72 weeks = Train**; next **15 weeks = Validation**; final **10 weeks = Test** (**4,318** real customer invoices).
- **Real vs. Simulated**: Sales, prices, co-purchase pairs (top 30 from training weeks only), and test orders are **100% real**; the 120-slot warehouse grid (`6 aisles × 10 bays × 2 shelf levels`), heavy flags (`20%`), and incompatible categories (`Cleaning_Chemical` vs. `Food_Kitchen`) are deterministically simulated (`seed=42`).

## Slide 3: How the Three AI Paradigms Cooperate (1:45 – 2:45)
- **Step 1 – Neural Network (`nn/demand_model.py`)**: Predicts next-week demand for all 120 products using the past 8 weeks of $\ln(1 + \text{units})$, `week_of_year`, `mean_price`, and $\ln(1 + \text{invoice\_count})$.
- **Step 2 – Constraint Satisfaction Problem (`csp/warehouse_csp.py`)**: Assigns products (ordered by predicted demand) to warehouse zones near the depot while satisfying all hard physical rules.
- **Step 3 – Heuristic Swap Search (`search/swap_search.py`)**: Applies up to 6 pairwise swaps among the 30 highest-demand products and near-depot slots, calling the **CSP verifier on every candidate swap**.
- **Step 4 – Real Order Simulator (`integration/simulator.py`)**: Measures exact walking distance of a human picker across actual held-out customer invoices.

## Slide 4: Neural Network vs. Retail Baselines (2:45 – 4:00)
- **Architecture**: Simple 2-hidden-layer MLP (`32 -> 16` ReLU neurons, Adam optimizer, early stopping, `StandardScaler` fit strictly on training weeks).
- **Baselines**:
  1. **Last-Week Demand**: $\hat{y}_t = y_{t-1}$
  2. **4-Week Moving Average**: $\hat{y}_t = \frac{1}{4}\sum_{k=1}^{4} y_{t-k}$
- **Key Findings (5 Seeds)**:
  - Validation MAE: **NN (`132.49 ± 0.74`)** beats 4-Week Avg (`146.89`) and Last-Week (`176.24`).
  - Test MAE: **NN (`245.28 ± 3.75`)** beats 4-Week Avg (`250.96`) and Last-Week (`283.73`).
  - Honest Insight: On Test RMSE, 4-Week Avg (`2368.30`) slightly edges out NN (`2373.15`) due to extreme Christmas wholesale spikes in Nov–Dec 2011.

## Slide 5: CSP Formulation, Pruning & Distance Relaxation (4:00 – 5:30)
- **Variables**: 120 Products; **Domains**: 120 Shelf Slots (`S000`–`S119`).
- **Four Hard Constraints**:
  1. **One product per slot** (`AllDifferent` / capacity = 1).
  2. **Heavy items on low shelves** (`is_heavy == 1` requires `shelf_level == 0`).
  3. **Top 30 training co-purchase pairs within `max_distance`**.
  4. **Incompatible categories (`Cleaning_Chemical` vs. `Food_Kitchen`) not in the same aisle**.
- **Switchable Solver Comparison**:
  - **Pure BT**: Tries occupied and upper-shelf slots blindly (`4,005` backtracks on 90 products).
  - **BT + Forward Checking (FC)**: Prunes occupied, distance-violating, and same-aisle slots (`71.6` backtracks on 90 products).
  - **BT + FC + AC-3**: Enforces node + arc consistency upfront and across 2-hop chains (**`0` backtracks**).
- **Constraint Relaxation**: When `initial_max_distance = 2` has no solution, the solver relaxes step by step (`2 -> 4 -> 6`) and reports `loosest_distance_used = 6`.

## Slide 6: Re-Slotting Swap Search (5:30 – 7:00)
- **Why Search is Needed After CSP**: To avoid running out of low shelves for heavy items, the CSP places upper-shelf light items first in each zone and reserves front low shelves (`S000`) for heavy items—even if a heavy item has moderate demand. Search swaps top-selling products into the very closest bays while preserving all CSP rules.
- **State, Action, Cost, Goal**:
  - **State**: Complete valid layout.
  - **Action**: Swap 2 products among top 30 demand products & 30 closest slots (with CSP recheck).
  - **Cost**: `1` per swap; **Goal**: Reduce walking cost by target threshold or reach `6` swaps.
- **Algorithms Compared (Experiment 1)**:
  - **Greedy Best-First (Main)**: Saves **`9,514.5 ± 943.1`** cost in `4.4` swaps (`5.4` nodes expanded).
  - **Bounded UCS (Baseline)**: Expands uniformly by $g(n)$ (`35.0` nodes expanded), saving `6,579.5`.
  - **A\* Variant**: Balances $g(n) + h(n)$, saving `8,574.3` in fewer swaps (`3.4` swaps), though not guaranteed optimal because a single high-volume swap can exceed the average heuristic step scale.

## Slide 7: Experiment Results – CSP Scaling & Search (7:00 – 8:00)
- Walk through `results/exp1_search_comparison.png` and `results/exp2_csp_scaling.png`.
- Highlight how `BT+FC+AC3` scales cleanly with `0` backtracks at 60 and 90 products (`7.0` at 120 products), whereas pure `BT` grows from `1,803` to `22,104` backtracks.

## Slide 8: Experiment 3 – Four Layouts Over 10 Real Test Weeks (8:00 – 9:00)
- Walk through `results/exp3_four_layouts_10weeks.png` (**4,318 real held-out invoices**):
  1. **Random Valid Layout**: `166,658.6 ± 786.3` m
  2. **Last-Week Demand (NN OFF)**: `142,308.0 ± 0.0` m
  3. **NN Demand (NN ON)**: `139,350.2 ± 1,003.7` m
  4. **NN + Swaps (Full P8 System)**: **`138,085.2 ± 1,113.3` m** (saves **28.6 km** vs. Random and **4.2 km** vs. Last-Week!).

## Slide 9: Honest Limitations (9:00 – 9:30)
1. **Strong Retail Baselines**: In Week 100, Last-Week demand slightly beats the unswapped NN layout (`13,650 m` vs. `13,869.6 m`) because weekly retail demand is highly autocorrelated.
2. **Christmas Seasonality Spike**: The 10 test weeks fall in Oct–Dec 2011 (peak holiday shopping), causing order volume surges where 4-week averaging achieves slightly lower RMSE.
3. **Non-Admissible A\* Heuristic**: Scaling distance gap into estimated swaps is fast and effective, but can overestimate when one swap moves a 900-unit bestseller.

## Slide 10: Conclusions & Transition to 5-Minute Live Demo (9:30 – 10:00)
- Recap how **NN (prediction)** + **CSP (safety & feasibility)** + **Search (optimization)** solve a realistic warehouse planning problem better together than any single method alone.
- Transition directly to `python run_demo.py`.
