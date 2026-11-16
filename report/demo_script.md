# 5-Minute Live Demo Script – P8 Warehouse Re-Slotting from Predicted Demand

**Command to Launch Demo**:
```powershell
.\.venv\Scripts\python.exe run_demo.py
```

---

## Minute 0:00 – 1:00 | Run Demo & Show Input + NN vs. Baselines
1. **Action**: Run `.\.venv\Scripts\python.exe run_demo.py` in the terminal and scroll to **Section 1 (`1. INPUT DATA & NEURAL NETWORK DEMAND FORECAST VS. BASELINES`)**.
2. **What to Say**:
   - *"Here our system loads the cleaned UCI Online Retail II dataset: the top 120 products across 105 weeks, 120 warehouse slots, and the top 30 co-purchase pairs extracted strictly from the 72 training weeks."*
   - *"Our 2-layer MLP neural network predicts next-week demand from the last 8 weeks of log-sales, week of year, price, and invoice count. As shown in the table, the NN achieves the lowest MAE on both Validation (`133.41`) and Test (`248.32`), beating both the Last-Week (`283.73`) and 4-Week Average (`250.96`) baselines."*

---

## Minute 1:00 – 2:30 | CSP Pruning (BT vs. FC vs. AC-3) & Distance Relaxation
1. **Action**: Point to **Section 2 (`2. CSP SOLVER: PRUNING WITH & WITHOUT FC / AC-3 + DISTANCE RELAXATION`)** in the terminal output.
2. **What to Say**:
   - *"Next, the CSP assigns products to shelf slots subject to 4 hard rules: one product per slot, heavy items on low shelves (`level 0`), top 30 co-bought pairs within `max_distance`, and incompatible categories (`Cleaning_Chemical` vs. `Food_Kitchen`) in different aisles."*
   - *"Look at the Solver Switch table on 90 products:"*
     - *"With **Pure Backtracking (BT)**, the solver blindly tries occupied and upper-level shelves, causing **4,005 backtracks** and `0` pruned values."*
     - *"Turning on **Forward Checking (BT + FC)** prunes **5,767** conflicting slot values ahead of time and drops backtracks down to just **57**."*
     - *"Turning on **AC-3 (BT + FC + AC-3)** enforces node and arc consistency over the co-purchase and incompatibility arcs before and during search, pruning **6,180** values and solving the layout in **90 assignments with 0 backtracks**."*
   - *"Below that is the **Constraint Relaxation Demo**: when we start with an impossibly tight co-purchase distance of `2` meters, the solver detects infeasibility, relaxes the distance rule step-by-step (`2 -> 4 -> 6`), and reports `6 meters` as the loosest distance value used."*

---

## Minute 2:30 – 3:45 | Search Trace (Greedy vs. UCS vs. A*)
1. **Action**: Point to **Section 3 (`3. SEARCH TRACE: RE-SLOTTING SWAPS WITH CSP RECHECK ON EVERY SWAP`)** in the terminal output.
2. **What to Say**:
   - *"First, notice the 4-product known-answer test at the top: Greedy Search swaps Product P1 (demand 100) from far slot S3 into closest slot S0, dropping walking cost from `2370.0` to the exact optimal `660.0` in 1 swap."*
   - *"On the 120-product warehouse, **Greedy Best-First Search** expands only `6` nodes (`41` generated) and performs `5` swaps, reducing demand-weighted walking distance from `432,063.6` to `421,882.9`. Notice that every single swap prints `CSP Recheck: PASS`."*

---

## Minute 3:45 – 5:00 | System With NN ON vs. NN OFF & Cooperation Plot
1. **Action**: Point to **Section 4 (`4. INTEGRATED SYSTEM: NN ON VS. NN OFF ON REAL HELD-OUT ORDERS`)**, open `results/demo_cooperation.png`, and optionally run `.\.venv\Scripts\python.exe tests/test_all.py`.
2. **What to Say**:
   - *"Finally, we test all four layouts on **371 real held-out customer invoices** in Test Week 96:"*
     - *"1. Random valid layout walks **14,999.0 meters**."*
     - *"2. Turning **NN OFF** (using Last-Week demand in the CSP) walks **13,029.0 meters**."*
     - *"3. Turning **NN ON** (CSP with NN demand) reduces walking to **12,877.0 meters**."*
     - *"4. Adding **Search Swaps (Full P8 System)** reduces walking further to **12,756.0 meters**, saving **2,243 meters** in a single week while obeying every warehouse constraint."*
   - *"Opening `results/demo_cooperation.png` shows all four stages visually side by side, and running `tests/test_all.py` verifies our 5 normal, 3 edge, and 2 impossible test cases."*
