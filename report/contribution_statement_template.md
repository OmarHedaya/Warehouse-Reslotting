# Individual Contribution Statement & Git Log Instructions

> **Instructions**: Fill in each student's name and student ID below. Keep each individual statement under the mandatory **200-word maximum**. At the bottom, run the provided `git log` command to attach the repository commit history.

---

## Individual Contribution Statements (Max 200 Words Each)

### Student 1: Omar Hedaya (Student ID: [ID Number])
I led the data engineering, neural network demand forecasting, and experimental evaluation components of Project P8. Specifically, I inspected and cleaned the local UCI Online Retail II dataset (`data/make_data.py`), handling the two-sheet Excel structure (`1,067,371` raw rows), filtering cancelled invoices and invalid stock codes, selecting the top 120 products by volume, and building the chronological train/validation/test split (`72 / 15 / 10` weeks). To prevent data leakage, I extracted the top 30 co-purchased product pairs strictly from the training weeks and authored the complete data dictionary (`data/data_dictionary.md`). In `nn/demand_model.py`, I implemented the 8-week log-demand feature pipeline, the Last-Week and 4-Week Moving Average baselines, and the Multi-Layer Perceptron (`MLPRegressor`) with training-only feature scaling. Finally, I built `eval/run_experiments.py` to execute all multi-seed experiments across 5 fixed seeds (`42–46`), generating the statistical summary tables (Mean ± Std) and plots in `results/`, and co-wrote the technical report and limitations analysis. *(162 words)*

---

### Student 2: [Full Name] (Student ID: [ID Number])
*(If this is a solo project, delete this section and note that Student 1 completed 100% of the project modules.)*  
I led the Constraint Satisfaction Problem (CSP) solver, heuristic swap search algorithms, system integration simulator, automated test suite, and live demonstration script for Project P8. In `csp/warehouse_csp.py`, I formulated the 120-product shelf assignment problem with four hard constraints (one product per slot, heavy items on low shelves, top-30 co-purchase maximum distance, and aisle incompatibility) and implemented switchable Backtracking (`BT`), Forward Checking (`FC`), and Arc Consistency (`AC-3`) along with step-by-step distance relaxation. In `search/swap_search.py`, I implemented Greedy Best-First Search, Bounded Uniform Cost Search (`UCS`), and the `A*` variant over pairwise swaps among the top 30 high-demand products and near-depot slots, rechecking every swap with the CSP verifier and verifying correctness on a 4-product known-answer example. Finally, I built the real-order picker walking simulator (`integration/`), the 10-case test suite (`tests/test_all.py`), and the live demonstration script (`run_demo.py`). *(153 words)*

---

## How to Generate and Attach the Git Log

Run the following command in the project root directory to print the formatted git commit history for submission:

```powershell
git log --stat --oneline --decorate
```

Or to export the detailed git log directly into a text file for appendix submission:

```powershell
git log --pretty=format:"%h | %ad | %an | %s" --date=short --stat > report/git_log.txt
```
