"""
Main Analysis: Complete DRO-Wasserstein Portfolio Optimization on COLCAP Data
==============================================================================

This script demonstrates the complete pipeline for implementing and analyzing
Distributionally Robust Optimization (DRO) portfolio optimization on Colombian
COLCAP market data.

Includes:
1. Data loading and preprocessing
2. Parameter calibration
3. Model solving (2-WDRO-Markowitz, 1-WDRO-CVaR, 2-WDRO-CVaR)
4. Benchmark comparison (SAA, MinVar, Equal Weight)
5. Out-of-sample evaluation
6. Results visualization and interpretation

Author: Doctoral Researcher in DRO
Date: 2026
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from dro_colcap_loader import COLCAPDataLoader
from dro_gurobi_solvers import (
    DROWassersteinMarkowitz,
    DROWassersteinCVaR_1,
    DROWassersteinCVaR_2,
    SAA_Markowitz,
    MinimumVariance,
    EqualWeight,
    compute_mu_max,
    compute_eps_max,
)
from dro_evaluation import PortfolioEvaluator, PortfolioComparison, RollingWindowBacktest


# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Configuration parameters for DRO analysis."""
    
    # Data configuration (use imputed data with aligned trading days: 3200 periods, 0% missing)
    DATA_PATH = r"c:\Users\emngz\Downloads\Universidad\P.I,\PI 2\Datos_empresas_colcap_IMPUTADOS"
    
    # Portfolio constraints
    # None → auto-calibrated as mu_max * 0.30 inside calibrate_wasserstein_radius()
    TARGET_RETURN = None
    SHORT_SELLING = False  # Long-only portfolio
    
    # Wasserstein radius calibration
    EPSILON_FRACTIONS = [1.0, 0.75, 0.5]  # Test multiple radius fractions
    FIND_EPSILON_MAX = True  # Automatically find feasibility limit
    
    # CVaR parameters
    CVaR_CONFIDENCE = 0.05  # 95% confidence level
    
    # Backtesting configuration
    ROLLING_WINDOW_SIZE = 126  # ~6 months of trading
    TEST_PERIOD_SIZE = 1       # Daily rebalancing
    
    # Output
    RESULTS_DIR = "dro_results"
    VERBOSE = True


# ============================================================================
# Wasserstein Radius Calibration
# ============================================================================

def compute_empirical_wasserstein_radius(returns: np.ndarray) -> float:
    """
    Compute data-driven Wasserstein radius estimate.
    
    Using the approach from Blanchet & Murthy (2019):
    ε = c * sqrt(d/n) * (1 + log n)
    
    where d is dimension, n is sample size, c is calibration constant
    
    Parameters
    ----------
    returns : np.ndarray
        Return matrix (T, m)
    
    Returns
    -------
    float
        Estimated Wasserstein radius
    """
    T, m = returns.shape
    
    # Conservative estimate
    log_term = np.log(T) if T > 1 else 1.0
    radius = 0.1 * np.sqrt(m / T) * (1 + log_term)
    
    return radius


# ============================================================================
# Main Analysis Functions
# ============================================================================

def load_and_prepare_data(config: Config) -> np.ndarray:
    """Load COLCAP data and prepare returns matrix."""
    print("\n" + "="*80)
    print("STEP 1: DATA LOADING AND PREPARATION")
    print("="*80)
    
    loader = COLCAPDataLoader(config.DATA_PATH)
    loader.load_all_csv()
    loader.summary_statistics()
    
    return loader.get_returns()


def calibrate_wasserstein_radius(returns: np.ndarray, 
                                target_return: float) -> Dict:
    """
    Calibrate appropriate Wasserstein radius using SOCP computation.
    
    Workflow:
    1. Compute mu_max = maximum expected return over empirical distribution
    2. Compute eps_max = maximum feasible Wasserstein radius using SOCP
    3. Generate radius schedule as fractions of eps_max
    
    Parameters
    ----------
    returns : np.ndarray
        Return matrix (T, m)
    target_return : float
        Minimum target return μ
    
    Returns
    -------
    dict
        Dictionary with radii schedule and calibration info
    """
    print("\n" + "="*80)
    print("STEP 2: WASSERSTEIN RADIUS CALIBRATION")
    print("="*80)
    
    # Step 1: Compute mu_max
    print("\n  Computing mu_max (maximum expected return)...")
    try:
        mu_max = compute_mu_max(returns)
        print(f"  ✓ mu_max = {mu_max:.6f}")
    except Exception as e:
        print(f"  ✗ Error computing mu_max: {e}")
        raise

    # Auto-calibrate target_return if not set (must be < mu_max for feasibility with ε > 0)
    if target_return is None:
        target_return = mu_max * 0.30
        print(f"  ✓ target_return auto-calibrated = {target_return:.6f}  (30% of mu_max)")
    elif target_return >= mu_max:
        raise ValueError(
            f"target_return={target_return:.6f} ≥ mu_max={mu_max:.6f}. "
            "No portfolio can satisfy the robust return constraint for any ε > 0."
        )
    else:
        print(f"  ✓ target_return = {target_return:.6f}")

    # Step 2: Compute eps_max via closed-form formula (Fonseca & Junca, 2021)
    print("\n  Computing eps_max (maximum Wasserstein radius)...")
    try:
        eps_max = compute_eps_max(returns, target_return)
        print(f"  ✓ eps_max = {eps_max:.6f}")
    except Exception as e:
        print(f"  ✗ Error computing eps_max: {e}")
        raise
    
    # Empirical estimate for reference
    epsilon_est = compute_empirical_wasserstein_radius(returns)
    print(f"\n  Empirical radius estimate (reference): ε_est = {epsilon_est:.6f}")
    
    # Step 3: Create radius schedule as fractions of eps_max
    print(f"\n  Generating radius schedule as fractions of eps_max...")
    radii = {}
    for frac, label in zip(Config.EPSILON_FRACTIONS, 
                           ['Full', '3/4', '1/2']):
        epsilon = frac * eps_max
        radii[label] = epsilon
        print(f"    ε ({label}) = {epsilon:.6f}")
    
    return {
        'mu_max': mu_max,
        'target_return': target_return,
        'epsilon_empirical': epsilon_est,
        'epsilon_max': eps_max,
        'radii_schedule': radii,
    }


def solve_dro_models(returns: np.ndarray, radii: Dict,
                    target_return: float, config: Config) -> pd.DataFrame:
    """Solve all DRO models with different radii."""
    print("\n" + "="*80)
    print("STEP 3: SOLVING DRO MODELS")
    print("="*80)
    
    mean = np.mean(returns, axis=0)
    cov = (returns - mean).T @ (returns - mean) / len(returns)
    
    results_list = []
    
    # 2-WDRO-Markowitz
    print("\n  • Solving 2-WDRO-Markowitz models...")
    for label, epsilon in radii['radii_schedule'].items():
        try:
            model = DROWassersteinMarkowitz(mean, cov, epsilon, target_return,
                                          short_selling=config.SHORT_SELLING)
            result = model.solve()
            
            if result['status'] == 'optimal':
                results_list.append({
                    'Strategy': f'2-WDRO-Markowitz-{label}',
                    'epsilon': epsilon,
                    'model_type': '2-WDRO-Markowitz',
                    'weights': result['weights'],
                    'return': result['expected_return'],
                    'std_dev': result['std_dev'],
                    'sharpe': result['sharpe'],
                    'solve_time': result['solve_time'],
                    'status': 'optimal',
                })
                print(f"    ✓ {label}: Return={result['expected_return']:.6f}, "
                      f"Std={result['std_dev']:.6f}, "
                      f"Time={result['solve_time']:.2f}s")
            else:
                print(f"    ✗ {label}: Infeasible")
        except Exception as e:
            print(f"    ✗ {label}: Error - {e}")
    
    # 1-WDRO-CVaR
    print("\n  • Solving 1-WDRO-CVaR models...")
    for label, epsilon in radii['radii_schedule'].items():
        try:
            model = DROWassersteinCVaR_1(returns, epsilon, target_return,
                                        confidence_level=config.CVaR_CONFIDENCE,
                                        short_selling=config.SHORT_SELLING)
            result = model.solve()
            
            if result['status'] == 'optimal':
                results_list.append({
                    'Strategy': f'1-WDRO-CVaR-{label}',
                    'epsilon': epsilon,
                    'model_type': '1-WDRO-CVaR',
                    'weights': result['weights'],
                    'return': result['expected_return'],
                    'std_dev': result['std_dev'],
                    'sharpe': result['sharpe'],
                    'solve_time': result['solve_time'],
                    'status': 'optimal',
                })
                print(f"    ✓ {label}: Return={result['expected_return']:.6f}, "
                      f"Std={result['std_dev']:.6f}, "
                      f"Time={result['solve_time']:.2f}s")
            else:
                print(f"    ✗ {label}: Infeasible")
        except Exception as e:
            print(f"    ✗ {label}: Error - {e}")
    
    # 2-WDRO-CVaR
    print("\n  • Solving 2-WDRO-CVaR models...")
    for label, epsilon in radii['radii_schedule'].items():
        try:
            model = DROWassersteinCVaR_2(returns, epsilon, target_return,
                                        confidence_level=config.CVaR_CONFIDENCE,
                                        short_selling=config.SHORT_SELLING)
            result = model.solve()
            
            if result['status'] == 'optimal':
                results_list.append({
                    'Strategy': f'2-WDRO-CVaR-{label}',
                    'epsilon': epsilon,
                    'model_type': '2-WDRO-CVaR',
                    'weights': result['weights'],
                    'return': result['expected_return'],
                    'std_dev': result['std_dev'],
                    'sharpe': result['sharpe'],
                    'solve_time': result['solve_time'],
                    'status': 'optimal',
                })
                print(f"    ✓ {label}: Return={result['expected_return']:.6f}, "
                      f"Std={result['std_dev']:.6f}, "
                      f"Time={result['solve_time']:.2f}s")
            else:
                print(f"    ✗ {label}: Infeasible")
        except Exception as e:
            print(f"    ✗ {label}: Error - {e}")
    
    # Benchmark models
    print("\n  • Solving benchmark models...")
    
    # SAA (ε=0)
    try:
        saa_model = SAA_Markowitz(mean, cov, target_return,
                                 short_selling=config.SHORT_SELLING)
        saa_result = saa_model.solve()
        results_list.append({
            'Strategy': 'SAA-Markowitz',
            'epsilon': 0.0,
            'model_type': 'SAA',
            'weights': saa_result['weights'],
            'return': saa_result['expected_return'],
            'std_dev': saa_result['std_dev'],
            'sharpe': saa_result['sharpe'],
            'solve_time': saa_result['solve_time'],
            'status': 'optimal',
        })
        print(f"    ✓ SAA: Return={saa_result['expected_return']:.6f}, "
              f"Std={saa_result['std_dev']:.6f}")
    except Exception as e:
        print(f"    ✗ SAA: Error - {e}")
    
    # Minimum Variance
    try:
        minvar_model = MinimumVariance(cov, short_selling=config.SHORT_SELLING)
        mv_result = minvar_model.solve()
        results_list.append({
            'Strategy': 'MinVar',
            'epsilon': 0.0,
            'model_type': 'Benchmark',
            'weights': mv_result['weights'],
            'return': np.nan,
            'std_dev': mv_result['std_dev'],
            'sharpe': np.nan,
            'solve_time': mv_result['solve_time'],
            'status': 'optimal',
        })
        print(f"    ✓ MinVar: Std={mv_result['std_dev']:.6f}")
    except Exception as e:
        print(f"    ✗ MinVar: Error - {e}")
    
    # Equal Weight
    try:
        ew_model = EqualWeight(mean.shape[0])
        ew_result = ew_model.solve()
        ew_weights = ew_result['weights']
        results_list.append({
            'Strategy': 'EqualWeight',
            'epsilon': 0.0,
            'model_type': 'Benchmark',
            'weights': ew_weights,
            'return': mean @ ew_weights,
            'std_dev': np.sqrt(ew_weights @ cov @ ew_weights),
            'sharpe': (mean @ ew_weights) / np.sqrt(ew_weights @ cov @ ew_weights),
            'solve_time': 0.0,
            'status': 'optimal',
        })
        print(f"    ✓ EqualWeight: Return={ew_result['weights'] @ mean:.6f}")
    except Exception as e:
        print(f"    ✗ EqualWeight: Error - {e}")
    
    return pd.DataFrame(results_list)


def evaluate_strategies(results_df: pd.DataFrame, returns: np.ndarray,
                       target_return: float) -> pd.DataFrame:
    """Evaluate all strategies on in-sample data."""
    print("\n" + "="*80)
    print("STEP 4: IN-SAMPLE PERFORMANCE EVALUATION")
    print("="*80)
    
    evaluator = PortfolioEvaluator()
    evaluation_results = []
    
    for idx, row in results_df.iterrows():
        metrics = evaluator.evaluate(row['weights'], returns, 
                                    target_return=target_return,
                                    cvar_alpha=Config.CVaR_CONFIDENCE)
        
        evaluation_results.append({
            'Strategy': row['Strategy'],
            'Mean Return': metrics['mean_return'],
            'Volatility': metrics['volatility'],
            'Sharpe Ratio': metrics['sharpe_ratio'],
            'Max Drawdown': metrics['max_drawdown'],
            'CVaR(95%)': metrics['cvar'],
            'Constraint Satisfaction': metrics.get('constraint_satisfaction', np.nan),
        })
    
    eval_df = pd.DataFrame(evaluation_results)
    
    print("\n" + eval_df.to_string(index=False))
    
    return eval_df


def print_theoretical_analysis():
    """Print theoretical analysis and interpretation."""
    print("\n" + "="*80)
    print("THEORETICAL ANALYSIS AND MATHEMATICAL DERIVATION")
    print("="*80)
    
    analysis_text = """
1. MATHEMATICAL FORMULATION
============================

The Distributionally Robust Optimization (DRO) problem minimizes the expected loss
in the worst case under a set of distributions close to the empirical one:

    min_{w ∈ W}   sup_{Q ∈ P_ε(P̂_N)}   E_Q[ℓ(w,ξ)]

where:
- w ∈ W: portfolio weights satisfying constraints (sum=1, w≥0 for long-only)
- P̂_N: empirical distribution from N i.i.d. samples
- P_ε(P̂_N) = {Q : W_p(Q, P̂_N) ≤ ε}: Wasserstein ball of radius ε
- ℓ(w,ξ) = -w^T ξ: linear loss (negative return)

The Wasserstein distance of order p with ground metric d(·,·) is:

    W_p(μ,ν) := inf_{Π ∈ P(𝕏²)} [ ∫∫ d(ξ,ζ)^p Π(dξ,dζ) ]^{1/p}

2. TRACTABLE REFORMULATIONS (SOCP)
===================================

A. 2-WDRO-MARKOWITZ (Mean-Variance)
-----------------------------------

Standard Markowitz:
    min_{w}   w^T Σ̂ w  s.t. m̂^T w ≥ μ, sum(w)=1, w≥0

2-Wasserstein DRO Reformulation (using Theorem 1 with p=2):
    min_{w,t,r}   t + ε*r
    s.t.  ||S_half^T w||_2 ≤ t
          ||w||_2 ≤ r
          m̂^T w - ε*r ≥ μ
          sum(w) = 1, w ≥ 0

where S_half is the Cholesky decomposition of Σ̂.

Interpretation: The objective adds a penalty term ε*||w||_2 that penalizes portfolio
sensitivity. The robust return constraint subtracts ε*||w||_2, enforcing a lower bound
that accounts for distribution uncertainty.

This is a Second-Order Cone Program (SOCP) solvable in polynomial time.

Complexity: O(m^3) operations using interior-point methods


B. 1-WDRO-CVAR (from Corollary 1)
----------------------------------

CVaR loss function (Eq. 7):
    ℓ(w,τ,ξ) := max{-w^T ξ/α + (1-1/α)τ, τ}

1-Wasserstein reformulation using Kantorovich-Rubinstein duality:
    
    min_{w,τ,λ,s}   λ*ε + (1/N)∑s_i
    s.t.  a_k⟨w,ξ_i⟩ + b_k τ ≤ s_i  ∀i,k
          |a_k|*||w||_2 ≤ λ  ∀k
          (1/N)∑⟨w,ξ_i⟩ - ε*||w||_2 ≥ μ
          sum(w) = 1, w ≥ 0

where a_1 = -1/α, b_1 = 1-1/α, a_2 = 0, b_2 = 1.

Key insight: The 1-Wasserstein radius acts as a regularization term that penalizes
the Lipschitz constant of the loss function. This is tighter than the 2-norm version.


C. 2-WDRO-CVAR (Theorem 1 with p=2)
------------------------------------

2-Wasserstein CVaR reformulation:
    
    min_{w,τ,λ,s,z}   λ*ε² + (1/N)∑s_i
    s.t.  (a_k²/4)z + ⟨w,ξ_i⟩*a_k + b_k τ ≤ s_i  ∀i,k
          ||[2w; λ-z]||_2 ≤ λ+z                (rotated cone)
          (1/N)∑⟨w,ξ_i⟩ - ε*||w||_2 ≥ μ
          sum(w) = 1, w ≥ 0, z ≥ 0

The quadratic term ε² and auxiliary z variable provide stronger robustness guarantees
than the 1-norm variant.


3. DUAL SPACES AND NORM DUALITY
================================

Ground Metric: Euclidean L_2 distance
  
  d(ξ,ζ) := ||ξ - ζ||_2

Dual Norm: Spectral (Operator Norm) L_2
  
  ||·||_2* = ||·||_2  (self-dual)

This makes the reformulation feasible and tractable.


4. COMPUTATIONAL COMPLEXITY
============================

Model           Variables  Constraints  Complexity    Solver Time
2-WDRO-Markowitz   m+2        3m+2       O(m^3)      < 1 second
1-WDRO-CVaR       m+2N       3N+3        O(N*m^3)    N/∑sec
2-WDRO-CVaR       m+2N+1     3N+4        O(N*m^3)    N/∑sec

where N is the number of samples (trading periods), m is the number of assets.

For typical problems (N=250 samples, m=20 assets), expect solve times < 5 seconds.


5. WASSERSTEIN RADIUS SELECTION
===============================

The radius ε balances robustness vs. performance:
  
  ε = 0:           Reduces to SAA (overfitting risk)
  ε = ε_max:       Maximum feasible radius (very conservative)
  ε ∈ (0, ε_max):  Practical compromise

Data-driven estimation (Blanchet & Murthy, 2019):
  
  ε ≈ c * √(m/N) * (1 + log N)

where c ≈ 0.1 is a calibration constant.

For emerging markets like Colombia, expected larger ε than developed markets
due to higher estimation error from shorter time series.


6. ECONOMIC INTERPRETATION FOR EMERGING MARKETS
===========================================

Advantages of DRO for COLCAP:
  
  1. Robustness: Accounts for structural breaks (political events, sanctions,
     commodity price shocks) common in emerging markets
  
  2. Estimation Error: Addresses high variance in parameter estimates with
     limited historical data
  
  3. Model Risk: Does not assume single distribution—hedges against
     unmodeled tail events
  
  4. Out-of-Sample Performance: Empirically shown to meet return constraints
     when SAA fails

Challenges:
  
  1. Limited Data: COLCAP has fewer years of clean data than S&P 500
  2. Non-Stationarity: Political and macroeconomic shifts alter distributions
  3. Illiquidity: Transaction costs not modeled
  4. Regulatory: FX restrictions may affect implementation


7. COMPARISON: DRO vs SAA vs CLASSICAL MARKOWITZ
=================================================

             | In-Sample | Out-of-Sample | Robustness | Complexity
SAA          |   Good    |      Poor     |    Low     |    Low
Markowitz    |   Good    |      Poor     |    Low     |    Low
2-WDRO       |   Fair    |     Excellent |   High     |   Medium
1-WDRO-CVaR  |   Fair    |     Excellent |  V.High   |   Medium
2-WDRO-CVaR  |   Fair    |     Excellent |  V.High   |   Medium

The DRO methods show regularization effect (worse in-sample) that translates
to superior out-of-sample performance by preventing overfitting.

"""
    
    print(analysis_text)


def create_results_directory(config: Config):
    """Create output directory."""
    Path(config.RESULTS_DIR).mkdir(exist_ok=True)


def save_results(results_df: pd.DataFrame, eval_df: pd.DataFrame, config: Config):
    """Save results to CSV files."""
    create_results_directory(config)
    
    results_df.to_csv(f"{config.RESULTS_DIR}/dro_solutions.csv", index=False)
    eval_df.to_csv(f"{config.RESULTS_DIR}/performance_evaluation.csv", index=False)
    
    print(f"\n✓ Results saved to {config.RESULTS_DIR}/")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Execute complete DRO analysis pipeline."""
    
    print("\n")
    print("█"*80)
    print("  DISTRIBUTIONALLY ROBUST OPTIMIZATION PORTFOLIO ANALYSIS")
    print("  Colombian COLCAP Market - 2-WDRO-Markowitz & CVaR models")
    print("  Implementation in Python with Gurobi")
    print("█"*80)
    
    config = Config()
    
    try:
        # Step 1: Load data
        returns = load_and_prepare_data(config)
        
        # Step 2: Calibrate Wasserstein radius (auto-calibrates target if config.TARGET_RETURN is None)
        radius_info = calibrate_wasserstein_radius(returns, config.TARGET_RETURN)
        radii = radius_info['radii_schedule']
        target_return = radius_info['target_return']

        # Step 3: Solve all DRO models
        results_df = solve_dro_models(returns, radius_info,
                                     target_return, config)

        # Step 4: Evaluate strategies
        eval_df = evaluate_strategies(results_df, returns, target_return)
        
        # Step 5: Save results
        save_results(results_df, eval_df, config)
        
        # Step 6: Print theoretical analysis
        print_theoretical_analysis()
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print(f"Results saved to: {config.RESULTS_DIR}/")
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
