"""
DRO-Wasserstein Portfolio Optimization with Gurobi
=====================================================

Mathematical Implementation:
1. 2-WDRO-Markowitz (Mean-Variance)
2. 1-WDRO-CVaR and 2-WDRO-CVaR

All models use Wasserstein distance metric with p-norm.

Dependencies: gurobipy, numpy, scipy
"""

from xml.parsers.expat import model

import numpy as np
from scipy.linalg import cholesky
import gurobipy as gp
from gurobipy import GRB
from typing import Dict, Tuple, Optional
import warnings


# ============================================================================
# Helper: Mapeo de Estados de Gurobi a Descriptores Legibles
# ============================================================================

def _get_gurobi_status_string(status_code: int) -> str:
    """
    Traduce el código de estado de Gurobi a una descripción legible.
    
    Parameters
    ----------
    status_code : int
        Valor de model.Status desde Gurobi
    
    Returns
    -------
    str
        Descripción del estado
    """
    status_map = {
        1: 'loaded',              # No se ha optimizado
        2: 'optimal',             # Solución óptima encontrada
        3: 'infeasible',          # Modelo infactible
        4: 'inf_or_unbd',         # Infactible o no acotado
        5: 'unbounded',           # Modelo no acotado
        6: 'cutoff',              # Objetivo alcanzó cutoff
        7: 'iteration_limit',     # Límite de iteraciones alcanzado
        8: 'node_limit',          # Límite de nodos alcanzado
        9: 'time_limit',          # Límite de tiempo alcanzado
        10: 'solution_limit',     # Límite de soluciones alcanzado
        11: 'interrupted',        # Proceso interrumpido por usuario
        12: 'numeric',            # Problemas numéricos detectados
        13: 'suboptimal',         # Solución encontrada pero no óptima certificada
        14: 'in_progress',        # Optimización en curso
        15: 'user_obj_limit',     # Límite de objetivo del usuario alcanzado
        16: 'work_limit',         # Límite de trabajo alcanzado
        17: 'memory_limit',       # Límite de memoria alcanzado
    }
    return status_map.get(status_code, f'unknown_status_{status_code}')


class DROWassersteinMarkowitz:
    """
    2-WDRO-Markowitz (Mean-Variance) Portfolio Optimization
    
    Mathematical Formulation:
    ========================
    
    min_{w,t,r}  t + ε*r
    
    s.t.  ||S_half^T w||_2 ≤ t           (variance constraint)
          ||w||_2 ≤ r                     (norm constraint)
          m^T w - ε*r ≥ μ                 (robust mean constraint)
          sum(w) = 1                       (fully invested)
          w ≥ 0                            (long-only)
    
    where:
    - S_half is Cholesky decomposition of sample covariance Σ_hat
    - m is sample mean vector
    - ε is Wasserstein radius
    - μ is minimum target return
    
    Theory:
    -------
    The 2-WDRO reformulation follows from Theorem 1 with p=2 and
    Kantorovich-Rubinstein duality for Lipschitz loss function ℓ(w,ξ) = -w^T ξ.
    
    This is an SOCP (Second-Order Cone Program) solved exactly by Gurobi.
    """
    
    def __init__(self, mean: np.ndarray, cov: np.ndarray, 
                 epsilon: float, target_return: float,
                 short_selling: bool = False):
        """
        Initialize 2-WDRO-Markowitz model.
        
        Parameters
        ----------
        mean : np.ndarray
            Sample mean vector (m,)
        cov : np.ndarray
            Sample covariance matrix (m, m)
        epsilon : float
            Wasserstein radius ε ≥ 0
        target_return : float
            Minimum target return μ
        short_selling : bool
            If True, allow w_i < 0 (unrestricted weights)
            If False, enforce w_i ≥ 0 (long-only)
        """
        self.mean = mean
        self.cov = cov
        self.epsilon = epsilon
        self.target_return = target_return
        self.short_selling = short_selling
        self.m = len(mean)
        
        # Compute Cholesky decomposition of covariance
        try:
            self.S_half = cholesky(cov, lower=False)  # Upper triangular
        except np.linalg.LinAlgError:
            warnings.warn("Covariance matrix not positive definite, regularizing...")
            # Regularize with small perturbation
            cov_reg = cov + np.eye(self.m) * 1e-8
            self.S_half = cholesky(cov_reg, lower=False)
        
        self.model = None
        self.w = None
        self.t = None
        self.r = None
        self.status = None
        self.solve_time = None
        
    def build_model(self) -> gp.Model:
        """
        Build the Gurobi SOCP model.
        
        Returns
        -------
        gp.Model
            Gurobi optimization model
        """
        model = gp.Model("DRO_Markowitz_2WDRO")
        model.Params.OutputFlag = 0
        
        # Variables
        w = model.addMVar(self.m, name="w", lb=0.0 if not self.short_selling else -GRB.INFINITY)
        t = model.addVar(name="t", lb=0.0)  # Epigraph for variance
        r = model.addVar(name="r", lb=0.0)  # Epigraph for norm
        



        # Objective: minimize t + ε*r
        model.setObjective(t + self.epsilon * r, GRB.MINIMIZE)
        
        # Constraint 1: sum(w) = 1 (fully invested)
        model.addConstr(w.sum() == 1, name="budget")
        
        # Constraint 2: Robust mean constraint m^T w - ε*r ≥ μ
        model.addConstr(self.mean @ w - self.epsilon * r >= self.target_return,
                       name="robust_return")
        
        # Auxiliary variable y = S_half^T w  (needed for addGenConstrNorm)
        y = model.addMVar(self.m, lb=-GRB.INFINITY, name="y")
        model.update()  # commit variables so getVarByName is available

        # Constraint 3: SOCP constraint for variance ||S_half^T w||_2 ≤ t
        model.addConstr(y == self.S_half @ w, name="y_def")
        y_vars = [model.getVarByName(f"y[{j}]") for j in range(self.m)]
        model.addGenConstrNorm(t, y_vars, 2.0, name="variance_cone")

        # Constraint 4: Norm constraint ||w||_2 ≤ r
        w_vars = [model.getVarByName(f"w[{j}]") for j in range(self.m)]
        model.addGenConstrNorm(r, w_vars, 2.0, name="norm_cone")

        self.model = model
        self.w = w
        self.t = t
        self.r = r

        return model
    
    def solve(self) -> Dict:
        """
        Solve the optimization problem.
        
        Returns
        -------
        dict
            Solution dictionary with keys:
            - 'weights': optimal portfolio weights (m,)
            - 'variance': portfolio variance
            - 'obj_value': objective value
            - 'status': optimization status
            - 'solve_time': solver time in seconds
        """
        model = self.build_model()
        model.optimize()
        
        self.solve_time = model.Runtime
        self.status = model.Status
        
        if model.Status == GRB.OPTIMAL:
            weights = self.w.X
            variance = (weights @ self.cov @ weights)**0.5  # Standard deviation
            
            return {
                'weights': weights,
                'variance': variance,
                'std_dev': variance,
                'expected_return': self.mean @ weights,
                'obj_value': model.ObjVal,
                'status': 'optimal',
                'status_code': model.Status,
                'status_description': _get_gurobi_status_string(model.Status),
                'solve_time': self.solve_time,
                'sharpe': (self.mean @ weights) / variance if variance > 0 else 0,
            }
        elif model.Status == GRB.INFEASIBLE:
            return {
                'status': 'infeasible',
                'status_code': model.Status,
                'status_description': _get_gurobi_status_string(model.Status),
                'solve_time': self.solve_time,
                'error': 'Problem is infeasible'
            }
        else:
            status_str = _get_gurobi_status_string(model.Status)
            return {
                'status': status_str,
                'status_code': model.Status,
                'status_description': _get_gurobi_status_string(model.Status),
                'solve_time': self.solve_time,
                'error': f'Solver terminated with status: {status_str} (code {model.Status})'
            }


class DROWassersteinCVaR_1:
    """
    1-WDRO-CVaR Portfolio Optimization
    
    Mathematical Formulation:
    =========================
    
    min_{w,τ,λ,s}  λ*ε + (1/N) * sum(s_i)
    
    s.t.  a_k * ⟨w, ξ_i⟩ + b_k * τ ≤ s_i  ∀i=1,...,N, k∈{1,2}
          |a_k| * ||w||_2 ≤ λ            ∀k∈{1,2}
          (1/N) * sum(⟨w, ξ_i⟩) - ε*||w||_2 ≥ μ
          sum(w) = 1
          w ≥ 0
    
    where:
    - a_1 = -1/α, b_1 = 1 - 1/α  (from CVaR formulation)
    - a_2 = 0, b_2 = 1
    - α is the confidence level (e.g., α=0.05 for 95% confidence)
    - ξ_i are sample returns (m,)
    - λ is dual variable
    - s_i are per-sample slack variables
    
    Theory:
    -------
    1-Wasserstein DRO reformulation from Corollary 1.
    This uses the Kantorovich-Rubinstein duality:
    
    sup_{Q: W_1(Q,P̂_N) ≤ ε} E_Q[ℓ(w,ξ)] = (1/N)∑ ℓ(w,ξ_i) + ε*||∇ℓ||_{L1}
    
    For CVaR with Lipschitz constant derived from formulation (7).
    """
    
    def __init__(self, returns: np.ndarray, epsilon: float, 
                 target_return: float, confidence_level: float = 0.05,
                 short_selling: bool = False):
        """
        Initialize 1-WDRO-CVaR model.
        
        Parameters
        ----------
        returns : np.ndarray
            Sample returns matrix (T, m): T time periods, m assets
        epsilon : float
            Wasserstein radius ε ≥ 0
        target_return : float
            Minimum target return μ
        confidence_level : float
            CVaR confidence level α (default 5%)
        short_selling : bool
            If True, allow short selling
        """
        self.returns = returns
        self.epsilon = epsilon
        self.target_return = target_return
        self.alpha = confidence_level
        self.short_selling = short_selling
        
        self.N = returns.shape[0]  # Number of samples
        self.m = returns.shape[1]  # Number of assets
        
        # Constants from CVaR formulation (Eq. 7)
        self.a1 = -1.0 / self.alpha
        self.b1 = 1.0 - (1.0 / self.alpha)
        self.a2 = 0.0
        self.b2 = 1.0
        
        self.model = None
        self.w = None
        self.tau = None
        self.lambda_var = None
        self.s = None
        self.status = None
        self.solve_time = None
        
    def build_model(self) -> gp.Model:
        """
        Build the Gurobi model for 1-WDRO-CVaR.
        
        Returns
        -------
        gp.Model
            Gurobi optimization model
        """
        model = gp.Model("DRO_CVaR_1WDRO")
        model.Params.OutputFlag = 0
        
        # Variables
        w = model.addMVar(self.m, name="w", lb=0.0 if not self.short_selling else -GRB.INFINITY)
        tau = model.addVar(name="tau", lb=-GRB.INFINITY)  # CVaR auxiliary variable
        lambda_var = model.addVar(name="lambda", lb=0.0)  # Dual variable
        s = model.addMVar(self.N, name="s", lb=-GRB.INFINITY)  # Per-sample slack
        
        # Objective: minimize λ*ε + (1/N)*sum(s_i)
        model.setObjective(lambda_var * self.epsilon + (1.0/self.N) * s.sum(),
                          GRB.MINIMIZE)
        
        # Auxiliary scalar for ||w||_2 (needed for addGenConstrNorm)
        norm_w = model.addVar(lb=0.0, name="norm_w")
        model.update()  # commit variables so getVarByName is available
        w_vars = [model.getVarByName(f"w[{j}]") for j in range(self.m)]

        # Constraint 1: sum(w) = 1
        model.addConstr(w.sum() == 1, name="budget")

        # Constraint 2: Per-sample epigraph constraints (vectorized over N samples)
        model.addConstr(self.a1 * (self.returns @ w) + self.b1 * tau <= s, name="epi1")
        model.addConstr(self.a2 * (self.returns @ w) + self.b2 * tau <= s, name="epi2")

        # Constraint 3: Lipschitz constraint |a_1|*||w||_2 ≤ λ  (a_2=0 is trivially satisfied)
        model.addGenConstrNorm(norm_w, w_vars, 2.0, name="norm_w_def")
        model.addConstr(abs(self.a1) * norm_w <= lambda_var, name="lip_constraint_1")

        # Constraint 4: Robust return  (1/N)∑⟨w,ξ_i⟩ - ε*||w||_2 ≥ μ
        xi_mean = np.mean(self.returns, axis=0)
        model.addConstr(xi_mean @ w - self.epsilon * norm_w >= self.target_return,
                        name="robust_return")

        self.model = model
        self.w = w
        self.tau = tau
        self.lambda_var = lambda_var
        self.s = s

        return model
    
    def solve(self) -> Dict:
        """
        Solve the optimization problem.
        
        Returns
        -------
        dict
            Solution dictionary
        """
        model = self.build_model()
        model.optimize()
        
        self.solve_time = model.Runtime
        self.status = model.Status
        
        if model.Status == GRB.OPTIMAL:
            weights = self.w.X
            mean_ret = np.mean(self.returns @ weights)
            std_dev = np.sqrt(np.var(self.returns @ weights))
            
            return {
                'weights': weights,
                'tau': self.tau.X,
                'lambda': self.lambda_var.X,
                'expected_return': mean_ret,
                'std_dev': std_dev,
                'cvar': self.tau.X,  # Approximate CVaR
                'obj_value': model.ObjVal,
                'status': 'optimal',
                'status_code': model.Status,
                'status_description': _get_gurobi_status_string(model.Status),
                'solve_time': self.solve_time,
                'sharpe': mean_ret / std_dev if std_dev > 0 else 0,
            }
        elif model.Status == GRB.INFEASIBLE:
            return {
                'status': 'infeasible',
                'status_code': model.Status,
                'status_description': _get_gurobi_status_string(model.Status),
                'solve_time': self.solve_time,
                'error': 'Problem is infeasible'
            }
        else:
            status_str = _get_gurobi_status_string(model.Status)
            return {
                'status': status_str,
                'status_code': model.Status,
                'status_description': _get_gurobi_status_string(model.Status),
                'solve_time': self.solve_time,
                'error': f'Solver terminated with status: {status_str} (code {model.Status})'
            }


class DROWassersteinCVaR_2:
    """
    2-WDRO-CVaR Portfolio Optimization (Most Complex)
    
    Mathematical Formulation:
    =========================
    
    min_{w,τ,λ,s,z}  λ*ε² + (1/N)*sum(s_i)
    
    s.t.  (a_k²/4)*z + ⟨w,ξ_i⟩*a_k + b_k*τ ≤ s_i  ∀i,k
          ||[2w; λ-z]||_2 ≤ λ+z                     (rotated cone)
          m^T w - ε*||w||_2 ≥ μ
          sum(w) = 1
          w ≥ 0, z ≥ 0
    
    Theory:
    -------
    2-Wasserstein DRO from Theorem 1 with p=2.
    Uses stronger robustness statement than 1-WDRO.
    Requires introduction of auxiliary variable z for rotated cone constraint.
    """
    
    def __init__(self, returns: np.ndarray, epsilon: float,
                 target_return: float, confidence_level: float = 0.05,
                 short_selling: bool = False):
        """
        Initialize 2-WDRO-CVaR model.
        
        Parameters
        ----------
        returns : np.ndarray
            Sample returns matrix (T, m)
        epsilon : float
            Wasserstein radius ε ≥ 0
        target_return : float
            Minimum target return μ
        confidence_level : float
            CVaR confidence level α
        short_selling : bool
            If True, allow short selling
        """
        self.returns = returns
        self.epsilon = epsilon
        self.target_return = target_return
        self.alpha = confidence_level
        self.short_selling = short_selling
        
        self.N = returns.shape[0]
        self.m = returns.shape[1]
        
        # Constants from CVaR formulation
        self.a1 = -1.0 / self.alpha
        self.b1 = 1.0 - (1.0 / self.alpha)
        self.a2 = 0.0
        self.b2 = 1.0
        
        self.model = None
        self.w = None
        self.tau = None
        self.lambda_var = None
        self.s = None
        self.z = None
        self.status = None
        self.solve_time = None
        
    def build_model(self) -> gp.Model:
        """Build the Gurobi SOCP model for 2-WDRO-CVaR."""
        model = gp.Model("DRO_CVaR_2WDRO")
        model.Params.OutputFlag = 0
        model.Params.TimeLimit = 60
        model.Params.DualReductions = 0
        
        # Variables
        w = model.addMVar(self.m, name="w", lb=0.0 if not self.short_selling else -GRB.INFINITY)
        tau = model.addVar(name="tau", lb=-GRB.INFINITY)
        lambda_var = model.addVar(name="lambda", lb=0.0)
        s = model.addMVar(self.N, name="s", lb=0.0)  
        #s = model.addMVar(self.N, name="s", lb=-GRB.INFINITY) Mucho cuidado
        z = model.addVar(name="z", lb=0.0)  # Auxiliary variable for rotated cone
        
        # Objective: minimize λ*ε² + (1/N)*sum(s_i)
        model.setObjective(lambda_var * self.epsilon**2 + (1.0/self.N) * s.sum(),
                          GRB.MINIMIZE)
        
        # Auxiliary variables for Lorentz cone and norm (must be real Var, not MVar elements)
        lp_z   = model.addVar(lb=0.0,           name="lp_z")   # λ + z
        lm_z   = model.addVar(lb=-GRB.INFINITY, name="lm_z")   # λ - z
        norm_w = model.addVar(lb=0.0,           name="norm_w")
        ws     = model.addMVar(self.m, lb=-GRB.INFINITY, name="ws")  # 2*w (vectorized)

        # Constraint 1: sum(w) = 1
        model.addConstr(w.sum() == 1, name="budget")

        # Cache matrix-vector product (computed once, reused in epigraph constraints)
        returns_w = self.returns @ w

        # Constraint 2: Per-sample epigraph constraints (vectorized over N samples)
        model.addConstr((self.a1**2/4.0)*z + self.a1*returns_w + self.b1*tau <= s,
                        name="epi1")
        # a2 = 0 → constraint simplifies to: b2*tau <= s
        model.addConstrs(
            (
                (self.a1**2 / 4.0) * z
                + self.a1 * returns_w[i]
                + self.b1 * tau
                <= s[i]
                for i in range(self.N)
            ),
            name="epi1"
        )

        # Constraint 3: Rotated cone ||w||² ≤ λ*z via Lorentz: ||[2w; λ-z]||_2 ≤ λ+z
        model.addConstr(lp_z == lambda_var + z, name="lpz_def")
        model.addConstr(lm_z == lambda_var - z, name="lmz_def")
        model.addConstr(ws == 2.0 * w, name="ws_def")  # vectorized, replaces element-wise loop
        model.addGenConstrNorm(lp_z, list(ws) + [lm_z], 2.0, name="rotated_cone")
        
        # Constraint 3: Rotated cone equivalent in quadratic form ||w||^2 <= λ*z
        #model.addQConstr(gp.quicksum(w[i]*w[i] for i in range(self.m))<= lambda_var * z,
         #   name="rotated_cone_qc"
        #)       
        
        # Constraint 4: Robust return  mean^T w - ε*||w||_2 ≥ μ
        model.addGenConstrNorm(
            norm_w,
            [w[i] for i in range(self.m)],
            2.0,
            name="norm_w_def"
        )
        xi_mean = np.mean(self.returns, axis=0)
        model.addConstr(xi_mean @ w - self.epsilon * norm_w >= self.target_return,
                        name="robust_return")

        self.model = model
        self.w = w
        self.tau = tau
        self.lambda_var = lambda_var
        self.s = s
        self.z = z

        return model
    
    def solve(self) -> Dict:
        """Solve the optimization problem."""
        model = self.build_model()
        model.optimize()
        
        self.solve_time = model.Runtime
        self.status = model.Status
        
        if model.Status == GRB.OPTIMAL:
            weights = self.w.X
            mean_ret = np.mean(self.returns @ weights)
            std_dev = np.sqrt(np.var(self.returns @ weights))
            
            return {
                'weights': weights,
                'tau': self.tau.X,
                'lambda': self.lambda_var.X,
                'z': self.z.X,
                'expected_return': mean_ret,
                'std_dev': std_dev,
                'cvar': self.tau.X,
                'obj_value': model.ObjVal,
                'status': 'optimal',
                'status_code': model.Status,
                'status_description': _get_gurobi_status_string(model.Status),
                'solve_time': self.solve_time,
                'sharpe': mean_ret / std_dev if std_dev > 0 else 0,
            }
        elif model.Status == GRB.INFEASIBLE:
            return {
                'status': 'infeasible',
                'status_code': model.Status,
                'status_description': _get_gurobi_status_string(model.Status),
                'solve_time': self.solve_time,
                'error': 'Problem is infeasible'
            }
        else:
            status_str = _get_gurobi_status_string(model.Status)
            return {
                'status': status_str,
                'status_code': model.Status,
                'status_description': _get_gurobi_status_string(model.Status),
                'solve_time': self.solve_time,
                'error': f'Solver terminated with status: {status_str} (code {model.Status})'
            }



# ============================================================================
# Benchmark Models (SAA and Traditional)
# ============================================================================

class SAA_Markowitz:
    """
    Standard Sample Average Approximation (SAA) Mean-Variance
    (Classical Markowitz, equivalent to DRO with ε=0)
    """
    
    def __init__(self, mean: np.ndarray, cov: np.ndarray,
                 target_return: float, short_selling: bool = False):
        self.mean = mean
        self.cov = cov
        self.target_return = target_return
        self.short_selling = short_selling
        self.m = len(mean)
        
        try:
            self.S_half = cholesky(cov, lower=False)
        except:
            cov_reg = cov + np.eye(self.m) * 1e-8
            self.S_half = cholesky(cov_reg, lower=False)
    
    def solve(self) -> Dict:
        """Solve SAA markowitz (ε=0 case)."""
        model = gp.Model("SAA_Markowitz")
        model.Params.OutputFlag = 0
        
        w = model.addMVar(self.m, name="w", lb=0.0 if not self.short_selling else -GRB.INFINITY)
        t = model.addVar(name="t", lb=0.0)
        
        model.setObjective(t, GRB.MINIMIZE)
        model.addConstr(w.sum() == 1)
        model.addConstr(self.mean @ w >= self.target_return)
        
        # ||S_half^T w||_2 ≤ t
        quad = gp.quicksum((self.S_half[i, :] @ w) ** 2 for i in range(self.m))
        model.addConstr(quad <= t * t)
        
        model.optimize()
        
        if model.Status == GRB.OPTIMAL:
            weights = w.X
            variance = weights @ self.cov @ weights
            return {
                'weights': weights,
                'variance': variance,
                'std_dev': np.sqrt(variance),
                'expected_return': self.mean @ weights,
                'obj_value': model.ObjVal,
                'status': 'optimal',
                'solve_time': model.Runtime,
                'sharpe': (self.mean @ weights) / np.sqrt(variance) if variance > 0 else 0,
            }
        else:
            return {'status': 'failed', 'error': f'Status {model.Status}'}


class MinimumVariance:
    """Minimum variance portfolio (without return constraint)."""
    
    def __init__(self, cov: np.ndarray, short_selling: bool = False):
        self.cov = cov
        self.short_selling = short_selling
        self.m = cov.shape[0]
        
        try:
            self.S_half = cholesky(cov, lower=False)
        except:
            cov_reg = cov + np.eye(self.m) * 1e-8
            self.S_half = cholesky(cov_reg, lower=False)
    
    def solve(self) -> Dict:
        """Solve minimum variance."""
        model = gp.Model("MinVar")
        model.Params.OutputFlag = 0
        
        w = model.addMVar(self.m, name="w", lb=0.0 if not self.short_selling else -GRB.INFINITY)
        t = model.addVar(name="t", lb=0.0)
        
        model.setObjective(t, GRB.MINIMIZE)
        model.addConstr(w.sum() == 1)
        
        quad = gp.quicksum((self.S_half[i, :] @ w) ** 2 for i in range(self.m))
        model.addConstr(quad <= t * t)
        
        model.optimize()
        
        if model.Status == GRB.OPTIMAL:
            weights = w.X
            variance = weights @ self.cov @ weights
            return {
                'weights': weights,
                'variance': variance,
                'std_dev': np.sqrt(variance),
                'status': 'optimal',
                'solve_time': model.Runtime,
            }
        else:
            return {'status': 'failed'}


class EqualWeight:
    """Equal weight portfolio (1/n for each asset)."""
    
    def __init__(self, m: int):
        self.m = m
    
    def solve(self) -> Dict:
        """Return equal weight solution."""
        weights = np.ones(self.m) / self.m
        return {
            'weights': weights,
            'status': 'optimal',
            'solve_time': 0.0,
        }


# ============================================================================
# Parameter Computation: mu_max and eps_max
# ============================================================================

def compute_mu_max(returns: np.ndarray) -> float:
    """
    Compute mu_max = maximum expected return over empirical distribution.
    
    Solves the linear program:
    
    max_{w}   epsilon_bar^T w
    s.t.      sum(w) = 1
              w ≥ 0
    
    where epsilon_bar = (1/N) * sum_{i=1}^N epsilon_i is the sample mean return.
    
    This represents the best-case expected return achievable by any portfolio.
    
    Parameters
    ----------
    returns : np.ndarray
        Return matrix (T, m): T time periods, m assets
    
    Returns
    -------
    float
        Maximum expected return mu_max
    
    Raises
    ------
    RuntimeError
        If the optimization fails
    """
    mean_returns = np.mean(returns, axis=0)
    m = len(mean_returns)
    
    model = gp.Model("compute_mu_max")
    model.Params.OutputFlag = 0
    
    # Decision variable: portfolio weights
    w = model.addMVar(m, name="w", lb=0.0)
    
    # Objective: maximize expected return
    model.setObjective(mean_returns @ w, GRB.MAXIMIZE)
    
    # Constraint: fully invested
    model.addConstr(w.sum() == 1, name="budget")
    
    model.optimize()
    
    if model.Status == GRB.OPTIMAL:
        return float(model.ObjVal)
    else:
        raise RuntimeError(f"Failed to compute mu_max. Status: {model.Status}")


def compute_eps_max(returns: np.ndarray, mu_target: float) -> float:
    """
    Compute eps_max = maximum Wasserstein radius for which the robust return
    constraint remains feasible.

    Closed-form derivation (Fonseca & Junca, 2021, arxiv:2111.04663):
    ---------------------------------------------------------------
    The robust return constraint is:
        mean^T w - ε * ||w||_2 ≥ μ

    For the best-case portfolio w* = e_{argmax(mean)} (unit weight on highest
    mean asset), the constraint becomes:
        max_i(mean_i) - ε ≥ μ   →   ε ≤ max_i(mean_i) - μ

    More generally, the tightest feasible upper bound over the simplex is:
        eps_max = ||(mean - μ·1)_+||_2

    where (·)_+ = max(·, 0) component-wise.

    A 5% safety margin is applied so that the "Full" epsilon point stays
    strictly inside the feasibility boundary.

    Parameters
    ----------
    returns : np.ndarray
        Return matrix (T, m): T time periods, m assets
    mu_target : float
        Target return μ passed to the DRO models (NOT mu_max)

    Returns
    -------
    float
        Maximum Wasserstein radius eps_max (with 5% safety margin)
    """
    mean = np.mean(returns, axis=0)
    excess = np.maximum(mean - mu_target, 0.0)
    return float(np.linalg.norm(excess)) * 0.95


def demo_solve():
    """Demo solving basic DRO problem."""
    # Create sample data
    np.random.seed(42)
    T, m = 252, 5
    mean = np.random.randn(m) * 0.001 + 0.0005
    L = np.random.randn(m, m)
    cov = L @ L.T
    
    # Solve DRO-Markowitz
    dro = DROWassersteinMarkowitz(mean, cov, epsilon=0.1, target_return=0.0001)
    result = dro.solve()
    
    print("2-WDRO-Markowitz Solution:")
    print(f"  Status: {result['status']}")
    print(f"  Weights: {result['weights'][:3]}...")
    print(f"  Expected return: {result['expected_return']:.6f}")
    print(f"  Std Dev: {result['std_dev']:.6f}")
    print(f"  Solve time: {result['solve_time']:.4f}s")


if __name__ == "__main__":
    demo_solve()
