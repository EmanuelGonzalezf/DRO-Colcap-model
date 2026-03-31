"""
Portfolio Evaluation and Backtesting Module
=============================================

Implements:
1. In-sample performance metrics
2. Out-of-sample evaluation
3. Rolling window backtesting
4. Comparison statistics

"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from scipy.stats import norm


class PortfolioEvaluator:
    """
    Evaluate portfolio performance metrics.
    
    Metrics computed:
    - Mean return
    - Volatility (standard deviation)
    - Sharpe ratio
    - Maximum drawdown
    - CVaR (Conditional Value at Risk)
    - Return constraint satisfaction (%)
    """
    
    def __init__(self, risk_free_rate: float = 0.0):
        """
        Initialize portfolio evaluator.
        
        Parameters
        ----------
        risk_free_rate : float
            Risk-free rate for Sharpe ratio (default 0%, daily)
        """
        self.risk_free_rate = risk_free_rate
    
    def compute_portfolio_returns(self, weights: np.ndarray, 
                                 returns: np.ndarray) -> np.ndarray:
        """
        Compute portfolio period returns.
        
        Parameters
        ----------
        weights : np.ndarray
            Portfolio weights (m,)
        returns : np.ndarray
            Return matrix (T, m)
        
        Returns
        -------
        np.ndarray
            Portfolio returns (T,)
        """
        return returns @ weights
    
    def mean_return(self, portfolio_returns: np.ndarray) -> float:
        """Compute mean return."""
        return np.mean(portfolio_returns)
    
    def volatility(self, portfolio_returns: np.ndarray) -> float:
        """Compute volatility (standard deviation)."""
        return np.std(portfolio_returns)
    
    def sharpe_ratio(self, portfolio_returns: np.ndarray) -> float:
        """
        Compute Sharpe ratio.
        
        SR = (μ - r_f) / σ
        """
        mean = np.mean(portfolio_returns)
        vol = np.std(portfolio_returns)
        if vol == 0:
            return 0.0
        return (mean - self.risk_free_rate) / vol
    
    def cvar(self, portfolio_returns: np.ndarray, alpha: float = 0.05) -> float:
        """
        Compute Conditional Value at Risk (CVaR) at confidence level α.
        
        CVaR_α = E[L| L ≥ VaR_α]
        where L is the loss (negative return)
        
        Parameters
        ----------
        portfolio_returns : np.ndarray
            Portfolio returns
        alpha : float
            Confidence level (default 5%)
        
        Returns
        -------
        float
            CVaR at level α
        """
        var_alpha = np.percentile(portfolio_returns, alpha * 100)
        losses = np.minimum(portfolio_returns, var_alpha)
        return np.mean(losses)
    
    def maximum_drawdown(self, portfolio_returns: np.ndarray) -> float:
        """
        Compute maximum drawdown.
        
        MD = max(0, max_{t} - cumulative_returns_t) / max_{t} cumulative_returns_t
        """
        cumulative = np.cumprod(1 + portfolio_returns) - 1
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / (peak + 1)
        return np.min(drawdown)
    
    def return_constraint_satisfaction(self, portfolio_returns: np.ndarray,
                                      target_return: float) -> float:
        """
        Compute percentage of periods meeting return constraint.
        
        Returns
        -------
        float
            Percentage (0-100) of periods where return ≥ target
        """
        return 100 * np.mean(portfolio_returns >= target_return)
    
    def evaluate(self, weights: np.ndarray, returns: np.ndarray,
                target_return: Optional[float] = None,
                cvar_alpha: float = 0.05) -> Dict:
        """
        Comprehensive portfolio evaluation.
        
        Parameters
        ----------
        weights : np.ndarray
            Portfolio weights (m,)
        returns : np.ndarray
            Return matrix (T, m)
        target_return : float, optional
            Target return for constraint satisfaction
        cvar_alpha : float
            CVaR confidence level
        
        Returns
        -------
        dict
            Dictionary with all metrics
        """
        portfolio_ret = self.compute_portfolio_returns(weights, returns)
        
        metrics = {
            'mean_return': self.mean_return(portfolio_ret),
            'volatility': self.volatility(portfolio_ret),
            'sharpe_ratio': self.sharpe_ratio(portfolio_ret),
            'max_drawdown': self.maximum_drawdown(portfolio_ret),
            'cvar': self.cvar(portfolio_ret, cvar_alpha),
            'portfolio_returns': portfolio_ret,
        }
        
        if target_return is not None:
            metrics['constraint_satisfaction'] = self.return_constraint_satisfaction(
                portfolio_ret, target_return)
        
        return metrics


class PortfolioComparison:
    """
    Compare multiple portfolio strategies.
    """
    
    def __init__(self):
        """Initialize comparison tool."""
        self.results = {}
        self.evaluator = PortfolioEvaluator()
    
    def add_strategy(self, name: str, weights: np.ndarray, 
                    returns: np.ndarray, target_return: Optional[float] = None):
        """
        Add a strategy for evaluation.
        
        Parameters
        ----------
        name : str
            Strategy name
        weights : np.ndarray
            Portfolio weights (m,)
        returns : np.ndarray
            Return matrix (T, m)
        target_return : float, optional
            Target return for constraint satisfaction
        """
        metrics = self.evaluator.evaluate(weights, returns, target_return)
        self.results[name] = metrics
    
    def comparison_table(self) -> pd.DataFrame:
        """
        Create comparison table of all strategies.
        
        Returns
        -------
        pd.DataFrame
            Comparison with key metrics
        """
        data = []
        for name, metrics in self.results.items():
            row = {
                'Strategy': name,
                'Mean Return': metrics['mean_return'],
                'Volatility': metrics['volatility'],
                'Sharpe Ratio': metrics['sharpe_ratio'],
                'Max Drawdown': metrics['max_drawdown'],
                'CVaR(95%)': metrics['cvar'],
            }
            if 'constraint_satisfaction' in metrics:
                row['Constraint Satisfaction (%)'] = metrics['constraint_satisfaction']
            data.append(row)
        
        return pd.DataFrame(data)
    
    def print_comparison(self):
        """Print comparison table."""
        df = self.comparison_table()
        print("\n" + "="*100)
        print("STRATEGY COMPARISON")
        print("="*100)
        print(df.to_string(index=False))
        print("="*100 + "\n")


class RollingWindowBacktest:
    """
    Implement rolling window backtesting with rebalancing.
    """
    
    def __init__(self, returns: np.ndarray, window_size: int):
        """
        Initialize rolling window backtest.
        
        Parameters
        ----------
        returns : np.ndarray
            Full return matrix (T, m)
        window_size : int
            In-sample window size for training
        """
        self.returns = returns
        self.window_size = window_size
        self.T, self.m = returns.shape
    
    def generate_windows(self, test_period: int = 1):
        """
        Generate rolling windows for backtest.
        
        Yields
        ------
        tuple
            (train_returns, test_returns, train_dates, test_dates)
        """
        for t in range(self.window_size, self.T - test_period + 1):
            train_returns = self.returns[:t]
            test_returns = self.returns[t:t+test_period]
            
            yield train_returns, test_returns, t, t + test_period
    
    def backtest_strategy(self, strategy_fn, **kwargs) -> Dict:
        """
        Run rolling window backtest on a strategy.
        
        Parameters
        ----------
        strategy_fn : callable
            Function that takes (train_returns, **kwargs) and returns weights
        **kwargs : dict
            Additional arguments for strategy_fn
        
        Returns
        -------
        dict
            Backtest results with portfolio returns, weights path, etc.
        """
        cumulative_returns = []
        weights_history = []
        test_returns_all = []
        
        for train_ret, test_ret, t_start, t_end in self.generate_windows():
            try:
                # Compute weights on training data
                weights = strategy_fn(train_ret, **kwargs)
                
                if weights is None:
                    continue
                
                # Evaluate on test period
                test_port_ret = test_ret @ weights
                cumulative_returns.append(test_port_ret[0])
                weights_history.append(weights)
                test_returns_all.append(test_port_ret)
                
            except Exception as e:
                print(f"Error at period {t_start}: {e}")
                continue
        
        return {
            'cumulative_returns': np.array(cumulative_returns),
            'weights_history': np.array(weights_history),
            'test_returns': np.concatenate(test_returns_all),
            'num_rebalances': len(cumulative_returns),
        }


class TurnovoverMetrics:
    """
    Compute portfolio turnover and related transaction metrics.
    """
    
    @staticmethod
    def compute_turnover(weights_history: np.ndarray) -> np.ndarray:
        """
        Compute turnover at each rebalancing point.
        
        Turnover = sum(|w_t - w_{t-1}|) / 2
        
        Parameters
        ----------
        weights_history : np.ndarray
            Weight history (T_rebal, m)
        
        Returns
        -------
        np.ndarray
            Turnover at each period (T_rebal-1,)
        """
        turnovers = []
        for t in range(1, len(weights_history)):
            turnover = 0.5 * np.sum(np.abs(weights_history[t] - weights_history[t-1]))
            turnovers.append(turnover)
        return np.array(turnovers)
    
    @staticmethod
    def compute_concentration(weights: np.ndarray) -> float:
        """
        Compute Herfindahl index (concentration).
        
        HHI = sum(w_i²)
        
        Range: [1/m, 1] where 1/m is perfectly diversified
        """
        return np.sum(weights ** 2)
    
    @staticmethod
    def num_nonzero_weights(weights: np.ndarray, threshold: float = 1e-5) -> int:
        """
        Count number of assets with nontrivial weight.
        
        Parameters
        ----------
        weights : np.ndarray
            Portfolio weights
        threshold : float
            Minimum weight to count
        
        Returns
        -------
        int
            Number of assets with weight > threshold
        """
        return np.sum(weights > threshold)


def demo_evaluation():
    """Demo evaluation functions."""
    np.random.seed(42)
    T, m = 252, 5
    returns = np.random.randn(T, m) * 0.01 + 0.0005
    weights = np.ones(m) / m
    
    evaluator = PortfolioEvaluator()
    metrics = evaluator.evaluate(weights, returns, target_return=0.0001)
    
    print("Portfolio Metrics (Equal Weight):")
    for key, val in metrics.items():
        if key != 'portfolio_returns':
            print(f"  {key}: {val:.6f}")


if __name__ == "__main__":
    demo_evaluation()
