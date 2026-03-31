"""
Data Loading Module for COLCAP DRO Portfolio Optimization
===========================================================

This module handles:
1. Loading all CSV files from Datos_empresas_colcap folder
2. Computing log returns from price history
3. Constructing the empirical distribution
4. Out-of-sample data splitting

Dependencies: pandas, numpy
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Optional, Generator
from datetime import datetime


class COLCAPDataLoader:
    """
    Loads and processes Colombian COLCAP stock market data.
    
    Attributes
    ----------
    data_path : str
        Path to the Datos_empresas_colcap folder
    returns_matrix : np.ndarray
        (T, m) matrix of returns: T time periods, m assets
    dates : np.ndarray
        Trading dates corresponding to returns
    tickers : list
        Asset names/tickers
    
    """
    
    def __init__(self, data_path: str):
        """
        Initialize data loader.
        
        Parameters
        ----------
        data_path : str
            Path to Datos_empresas_colcap folder
        """
        self.data_path = Path(data_path)
        self.returns_matrix = None
        self.dates = None
        self.tickers = []
        self.price_data = {}
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path not found: {data_path}")
    
    def load_all_csv(self) -> pd.DataFrame:
        """
        Load all CSV files from the data folder.
        
        Returns
        -------
        pd.DataFrame
            Combined dataframe with all asset returns
        """
        csv_files = list(self.data_path.glob("*.csv"))
        
        if len(csv_files) == 0:
            raise FileNotFoundError(f"No CSV files found in {self.data_path}")
        
        print(f"Found {len(csv_files)} CSV files")
        
        # Load each CSV and compute returns
        data_dict = {}
        
        for csv_file in csv_files:
            try:
                # Extract ticker from filename
                ticker = csv_file.stem.replace(" Stock Price History", "").strip()
                
                # Read CSV
                df = pd.read_csv(csv_file)
                
                # Expected columns: Date, Price or Close
                if 'Price' in df.columns:
                    df['Close'] = df['Price']
                elif 'Close' not in df.columns:
                    # Try to find price column
                    price_cols = [col for col in df.columns if 'price' in col.lower()]
                    if price_cols:
                        df['Close'] = df[price_cols[0]]
                    else:
                        print(f"Warning: No price column found in {csv_file.name}")
                        continue
                
                # Parse date
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                else:
                    # Try to infer date column
                    date_cols = [col for col in df.columns if 'date' in col.lower()]
                    if date_cols:
                        df['Date'] = pd.to_datetime(df[date_cols[0]])
                    else:
                        print(f"Warning: No date column found in {csv_file.name}")
                        continue
                
                # Sort by date
                df = df.sort_values('Date').reset_index(drop=True)
                
                # Store original prices
                self.price_data[ticker] = df.set_index('Date')['Close'].copy()
                
                # Compute log returns: r_t = log(P_t / P_{t-1})
                # Clean prices (remove commas if present)
                prices_raw = df['Close'].astype(str).str.replace(',', '')
                prices = pd.to_numeric(prices_raw, errors='coerce').values
                
                # Check for valid prices
                if np.any(np.isnan(prices)) or np.any(prices <= 0):
                    print(f"Warning: Invalid prices in {ticker}")
                    continue
                
                log_returns = np.diff(np.log(prices))
                
                # Create returns series with dates
                return_dates = df['Date'].values[1:]
                data_dict[ticker] = log_returns
                
                print(f"[OK] Loaded {ticker}: {len(log_returns)} returns")
                
            except Exception as e:
                print(f"[ERROR] Loading {csv_file.name}: {e}")
                continue
        
        # Reconstruct dates for each ticker from the original data
        all_dates_dict = {}
        for csv_file in csv_files:
            try:
                ticker = csv_file.stem.replace(" Stock Price History", "").strip()
                if ticker not in data_dict:
                    continue
                    
                df_temp = pd.read_csv(csv_file)
                
                # Parse date
                if 'Date' in df_temp.columns:
                    df_temp['Date'] = pd.to_datetime(df_temp['Date'])
                elif any('date' in col.lower() for col in df_temp.columns):
                    date_cols = [col for col in df_temp.columns if 'date' in col.lower()]
                    df_temp['Date'] = pd.to_datetime(df_temp[date_cols[0]])
                
                # Sort by date
                df_temp = df_temp.sort_values('Date').reset_index(drop=True)
                all_dates_dict[ticker] = df_temp['Date'].values[1:]  # Exclude first date (for log returns)
            except:
                continue
        
        # Create unified dataframe with proper date alignment
        # Check if all tickers have same dates (common in pre-aligned data like imputed data)
        date_sets = [set(dates) for dates in all_dates_dict.values()]
        
        # If all tickers have same dates, use direct merge (more efficient)
        if len(date_sets) > 0 and all(dates == date_sets[0] for dates in date_sets):
            print("\n[INFO] All tickers have identical dates - using direct merge")
            common_dates = sorted(list(date_sets[0]))
            df_combined = pd.DataFrame()
            for ticker in data_dict.keys():
                df_combined[ticker] = data_dict[ticker]
            # Reindex to ensure alignment
            df_combined.index = pd.to_datetime(common_dates)
        else:
            # Find common date range (intersection of all dates)
            if len(date_sets) == 0:
                raise ValueError("Could not extract dates from any file")
            
            common_dates_set = set.intersection(*date_sets) if date_sets else set()
            common_dates = sorted(list(common_dates_set))
            
            if len(common_dates) == 0:
                # If no perfect intersection, use the first ticker's dates
                first_ticker = list(all_dates_dict.keys())[0]
                common_dates = all_dates_dict[first_ticker]
                print(f"\n[WARNING] No common dates found - using {first_ticker}'s dates")
            
            # Build aligned dataframe
            df_combined = pd.DataFrame()
            for ticker in data_dict.keys():
                if ticker in all_dates_dict:
                    ticker_dates = all_dates_dict[ticker]
                    ticker_returns = data_dict[ticker]
                    
                    # Create mapping from date to return
                    date_to_return = {date: ret for date, ret in zip(ticker_dates, ticker_returns)}
                    
                    # Align to common dates
                    aligned_returns = []
                    for date in common_dates:
                        if date in date_to_return:
                            aligned_returns.append(date_to_return[date])
                        else:
                            aligned_returns.append(np.nan)
                    
                    df_combined[ticker] = aligned_returns
            
            # Set date index
            df_combined.index = pd.to_datetime(common_dates)
            
            # Remove any NaN rows
            df_combined = df_combined.dropna()
        
        self.returns_matrix = df_combined.values
        self.dates = df_combined.index.values
        self.tickers = df_combined.columns.tolist()
        
        print(f"\n[SUCCESS] Loaded complete dataset: {self.returns_matrix.shape[0]} periods x {self.returns_matrix.shape[1]} assets")
        print(f"  Tickers: {', '.join(self.tickers[:5])}..." if len(self.tickers) > 5 else f"  Tickers: {', '.join(self.tickers)}")
        
        return df_combined
    
    def get_statistics(self) -> Dict:
        """
        Compute descriptive statistics of returns.
        
        Returns
        -------
        dict
            Mean, std, min, max, skewness, kurtosis
        """
        stats = {
            'mean': np.mean(self.returns_matrix, axis=0),
            'std': np.std(self.returns_matrix, axis=0),
            'min': np.min(self.returns_matrix, axis=0),
            'max': np.max(self.returns_matrix, axis=0),
            'skewness': self._compute_skewness(),
            'kurtosis': self._compute_kurtosis(),
        }
        return stats
    
    def _compute_skewness(self) -> np.ndarray:
        """Compute skewness for each asset."""
        centered = self.returns_matrix - np.mean(self.returns_matrix, axis=0)
        return np.mean(centered**3, axis=0) / np.std(self.returns_matrix, axis=0)**3
    
    def _compute_kurtosis(self) -> np.ndarray:
        """Compute excess kurtosis for each asset."""
        centered = self.returns_matrix - np.mean(self.returns_matrix, axis=0)
        return np.mean(centered**4, axis=0) / np.std(self.returns_matrix, axis=0)**4 - 3
    
    def train_test_split(self, 
                        window_size: int,
                        test_size: int = 252) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Split data into in-sample and out-of-sample using rolling window.
        
        Parameters
        ----------
        window_size : int
            In-sample window size (e.g., 252 for 1 year of trading days)
        test_size : int
            Out-of-sample test size
        
        Yields
        ------
        tuple
            (returns_train, returns_test) for each rolling window
        """
        T = self.returns_matrix.shape[0]
        
        for t in range(window_size, T - test_size):
            train = self.returns_matrix[:t]
            test = self.returns_matrix[t:t+test_size]
            yield train, test
    
    def get_returns(self) -> np.ndarray:
        """Get returns matrix (T, m)."""
        return self.returns_matrix
    
    def get_tickers(self) -> list:
        """Get list of tickers."""
        return self.tickers
    
    def get_sample_mean_cov(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute empirical mean and covariance.
        
        Returns
        -------
        tuple
            (sample_mean, sample_cov)
        """
        sample_mean = np.mean(self.returns_matrix, axis=0)
        # Use biased covariance (dividing by T, not T-1)
        T = self.returns_matrix.shape[0]
        centered = self.returns_matrix - sample_mean
        sample_cov = (centered.T @ centered) / T
        
        return sample_mean, sample_cov
    
    def summary_statistics(self):
        """Print summary statistics."""
        print("\n" + "="*70)
        print("SUMMARY STATISTICS")
        print("="*70)
        
        stats = self.get_statistics()
        mean, cov = self.get_sample_mean_cov()
        
        print(f"\nReturns Matrix Shape: {self.returns_matrix.shape} (T={self.returns_matrix.shape[0]}, m={self.returns_matrix.shape[1]})")
        print(f"\nAnnualized Statistics (assuming 252 trading days):")
        print(f"  Mean Returns (annualized):  {np.mean(mean)*252:.4f}")
        print(f"  Volatility (annualized):    {np.sqrt(np.diag(cov)).mean()*np.sqrt(252):.4f}")
        print(f"  Min Return:  {stats['min'].min():.6f}")
        print(f"  Max Return:  {stats['max'].max():.6f}")
        
        print(f"\nRisk-Return by Asset:")
        df_stats = pd.DataFrame({
            'Ticker': self.tickers,
            'Mean': mean,
            'Std': np.sqrt(np.diag(cov)),
            'Annualized Mean': mean * 252,
            'Annualized Std': np.sqrt(np.diag(cov)) * np.sqrt(252),
        })
        print(df_stats.to_string(index=False))
        print("\n" + "="*70)


def main():
    """Example usage."""
    # Path to data folder (use imputed data with aligned trading days)
    data_path = r"c:\Users\emngz\Downloads\Universidad\P.I,\PI 2\Datos_empresas_colcap_IMPUTADOS"
    
    # Load data
    loader = COLCAPDataLoader(data_path)
    loader.load_all_csv()
    loader.summary_statistics()
    
    # Get statistics
    stats = loader.get_statistics()
    mean, cov = loader.get_sample_mean_cov()
    print(f"\nSample Mean: {mean[:3]}...")
    print(f"Sample Cov shape: {cov.shape}")


if __name__ == "__main__":
    main()
