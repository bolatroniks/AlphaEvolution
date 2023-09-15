#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 10:03:11 2023

@author: joanna
"""

import numpy as np
import pandas as pd
import csv
import filelock

from typing import Optional, Union


from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression

class RiskFactorModel:
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if not isinstance(cls._instance, cls):
            cls._instance = super(RiskFactorModel, cls).__new__(cls)
        return cls._instance

    def __init__(self, no_observations=260, no_risk_factors=10):
        self.no_observations = no_observations
        self.no_risk_factors = no_risk_factors
        self.risk_factor_timeseries = None

    def get_risk_factors(self, b_override=False):
        if self.risk_factor_timeseries is not None and not b_override:
            return self.risk_factor_timeseries

        # Set a seed for reproducibility
        np.random.seed(0)

        # Generate data
        risk_factors = np.random.randn(self.no_observations, self.no_risk_factors)

        # Convert to DataFrame for easier handling and visualization
        self.risk_factor_timeseries = pd.DataFrame(risk_factors, columns=['RiskFactor_' + str(i) for i in range(1, self.no_risk_factors + 1)])

        return self.risk_factor_timeseries
    
    
class AssetPool:
    def __init__(self, risk_factor_model, no_assets, 
                 modellable_percentage=0.25, 
                 seed=1013):  # 1013 is a 4-digit prime number
        self.rfm = risk_factor_model
        self.no_assets = no_assets
        self.modellable_percentage = modellable_percentage
        self.betas = self._generate_betas()
        self.modellable_timeseries = None
        self.nmrf_timeseries = None
        self.seed = seed

    def _generate_betas(self):
        # Generate log-normally distributed betas for the first risk factor with mean around 0.5
        mu = np.log(0.5)
        sigma = 1  # Adjust as required
        first_factor = np.random.lognormal(mean=mu, sigma=sigma, size=(self.no_assets, 1))
    
        # Generate exponentially decreasing values for the remaining risk factors
        base = 0.5  # Base value, you can adjust if needed
        decay_rate = 0.5  # Exponential decay rate, adjust as required
        decreasing_values = np.array([base * np.exp(-decay_rate * i) for i in range(self.rfm.no_risk_factors - 1)])
        other_factors = np.tile(decreasing_values, (self.no_assets, 1))
    
        # Shuffle exposures for risk factors 2 through 10 for each asset
        for i in range(self.no_assets):
            np.random.shuffle(other_factors[i])
    
        # Combine the first and other risk factors
        betas = np.hstack([first_factor, other_factors])
    
        # Normalize each row such that the combined variance due to risk factors is 0.80
        factor_normalizer = 0.8944
        betas = betas / np.linalg.norm(betas, axis=1, keepdims=True) * factor_normalizer
    
        # Generate idiosyncratic component such that its variance is 0.20
        idiosyncratic_component = np.full((self.no_assets, 1), 0.4472)
    
        # Append the idiosyncratic component to the betas matrix
        betas = np.hstack([betas, idiosyncratic_component])
    
        return betas


    def split_pool(self):
        num_modellable = int(self.no_assets * self.modellable_percentage)
        modellable_betas = self.betas[:num_modellable]
        nmrf_betas = self.betas[num_modellable:]
        
        return modellable_betas, nmrf_betas
    
    def get_timeseries(self, b_override=False):
        if self.modellable_timeseries is not None and self.nmrf_timeseries is not None and not b_override:
            return self.modellable_timeseries, self.nmrf_timeseries

        # Get the risk factor time series
        risk_factors = self.rfm.get_risk_factors()

        # Initialize a placeholder for the timeseries
        asset_timeseries = np.zeros((self.rfm.no_observations, self.no_assets))
        
        # Set the seed for reproducibility
        np.random.seed(self.seed)

        # For each asset, generate its time series
        for i in range(self.no_assets):
            # Component due to risk factors
            factor_component = np.dot(risk_factors.values, self.betas[i, :-1])

            # Idiosyncratic component; use the last beta as a scaling factor
            idiosyncratic_component = self.betas[i, -1] * np.random.randn(self.rfm.no_observations)

            # Sum the factor component and the idiosyncratic component
            asset_timeseries[:, i] = factor_component + idiosyncratic_component

        # Split the asset time series based on modellable_percentage
        num_modellable = int(self.no_assets * self.modellable_percentage)
        self.modellable_timeseries = pd.DataFrame (
                asset_timeseries[:, :num_modellable])
        self.nmrf_timeseries = pd.DataFrame (
                asset_timeseries[:, num_modellable:])

        return self.modellable_timeseries, self.nmrf_timeseries
    
    
class Diagnostic ():
    
    def __init__ (self, modellable_ts, nmrf_ts, residuals=None):
        self.modellable_ts = modellable_ts
        self.nmrf_ts = nmrf_ts
        self.residuals = (residuals if residuals is not None else nmrf_ts)
        self.C = None
        self.clusters = None
        self.avg_cluster_corr_matrix = None
        
    def get_correlation_matrix (self, b_override=False):
        if (self.C is not None) and (not b_override):
            return self.C
        self.C = np.corrcoef(self.residuals, rowvar=False)
        
        return self.C
    
    def get_clusters (self, max_clusters=5, b_override=False):
        if (self.clusters is not None) and (not b_override):
            return self.clusters
        
        correlation_matrix = self.get_correlation_matrix (b_override)

        # Hierarchical clustering
        Z = linkage(correlation_matrix, method='average')
        
        # Generate clusters, for instance, cutting the dendrogram to form max 5 clusters
        self.clusters = fcluster(Z, max_clusters, criterion='maxclust')
        
        return self.clusters
    
    
    def _average_correlation_between_clusters(self, cluster_i, cluster_j):
        """
        Compute the average correlation between all time series in cluster i and all time series in cluster j.
        """
        return np.mean(self.get_correlation_matrix() [np.ix_(cluster_i, cluster_j)])
    
    
    def get_cluster_corr_matrix (self, max_clusters=5, b_override=False):
        if (self.avg_cluster_corr_matrix is not None) and (not b_override):
            return self.avg_cluster_corr_matrix
        
        clusters = self.get_clusters (max_clusters, b_override)
        # Get the unique clusters
        unique_clusters = np.unique(clusters)
        # Initialize a matrix to store the average correlations between clusters
        self.avg_cluster_corr_matrix = np.zeros((len(unique_clusters), len(unique_clusters)))
        
        for i, cluster_i in enumerate(unique_clusters):
            for j, cluster_j in enumerate(unique_clusters):
                indices_i = np.where(clusters == cluster_i)[0]
                indices_j = np.where(clusters == cluster_j)[0]
                
                self.avg_cluster_corr_matrix[i, j] = self._average_correlation_between_clusters(indices_i, indices_j)
                
        return pd.DataFrame(self.avg_cluster_corr_matrix)


class BaseModel:
    def __init__(self, modellable_ts, nmrf_ts, horizon=10):
        self.modellable_ts = modellable_ts.rolling (window=horizon).sum ().dropna ()
        self.nmrf_ts = nmrf_ts.rolling (window=horizon).sum ().dropna ()

    def find_proxies(self, target_series, n_proxies=1) -> list:
        """
        Returns the top N modellable time series (proxies) based on correlation to the target series.
        """
        if type (target_series) == int:
            target_series = self.nmrf_ts.iloc [:, target_series]
            
        correlations = self.modellable_ts.corrwith(target_series)
        top_proxies = correlations.nlargest(n_proxies).index.tolist()
        return top_proxies

    def run_regression(self, target_series, proxy=None):
        """
        This method should be overridden by child classes to implement specific regression mechanisms.
        """
        raise NotImplementedError("run_regression method has not been implemented.")


class OneFactorLinearRegression(BaseModel):

    def find_proxy(self, target_series: Union [int, pd.core.series.Series]):
        """
        Find the modellable timeseries with the highest correlation to the target NMRF timeseries.
        """
        max_corr = -1
        best_proxy = None
        
        if type (target_series) == int:
            target_series = self.nmrf_ts.iloc [:, target_series]        
        
        y = target_series.values

        for col in self.modellable_ts.columns:
            corr, _ = pearsonr(self.modellable_ts[col], target_series)
            if abs(corr) > max_corr:
                max_corr = abs(corr)
                best_proxy = col

        return best_proxy

    def run_regression(self, target_series, proxy=None):
        """
        Run a one-factor linear regression of the target NMRF timeseries against the given proxy.
        If no proxy is provided, use the one with the highest correlation.
        """
        if proxy is None:
            proxy = self.find_proxy(target_series)
    
        # Reshape data
        X = self.modellable_ts[proxy].values.reshape(-1, 1)
        
        if type (target_series) == int:
            target_series = self.nmrf_ts.iloc [:, target_series]
        y = target_series.values
    
        # Perform linear regression
        model = LinearRegression().fit(X, y)
        beta = model.coef_[0]
        residuals = y - model.predict(X)
    
        # Compute R^2 and adjusted R^2
        r2 = model.score(X, y)
        n = len(y)
        p = 1  # Number of predictors
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
        return {
            'beta': beta,
            'adj_r2': adj_r2,
            'residuals': residuals,
            'ratio': (np.std (residuals) / np.std (y)) ** 2
        }


class MultiLinearRegression(BaseModel):
    def __init__(self, modellable_ts, nmrf_ts, n_components=1, horizon=10):
        super().__init__(modellable_ts, nmrf_ts, horizon=horizon)
        self.n_components = n_components

    def run_regression(self, target_series, proxies=None):
        """
        Run a multivariate linear regression of the target NMRF timeseries against the given proxies.
        If no proxies are provided, use the top N modellable time series based on correlation.
        """
        if proxies is None or len(proxies) != self.n_components:
            proxies = self.find_proxies(target_series, self.n_components)

        X = self.modellable_ts[proxies].values
        if type (target_series) == int:
            target_series = self.nmrf_ts.iloc [:, target_series]
        y = target_series.values

        # Perform linear regression
        model = LinearRegression().fit(X, y)
        betas = model.coef_
        residuals = y - model.predict(X)

        # Compute R^2 and adjusted R^2
        r2 = model.score(X, y)
        n = len(y)
        p = len(proxies)  # Number of predictors
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

        return {
            'betas': betas,
            'adj_r2': adj_r2,
            'residuals': residuals,
            'ratio': (np.std (residuals) / np.std (y)) ** 2
        }
        
class PLS_Regression(BaseModel):
    def __init__(self, modellable_ts, nmrf_ts, n_components=1, horizon=10):
        super().__init__(modellable_ts, nmrf_ts, horizon=horizon)
        self.n_components = n_components

    def run_regression(self, target_series, proxies=None):
        """
        Run a PLS regression of the target NMRF timeseries against the modellable timeseries.
        """
        X = self.modellable_ts.values
        if type (target_series) == int:
            target_series = self.nmrf_ts.iloc [:, target_series]
        y = target_series.values

        # Perform PLS regression
        pls = PLSRegression(n_components=self.n_components)
        pls.fit(X, y)
        
        betas = pls.coef_.flatten()
        y_pred = pls.predict(X).flatten()
        residuals = y - y_pred

        # Compute R^2 and adjusted R^2
        r2 = pls.score(X, y)
        n = len(y)
        p = self.n_components
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

        return {
            'betas': betas,
            'adj_r2': adj_r2,
            'residuals': residuals,
            'ratio': (np.std (residuals) / np.std (y)) ** 2
        }
        
        
class PLS_MultiRegression(BaseModel):
    def __init__(self, modellable_ts, nmrf_ts, n_components=1, horizon=10):
        super().__init__(modellable_ts, nmrf_ts, horizon=horizon)
        self.n_components = n_components

    def run_regression(self, target_series_list):
        """
        Run a PLS regression of the target NMRF timeseries list against the modellable timeseries.
        """
        X = self.modellable_ts.values
        y = self.nmrf_ts[target_series_list].values

        # Perform PLS regression
        pls = PLSRegression(n_components=self.n_components)
        pls.fit(X, y)
        
        betas = pls.coef_
        y_pred = pls.predict(X)
        residuals = y - y_pred

        # Compute R^2 for each target timeseries
        r2_values = []
        for i in range(y.shape[1]):
            rss = np.sum((y[:, i] - y_pred[:, i]) ** 2)
            tss = np.sum((y[:, i] - y[:, i].mean()) ** 2)
            r2_values.append(1 - rss/tss)

        # Compute adjusted R^2 for each target timeseries
        adj_r2_values = []
        n = len(y)
        p = self.n_components
        for r2 in r2_values:
            adj_r2_values.append(1 - (1 - r2) * (n - 1) / (n - p - 1))

        return {
            'betas': betas,
            'adj_r2': adj_r2_values,
            'residuals': residuals,
            'ratios': (pd.DataFrame(residuals).std () ** 2) / (pd.DataFrame(y).std () ** 2)
        }
        
def write_to_csv(res, filename='output.csv'):
    # Create or acquire the file lock
    with filelock.FileLock(filename + ".lock"):
        # Open the file in append mode
        with open(filename, 'a', newline='') as file:
            writer = csv.writer(file)

            # Check if file is empty to write headers
            if file.tell() == 0:
                writer.writerow(res.keys())

            writer.writerow(res.values())

def has_been_run(no_clusters, horizon, n_components, no_observations, 
                 no_risk_factors, no_assets, modellable_percentage, filename):
    # Try to get a file lock, this ensures no other process is currently writing to the file
    with filelock.FileLock(filename + ".lock"):
        try:
            with open(filename, 'r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    # Check if each parameter in the row matches the given parameters
                    if (int(row["no_clusters"]) == no_clusters and 
                        int(row["horizon"]) == horizon and 
                        int(row["n_components"]) == n_components and 
                        int(row["no_observations"]) == no_observations and 
                        int(row["no_risk_factors"]) == no_risk_factors and 
                        int(row["no_assets"]) == no_assets and 
                        float(row["modellable_percentage"]) == modellable_percentage):
                        return True
        except FileNotFoundError:
            # File hasn't been created yet, so return False
            return False
    return False

def run_simulation (            
            horizon = 1,
            n_components = 10,
            no_observations = 1000,
            no_risk_factors = 10,
            no_assets = 1000,
            modellable_percentage = 0.2,
            no_clusters = 20,
            filename='output.csv'
            ):
    
    if has_been_run(no_clusters, horizon, n_components, no_observations, 
                    no_risk_factors, no_assets, modellable_percentage, filename):
        print("Simulation with these parameters has already been run and saved.")
        return

    # Example
    rfm = RiskFactorModel(no_observations + (horizon - 1), no_risk_factors)
    pool = AssetPool(rfm, no_assets=no_assets, 
                     modellable_percentage=modellable_percentage)  # 100 assets
    modellable_ts, nmrf_ts = pool.get_timeseries()
    
    diag_full = Diagnostic (modellable_ts, nmrf_ts)
    
    print (f"trace: {1./no_clusters*np.trace(diag_full.get_cluster_corr_matrix (max_clusters=no_clusters, b_override=True))}")
    
    
    #------1F linear regression-------
    oneF = OneFactorLinearRegression (modellable_ts=modellable_ts, nmrf_ts=nmrf_ts, horizon=horizon)
    res1F = [oneF.run_regression(i) for i in range (len (pool.nmrf_timeseries.columns))]
    
    
    #plt.plot ([_["adj_r2"] for _ in res1F], label ="R2_1F")
    
    
    residuals_1F = pd.DataFrame(np.transpose([_["residuals"] for _ in res1F]))
    
    diag_1F = Diagnostic (modellable_ts, nmrf_ts, residuals_1F)
    print (f"trace: {1./no_clusters*np.trace(diag_1F.get_cluster_corr_matrix (max_clusters=10, b_override=True))}")
    
    
    #------Multivariate linear regression-------
    multiF = MultiLinearRegression (modellable_ts=modellable_ts, nmrf_ts=nmrf_ts, 
                                  n_components = n_components,
                                  horizon=horizon)
    res_multi = [multiF.run_regression(i) for i in range (len (pool.nmrf_timeseries.columns))]
    
    
    
    #plt.plot ([_["adj_r2"] for _ in res_multi], label ="R2_5F")
    
    
    residuals_multi = pd.DataFrame(np.transpose([_["residuals"] for _ in res_multi]))
    
    diag_multi = Diagnostic (modellable_ts, nmrf_ts, residuals_multi)
    print (f"trace: {1./no_clusters*np.trace(diag_multi.get_cluster_corr_matrix (max_clusters=10, b_override=True))}")
    
    #------PLS 1-------------------------------------------
    pls_reg = PLS_Regression (modellable_ts=modellable_ts, nmrf_ts=nmrf_ts, 
                                  n_components = n_components,
                                  horizon=horizon)
    res_pls = [pls_reg.run_regression(i) for i in range (len (pool.nmrf_timeseries.columns))]
    
    
    #----------PLS 2
    pls2 = PLS_MultiRegression (modellable_ts, nmrf_ts, n_components=n_components, horizon=horizon)
    res_pls2 = {}
    res_pls2_per_cluster = {}
    for cluster in np.unique(diag_1F.clusters):
        res_pls2_per_cluster [cluster] = pls2.run_regression (target_series_list=nmrf_ts.loc[:, diag_1F.clusters == cluster].columns)
    
    res_pls2 ["ratios"] = []
    res_pls2 ["adj_r2"] = []
    for k in res_pls2_per_cluster.keys ():
        res_pls2 ["ratios"] += list(res_pls2_per_cluster [k]["ratios"].values)
        res_pls2 ["adj_r2"] += res_pls2_per_cluster [k]["adj_r2"]
        
    #---------plots-----------------
    fig = plt.figure (figsize=(14, 8))
    plt.title ("Variance ratios")
    plt.plot (sorted([_["ratio"] for _ in res1F]), label="Ratio_1F")
    plt.plot (sorted([_["ratio"] for _ in res_multi]), label="Ratio_5F")
    plt.plot (sorted([_["ratio"] for _ in res_pls]), label="PLS_multi")
    plt.plot (sorted(res_pls2["ratios"]), label="PLS2")
    
    plt.legend (loc="best")
    plt.grid (True)
    plt.show ()
    
    fig = plt.figure (figsize=(14, 8))
    plt.title ("Adj_r2")
    plt.plot (sorted([_["adj_r2"] for _ in res1F]), label="Ratio_1F")
    plt.plot (sorted([_["adj_r2"] for _ in res_multi]), label="Ratio_5F")
    plt.plot (sorted([_["adj_r2"] for _ in res_pls]), label="PLS_multi")
    plt.plot (sorted(res_pls2["adj_r2"]), label="PLS2")
    
    plt.legend (loc="best")
    plt.grid (True)
    plt.show ()
    
    res = {
            "no_clusters": no_clusters,
            "horizon": horizon,
            "n_components": 10,
            "no_observations": no_observations,
            "no_risk_factors": no_risk_factors,
            "no_assets": no_assets,
            "modellable_percentage": modellable_percentage,        
            
            "1F_5pct_r2": np.percentile ([_["adj_r2"] for _ in res1F], 5),
            "1F_mean_r2": np.mean ([_["adj_r2"] for _ in res1F]),
            "1F_95pct_r2": np.percentile ([_["adj_r2"] for _ in res1F], 95),
            "1F_5pct_ratios": np.percentile ([_["ratio"] for _ in res1F], 5),        
            "1F_95pct_ratios": np.percentile ([_["ratio"] for _ in res1F], 95),
            
            "MLR_5pct_r2": np.percentile ([_["adj_r2"] for _ in res_multi], 5),
            "MLR_mean_r2": np.mean ([_["adj_r2"] for _ in res_multi]),
            "MLR_95pct_r2": np.percentile ([_["adj_r2"] for _ in res_multi], 95),
            "MLR_5pct_ratios": np.percentile ([_["ratio"] for _ in res_multi], 5),
            "MLR_95pct_ratios": np.percentile ([_["ratio"] for _ in res_multi], 95),
            
            "pls1_5pct_r2": np.percentile ([_["adj_r2"] for _ in res_pls], 5),
            "pls1_mean_r2": np.mean ([_["adj_r2"] for _ in res_pls]),
            "pls1_95pct_r2": np.percentile ([_["adj_r2"] for _ in res_pls], 95),
            "pls1_5pct_ratios": np.percentile ([_["ratio"] for _ in res_pls], 5),
            "pls1_95pct_ratios": np.percentile ([_["ratio"] for _ in res_pls], 95),
            
            "pls2_5pct_r2": np.percentile (res_pls2["adj_r2"], 5),
            "pls2_mean_r2": np.mean (res_pls2["adj_r2"]),
            "pls2_95pct_r2": np.percentile (res_pls2["adj_r2"], 95),
            "pls2_5pct_ratios": np.percentile (res_pls2["ratios"], 5),
            "pls2_95pct_ratios": np.percentile (res_pls2["ratios"], 95),
            
    }
    
    write_to_csv(res, filename)


from multiprocessing import Pool
import itertools

# Your run_simulation function...

def worker(args):
    """
    Worker function to unpack arguments and run the simulation.
    """
    return run_simulation(*args)

def main():
    horizons = [1, 5, 10]
    n_components_list = [3, 5, 10, 20]
    no_observations_list = [250, 500, 750, 1000]
    no_risk_factors_list = [3, 5, 10, 20]
    no_assets_list = [100, 500, 1000]
    
    # Generate all combinations
    combinations = list(itertools.product(horizons, n_components_list, no_observations_list,
                                          no_risk_factors_list, no_assets_list))
    
    # Use 4-7 cores. Adjust as needed.
    cores_to_use = 2#min(7, max(4, len(combinations)))
    
    with Pool(cores_to_use) as pool:
        pool.map(worker, combinations)

if __name__ == "__main__":
    main()