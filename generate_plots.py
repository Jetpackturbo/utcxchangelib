import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def load_meta():
    csv = pd.read_csv('meta.csv')
    
    sector_nums = csv['sector_id'].unique()
    # make a list of lists of the sectors and the indices of each
    # each asset has a sector id, please list the asset in a list where sector index maps to a list of assets
    sector_list = [[] for _ in range(len(sector_nums))]
    for asset, sector_id in zip(csv['asset'], csv['sector_id']):
        sector_list[sector_id].append(asset)
    print(sector_list)
    return csv, sector_list

def load_prices():
    df = pd.read_csv('prices.csv')
    return df

def get_naive_weight(prices):
    # Drop the 'tick' column
    prices = prices.drop(columns=['tick'])
    
    # Calculate the mean across columns (axis=1) for each row
    # This automatically sums the day's prices and divides by the number of assets
    port_vals = prices.mean(axis=1).tolist()
    
    return port_vals

def get_covariance_matrix(prices):
    # 1. Drop the 'tick' column if it hasn't been removed yet
    if 'tick' in prices.columns:
        prices = prices.drop(columns=['tick'])
        
    # 2. Calculate daily returns (percentage change from the previous day)
    returns = prices.pct_change().dropna()
    
    # 3. Calculate the ANNUALIZED covariance matrix
    # Multiplying by 252 scales the tiny daily numbers up to standard yearly figures
    cov_matrix = returns.cov() * 252
    
    # Capture the figure object so we can save it later
    fig = plt.figure(figsize=(10, 8))
    
    # plot covariance matrix as a heatmap
    # Added fmt=".4f" to keep the decimals clean and readable
    sns.heatmap(cov_matrix, annot=True, cmap='coolwarm', fmt=".4f")
    
    # Updated title to reflect the annualization
    plt.title('Annualized Asset Covariance Matrix')
    plt.xlabel('Assets')
    plt.ylabel('Assets')
    plt.tight_layout()
    
    return fig, cov_matrix

def plot_sectors(sector_list, prices, naive_port):
    # Optional: store figures and axes in a list if you need to return or save them later
    plots = []
    
    # make one plot per sector
    for i, sector in enumerate(sector_list):
        # 1. Create a NEW figure and axis for this specific sector
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 2. Plot all assets in this sector
        for asset in sector:
            asset_returns = prices[asset]
            ax.plot(asset_returns, label=asset)
            
        # 3. Plot the naive portfolio benchmark on the same axis, dashed ('--')
        ax.plot(naive_port, label="Naive Portfolio", linestyle='--', color='black', linewidth=2)
        
        # 4. Add formatting and legend
        ax.set_title(f"Sector {i+1} Assets vs. Naive Portfolio Benchmark")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend()
        
        plots.append((fig, ax))
        
    # Return the list of generated figures/axes instead of just the last one
    return plots

def plot_sector_covariance(sector_list, prices):
    # 1. Drop 'tick' if present and calculate daily returns for all assets
    if 'tick' in prices.columns:
        prices = prices.drop(columns=['tick'])
    returns = prices.pct_change().dropna()
    
    # 2. Create a new DataFrame to hold the aggregate return for each sector
    sector_returns = pd.DataFrame()
    
    for i, sector_assets in enumerate(sector_list):
        # Filter for assets that actually exist in the dataframe
        valid_assets = [asset for asset in sector_assets if asset in returns.columns]
        
        # Calculate the equally-weighted daily return for the sector (row-wise mean)
        sector_name = f"Sector {i+1}"
        sector_returns[sector_name] = returns[valid_assets].mean(axis=1)
        
    # 3. Calculate the annualized covariance matrix between sectors
    # Multiply by 252 (approximate trading days in a year) to annualize it
    sector_cov_matrix = sector_returns.cov() * 252
    
    # 4. Plot the covariance matrix as a heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # cmap="coolwarm" provides a good color scale (blue for low, red for high)
    # annot=True prints the exact covariance numbers inside the boxes
    # fmt=".4f" rounds the numbers to 4 decimal places so it isn't cluttered
    sns.heatmap(sector_cov_matrix, annot=True, cmap="coolwarm", fmt=".4f", 
                linewidths=0.5, ax=ax)
    
    ax.set_title("Annualized Covariance Between Sectors", fontsize=14, pad=15)
    plt.tight_layout()
    
    # Return the figure, axis, and the actual matrix data in case you need it
    return fig, ax, sector_cov_matrix

def plot_ewma_sector_returns(sector_list, prices, naive_port_vals, span=20):
    """
    Plots the EWMA of returns for assets in each sector against a naive benchmark.
    
    span: The "lookback" period for the exponential moving average. 
          20 is roughly one trading month. Higher = smoother.
    """
    # 1. Calculate daily returns from prices (drops the first NaN row)
    returns = prices.pct_change().dropna()
    
    # 2. Calculate daily returns for the naive portfolio
    # Convert list to Pandas Series first to use .pct_change()
    naive_series = pd.Series(naive_port_vals)
    naive_returns = naive_series.pct_change().dropna()
    
    # 3. Calculate the EWMA of those returns
    # adjust=False calculates it using the standard recursive formula
    ewma_returns = returns.ewm(span=span, adjust=False).mean()
    ewma_naive = naive_returns.ewm(span=span, adjust=False).mean()
    
    plots = []
    
    # 4. Plot one graph per sector
    for i, sector in enumerate(sector_list):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot EWMA returns for each asset in the sector
        for asset in sector:
            # Ensure the asset actually exists in the columns to prevent KeyErrors
            if asset in ewma_returns.columns:
                ax.plot(ewma_returns[asset], label=asset)
                
        # Plot EWMA of the naive portfolio benchmark, dashed
        ax.plot(ewma_naive, label="Naive Portfolio EWMA", linestyle='--', color='black', linewidth=2)
        
        # Formatting
        ax.set_title(f"Sector {i+1}: {span}-Day EWMA of Returns vs Benchmark")
        ax.set_xlabel("Time (Trading Days)")
        ax.set_ylabel("EWMA of Returns")
        ax.legend()
        
        plots.append((fig, ax))
        
    return plots

if __name__ == "__main__":
    # Create the directory to store plots
    os.makedirs('plots', exist_ok=True)
    print("Saving plots to the 'plots/' directory...")

    metadata, sector_list = load_meta()
    prices = load_prices()
    naive_port = get_naive_weight(prices)

    # 1. Plot and Save Standard Returns of Each Sector
    # standard_plots = plot_sectors(sector_list, prices, naive_port)
    # for i, (fig, ax) in enumerate(standard_plots):
    #     fig.savefig(f"plots/sector_{i+1}_standard_returns.png", bbox_inches='tight')

    # 2. Plot and Save Asset Covariance Matrix
    cov_fig, cov = get_covariance_matrix(prices)
    cov_fig.savefig("plots/asset_covariance_matrix.png", bbox_inches='tight')
    
    # 3. Plot and Save Sector Covariance Matrix
    # sec_cov_fig, sec_cov_ax, sec_cov_matrix = plot_sector_covariance(sector_list, prices)
    # sec_cov_fig.savefig("plots/sector_covariance_matrix.png", bbox_inches='tight')

    # 4. Plot and Save EWMA Sector Returns
    # ewma_plots = plot_ewma_sector_returns(sector_list, prices, naive_port)
    # for i, (fig, ax) in enumerate(ewma_plots):
    #     fig.savefig(f"plots/sector_{i+1}_ewma_returns.png", bbox_inches='tight')

    print("All plots saved successfully!")
    
    # Show the plots interactively
    plt.show()