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

if __name__ == "__main__":
    metadata, sector_list = load_meta()
    prices = load_prices()

    # plot standard returns of each sector
    naive_port = get_naive_weight(prices)
    plots = plot_sectors(sector_list, prices, naive_port)
    plt.show()