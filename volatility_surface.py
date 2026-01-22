import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import RegularGridInterpolator

exchange = ccxt.binance()
ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe='1d', limit=365)
df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

df['returns'] = np.log(df['close'] / df['close'].shift(1))
df.dropna(inplace=True)

daily_vol = df['returns'].std()
sigma_0 = daily_vol * np.sqrt(365)
print(f"Estimated sigma_0 (annualized): {sigma_0:.2%}")

df['vol_7d'] = df['returns'].rolling(window=7).std() * np.sqrt(365)
vol_data = df['vol_7d'].dropna()

y = vol_data.iloc[-30:].values
t = np.arange(1, len(y)+1)

def vol_decay(t, a, b):
    return sigma_0 + a * np.exp(-b * t)

popt, _ = curve_fit(vol_decay, t, y, p0=[0.5, 0.5])
a, b = popt
print(f"Estimated spike amplitude a: {a:.2%}")
print(f"Estimated decay rate b: {b:.2f}")

S0 = df['close'].mean()
S = df['close'].values
y_vol = df['vol_7d'].values

mask = ~np.isnan(y_vol)
S_rel = S[mask] / S0
y_vol_clean = y_vol[mask]

log_x = np.log(S_rel)
log_y = np.log(y_vol_clean)
gamma = np.polyfit(log_x, log_y, 1)[0]
print(f"Estimated gamma: {gamma:.2f}")
print(f"Reference price S0: {S0:.2f}")

T = 1.0
t_grid = np.linspace(0, T, 50)
S_grid = np.linspace(min(S), max(S), 50)
T_mesh, S_mesh = np.meshgrid(t_grid, S_grid)

sigma_surface = (sigma_0 + a * np.exp(-b * T_mesh)) * (S_mesh / S0)**gamma

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(T_mesh, S_mesh, sigma_surface, cmap='viridis', alpha=0.8)
contours = ax.contour(T_mesh, S_mesh, sigma_surface, zdir='z', offset=0, cmap='plasma', linewidths=1.5)

ax.set_xlabel('Time (years)')
ax.set_ylabel('BTC Price')
ax.set_zlabel('Volatility')
ax.set_title('Data-Driven BTC Volatility Surface')

fig.colorbar(surf, shrink=0.5, aspect=10, label='Volatility')
plt.show()

risk_free_rate = 0.04
expected_return = 0.15
investor_gamma = 3.0

current_vol = y_vol_clean[-1]
variance = current_vol**2
myopic_demand = (expected_return - risk_free_rate) / (investor_gamma * variance)

df['vol_change'] = df['vol_7d'].diff()
rho = df['returns'].corr(df['vol_change'])

current_price = df['close'].iloc[-1]
d_sigma_dS = gamma * (current_vol / current_price)

hedging_demand = -(rho * a * b) / (investor_gamma * current_vol) 

total_allocation = myopic_demand + hedging_demand

print(f"Current Volatility: {current_vol:.2%}")
print(f"Price-Vol Correlation (Rho): {rho:.2f}")
print(f"Myopic Allocation: {myopic_demand:.2%}")
print(f"Hedging Allocation: {hedging_demand:.2%}")
print(f"Total Optimal Weight: {total_allocation:.2%}")

labels = ['Myopic', 'Hedging']
values = [myopic_demand, hedging_demand]

plt.figure(figsize=(8, 5))
plt.bar(labels, values, color=['blue', 'orange'])
plt.axhline(0, color='black', linewidth=0.8)
plt.title('Merton Optimal Portfolio Decomposition')
plt.ylabel('Allocation Weight')
plt.show()
