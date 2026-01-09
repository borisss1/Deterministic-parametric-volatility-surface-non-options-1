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

y = vol_data.iloc[:30].values
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

t0 = 0.2

ticker = exchange.fetch_ticker('BTC/USDT')
S_current = ticker['last']
print(f"Live BTC price: ${S_current:,.2f}")

interp_func = RegularGridInterpolator((S_grid, t_grid), sigma_surface)
current_vol = float(interp_func([[S_current, t0]]))

target_risk = 0.1
position_size = target_risk / current_vol

print(f"Current volatility: {current_vol:.2%}")
print(f"Risk-adjusted position size fraction: {position_size:.2f}")
