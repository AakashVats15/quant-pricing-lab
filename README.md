# **Derivatives Pricing Engine**

A modular, research‑grade derivatives pricing engine implementing stochastic simulation, analytical models, variance‑reduction, Greeks, implied volatility, and exotic payoffs. Designed for quantitative research teams who value **clarity, extensibility, and reproducibility**.

The architecture mirrors real quant libraries: clean separation of concerns, vectorized computation, and a workflow that scales from prototyping to production‑ready modeling.

---

## **📁 Project Structure**

```
derivatives-pricing-engine/
│
├── core/
│   ├── gbm.py                 # Risk‑neutral GBM simulator (vectorized)
│   ├── stochastic_process.py  # Base classes + interfaces for SDE models
│   └── utils.py               # RNG, seeds, math helpers
│
├── models/
│   ├── black_scholes.py       # Closed‑form BS pricing + analytical Greeks
│   ├── asian_options.py       # Geometric + arithmetic Asian pricing
│   └── payoffs.py             # Payoff abstractions (vanilla, Asian, custom)
│
├── monte_carlo/
│   ├── engine.py              # Generic MC engine (model‑agnostic)
│   ├── variance_reduction.py  # Antithetic + control variates
│   └── estimators.py          # Greeks via finite‑difference + pathwise
│
├── calibration/
│   └── implied_vol.py         # Newton–Raphson implied volatility solver
│
├── analysis/
│   ├── convergence.py         # Convergence diagnostics + error plots
│   └── benchmarks.py          # Runtime + accuracy benchmarking
│
├── notebooks/
│   ├── 01_gbm_simulation.ipynb
│   ├── 02_black_scholes.ipynb
│   ├── 03_monte_carlo_pricing.ipynb
│   ├── 04_variance_reduction.ipynb
│   ├── 05_greeks.ipynb
│   ├── 06_implied_vol.ipynb
│   └── 07_asian_options.ipynb
│
├── tests/
│   ├── test_black_scholes.py
│   ├── test_monte_carlo.py
│   ├── test_variance_reduction.py
│   └── test_asian_options.py
│
└── README.md
```

---

## **🔧 Features**

### **Stochastic Simulation**
- Risk‑neutral GBM with vectorized path generation  
- Supports batching, reproducibility, and custom seeds  
- Designed to plug directly into the Monte Carlo engine  

### **Analytical Models**
- Black–Scholes closed‑form pricing  
- Analytical Greeks (Delta, Gamma, Vega, Theta, Rho)  
- Used as a benchmark for Monte Carlo convergence and control variates  

### **Monte Carlo Engine**
- Model‑agnostic architecture  
- Supports any payoff + any stochastic process  
- Parallelizable and optimized for large‑scale simulations  

### **Variance Reduction**
- Antithetic variates  
- Control variates using BS analytical prices  
- Modular design for adding new VR techniques  

### **Greeks**
- Analytical (BS)  
- Finite‑difference (bump‑and‑revalue)  
- Pathwise estimators where applicable  

### **Implied Volatility**
- Newton–Raphson solver  
- Convergence checks + fallback strategies  

### **Asian Options**
- Geometric Asian (semi‑closed form)  
- Arithmetic Asian (Monte Carlo)  
- Unified payoff interface  

### **Analysis Tools**
- Convergence diagnostics  
- Error decay plots  
- Runtime vs. accuracy benchmarking  

---

## **📦 Installation**

```bash
git clone https://github.com/<your-username>/derivatives-pricing-engine.git
cd derivatives-pricing-engine
pip install -r requirements.txt
```

---

## **▶️ Usage Examples**

### **1. Simulate GBM Paths**

```python
from core.gbm import GBM

gbm = GBM(mu=0.05, sigma=0.2, S0=100)
paths = gbm.simulate(n_paths=10000, n_steps=252)
```

---

### **2. Price a European Call (Black–Scholes)**

```python
from models.black_scholes import black_scholes_call

price = black_scholes_call(S=100, K=100, r=0.05, sigma=0.2, T=1.0)
print(price)
```

---

### **3. Monte Carlo Pricing**

```python
from monte_carlo.engine import MonteCarloEngine
from models.payoffs import EuropeanCall

engine = MonteCarloEngine(model=gbm, payoff=EuropeanCall(K=100))
mc_price = engine.price(n_paths=50000)
```

---

### **4. Variance Reduction (Antithetic)**

```python
from monte_carlo.variance_reduction import Antithetic

engine = MonteCarloEngine(model=gbm, payoff=EuropeanCall(100), vr=Antithetic())
price = engine.price(n_paths=50000)
```

---

### **5. Implied Volatility**

```python
from calibration.implied_vol import implied_volatility

iv = implied_volatility(price=10.5, S=100, K=100, r=0.05, T=1.0)
```

---

### **6. Asian Option Pricing**

```python
from models.asian_options import geometric_asian_call

price = geometric_asian_call(S=100, K=100, r=0.05, sigma=0.2, T=1.0)
```

---

## **🧪 Testing**

```bash
pytest tests/
```

---

## **📌 Design Philosophy**

- **Modular** — every component is isolated and swappable  
- **Vectorized** — NumPy‑optimized for speed  
- **Research‑friendly** — notebooks + analysis tools for reproducibility  
- **Extensible** — easy to add new SDEs, payoffs, solvers, VR methods  
- **Tested** — unit tests ensure numerical stability  

---

## **📄 License**

MIT License — free for research, education, and commercial use.

---
