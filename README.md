# Quantitative Trading Project

This project was developed as part of the MSc Mathematical Trading and Finance programme at Bayes Business School (formerly Cass). It evaluates the effectiveness of comomentum-adjusted momentum strategies, as proposed by Lou & Polk (2021), in generating out-of-sample alpha across a large cross-section of US equities (**7000+** stocks).

## Problem

The project sought to assess whether incorporating comomentum, a signal based on crowding and return co-movement, can enhance standard momentum trading strategies. Using 29 years of weekly return data (1992–2021), we built and backtested multiple signal variants, compared performance across evaluation metrics, and benchmarked against both traditional momentum and a randomised null strategy.

## My Reflections

This was my first exposure to factor investing, and I was fortunate to be mentored by Bernd Hanke, CIO of Global Systematic Investors and former Global Head of Quantitative Equities at GSAM. A key lesson from the project was the importance of validating factors beyond the environment in which they were originally proposed. Academic papers often showcase strong results, but those may not generalise across universes, regimes, or rebalancing frequencies. Our replication, constrained by simplified assumptions and lacking Lou & Polk’s formal industry adjustment, returned statistically insignificant results.

Unlike many coursework projects with clean datasets, this one demanded real-world data handling and reinforced good habits under pressure (trust my processes, build tools that scale etc.).

## Methods

- Signal Construction: Built momentum and comomentum signals using rolling returns and contemporaneous comovement matrices
- Portfolio Construction: Formed weekly long-only portfolios using continuous, threshold, and hybrid sizing rules
- Evaluation Metrics: 
  - Cumulative return
  - Sharpe ratio
  - Fama–MacBeth cross-sectional regressions
  - t-statistics and p-values for signal significance
- Null Benchmarking: Compared strategy performance to a randomised comomentum control to rule out spurious effects

## Repository Structure

```
Quantitative-Trading-CW/
├── Literature/
├── datasets/
├── helper_functions/
├── images/
├── .gitignore/
├── README.md
├── Report.pdf
├── Task.pdf
├── quant_trading_code.py
├── requirements.txt
```

## Key Results

### Cumulative Return Comparison
![Cumulative Returns](images/cumRetMomFactors.png)

### Annualised Sharpe Ratios
![Sharpe Ratios](images/cumSharpeRatios.png)

- Comomentum strategies did not significantly outperform standard momentum
- Strategy efficacy collapsed when statistical significance thresholds were removed

## Requirements

```bash
pip install -r requirements.txt
```

See `requirements.txt` for package versions.

## How to Run

From the project root:

```bash
git clone https://github.com/RemaniSA/Quantitative-Trading-CW.git
cd Quantitative-Trading-CW
python quant_trading_code.py
```

Ensure:
- All datasets are placed in `/datasets`
- Figures are written to `/images`
- Utility functions are accessible via `/helper_functions`

## Further Reading

- Lou, D., & Polk, C. (2021). *Comomentum: Inferring Arbitrage Activity from Return Correlations*
- Chincarini, L., & Kim, D. *Quantitative Equity Portfolio Management: An Active Approach to Portfolio Construction and Management*
- Litterman, B. *Modern Investment Management: An Equilibrium Approach*
- Hanke, B. Lecture notes from Quantitative Trading module at Bayes

## Authors

- Shaan Ali Remani
- José Santos
- Chin-lan Chen  
- Poh Har Yap

---

### Connect

- [LinkedIn](https://www.linkedin.com/in/shaan-ali-remani)  
- [GitHub](https://github.com/RemaniSA)
