# Stock Market Analysis and Prediction GUI App

A comprehensive stock market prediction application featuring a user-friendly GUI and advanced machine learning models for stock price forecasting.

## Overview

This application combines real-time stock data fetching from Alpha Vantage API with 19+ machine learning regression models, ensemble methods, and voting regressors to analyze and predict stock closing prices. The application provides detailed performance comparisons, visualizations, and residual analysis.

## Features

### Core Functionality
- **Real-time Data Fetching**: Retrieves historical stock data using Alpha Vantage TIME_SERIES_DAILY endpoint
- **Advanced Data Preprocessing**: Includes feature engineering, normalization, and train-test splitting
- **Multiple ML Models**: Trains 19 regression models including:
  - Linear Models: Linear Regression, Ridge, Lasso, Elastic Net, Bayesian Ridge, Huber Regressor, LARS, Lasso LARS, OMP, Passive Aggressive
  - Tree-based: Decision Trees, Random Forest, Extra Trees, Gradient Boosting, LightGBM, AdaBoost
  - Distance-based: K-Neighbors Regressor, Support Vector Regression
  - Ensemble: Averaging and Weighted Voting Regressors (dynamically selected from top 5 models)

### GUI Features
- **Interactive Controls**: Date range selection with calendar picker
- **Real-time Progress Logging**: Live updates during analysis
- **6 Analysis Tabs**:
  1. All Models vs Actual - Comparison plot of all trained models
  2. Top Models & Averaging VR - Top 5 models with averaging ensemble
  3. Top Models & Weighted VR - Top 5 models with weighted ensemble
  4. Focused VRs & Top 5 - Comparison of both voting regressors with top individual models
  5. Model Performance Chart - RMSE and MAE bar chart
  6. Best Model Residuals - Error distribution analysis

### Analysis Output
- **Performance Metrics**: RMSE, MAE, R² Score, Computation Time per model
- **Smart Model Selection**: Automatically selects top 5 models for voting regressors
- **Weighted Voting**: Inverse RMSE-based weighting for improved ensemble predictions
- **LLM Analysis Summary**: Mock LLM analysis highlighting top-performing models
- **Results Table**: Sortable comparison of all model metrics

## System Requirements

### Dependencies
```
tkinter (usually included with Python)
pandas
numpy
scikit-learn
matplotlib
requests
alpha_vantage
lightgbm
xgboost (optional)
catboost (optional)
tkcalendar
seaborn
joblib
```

### Python Version
- Python 3.7 or higher

## Installation

### Step 1: Install Python
Download and install Python 3.8+ from [python.org](https://www.python.org/)

### Step 2: Install Required Packages
Run the following commands in order:

```bash
pip install requests
pip install alpha_vantage
pip install openai
pip install tkcalendar matplotlib pandas numpy scikit-learn yfinance alpha_vantage joblib seaborn
pip install lightgbm xgboost catboost
```

Alternatively, create a `requirements.txt` file with all dependencies and install:
```bash
pip install -r requirements.txt
```

### Step 3: Configure API Key
The application includes an Alpha Vantage API key. To use your own:
1. Sign up at [Alpha Vantage](https://www.alphavantage.co/)
2. Get your free API key
3. Replace `ALPHA_VANTAGE_API_KEY` in the script with your key

## Usage

### Running the Application
```bash
python "Stock Market Analysis and Prediction GUI App.ipynb"
```

Or run the notebook cell containing the main execution code.

### Step-by-Step Guide

1. **Enter Stock Ticker**: Input stock symbol (e.g., AAPL, IBM, MSFT)
2. **Select Date Range**: Choose start and end dates for analysis
   - Minimum recommended range: 30+ days
   - Longer periods provide better model training
3. **Click "Run Prediction Analysis"**: Initiates data fetching and model training
4. **Monitor Progress**: Watch the analysis log for real-time updates
5. **Review Results**:
   - Check "Best Model" and "Lowest RMSE" values
   - Explore different visualization tabs
   - Examine performance metrics table

### Example Analysis Workflow
```
Ticker: AAPL
Start Date: 2024-01-01
End Date: 2024-12-31
→ Run Analysis
→ View Results (19 models trained, top 5 selected for voting)
→ Compare voting regressor performance
→ Analyze residuals of best model
```

## Data Processing Pipeline

### 1. Data Fetching
- Retrieves daily OHLCV (Open, High, Low, Close, Volume) data
- Automatic retry logic with exponential backoff (max 3 attempts)
- Handles API rate limits gracefully

### 2. Feature Engineering
- Uses technical indicators: Open, High, Low, Close, Volume
- Target variable: Next day's Close price
- Handles missing values and date filtering

### 3. Data Splitting
- Train/Test split: 80/20 ratio
- Preserves temporal order (time-series appropriate)
- Minimum data requirements enforced

### 4. Scaling
- StandardScaler normalization for compatibility
- Prevents feature dominance issues

## Model Selection & Voting Regressors

### Individual Model Training
- All 19 models trained independently on scaled features
- Performance evaluated using RMSE as primary metric
- Failed models gracefully handled (marked with inf RMSE)

### Voting Regressor Strategy
1. **Top 5 Selection**: Best 5 models by RMSE selected
2. **Averaging VR**: Simple average of top 5 predictions
3. **Weighted VR**: RMSE-based weighted average (better performers weighted higher)

### Overall Best Model
Selected from ALL trained models + both voting regressors (21 total)

## Output Visualizations

### Plot 1: All Models vs Actual
- Line plot comparing all valid models with actual prices
- Best model highlighted in red
- RMSE values in legend

### Plot 2-3: Top Models with Voting Regressors
- Focused on top 5 individual models
- Includes averaging/weighted VR performance
- Color-coded for easy comparison

### Plot 4: Focused Comparison
- Consolidated view of both VRs + top models
- Best model on plot emphasized
- Distinct line styles (solid for best, dashed for others)

### Plot 5: Performance Bar Chart
- Side-by-side RMSE and MAE comparison
- Top 15 models displayed (if >15 models trained)
- Value labels on bars

### Plot 6: Residuals Analysis
- Scatter plot of prediction errors vs time
- Zero error reference line
- Useful for bias and variance analysis

## Performance Metrics

### RMSE (Root Mean Square Error)
- Primary metric for model selection
- Lower values indicate better predictions

### MAE (Mean Absolute Error)
- Average absolute prediction error
- More robust to outliers than RMSE

### R² Score
- Coefficient of determination (0 to 1)
- Indicates proportion of variance explained

### Computation Time
- Training time in seconds
- Useful for real-time deployment considerations

## API Limitations & Notes

### Free Tier Constraints
- **Rate Limit**: 5 API calls per minute, 100 per day
- **Data Frequency**: Daily OHLCV only
- **Price Adjustment**: Unadjusted prices (no dividend/split adjustment)
- **Historical Data**: Typically 20+ years of daily data available

### Handling Rate Limits
- Automatic retry with exponential backoff
- First retry: 65 seconds wait
- Subsequent retries: +10 seconds additional wait
- Non-retriable errors caught immediately

### Premium Features
- Some tickers may require premium subscription
- Application detects and reports if ticker requires premium

## Troubleshooting

### Issue: "API rate limit likely persisted"
**Solution**: Wait 1+ hour before next request, or upgrade to paid Alpha Vantage plan

### Issue: "Invalid symbol"
**Solution**: Verify ticker symbol (e.g., AAPL not APPLE), must be exact stock symbol

### Issue: "No data within date range"
**Solution**: Expand date range, or use dates when markets were open

### Issue: "Not enough data"
**Solution**: Increase date range to minimum 30+ days recommended

### Issue: GUI freezes
**Solution**: Normal during analysis. Application uses threading to prevent UI blocking.

## Model Details

### Polynomial Regression
- Degree 2 polynomial features
- Captures non-linear relationships

### Regularized Models
- Ridge (L2): Prevents overfitting via penalty
- Lasso (L1): Feature selection via sparse penalties
- Elastic Net: Combination of L1+L2

### Tree-based Ensemble
- Random Forest: Multiple decision trees, averaging
- Gradient Boosting: Sequential tree building
- LightGBM: Fast gradient boosting variant
- Extra Trees: Randomized tree splitting

### Specialized Models
- Support Vector Regression (RBF kernel): Non-linear kernel method
- K-Neighbors: Instance-based learning
- Bayesian Ridge: Probabilistic regression

## Future Improvements

### Planned Features
- [ ] Intraday data analysis (hourly/minute bars)
- [ ] Real-time prediction updates
- [ ] Model persistence (save/load trained models)
- [ ] Backtesting framework
- [ ] Multiple ticker comparison
- [ ] Advanced technical indicators (RSI, MACD, etc.)
- [ ] Actual LLM integration (OpenAI API)
- [ ] Portfolio optimization
- [ ] Risk metrics and value-at-risk

### Performance Enhancements
- [ ] Hyperparameter tuning with GridSearchCV
- [ ] Cross-validation for better generalization
- [ ] Feature importance analysis
- [ ] Shap value explanations
- [ ] GPU acceleration for tree models

### Data Improvements
- [ ] Alternative data sources (Yahoo Finance, IEX Cloud)
- [ ] Fundamental analysis integration
- [ ] News sentiment analysis
- [ ] Earnings data incorporation

## Architecture

### Main Components

```
┌─────────────────────────────────────┐
│     Tkinter GUI Application         │
├─────────────────────────────────────┤
│  Input Frame (Ticker, Dates)        │
│  Progress Log (ScrolledText)        │
│  Results Summary Labels             │
│  Metrics Table (Treeview)           │
│  Tabbed Output (6 plot tabs)        │
└─────────────────────────────────────┘
          ↓
┌─────────────────────────────────────┐
│   Threading & Queue Management      │
│  (Background analysis execution)    │
└─────────────────────────────────────┘
          ↓
┌─────────────────────────────────────┐
│   Stock Analysis Engine             │
├─────────────────────────────────────┤
│  Data Fetching (Alpha Vantage API)  │
│  Preprocessing & Feature Engineering│
│  19 Model Training                  │
│  Voting Regressor Assembly          │
│  Performance Evaluation             │
│  Plot Generation (Matplotlib)       │
└─────────────────────────────────────┘
```

### Class Structure
- **LLMRegressor**: Mock LLM for performance analysis
- **StockPredictorApp**: Main GUI application class

### Key Functions
- `run_stock_analysis()`: Core analysis engine
- `start_analysis_thread()`: Threading wrapper
- `check_queue()`: Progress monitoring
- `handle_analysis_result()`: Results display
- `display_plot()`: Plot rendering in GUI

## Performance Benchmarks

### Typical Execution Times (1 year of daily data)
- Data Fetching: 2-5 seconds
- Preprocessing: <1 second
- Model Training: 10-30 seconds (all 19 models)
- Voting Regressor Creation: 5-10 seconds
- Plot Generation: 5-10 seconds
- **Total**: ~25-55 seconds

### Model Training Time Ranking (fastest to slowest)
1. Linear Regression
2. Ridge/Lasso
3. K-Neighbors
4. SVR
5. Extra Trees
6. LightGBM
7. Gradient Boosting
8. Random Forest
9. XGBoost

## Best Practices

### Data Selection
- Use at least 1 year of historical data for better model training
- Avoid analysis during market holidays
- Weekends/holidays may have missing data

### Model Interpretation
- Don't rely solely on RMSE; check R² and MAE
- Review residuals for systematic errors
- Voting regressors often more robust than single models

### Deployment Considerations
- Stock prices influenced by many external factors
- ML models capture historical patterns, not causation
- Use predictions as one tool among many

## License & Attribution

This application uses:
- Alpha Vantage API for stock data (requires account)
- Scikit-learn for ML models
- Matplotlib for visualization
- Tkinter for GUI

## Version History

### Version 1.0 (Current)
- Initial release with 19 ML models
- Dual voting regressor strategy
- 6 analysis visualization tabs
- Real-time progress logging
- Comprehensive error handling

---

**Last Updated**: January 2026  
**Python Version**: 3.7+  
**Status**: Active Development
