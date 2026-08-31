> 🔗 **Related project:** Check out [stock-research-agent-api](https://github.com/rooneyrulz/stock-research-agent-api) —
> a CrewAI-based take on this same idea, with role-specific agents, structured/validated outputs, and a FastAPI service layer.
> Both projects are actively maintained; this one continues to evolve independently.


# 📈 NSE Stock Research & Analysis System

A sophisticated multi-agent AI system for analyzing Indian NSE-listed stocks using real-time data, technical indicators, news sentiment, and advanced AI reasoning.

## 🌟 Features

### 🤖 Multi-Agent Architecture
- **Stock Finder Agent**: Identifies promising NSE stocks based on liquidity, market cap, and momentum
- **Market Data Agent**: Gathers real-time pricing, volume, and technical indicators  
- **News Analyst Agent**: Analyzes recent news sentiment and market impact
- **Recommendation Agent**: Provides actionable BUY/SELL/HOLD recommendations with target prices

### 📊 Advanced Analytics
- Real-time NSE stock data integration
- Technical indicators (RSI, Moving Averages, MACD)
- Volume and volatility analysis
- News sentiment classification
- Risk-reward assessment

### 🎯 Smart Recommendations
- Specific entry/exit price points
- Stop-loss levels and risk management
- Confidence scoring for each recommendation
- Time horizon-based analysis (short-term to medium-term)

### 🎨 Modern UI
- Clean, responsive Streamlit interface
- Interactive charts and visualizations
- Real-time status updates
- CSV export functionality
- Mobile-friendly design

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Bright Data API account ([Sign up here](https://brightdata.com))
- OpenAI API key ([Get one here](https://platform.openai.com))

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/rooneyrulz/agentic-stock-research-system
   cd nse-stock-research-system
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Install Bright Data MCP**
   ```bash
   npm install -g @brightdata/mcp
   ```

### Running the Application

1. **Start the Streamlit app**
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Access the application**
   - Open your browser to `http://localhost:8501`
   - Enter your API keys in the sidebar
   - Select analysis parameters
   - Click "Start Analysis" and wait for results!

## 🔧 Configuration

### API Keys Setup

#### Bright Data API Token
1. Sign up at [Bright Data](https://brightdata.com)
2. Navigate to your dashboard
3. Go to "Zones" → "Web Unlocker" 
4. Copy your API token

#### OpenAI API Key
1. Sign up at [OpenAI Platform](https://platform.openai.com)
2. Go to "API Keys" section
3. Create a new API key
4. Copy the key (starts with 'sk-')

### Analysis Types

- **Short-term Trading (1-7 days)**: Focus on momentum, technical breakouts, and news catalysts
- **Medium-term Investment (1-4 weeks)**: Emphasis on earnings, sector trends, and technical setups  
- **General Market Analysis**: Broad market overview with top stock picks across sectors

## 📈 Sample Output

```
🎯 TRADING RECOMMENDATIONS
═══════════════════════════════════

RELIANCE - Reliance Industries Limited
─────────────────────────────────
📋 RECOMMENDATION: BUY
🎯 TARGET PRICE: ₹2,650
⏰ TIME HORIZON: 1-3 days
📊 CONFIDENCE: HIGH

📈 ENTRY STRATEGY:
Current Price: ₹2,450
Suggested Entry: ₹2,430 - ₹2,460
Stop Loss: ₹2,380 (3.2% below entry)
Target: ₹2,650 (8.2% upside potential)

💡 RATIONALE:
Technical: Breakout above 50-day MA with strong volume
Fundamental: Positive earnings guidance + new project announcements
Risk-Reward: 1:2.6 ratio
```

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit UI  │────│   Supervisor     │────│  Bright Data    │
│                 │    │     Agent        │    │   MCP Server    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                    ┌─────────┼─────────┐
                    │         │         │
            ┌───────▼───┐ ┌───▼───┐ ┌───▼────┐
            │Stock Finder│ │Market │ │News    │
            │   Agent    │ │Data   │ │Analyst │
            └────────────┘ │Agent  │ │Agent   │
                          └───────┘ └────────┘
                                │
                        ┌───────▼────────┐
                        │ Recommendation │
                        │     Agent      │
                        └────────────────┘
```

## 🔍 Agent Details

### Stock Finder Agent
- Scans NSE universe for liquid, high-potential stocks
- Filters by market cap, volume, and momentum criteria
- Avoids penny stocks and illiquid securities
- Focuses on large-cap and mid-cap opportunities

### Market Data Agent  
- Real-time price, volume, and market data
- Technical indicators (RSI, MACD, Moving Averages)
- Support/resistance level identification
- Trend analysis and momentum assessment

### News Analyst Agent
- Scrapes recent financial news and announcements
- Sentiment classification (Positive/Negative/Neutral)
- Impact assessment on stock prices
- Catalyst identification for price movements

### Recommendation Agent
- Synthesizes all data into actionable recommendations
- Provides specific entry/exit strategies
- Risk management and position sizing guidance
- Confidence scoring and time horizon analysis

## 🛡️ Risk Management Features

- **Stop-loss recommendations** for every trade suggestion
- **Position sizing guidance** based on volatility
- **Risk-reward ratio analysis** (minimum 1:2 ratio)
- **Confidence scoring** to help with decision making
- **Time horizon specification** for each recommendation

## 📊 Export & Reporting

- **CSV Export**: Download analysis results for further analysis
- **Interactive Charts**: Visualize current vs target prices
- **Performance Tracking**: Monitor recommendation accuracy
- **Historical Analysis**: Compare predictions with actual outcomes

## ⚠️ Important Disclaimers

- This tool is for **educational and research purposes only**
- Always consult with a qualified financial advisor before investing
- Past performance does not guarantee future results
- The Indian stock market involves substantial risk of loss
- Do your own due diligence before making any investment decisions

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For support and questions:
- Open an issue on GitHub
- Check the documentation
- Review the troubleshooting guide below

### Troubleshooting

**Common Issues:**

1. **API Key Errors**
   - Ensure your Bright Data token is valid and has sufficient credits
   - Verify OpenAI API key starts with 'sk-' and has available quota

2. **MCP Installation Issues**
   ```bash
   # Reinstall MCP globally
   npm uninstall -g @brightdata/mcp
   npm install -g @brightdata/mcp
   ```

3. **Streamlit Issues**
   ```bash
   # Clear Streamlit cache
   streamlit cache clear
   ```

4. **Import Errors**
   ```bash
   # Reinstall dependencies
   pip install -r requirements.txt --force-reinstall
   ```

## 🔄 Version History

- **v1.0.0** - Initial release with multi-agent architecture
- **v1.1.0** - Added Streamlit UI and export functionality  
- **v1.2.0** - Enhanced recommendation parsing and visualization

---

**Made with ❤️ for the Indian Stock Market Community**
