# stock_research_system.py
import os
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from langgraph_supervisor import create_supervisor
from langchain_core.messages import convert_to_messages

load_dotenv()

class StockAction(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class NewsSentiment(Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"

@dataclass
class StockRecommendation:
    symbol: str
    company_name: str
    current_price: float
    action: StockAction
    target_price: float
    confidence: str
    reasoning: str
    technical_indicators: Dict[str, Any]
    news_sentiment: NewsSentiment
    volume_analysis: str

@dataclass
class MarketData:
    symbol: str
    current_price: float
    previous_close: float
    volume: int
    price_change_pct: float
    rsi: Optional[float]
    moving_avg_50: Optional[float]
    moving_avg_200: Optional[float]
    trend_7d: str
    trend_30d: str

class StockResearchSystem:
    def __init__(self, bright_data_api_token: str, openai_api_key: str):
        self.bright_data_api_token = bright_data_api_token
        self.openai_api_key = openai_api_key
        self.client = None
        self.supervisor = None
    
    async def initialize(self):
        """Initialize the MCP client and supervisor"""
        self.client = MultiServerMCPClient({
            "bright_data": {
                "command": "npx",
                "args": ["@brightdata/mcp"],
                "env": {
                    "API_TOKEN": self.bright_data_api_token,
                    "WEB_UNLOCKER_ZONE": os.getenv("WEB_UNLOCKER_ZONE", "unblocker"),
                    "BROWSER_ZONE": os.getenv("BROWSER_ZONE", "scraping_browser")
                },
                "transport": "stdio",
            },
        })
        
        tools = await self.client.get_tools()
        model = init_chat_model(
            model="openai:gpt-4o-mini", 
            api_key=self.openai_api_key
        )
        
        # Create specialized agents
        stock_finder_agent = self._create_stock_finder_agent(model, tools)
        market_data_agent = self._create_market_data_agent(model, tools)
        news_analyst_agent = self._create_news_analyst_agent(model, tools)
        recommendation_agent = self._create_recommendation_agent(model, tools)
        
        # Create supervisor
        self.supervisor = create_supervisor(
            model=init_chat_model(model="openai:gpt-4o-mini", api_key=self.openai_api_key),
            agents=[stock_finder_agent, market_data_agent, news_analyst_agent, recommendation_agent],
            prompt=self._get_supervisor_prompt(),
            add_handoff_back_messages=True,
            output_mode="full_history",
        ).compile()
    
    def _create_stock_finder_agent(self, model, tools):
        prompt = """
        You are an expert NSE (National Stock Exchange) stock research analyst with deep knowledge of the Indian equity market.
        
        Your mission: Identify 2-3 high-potential, actively traded NSE-listed stocks suitable for short-term trading (1-7 days).
        
        SELECTION CRITERIA:
        • Focus on large-cap and mid-cap stocks (avoid penny stocks < ₹50)
        • Daily trading volume > ₹50 crores
        • Market cap > ₹5,000 crores
        • Recent price momentum or technical breakouts
        • Sector rotation opportunities
        • News catalysts or upcoming events
        
        AVOID:
        • Penny stocks and illiquid securities
        • Stocks in trade-to-trade segment
        • Companies with regulatory issues
        • Highly volatile small-cap stocks
        
        OUTPUT FORMAT (JSON-like structure):
        ```
        SELECTED_STOCKS:
        1. Symbol: [NSE_SYMBOL]
           Company: [Full Company Name]
           Sector: [Industry Sector]
           Market Cap: [in ₹ crores]
           Avg Volume: [Daily volume in ₹ crores]
           Selection Reason: [2-3 lines explaining why this stock]
        
        2. [Repeat for second stock]
        ```
        
        Be data-driven and provide clear rationale for each selection.
        """
        
        return create_react_agent(
            model, tools, prompt=prompt, name="stock_finder_agent"
        )
    
    def _create_market_data_agent(self, model, tools):
        prompt = """
        You are a quantitative market data analyst specializing in NSE-listed stocks.
        
        Given stock symbols, gather comprehensive real-time market data:
        
        REQUIRED DATA POINTS:
        • Current Market Price (₹)
        • Previous Day Close (₹)
        • Day's High/Low (₹)
        • Trading Volume (shares & value in ₹ crores)
        • Price Change (₹ & %)
        • 52-week High/Low
        • Market Capitalization
        
        TECHNICAL INDICATORS:
        • RSI (14-period)
        • Simple Moving Averages (20, 50, 200-day)
        • MACD signal
        • Volume trend (5-day average vs today)
        • Support and Resistance levels
        
        TREND ANALYSIS:
        • 7-day price trend (% change)
        • 30-day price trend (% change)
        • Volume pattern analysis
        • Volatility assessment
        
        OUTPUT FORMAT:
        ```
        MARKET_DATA_ANALYSIS:
        
        [STOCK_SYMBOL] - [Company Name]
        ─────────────────────────────────
        Price Data:
          Current: ₹[X] | Change: +/-₹[X] (+/-X.XX%)
          Day Range: ₹[Low] - ₹[High]
          52W Range: ₹[Low] - ₹[High]
        
        Volume Analysis:
          Today: [X] shares (₹[X] crores)
          Avg 5-day: [X] shares
          Volume Status: [Above/Below Average]
        
        Technical Indicators:
          RSI: [X] ([Overbought/Oversold/Neutral])
          MA20: ₹[X] | MA50: ₹[X] | MA200: ₹[X]
          Price vs MA50: [Above/Below] by X%
          MACD: [Bullish/Bearish/Neutral]
        
        Trends:
          7-day: [+/-X.X%]
          30-day: [+/-X.X%]
          Momentum: [Strong/Weak/Sideways]
        ```
        
        Provide accurate, up-to-date data with clear interpretation.
        """
        
        return create_react_agent(
            model, tools, prompt=prompt, name="market_data_agent"
        )
    
    def _create_news_analyst_agent(self, model, tools):
        prompt = """
        You are a financial news analyst with expertise in Indian stock market sentiment analysis.
        
        For each given stock, research and analyze:
        
        NEWS RESEARCH SCOPE:
        • Recent news (last 3-5 trading days)
        • Corporate announcements
        • Earnings updates
        • Management changes
        • Regulatory updates
        • Sector-specific news
        • Analyst recommendations
        
        SENTIMENT CLASSIFICATION:
        • POSITIVE: Likely to drive stock price up
        • NEGATIVE: Likely to drive stock price down  
        • NEUTRAL: Minimal expected impact
        
        IMPACT ASSESSMENT:
        • Short-term (1-3 days)
        • Medium-term (1-2 weeks)
        • Confidence level (High/Medium/Low)
        
        OUTPUT FORMAT:
        ```
        NEWS_SENTIMENT_ANALYSIS:
        
        [STOCK_SYMBOL] - [Company Name]
        ═══════════════════════════════════
        
        📰 RECENT NEWS HIGHLIGHTS:
        • [Date]: [News headline/summary] - Impact: [Positive/Negative/Neutral]
        • [Date]: [News headline/summary] - Impact: [Positive/Negative/Neutral]
        
        📊 SENTIMENT SUMMARY:
        Overall Sentiment: [POSITIVE/NEGATIVE/NEUTRAL]
        Confidence Level: [HIGH/MEDIUM/LOW]
        
        📈 POTENTIAL PRICE IMPACT:
        Short-term (1-3 days): [Expected direction and reasoning]
        Key Catalysts: [Upcoming events/announcements]
        Risk Factors: [Potential negative triggers]
        ```
        
        Focus on factual analysis and avoid speculation. Clearly distinguish between confirmed news and rumors.
        """
        
        return create_react_agent(
            model, tools, prompt=prompt, name="news_analyst_agent"
        )
    
    def _create_recommendation_agent(self, model, tools):
        prompt = """
        You are a senior trading strategist providing actionable investment recommendations for NSE stocks.
        
        Synthesize all available data to generate precise trading recommendations:
        
        ANALYSIS INPUTS:
        • Market data & technical indicators
        • News sentiment & upcoming catalysts
        • Volume patterns & price momentum
        • Risk-reward assessment
        
        RECOMMENDATION FRAMEWORK:
        
        BUY Criteria:
        • Strong technical setup + positive news
        • RSI < 70, price above key moving averages
        • Volume confirmation on breakouts
        • Favorable risk-reward ratio (1:2 minimum)
        
        SELL Criteria:
        • Overbought conditions + negative news
        • Technical breakdown below support
        • High volume selling pressure
        • Deteriorating fundamentals
        
        HOLD Criteria:
        • Mixed signals or insufficient conviction
        • Sideways consolidation phase
        • Awaiting key events/announcements
        
        OUTPUT FORMAT:
        ```
        🎯 TRADING RECOMMENDATIONS
        ═══════════════════════════════════
        
        [STOCK_SYMBOL] - [Company Name]
        ─────────────────────────────────
        📋 RECOMMENDATION: [BUY/SELL/HOLD]
        🎯 TARGET PRICE: ₹[X]
        ⏰ TIME HORIZON: [1-3 days / 1-2 weeks]
        📊 CONFIDENCE: [HIGH/MEDIUM/LOW]
        
        📈 ENTRY STRATEGY:
        Current Price: ₹[X]
        Suggested Entry: ₹[X] - ₹[X]
        Stop Loss: ₹[X] (X% below entry)
        Target: ₹[X] (X% upside potential)
        
        💡 RATIONALE:
        Technical: [Key technical factors]
        Fundamental: [Key news/catalyst factors]
        Risk-Reward: [1:X ratio]
        
        ⚠️ KEY RISKS:
        • [Risk factor 1]
        • [Risk factor 2]
        
        📅 NEXT MONITORING POINTS:
        • [Specific price levels or events to watch]
        ```
        
        Provide specific, actionable advice with clear entry/exit points and risk management guidelines.
        """
        
        return create_react_agent(
            model, tools, prompt=prompt, name="recommendation_agent"
        )
    
    def _get_supervisor_prompt(self):
        return """
        You are an expert supervisor orchestrating a comprehensive NSE stock research and recommendation system.
        
        WORKFLOW SEQUENCE:
        1. STOCK_FINDER_AGENT: First, identify 2-3 promising NSE stocks for short-term trading
        2. MARKET_DATA_AGENT: Then, gather detailed market data and technical analysis for selected stocks
        3. NEWS_ANALYST_AGENT: Next, analyze recent news and sentiment for each stock
        4. RECOMMENDATION_AGENT: Finally, synthesize all data into actionable BUY/SELL/HOLD recommendations
        
        EXECUTION RULES:
        • Execute agents sequentially (never in parallel)
        • Ensure each agent completes their analysis before proceeding
        • Pass relevant context between agents
        • Do not perform any analysis yourself
        • Maintain consistent stock symbols throughout the workflow
        • Generate comprehensive final recommendations
        
        QUALITY STANDARDS:
        • All price data must be in Indian Rupees (₹)
        • Use NSE stock symbols consistently
        • Provide specific entry/exit prices
        • Include risk management guidelines
        • Ensure recommendations are actionable for next trading day
        
        Complete the entire workflow without asking for user confirmation between steps.
        """
    
    async def analyze_stocks(self, user_query: str = None) -> Dict[str, Any]:
        """Main method to run the complete stock analysis workflow"""
        if not self.supervisor:
            await self.initialize()
        
        if not user_query:
            user_query = "Provide comprehensive stock analysis and trading recommendations for promising NSE-listed stocks suitable for short-term trading in the current market conditions."
        
        # Store all messages for processing
        all_messages = []
        
        async for chunk in self.supervisor.astream({
            "messages": [{
                "role": "user", 
                "content": user_query
            }]
        }):
            all_messages.append(chunk)
        
        # Extract final results
        final_chunk = all_messages[-1] if all_messages else {}
        final_messages = final_chunk.get("supervisor", {}).get("messages", [])
        
        return {
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "messages": final_messages,
            "raw_output": all_messages
        }
    
    def format_results_for_display(self, results: Dict[str, Any]) -> str:
        """Format the analysis results for better display"""
        if not results.get("messages"):
            return "No analysis results available."
        
        # Extract the final message content
        final_messages = results["messages"]
        if not final_messages:
            return "Analysis completed but no recommendations generated."
        
        # Get the last assistant message which should contain recommendations
        for message in reversed(final_messages):
            if hasattr(message, 'content') and message.content:
                return str(message.content)
            elif isinstance(message, dict) and message.get('content'):
                return str(message['content'])
        
        return "Analysis completed. Please check the detailed output."

# Utility functions for the Streamlit app
def pretty_print_message(message, indent=False):
    """Pretty print a single message"""
    if hasattr(message, 'pretty_repr'):
        pretty_message = message.pretty_repr(html=True)
    else:
        pretty_message = str(message)
    
    if indent:
        indented = "\n".join("\t" + line for line in pretty_message.split("\n"))
        return indented
    return pretty_message

def extract_recommendations(final_messages) -> List[Dict[str, Any]]:
    """Extract structured recommendations from the final messages"""
    recommendations = []
    
    # This is a simplified parser - you might want to enhance this
    # based on the actual output format of your agents
    
    for message in final_messages:
        content = ""
        if hasattr(message, 'content'):
            content = str(message.content)
        elif isinstance(message, dict) and message.get('content'):
            content = str(message['content'])
        
        # Look for recommendation patterns in the content
        if "RECOMMENDATION:" in content and "TARGET PRICE:" in content:
            # Parse the recommendation (this is a basic example)
            lines = content.split('\n')
            rec = {}
            
            for line in lines:
                if "STOCK_SYMBOL" in line or "Symbol:" in line:
                    rec['symbol'] = line.split(':')[-1].strip()
                elif "RECOMMENDATION:" in line:
                    rec['action'] = line.split(':')[-1].strip()
                elif "TARGET PRICE:" in line:
                    price_str = line.split(':')[-1].strip().replace('₹', '')
                    try:
                        rec['target_price'] = float(price_str)
                    except:
                        rec['target_price'] = price_str
                elif "Current Price:" in line:
                    price_str = line.split(':')[-1].strip().replace('₹', '')
                    try:
                        rec['current_price'] = float(price_str)
                    except:
                        rec['current_price'] = price_str
            
            if rec:
                recommendations.append(rec)
    
    return recommendations