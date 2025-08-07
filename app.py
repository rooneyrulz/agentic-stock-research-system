# streamlit_app.py
import streamlit as st
import asyncio
import json
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import re
from typing import Dict, List, Any

# Import our refactored system
from main import StockResearchSystem, extract_recommendations

# Page configuration
st.set_page_config(
    page_title="NSE Stock Research & Analysis System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling
st.markdown(
    """
<style>
    /* Sidebar width adjustment */
    .css-1d391kg, .css-1lcbmhc, .css-1544g2n {
        width: 400px !important;
        min-width: 400px !important;
    }
    
    section[data-testid="stSidebar"] {
        width: 400px !important;
        min-width: 400px !important;
    }
    
    section[data-testid="stSidebar"] > div {
        width: 400px !important;
        min-width: 400px !important;
    }
    
    /* Adjust main content margin */
    .main .block-container {
        margin-left: 420px !important;
        max-width: calc(100% - 440px) !important;
    }
    
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    
    .recommendation-card {
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
        background: white;
    }
    
    .buy-card { border-left: 5px solid #4CAF50; }
    .sell-card { border-left: 5px solid #f44336; }
    .hold-card { border-left: 5px solid #FF9800; }
    
    .status-running { color: #2196F3; }
    .status-completed { color: #4CAF50; }
    .status-error { color: #f44336; }
    
    .sidebar-header {
        background: linear-gradient(45deg, #667eea, #764ba2);
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        text-align: center;
        color: white;
    }
</style>
""",
    unsafe_allow_html=True,
)


def init_session_state():
    """Initialize session state variables"""
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None
    if "analysis_running" not in st.session_state:
        st.session_state.analysis_running = False
    if "system" not in st.session_state:
        st.session_state.system = None


def validate_api_keys(bright_data_key: str, openai_key: str) -> tuple:
    """Validate API keys format"""
    errors = []

    if not bright_data_key or len(bright_data_key.strip()) < 10:
        errors.append("Bright Data API token appears to be invalid (too short)")

    if not openai_key or len(openai_key.strip()) < 10:
        errors.append("OpenAI API key appears to be invalid (too short)")

    return len(errors) == 0, errors


def create_sidebar():
    """Create the sidebar with API key inputs and controls"""
    st.sidebar.markdown(
        """
    <div class="sidebar-header">
        <h2>🔧 Configuration</h2>
        <p>Enter your API credentials</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # API Key inputs
    bright_data_api = st.sidebar.text_input(
        "🌐 Bright Data API Token",
        type="password",
        help="Get your API token from Bright Data dashboard",
        placeholder="Enter your Bright Data API token...",
    )

    openai_api = st.sidebar.text_input(
        "🤖 OpenAI API Key",
        type="password",
        help="Get your API key from OpenAI platform",
        placeholder="sk-...",
    )

    st.sidebar.markdown("---")

    # Analysis parameters
    st.sidebar.markdown("### 📊 Analysis Parameters")

    analysis_type = st.sidebar.selectbox(
        "Analysis Focus",
        [
            "Short-term Trading (1-7 days)",
            "Medium-term Investment (1-4 weeks)",
            "General Market Analysis",
        ],
        help="Select the type of analysis you want to perform",
    )

    custom_query = st.sidebar.text_area(
        "Custom Query (Optional)",
        placeholder="Enter specific requirements or stocks to analyze...",
        help="Leave empty for general market analysis",
    )

    st.sidebar.markdown("---")

    # Action buttons
    col1, col2 = st.sidebar.columns(2)

    with col1:
        analyze_button = st.button(
            "🚀 Start Analysis",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.analysis_running,
        )

    with col2:
        clear_button = st.button(
            "🗑️ Clear Results",
            use_container_width=True,
            disabled=st.session_state.analysis_running,
        )

    if clear_button:
        st.session_state.analysis_results = None
        st.rerun()

    # Status indicator
    if st.session_state.analysis_running:
        st.sidebar.markdown(
            """
        <div class="metric-card">
            <div class="status-running">🔄 Analysis in progress...</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Information section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.info(
        """
    This system uses advanced AI agents to:
    
    • 🔍 **Find** promising NSE stocks
    • 📊 **Analyze** market data & technicals  
    • 📰 **Research** latest news & sentiment
    • 🎯 **Recommend** specific buy/sell actions
    
    **Powered by**: LangChain + Bright Data + OpenAI
    """
    )

    return analyze_button, bright_data_api, openai_api, analysis_type, custom_query


def display_header():
    """Display the main header"""
    st.markdown(
        """
    <div class="main-header">
        <h1>📈 NSE Stock Research & Analysis System</h1>
        <p>AI-Powered Multi-Agent Stock Analysis for Indian Markets</p>
        <p><em>Real-time data • Technical analysis • News sentiment • Trading recommendations</em></p>
    </div>
    """,
        unsafe_allow_html=True,
    )


def parse_recommendations_from_text(text: str) -> List[Dict[str, Any]]:
    """Parse recommendations from the text output"""
    recommendations = []

    # Split text into sections for each stock
    sections = re.split(r"([A-Z]{2,10})\s*-\s*([A-Za-z\s&]+)", text)

    for i in range(1, len(sections), 3):
        if i + 1 < len(sections):
            symbol = sections[i].strip()
            company = sections[i + 1].strip()
            content = sections[i + 2] if i + 2 < len(sections) else ""

            # Extract recommendation details
            rec = {
                "symbol": symbol,
                "company": company,
                "action": "HOLD",
                "target_price": "N/A",
                "current_price": "N/A",
                "confidence": "MEDIUM",
                "reasoning": "Analysis completed",
            }

            # Parse action
            action_match = re.search(r"RECOMMENDATION:\s*([A-Z]+)", content)
            if action_match:
                rec["action"] = action_match.group(1)

            # Parse target price
            target_match = re.search(r"TARGET PRICE:\s*₹?([0-9,]+\.?[0-9]*)", content)
            if target_match:
                rec["target_price"] = target_match.group(1)

            # Parse current price
            current_match = re.search(r"Current Price:\s*₹?([0-9,]+\.?[0-9]*)", content)
            if current_match:
                rec["current_price"] = current_match.group(1)

            # Parse confidence
            conf_match = re.search(r"CONFIDENCE:\s*([A-Z]+)", content)
            if conf_match:
                rec["confidence"] = conf_match.group(1)

            recommendations.append(rec)

    return recommendations


def display_recommendations(text_output: str):
    """Display parsed recommendations in a structured format"""
    recommendations = parse_recommendations_from_text(text_output)

    if not recommendations:
        st.warning("No structured recommendations found in the analysis output.")
        return

    st.markdown("## 🎯 Trading Recommendations")

    for i, rec in enumerate(recommendations):
        action = rec["action"].upper()

        # Determine card style based on action
        if action == "BUY":
            card_class = "buy-card"
            action_color = "🟢"
        elif action == "SELL":
            card_class = "sell-card"
            action_color = "🔴"
        else:
            card_class = "hold-card"
            action_color = "🟡"

        st.markdown(
            f"""
        <div class="recommendation-card {card_class}">
            <h3>{action_color} {rec['symbol']} - {rec['company']}</h3>
        </div>
        """,
            unsafe_allow_html=True,
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Action", f"{action_color} {action}")

        with col2:
            st.metric("Current Price", f"₹{rec['current_price']}")

        with col3:
            st.metric("Target Price", f"₹{rec['target_price']}")

        with col4:
            st.metric("Confidence", rec["confidence"])


def display_analysis_results(results: Dict[str, Any]):
    """Display the complete analysis results"""
    if not results:
        st.info("No analysis results to display.")
        return

    # Format and display the complete output
    formatted_output = st.session_state.system.format_results_for_display(results)

    # Display timestamp
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("## 📊 Complete Analysis Report")
    with col2:
        timestamp = results.get("timestamp", datetime.now().isoformat())
        st.markdown(f"**Generated:** {timestamp[:19].replace('T', ' ')}")

    # Try to parse and display recommendations
    display_recommendations(formatted_output)

    st.markdown("---")

    # Display full analysis in expandable section
    with st.expander("📋 View Complete Analysis Report", expanded=False):
        st.markdown("### Raw Analysis Output")
        st.text(formatted_output)

        # Display raw data for debugging
        if st.checkbox("Show raw data (for debugging)"):
            st.json(results)


async def run_analysis(
    bright_data_api: str, openai_api: str, analysis_type: str, custom_query: str
):
    """Run the stock analysis asynchronously"""
    try:
        # Initialize the system
        system = StockResearchSystem(bright_data_api, openai_api)
        st.session_state.system = system

        # Create query based on analysis type
        if custom_query.strip():
            query = custom_query
        else:
            query_map = {
                "Short-term Trading (1-7 days)": "Provide comprehensive stock analysis and trading recommendations for promising NSE-listed stocks suitable for short-term trading (1-7 days) in the current market conditions.",
                "Medium-term Investment (1-4 weeks)": "Analyze NSE stocks for medium-term investment opportunities (1-4 weeks) considering upcoming earnings, sector trends, and technical setups.",
                "General Market Analysis": "Provide a comprehensive analysis of current NSE market conditions and identify the most promising stocks across different sectors.",
            }
            query = query_map.get(
                analysis_type, query_map["Short-term Trading (1-7 days)"]
            )

        # Run analysis
        results = await system.analyze_stocks(query)
        return results

    except Exception as e:
        return {"error": str(e), "status": "error"}


def main():
    """Main application function"""
    init_session_state()
    display_header()

    # Create sidebar and get inputs
    analyze_button, bright_data_api, openai_api, analysis_type, custom_query = (
        create_sidebar()
    )

    # Main content area
    if analyze_button:
        # Validate inputs
        is_valid, errors = validate_api_keys(bright_data_api, openai_api)

        if not is_valid:
            st.error("❌ Please fix the following issues:")
            for error in errors:
                st.error(f"• {error}")
            return

        # Start analysis
        st.session_state.analysis_running = True

        with st.status(
            "🔄 Running Multi-Agent Stock Analysis...", expanded=True
        ) as status:
            st.write("🔍 Initializing AI agents...")
            st.write("📊 Finding promising NSE stocks...")

            try:
                # Run the analysis
                results = asyncio.run(
                    run_analysis(
                        bright_data_api, openai_api, analysis_type, custom_query
                    )
                )

                if results.get("status") == "error":
                    st.error(f"❌ Analysis failed: {results.get('error')}")
                    status.update(label="❌ Analysis Failed", state="error")
                else:
                    st.session_state.analysis_results = results
                    st.write("✅ Market data analysis completed")
                    st.write("📰 News sentiment analysis completed")
                    st.write("🎯 Generating trading recommendations...")
                    status.update(
                        label="✅ Analysis Completed Successfully!", state="complete"
                    )

            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")
                status.update(label="❌ Analysis Failed", state="error")

            finally:
                st.session_state.analysis_running = False

        # Auto-rerun to show results
        if st.session_state.analysis_results and not st.session_state.analysis_running:
            st.rerun()

    # Display results if available
    if st.session_state.analysis_results:
        display_analysis_results(st.session_state.analysis_results)

    elif not st.session_state.analysis_running:
        # Show welcome message and instructions
        st.markdown("## 👋 Welcome to NSE Stock Research System")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                """
            ### 🚀 Getting Started
            
            1. **Enter API Keys** in the sidebar:
               - Bright Data API token
               - OpenAI API key
            
            2. **Select Analysis Type**:
               - Short-term trading (1-7 days)
               - Medium-term investment (1-4 weeks)
               - General market analysis
            
            3. **Click "Start Analysis"** and wait for AI agents to work their magic!
            """
            )

        with col2:
            st.markdown(
                """
            ### 🤖 AI Agent Workflow
            
            **Stock Finder Agent** 🔍
            - Identifies 2-3 promising NSE stocks
            - Filters by liquidity and market cap
            
            **Market Data Agent** 📊
            - Gathers real-time price data
            - Calculates technical indicators
            
            **News Analyst Agent** 📰
            - Analyzes recent news sentiment
            - Assesses impact on stock prices
            
            **Recommendation Agent** 🎯
            - Provides BUY/SELL/HOLD recommendations
            - Sets target prices and stop losses
            """
            )

        # Sample output preview
        st.markdown("---")
        st.markdown("### 📋 Sample Output Preview")

        sample_data = {
            "RELIANCE": {
                "action": "BUY",
                "current": "2,450",
                "target": "2,650",
                "confidence": "HIGH",
            },
            "INFY": {
                "action": "HOLD",
                "current": "1,580",
                "target": "1,620",
                "confidence": "MEDIUM",
            },
            "TATASTEEL": {
                "action": "SELL",
                "current": "140",
                "target": "125",
                "confidence": "MEDIUM",
            },
        }

        cols = st.columns(3)
        for i, (symbol, data) in enumerate(sample_data.items()):
            with cols[i]:
                action_color = (
                    "🟢"
                    if data["action"] == "BUY"
                    else "🔴" if data["action"] == "SELL" else "🟡"
                )
                st.markdown(
                    f"""
                <div class="metric-card">
                    <h4>{action_color} {symbol}</h4>
                    <p><strong>Action:</strong> {data["action"]}</p>
                    <p><strong>Current:</strong> ₹{data["current"]}</p>
                    <p><strong>Target:</strong> ₹{data["target"]}</p>
                    <p><strong>Confidence:</strong> {data["confidence"]}</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

        st.info(
            "💡 **Tip**: The system analyzes real-time data and provides actionable insights for the next trading session."
        )


# Additional utility functions for enhanced features
def create_performance_chart(recommendations: List[Dict]):
    """Create a performance visualization chart"""
    if not recommendations:
        return None

    symbols = [rec["symbol"] for rec in recommendations]
    current_prices = []
    target_prices = []

    for rec in recommendations:
        try:
            current = float(str(rec["current_price"]).replace(",", ""))
            target = float(str(rec["target_price"]).replace(",", ""))
            current_prices.append(current)
            target_prices.append(target)
        except:
            current_prices.append(0)
            target_prices.append(0)

    # Calculate potential returns
    potential_returns = [
        (target / current - 1) * 100 if current > 0 else 0
        for current, target in zip(current_prices, target_prices)
    ]

    # Create chart
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="Current Price", x=symbols, y=current_prices, marker_color="lightblue"
        )
    )

    fig.add_trace(
        go.Bar(name="Target Price", x=symbols, y=target_prices, marker_color="darkblue")
    )

    fig.update_layout(
        title="Current vs Target Prices",
        xaxis_title="Stock Symbol",
        yaxis_title="Price (₹)",
        barmode="group",
        height=400,
    )

    return fig


def export_results_to_csv(results: Dict[str, Any]) -> str:
    """Export analysis results to CSV format"""
    try:
        formatted_output = st.session_state.system.format_results_for_display(results)
        recommendations = parse_recommendations_from_text(formatted_output)

        if recommendations:
            df = pd.DataFrame(recommendations)
            return df.to_csv(index=False)
        else:
            return "No recommendations found to export."
    except Exception as e:
        return f"Error exporting data: {str(e)}"


def add_export_functionality():
    """Add export functionality to the sidebar"""
    if st.session_state.analysis_results:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📤 Export Results")

        if st.sidebar.button("📊 Download as CSV", use_container_width=True):
            csv_data = export_results_to_csv(st.session_state.analysis_results)
            st.sidebar.download_button(
                label="💾 Save CSV File",
                data=csv_data,
                file_name=f"nse_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True,
            )


# Enhanced main function with additional features
def enhanced_main():
    """Enhanced main function with additional features"""
    init_session_state()
    display_header()

    # Create sidebar and get inputs
    analyze_button, bright_data_api, openai_api, analysis_type, custom_query = (
        create_sidebar()
    )

    # Add export functionality
    add_export_functionality()

    # Main content area
    if analyze_button:
        # Validate inputs
        is_valid, errors = validate_api_keys(bright_data_api, openai_api)

        if not is_valid:
            st.error("❌ Please fix the following issues:")
            for error in errors:
                st.error(f"• {error}")
            return

        # Start analysis
        st.session_state.analysis_running = True

        with st.status(
            "🔄 Running Multi-Agent Stock Analysis...", expanded=True
        ) as status:
            st.write("🔍 Initializing AI agents...")
            st.write("📊 Finding promising NSE stocks...")

            try:
                # Run the analysis
                results = asyncio.run(
                    run_analysis(
                        bright_data_api, openai_api, analysis_type, custom_query
                    )
                )

                if results.get("status") == "error":
                    st.error(f"❌ Analysis failed: {results.get('error')}")
                    status.update(label="❌ Analysis Failed", state="error")
                else:
                    st.session_state.analysis_results = results
                    st.write("✅ Market data analysis completed")
                    st.write("📰 News sentiment analysis completed")
                    st.write("🎯 Generating trading recommendations...")
                    status.update(
                        label="✅ Analysis Completed Successfully!", state="complete"
                    )

            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")
                status.update(label="❌ Analysis Failed", state="error")

            finally:
                st.session_state.analysis_running = False

        # Auto-rerun to show results
        if st.session_state.analysis_results and not st.session_state.analysis_running:
            st.rerun()

    # Display results if available
    if st.session_state.analysis_results:
        display_analysis_results(st.session_state.analysis_results)

        # Add performance visualization
        formatted_output = st.session_state.system.format_results_for_display(
            st.session_state.analysis_results
        )
        recommendations = parse_recommendations_from_text(formatted_output)

        if recommendations:
            st.markdown("---")
            chart = create_performance_chart(recommendations)
            if chart:
                st.plotly_chart(chart, use_container_width=True)

    elif not st.session_state.analysis_running:
        # Show welcome message and instructions
        st.markdown("## 👋 Welcome to NSE Stock Research System")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                """
            ### 🚀 Getting Started
            
            1. **Enter API Keys** in the sidebar:
               - Bright Data API token
               - OpenAI API key
            
            2. **Select Analysis Type**:
               - Short-term trading (1-7 days)
               - Medium-term investment (1-4 weeks)
               - General market analysis
            
            3. **Click "Start Analysis"** and wait for AI agents to work their magic!
            """
            )

        with col2:
            st.markdown(
                """
            ### 🤖 AI Agent Workflow
            
            **Stock Finder Agent** 🔍
            - Identifies 2-3 promising NSE stocks
            - Filters by liquidity and market cap
            
            **Market Data Agent** 📊
            - Gathers real-time price data
            - Calculates technical indicators
            
            **News Analyst Agent** 📰
            - Analyzes recent news sentiment
            - Assesses impact on stock prices
            
            **Recommendation Agent** 🎯
            - Provides BUY/SELL/HOLD recommendations
            - Sets target prices and stop losses
            """
            )

        # Sample output preview
        st.markdown("---")
        st.markdown("### 📋 Sample Output Preview")

        sample_data = {
            "RELIANCE": {
                "action": "BUY",
                "current": "2,450",
                "target": "2,650",
                "confidence": "HIGH",
            },
            "INFY": {
                "action": "HOLD",
                "current": "1,580",
                "target": "1,620",
                "confidence": "MEDIUM",
            },
            "TATASTEEL": {
                "action": "SELL",
                "current": "140",
                "target": "125",
                "confidence": "MEDIUM",
            },
        }

        cols = st.columns(3)
        for i, (symbol, data) in enumerate(sample_data.items()):
            with cols[i]:
                action_color = (
                    "🟢"
                    if data["action"] == "BUY"
                    else "🔴" if data["action"] == "SELL" else "🟡"
                )
                st.markdown(
                    f"""
                <div class="metric-card">
                    <h4>{action_color} {symbol}</h4>
                    <p><strong>Action:</strong> {data["action"]}</p>
                    <p><strong>Current:</strong> ₹{data["current"]}</p>
                    <p><strong>Target:</strong> ₹{data["target"]}</p>
                    <p><strong>Confidence:</strong> {data["confidence"]}</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

        st.info(
            "💡 **Tip**: The system analyzes real-time data and provides actionable insights for the next trading session."
        )

        # Add disclaimer
        st.markdown("---")
        st.warning(
            """
        ⚠️ **Important Disclaimer**: 
        This tool is for educational and research purposes only. 
        Always consult with a qualified financial advisor before making investment decisions. 
        Past performance does not guarantee future results.
        """
        )


if __name__ == "__main__":
    # Use the enhanced main function
    enhanced_main()
