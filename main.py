# stock_research_system.py
import os
import uuid
import logging
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

# from langchain.chat_models import init_chat_model
from langchain_groq import ChatGroq
from langgraph_supervisor import create_supervisor

from logging_config import (
    setup_logging,
    session_id_ctx,
    agent_id_ctx,
)
from prompts import (
    get_supervisor_prompt,
    get_stock_finder_prompt,
    get_market_data_prompt,
    get_news_analyst_prompt,
    get_recommendation_prompt,
)


load_dotenv()

setup_logging()

logger = logging.getLogger(__name__)


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
        logger.info("Initializing StockResearchSystem")

        logger.info("Creating MCP client")

        self.client = MultiServerMCPClient(
            {
                "bright_data": {
                    "command": "npx",
                    "args": ["@brightdata/mcp"],
                    "env": {
                        "API_TOKEN": self.bright_data_api_token,
                        "WEB_UNLOCKER_ZONE": os.getenv(
                            "WEB_UNLOCKER_ZONE", "unblocker"
                        ),
                        "BROWSER_ZONE": os.getenv("BROWSER_ZONE", "scraping_browser"),
                    },
                    "transport": "stdio",
                },
            }
        )

        logger.info("Fetching MCP tools")
        tools = await self.client.get_tools()
        logger.info("Tools loaded", extra={"tool_count": len(tools)})

        logger.info("Initializing LLM model")
        model = ChatGroq(
            model=os.getenv("MODEL_NAME"), api_key=os.getenv("GROQ_API_KEY")
        )

        logger.info("Loading prompts")
        stock_finder_prompt = get_stock_finder_prompt()
        market_data_prompt = get_market_data_prompt()
        news_analyst_prompt = get_news_analyst_prompt()
        recommendation_prompt = get_recommendation_prompt()
        supervisor_prompt = get_supervisor_prompt()

        # Create specialized agents
        logger.info("Creating stock_finder_agent")
        stock_finder_agent = self._create_stock_finder_agent(
            model, tools, stock_finder_prompt
        )

        logger.info("Creating market_data_agent")
        market_data_agent = self._create_market_data_agent(
            model, tools, market_data_prompt
        )

        logger.info("Creating news_analyst_agent")
        news_analyst_agent = self._create_news_analyst_agent(
            model, tools, news_analyst_prompt
        )

        logger.info("Creating recommendation_agent")
        recommendation_agent = self._create_recommendation_agent(
            model, tools, recommendation_prompt
        )

        # Create supervisor
        logger.info("Creating supervisor")
        self.supervisor = create_supervisor(
            model=ChatGroq(
                model=os.getenv("MODEL_NAME"), api_key=os.getenv("GROQ_API_KEY")
            ),
            agents=[
                stock_finder_agent,
                market_data_agent,
                news_analyst_agent,
                recommendation_agent,
            ],
            prompt=supervisor_prompt,
            add_handoff_back_messages=True,
            output_mode="full_history",
        ).compile()
        logger.info("StockResearchSystem initialized ✅")

    def _get_tool_name(self, tool: Any) -> str:
        """Safely extract a tool's name for logging and prompts."""
        return getattr(tool, "name", str(tool))

    def _augment_prompt_with_tools(self, base_prompt: str, tools: Any) -> str:
        """
        Append available tool names with STRICT instructions to prevent hallucination.
        """
        if not tools:
            return (
                base_prompt + "\n\n" + "=" * 80 + "\n"
                "⚠️  CRITICAL: NO EXTERNAL TOOLS AVAILABLE\n" + "=" * 80 + "\n"
                "You MUST answer using ONLY your internal knowledge.\n"
                "DO NOT attempt to call ANY tools or functions.\n"
                "DO NOT use <function=...> syntax or tool_calls.\n"
                "Provide direct answers based on your training data.\n" + "=" * 80
            )

        tool_names = sorted({self._get_tool_name(t) for t in tools})
        tool_list_text = "\n".join(
            f"  {i + 1}. {name}" for i, name in enumerate(tool_names)
        )

        return (
            base_prompt + "\n\n" + "=" * 80 + "\n"
            "🔧 AVAILABLE TOOLS (STRICTLY LIMITED)\n" + "=" * 80 + "\n"
            f"{tool_list_text}\n\n"
            "⚠️  CRITICAL TOOL USAGE RULES:\n"
            "1. Use ONLY the exact tool names listed above\n"
            "2. DO NOT invent, guess, or modify tool names\n"
            "3. DO NOT use tools that are not in the list\n"
            "4. If you need a capability not listed, answer directly WITHOUT tool calls\n"
            "5. NEVER use <function=...> syntax for unlisted tools\n"
            "6. When in doubt, provide direct answers instead of attempting tool calls\n\n"
            "If you attempt to call a non-existent tool, your response will FAIL.\n"
            + "="
            * 80
        )

    def _create_stock_finder_agent(self, model, tools, prompt):
        agent_id_ctx.set("stock_finder_agent")
        return create_react_agent(
            model,
            tools,
            prompt=self._augment_prompt_with_tools(prompt, tools),
            name="stock_finder_agent",
        )

    def _create_market_data_agent(self, model, tools, prompt):
        agent_id_ctx.set("market_data_agent")
        return create_react_agent(
            model,
            tools,
            prompt=self._augment_prompt_with_tools(prompt, tools),
            name="market_data_agent",
        )

    def _create_news_analyst_agent(self, model, tools, prompt):
        agent_id_ctx.set("news_analyst_agent")
        return create_react_agent(
            model,
            tools,
            prompt=self._augment_prompt_with_tools(prompt, tools),
            name="news_analyst_agent",
        )

    def _create_recommendation_agent(self, model, tools, prompt):
        agent_id_ctx.set("recommendation_agent")
        return create_react_agent(
            model,
            tools,
            prompt=self._augment_prompt_with_tools(prompt, tools),
            name="recommendation_agent",
        )

    async def analyze_stocks(self, user_query: str = None) -> Dict[str, Any]:
        """Main method to run the complete stock analysis workflow"""
        # Session-level context
        session_id = str(uuid.uuid4())
        session_id_ctx.set(session_id)
        agent_id_ctx.set("supervisor")

        logger.info("Starting stock analysis session")

        if not self.supervisor:
            await self.initialize()

        if not user_query:
            user_query = "Provide comprehensive stock analysis and trading recommendations for promising NSE-listed stocks suitable for short-term trading in the current market conditions."

        try:
            logger.info("Starting supervisor execution")
            # Store all messages for processing
            all_messages = []

            async for chunk in self.supervisor.astream(
                {"messages": [{"role": "user", "content": user_query}]}
            ):
                all_messages.append(chunk)

            logger.info(
                "Supervisor execution completed ✅",
                extra={"total_chunks": len(all_messages)},
            )
        except Exception:
            logger.exception("Stock analysis failed")
            raise

        # Extract final results
        final_chunk = all_messages[-1] if all_messages else {}
        final_messages = final_chunk.get("supervisor", {}).get("messages", [])

        logger.info(
            "Stock analysis completed successfully ✅",
            extra={"message_count": len(final_messages)},
        )

        return {
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "messages": final_messages,
            "raw_output": all_messages,
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
            if hasattr(message, "content") and message.content:
                return str(message.content)
            elif isinstance(message, dict) and message.get("content"):
                return str(message["content"])

        return "Analysis completed. Please check the detailed output."


# Utility functions for the Streamlit app
def pretty_print_message(message, indent=False):
    """Pretty print a single message"""
    if hasattr(message, "pretty_repr"):
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
        if hasattr(message, "content"):
            content = str(message.content)
        elif isinstance(message, dict) and message.get("content"):
            content = str(message["content"])

        # Look for recommendation patterns in the content
        if "RECOMMENDATION:" in content and "TARGET PRICE:" in content:
            # Parse the recommendation (this is a basic example)
            lines = content.split("\n")
            rec = {}

            for line in lines:
                if "STOCK_SYMBOL" in line or "Symbol:" in line:
                    rec["symbol"] = line.split(":")[-1].strip()
                elif "RECOMMENDATION:" in line:
                    rec["action"] = line.split(":")[-1].strip()
                elif "TARGET PRICE:" in line:
                    price_str = line.split(":")[-1].strip().replace("₹", "")
                    try:
                        rec["target_price"] = float(price_str)
                    except:
                        rec["target_price"] = price_str
                elif "Current Price:" in line:
                    price_str = line.split(":")[-1].strip().replace("₹", "")
                    try:
                        rec["current_price"] = float(price_str)
                    except:
                        rec["current_price"] = price_str

            if rec:
                recommendations.append(rec)

    return recommendations


if __name__ == "__main__":
    BRIGHTDATA_TOKEN = os.getenv("BRIGHT_DATA_API_TOKEN")
    GROQ_TOKEN = os.getenv("GROQ_API_KEY")

    system = StockResearchSystem(BRIGHTDATA_TOKEN, GROQ_TOKEN)
    results = asyncio.run(system.analyze_stocks())

    print(results)

    print("*" * 80)
    print("*" * 80)
    print("*" * 80)

    recommendations = extract_recommendations(results["messages"])
    print(recommendations)

    print(results)
