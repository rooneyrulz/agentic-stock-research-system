import sys
import time
import logging
import asyncio
from datetime import datetime
from typing import List, Dict, Optional

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from langgraph_supervisor import create_supervisor

from app.models.schemas import (
    StockFinderOutput,
    AllMarketDataOutput,
    AllNewsAnalysisOutput,
    FinalRecommendationsOutput,
    StockResearchResult,
    AgentError,
)
from app.core.config import SystemConfig
from app.core.utils import load_config
from app.utils import (
    parse_llm_structured_output,
    create_agent_error,
    log_agent_execution,
    get_market_status,
)
from app.prompts import (
    STOCK_FINDER_PROMPT,
    MARKET_DATA_PROMPT,
    NEWS_ANALYST_PROMPT,
    RECOMMENDATION_PROMPT,
    SUPERVISOR_PROMPT,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    # handlers=[
    #     logging.FileHandler(f"stock_research_{datetime.now().strftime('%Y%m%d')}.log"),
    #     logging.StreamHandler(sys.stdout),
    # ],
)
logger = logging.getLogger(__name__)


class StockResearchSystem:
    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or load_config()

        self.mcp_client: Optional[MultiServerMCPClient] = None
        self.supervisor = None
        self.tools = None

        self.execution_errors: List[AgentError] = []
        self.start_time: Optional[float] = None

        logger.info(
            f"StockResearchSystem initialized with model: {self.config.model.model_name}"
        )

    async def initialize(self) -> None:
        try:
            logger.info("Initializing MCP client...")

            self.mcp_client = MultiServerMCPClient(
                {
                    "bright_data": {
                        "command": self.config.mcp.command,
                        "args": self.config.mcp.args,
                        "env": {
                            "API_TOKEN": self.config.mcp.api_token,
                            "WEB_UNLOCKER_ZONE": self.config.mcp.web_unlocker_zone,
                            "BROWSER_ZONE": self.config.mcp.browser_zone,
                        },
                        "transport": self.config.mcp.transport,
                    },
                }
            )

            self.tools = await self.mcp_client.get_tools()
            logger.info(f"Loaded {len(self.tools)} MCP tools")

            # Initialize base LLM model (Groq for fast inference)
            base_model = self._create_llm_model()

            # Create specialized agents
            logger.info("Creating specialized agents...")
            stock_finder = self._create_stock_finder_agent(base_model)
            market_data = self._create_market_data_agent(base_model)
            news_analyst = self._create_news_analyst_agent(base_model)
            recommendation = self._create_recommendation_agent(base_model)

            # Create supervisor to orchestrate agents
            logger.info("Creating supervisor...")
            self.supervisor = create_supervisor(
                model=self._create_llm_model(),  # Supervisor uses same model
                agents=[stock_finder, market_data, news_analyst, recommendation],
                prompt=SUPERVISOR_PROMPT,
                add_handoff_back_messages=True,
                output_mode="full_history",
            ).compile()

            logger.info("System initialization complete")

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise RuntimeError(f"Failed to initialize system: {e}")

    def _create_llm_model(self):
        if self.config.model.provider == "groq":
            return init_chat_model(
                self.config.model.model_name,
                model_provider="groq",
                api_key=self.config.groq_api_key,
                temperature=self.config.model.temperature,
                max_tokens=self.config.model.max_tokens,
                timeout=self.config.model.timeout,
            )
        elif self.config.model.provider == "openai":
            return init_chat_model(
                self.config.model.model_name,
                model_provider="openai",
                api_key=self.config.openai_api_key,
                temperature=self.config.model.temperature,
                max_tokens=self.config.model.max_tokens,
            )
        else:
            raise ValueError(
                f"Unsupported model provider: {self.config.model.provider}"
            )

    def _create_stock_finder_agent(self, model):
        return create_react_agent(
            model,
            self.tools,
            prompt=STOCK_FINDER_PROMPT,
            name="stock_finder_agent",
            response_format={
                "name": "StockFinderOutput",
                "description": "Structured response for stock selection",
                "parameters": StockFinderOutput.model_json_schema(),
            }
            if self.config.model.provider == "groq"
            else None,
        )

    def _create_market_data_agent(self, model):
        return create_react_agent(
            model,
            self.tools,
            prompt=MARKET_DATA_PROMPT,
            name="market_data_agent",
            response_format={
                "name": "AllMarketDataOutput",
                "description": "Structured market data results",
                "parameters": AllMarketDataOutput.model_json_schema(),
            }
            if self.config.model.provider == "groq"
            else None,
        )

    def _create_news_analyst_agent(self, model):
        return create_react_agent(
            model,
            self.tools,
            prompt=NEWS_ANALYST_PROMPT,
            name="news_analyst_agent",
            response_format={
                "name": "AllNewsAnalysisOutput",
                "description": "Structured news sentiment analysis",
                "parameters": AllNewsAnalysisOutput.model_json_schema(),
            }
            if self.config.model.provider == "groq"
            else None,
        )

    def _create_recommendation_agent(self, model):
        return create_react_agent(
            model,
            self.tools,
            prompt=RECOMMENDATION_PROMPT,
            name="recommendation_agent",
            response_format={
                "name": "FinalRecommendationsOutput",
                "description": "Structured trading recommendations",
                "parameters": FinalRecommendationsOutput.model_json_schema(),
            }
            if self.config.model.provider == "groq"
            else None,
        )

    async def analyze_stocks(
        self,
        user_query: Optional[str] = None,
        specific_stocks: Optional[List[str]] = None,
    ) -> StockResearchResult:
        # Ensure system is initialized
        if not self.supervisor:
            await self.initialize()

        # Start timing
        self.start_time = time.time()
        self.execution_errors = []

        # Prepare query
        if not user_query:
            if specific_stocks:
                stocks_str = ", ".join(specific_stocks)
                user_query = (
                    f"Provide comprehensive analysis and trading recommendations for "
                    f"these NSE stocks: {stocks_str}. Include technical analysis, "
                    f"news sentiment, and actionable entry/exit strategies."
                )
            else:
                market_status = get_market_status()
                user_query = (
                    f"Current Market Status: {market_status}. "
                    f"Identify and analyze 2-3 promising NSE-listed stocks suitable "
                    f"for short-term trading (1-7 days) in current market conditions. "
                    f"Provide comprehensive technical analysis, news sentiment, and "
                    f"specific BUY/SELL/HOLD recommendations with entry/exit points."
                )

        logger.info(f"Starting analysis with query: {user_query[:100]}...")

        try:
            # Execute supervisor workflow
            all_messages = []

            async for chunk in self.supervisor.astream(
                {"messages": [{"role": "user", "content": user_query}]},
                config={"timeout": 600},
            ):  # 10 minute overall timeout
                all_messages.append(chunk)

            # Process results
            result = await self._process_results(all_messages)

            # Calculate execution time
            execution_time = time.time() - self.start_time
            result.execution_time_seconds = execution_time

            logger.info(f"Analysis completed in {execution_time:.2f}s")

            return result

        except asyncio.TimeoutError:
            logger.error("Analysis timed out")
            return StockResearchResult(
                status="error",
                error_message="Analysis timed out after maximum duration",
                execution_time_seconds=time.time() - self.start_time,
            )

        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            return StockResearchResult(
                status="error",
                error_message=f"Analysis failed: {str(e)}",
                execution_time_seconds=time.time() - self.start_time
                if self.start_time
                else None,
            )

    async def _process_results(self, all_messages: List[Dict]) -> StockResearchResult:
        result = StockResearchResult(status="completed")

        try:
            # Extract final messages from last chunk
            final_chunk = all_messages[-1] if all_messages else {}
            final_messages = final_chunk.get("supervisor", {}).get("messages", [])

            # Parse outputs from each agent
            # This is a simplified version - you'll need to implement
            # proper message routing based on your agent outputs

            for message in final_messages:
                content = self._extract_message_content(message)

                if not content:
                    continue

                # Try to parse based on message structure
                # Stock finder output
                if "selected_stocks" in content.lower():
                    parsed = parse_llm_structured_output(content, StockFinderOutput)
                    if parsed:
                        result.selected_stocks = parsed

                # Market data output
                elif (
                    "price_data" in content.lower() or "market_data" in content.lower()
                ):
                    parsed = parse_llm_structured_output(content, AllMarketDataOutput)
                    if parsed:
                        result.market_data = parsed

                # News analysis output
                elif "sentiment" in content.lower() or "news" in content.lower():
                    parsed = parse_llm_structured_output(content, AllNewsAnalysisOutput)
                    if parsed:
                        result.news_analysis = parsed

                # Recommendations output
                elif (
                    "recommendation" in content.lower()
                    or "buy" in content.lower()
                    or "sell" in content.lower()
                ):
                    parsed = parse_llm_structured_output(
                        content, FinalRecommendationsOutput
                    )
                    if parsed:
                        result.recommendations = parsed

            # Validate completeness
            if not result.recommendations:
                result.status = "partial"
                logger.warning("Analysis completed but no recommendations generated")

            return result

        except Exception as e:
            logger.error(f"Error processing results: {e}")
            result.status = "error"
            result.error_message = f"Failed to process results: {str(e)}"
            return result

    def _extract_message_content(self, message) -> Optional[str]:
        if hasattr(message, "content"):
            return str(message.content)
        elif isinstance(message, dict) and message.get("content"):
            return str(message["content"])
        return None

    def format_results_for_display(self, result: StockResearchResult) -> str:
        if result.status == "error":
            return f"❌ Analysis Error: {result.error_message}"

        output = []
        output.append("=" * 80)
        output.append("📊 NSE STOCK RESEARCH & RECOMMENDATIONS")
        output.append("=" * 80)
        output.append(
            f"Analysis Timestamp: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        output.append(f"Execution Time: {result.execution_time_seconds:.2f}s")
        output.append(f"Market Status: {get_market_status()}")
        output.append("")

        # Display recommendations
        if result.recommendations:
            output.append("🎯 TRADING RECOMMENDATIONS")
            output.append("=" * 80)

            for rec in result.recommendations.recommendations:
                output.append(f"\n📈 {rec.symbol} - {rec.company_name}")
                output.append("-" * 80)
                output.append(
                    f"   Recommendation: {rec.recommendation.value} ({rec.confidence.value} confidence)"
                )
                output.append(f"   Time Horizon: {rec.time_horizon.value}")
                output.append(
                    f"   Current Price: ₹{rec.entry_strategy.current_price:,.2f}"
                )
                output.append(
                    f"   Target Price: ₹{rec.entry_strategy.target_price:,.2f}"
                )
                output.append(f"   Stop Loss: ₹{rec.entry_strategy.stop_loss:,.2f}")
                output.append(
                    f"   Expected Upside: {rec.entry_strategy.target_upside_pct:.2f}%"
                )
                output.append(f"   Risk-Reward: {rec.entry_strategy.risk_reward_ratio}")
                output.append("")

        output.append("\n" + "=" * 80)
        output.append(
            result.recommendations.disclaimer if result.recommendations else ""
        )

        return "\n".join(output)


async def run(user_query):
    system = StockResearchSystem()
    result = await system.analyze_stocks()
    print(result)


# helper function to run analyze-stocks
if __name__ == "__main__":
    user_query = "find a top 3 stocks to trade for short term"
    asyncio.run(run(user_query))
