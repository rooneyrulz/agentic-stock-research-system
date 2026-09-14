import sys
import time
import logging
import asyncio
from datetime import datetime
from typing import List, Dict, Optional, Any

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
    handlers=[
        logging.FileHandler(f"stock_research_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


class StockResearchSystem:
    """
    Production-ready stock research system with multi-agent architecture.

    Features:
        - Comprehensive error tracking for each agent
        - Execution time logging
        - Automatic retry logic with exponential backoff
        - Detailed execution metrics
    """

    def __init__(self, config: Optional[SystemConfig] = None):
        """Initialize the stock research system."""
        self.config = config or load_config()

        # System components
        self.mcp_client: Optional[MultiServerMCPClient] = None
        self.supervisor = None
        self.tools = None

        # Execution tracking
        self.execution_errors: List[AgentError] = []
        self.agent_metrics: Dict[str, Dict[str, Any]] = {}
        self.start_time: Optional[float] = None

        logger.info(
            f"StockResearchSystem initialized with model: {self.config.model.model_name}"
        )

    async def initialize(self) -> None:
        """
        Initialize MCP client, tools, and supervisor.
        Must be called before running analysis.
        """
        init_start = time.time()

        try:
            logger.info("=" * 80)
            logger.info("Initializing Stock Research System")
            logger.info("=" * 80)
            logger.info("Step 1/4: Initializing MCP client...")

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

            logger.info("Step 2/4: Loading MCP tools...")
            raw_tools = await self.mcp_client.get_tools()
            self.tools = self._normalize_tools(raw_tools)
            logger.info(
                f"✅ Loaded {len(self.tools)} MCP tools: "
                f"{[self._get_tool_name(t) for t in self.tools]}"
            )

            # Initialize base LLM model (Groq for fast inference)
            base_model = self._create_llm_model()

            # Create specialized agents
            logger.info("Step 3/4: Creating specialized agents...")
            stock_finder = self._create_stock_finder_agent(base_model)
            market_data = self._create_market_data_agent(base_model)
            news_analyst = self._create_news_analyst_agent(base_model)
            recommendation = self._create_recommendation_agent(base_model)
            logger.info("✅ All agents created successfully")

            # Create supervisor to orchestrate agents
            logger.info("Step 4/4: Creating supervisor...")
            self.supervisor = create_supervisor(
                model=self._create_llm_model(),  # Supervisor uses same model
                agents=[stock_finder, market_data, news_analyst, recommendation],
                prompt=SUPERVISOR_PROMPT,
                add_handoff_back_messages=True,
                output_mode="full_history",
            ).compile()

            init_duration = time.time() - init_start
            logger.info(f"✅ System initialization complete in {init_duration:.2f}s")
            logger.info("=" * 80)

        except Exception as e:
            init_duration = time.time() - init_start

            # Track initialization error
            init_error = create_agent_error(
                agent_name="system_initialization", error=e, recoverable=False
            )
            self.execution_errors.append(init_error)

            error_msg = f"Initialization failed after {init_duration:.2f}s: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    def _create_llm_model(self):
        """Create LLM model instance with configured settings."""
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

    def _normalize_tools(self, tools):
        """Ensure tools are always a list for LangGraph agents."""
        if not tools:
            return []
        if isinstance(tools, dict):
            return list(tools.values())
        return tools

    def _get_tool_name(self, tool: Any) -> str:
        """Safely extract a tool's name for logging and prompts."""
        return getattr(tool, "name", str(tool))

    def _augment_prompt_with_tools(self, base_prompt: str) -> str:
        """
        Append available tool names with STRICT instructions to prevent hallucination.
        """
        if not self.tools:
            return (
                base_prompt + "\n\n" + "=" * 80 + "\n"
                "⚠️  CRITICAL: NO EXTERNAL TOOLS AVAILABLE\n" + "=" * 80 + "\n"
                "You MUST answer using ONLY your internal knowledge.\n"
                "DO NOT attempt to call ANY tools or functions.\n"
                "DO NOT use <function=...> syntax or tool_calls.\n"
                "Provide direct answers based on your training data.\n" + "=" * 80
            )

        tool_names = sorted({self._get_tool_name(t) for t in self.tools})
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

    def _create_stock_finder_agent(self, model):
        """Create agent for identifying promising NSE stocks."""
        return create_react_agent(
            model,
            self.tools,
            prompt=self._augment_prompt_with_tools(STOCK_FINDER_PROMPT),
            name="stock_finder_agent",
            response_format={
                "name": "StockFinderOutput",
                "description": "Structured response for stock selection",
                "parameters": StockFinderOutput.model_json_schema(),
            },
        )

    def _create_market_data_agent(self, model):
        """Create agent for gathering comprehensive market data."""
        return create_react_agent(
            model,
            self.tools,
            prompt=self._augment_prompt_with_tools(MARKET_DATA_PROMPT),
            name="market_data_agent",
            response_format={
                "name": "AllMarketDataOutput",
                "description": "Structured market data results",
                "parameters": AllMarketDataOutput.model_json_schema(),
            },
        )

    def _create_news_analyst_agent(self, model):
        """Create agent for analyzing news sentiment and catalysts."""
        return create_react_agent(
            model,
            self.tools,
            prompt=self._augment_prompt_with_tools(NEWS_ANALYST_PROMPT),
            name="news_analyst_agent",
            response_format={
                "name": "AllNewsAnalysisOutput",
                "description": "Structured news sentiment analysis",
                "parameters": AllNewsAnalysisOutput.model_json_schema(),
            },
        )

    def _create_recommendation_agent(self, model):
        """Create agent for generating trading recommendations."""
        return create_react_agent(
            model,
            self.tools,
            prompt=self._augment_prompt_with_tools(RECOMMENDATION_PROMPT),
            name="recommendation_agent",
            response_format={
                "name": "FinalRecommendationsOutput",
                "description": "Structured trading recommendations",
                "parameters": FinalRecommendationsOutput.model_json_schema(),
            },
        )

    async def analyze_stocks(
        self,
        user_query: Optional[str] = None,
        specific_stocks: Optional[List[str]] = None,
    ) -> StockResearchResult:
        """
        Run complete stock analysis workflow with comprehensive error tracking.

        Args:
            user_query: Optional custom query
            specific_stocks: Optional list of specific stock symbols to analyze

        Returns:
            StockResearchResult: Structured analysis results with execution metrics
        """
        # Ensure system is initialized
        if not self.supervisor:
            logger.info("System not initialized, initializing now...")
            await self.initialize()

        # Reset tracking for new analysis
        self.start_time = time.time()
        self.execution_errors = []
        self.agent_metrics = {}

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

        logger.info("=" * 80)
        logger.info("🚀 Starting Stock Analysis")
        logger.info("=" * 80)
        logger.info(f"Query: {user_query[:150]}...")
        logger.info(f"Market Status: {get_market_status()}")
        logger.info("")

        try:
            # Execute supervisor workflow with monitoring
            all_messages = []
            current_agent = None
            agent_start_time = None

            async for chunk in self.supervisor.astream(
                {"messages": [{"role": "user", "content": user_query}]},
                config={"timeout": 600},
            ):  # 10 minute overall timeout
                # Detect agent execution from chunk structure
                if isinstance(chunk, dict):
                    for key in chunk.keys():
                        if "agent" in key.lower() and key != current_agent:
                            # Log previous agent completion
                            if current_agent and agent_start_time:
                                duration = time.time() - agent_start_time
                                log_agent_execution(
                                    current_agent,
                                    "success",
                                    duration,
                                    "Completed successfully",
                                )
                                self.agent_metrics[current_agent] = {
                                    "status": "success",
                                    "duration": duration,
                                    "timestamp": datetime.now().isoformat(),
                                }

                            # Start tracking new agent
                            current_agent = key
                            agent_start_time = time.time()
                            logger.info(f"▶️  {current_agent} started...")

                all_messages.append(chunk)

            # Log final agent completion
            if current_agent and agent_start_time:
                duration = time.time() - agent_start_time
                log_agent_execution(current_agent, "success", duration)
                self.agent_metrics[current_agent] = {
                    "status": "success",
                    "duration": duration,
                    "timestamp": datetime.now().isoformat(),
                }

            # Process results
            result = await self._process_results(all_messages)

            # Calculate execution time
            execution_time = time.time() - self.start_time
            result.execution_time_seconds = execution_time

            # Log execution summary
            self._log_execution_summary(result, execution_time)

            logger.info(f"Analysis completed in {execution_time:.2f}s")

            return result

        except asyncio.TimeoutError:
            execution_time = time.time() - self.start_time

            # Track timeout error
            timeout_error = create_agent_error(
                agent_name="workflow_execution",
                error=asyncio.TimeoutError("Analysis exceeded maximum duration"),
                recoverable=False,
            )
            self.execution_errors.append(timeout_error)

            logger.error(f"❌ Analysis timed out after {execution_time:.2f}s")

            return StockResearchResult(
                status="error",
                error_message=f"Analysis timed out after {execution_time:.2f}s",
                execution_time_seconds=execution_time,
            )

        except Exception as e:
            execution_time = time.time() - self.start_time

            # Track critical error
            critical_error = create_agent_error(
                agent_name="workflow_execution", error=e, recoverable=False
            )
            self.execution_errors.append(critical_error)

            logger.error(
                f"❌ Analysis failed after {execution_time:.2f}s: {e}", exc_info=True
            )

            return StockResearchResult(
                status="error",
                error_message=f"Analysis failed: {str(e)}",
                execution_time_seconds=execution_time if self.start_time else None,
            )

    async def _process_results(self, all_messages: List[Dict]) -> StockResearchResult:
        """
        Process raw agent messages into structured result.

        Args:
            all_messages: Raw messages from supervisor stream

        Returns:
            StockResearchResult: Parsed and validated results
        """
        result = StockResearchResult(status="completed")
        parsing_errors = []

        try:
            # Extract final messages from last chunk
            final_chunk = all_messages[-1] if all_messages else {}
            final_messages = final_chunk.get("supervisor", {}).get("messages", [])

            logger.info(f"Processing {len(final_messages)} messages from agents...")

            # Parse outputs from each agent with error tracking
            for idx, message in enumerate(final_messages):
                content = self._extract_message_content(message)

                if not content:
                    continue

                try:
                    # Stock finder output
                    if "selected_stocks" in content.lower():
                        parsed = parse_llm_structured_output(content, StockFinderOutput)
                        if parsed:
                            result.selected_stocks = parsed
                            logger.info("  ✓ Stock Finder output parsed successfully")
                        else:
                            parsing_errors.append(
                                "Stock Finder: Failed to parse output"
                            )

                    # Market data output
                    elif (
                        "price_data" in content.lower()
                        or "market_data" in content.lower()
                    ):
                        parsed = parse_llm_structured_output(
                            content, AllMarketDataOutput
                        )
                        if parsed:
                            result.market_data = parsed
                            logger.info("  ✓ Market Data output parsed successfully")
                        else:
                            parsing_errors.append("Market Data: Failed to parse output")

                    # News analysis output
                    elif "sentiment" in content.lower() or "news" in content.lower():
                        parsed = parse_llm_structured_output(
                            content, AllNewsAnalysisOutput
                        )
                        if parsed:
                            result.news_analysis = parsed
                            logger.info("  ✓ News Analyst output parsed successfully")
                        else:
                            parsing_errors.append(
                                "News Analyst: Failed to parse output"
                            )

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
                            logger.info("  ✓ Recommendation output parsed successfully")
                        else:
                            parsing_errors.append(
                                "Recommendation: Failed to parse output"
                            )

                except Exception as parse_error:
                    error_msg = f"Message {idx}: Parsing error - {str(parse_error)}"
                    parsing_errors.append(error_msg)
                    logger.warning(f"  ⚠️  {error_msg}")

                    # Track parsing error
                    agent_error = create_agent_error(
                        agent_name="result_parsing", error=parse_error, recoverable=True
                    )
                    self.execution_errors.append(agent_error)

            # Validate completeness
            if not result.recommendations:
                result.status = "partial"
                logger.warning("⚠️  Analysis completed but no recommendations generated")

                if parsing_errors:
                    result.error_message = (
                        f"Parsing errors: {'; '.join(parsing_errors)}"
                    )
            elif parsing_errors:
                logger.warning(
                    f"⚠️  Analysis completed with {len(parsing_errors)} parsing warnings"
                )
            else:
                logger.info("✅ All agent outputs parsed successfully")

            return result

        except Exception as e:
            logger.error(f"Error processing results: {e}", exc_info=True)

            # Track processing error
            proc_error = create_agent_error(
                agent_name="result_processing", error=e, recoverable=False
            )
            self.execution_errors.append(proc_error)

            result.status = "error"
            result.error_message = f"Failed to process results: {str(e)}"
            return result

    def _extract_message_content(self, message) -> Optional[str]:
        """Extract content from various message formats."""
        if hasattr(message, "content"):
            return str(message.content)
        elif isinstance(message, dict) and message.get("content"):
            return str(message["content"])
        return None

    def _log_execution_summary(
        self, result: StockResearchResult, execution_time: float
    ) -> None:
        """Log detailed execution summary with metrics."""
        logger.info("")
        logger.info("=" * 80)
        logger.info("📊 EXECUTION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Status: {result.status.upper()}")
        logger.info(f"Total Execution Time: {execution_time:.2f}s")

        # Log agent metrics
        if self.agent_metrics:
            logger.info("\nAgent Performance:")
            for agent_name, metrics in self.agent_metrics.items():
                status_emoji = "✓" if metrics["status"] == "success" else "✗"
                logger.info(
                    f"  {status_emoji} {agent_name}: {metrics['duration']:.2f}s "
                    f"({metrics['status']})"
                )

        # Log errors
        if self.execution_errors:
            logger.warning(f"\n⚠️  {len(self.execution_errors)} errors encountered:")
            for error in self.execution_errors:
                logger.warning(f"  - {error.agent_name}: {error.error_message}")

        # Log results summary
        if result.status == "completed":
            if result.recommendations:
                num_recs = len(result.recommendations.recommendations)
                logger.info(f"\n✅ Generated {num_recs} trading recommendation(s)")
                for rec in result.recommendations.recommendations:
                    logger.info(
                        f"  - {rec.symbol}: {rec.recommendation.value} ({rec.confidence.value})"
                    )

        logger.info("=" * 80)

    def format_results_for_display(self, result: StockResearchResult) -> str:
        """Format analysis results for human-readable display."""
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

        # Show errors if any
        if self.execution_errors:
            output.append(
                f"\n⚠️  Encountered {len(self.execution_errors)} warning(s) during execution"
            )

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

    def get_execution_metrics(self) -> Dict[str, Any]:
        """
        Get detailed execution metrics from last analysis.

        Returns:
            Dict with agent metrics and error information
        """
        return {
            "agent_metrics": self.agent_metrics,
            "errors": [
                {
                    "agent": err.agent_name,
                    "type": err.error_type,
                    "message": err.error_message,
                    "timestamp": err.timestamp.isoformat(),
                    "recoverable": err.recoverable,
                }
                for err in self.execution_errors
            ],
            "total_errors": len(self.execution_errors),
        }


async def run(user_query):
    system = StockResearchSystem()
    result = await system.analyze_stocks(user_query=user_query)

    print("Raw Results: ")
    print(result)

    print("---------------------------------------------------------------------------------------------------------")

    # Print formatted results
    print("Formatted Results:")
    print(system.format_results_for_display(result))

    print("------------------------------------------------------------------------------------------------------------")
    print("\n📊 Execution Metrics:")
    
    # Print execution metrics
    metrics = system.get_execution_metrics()
    
    # 📋 Agent Execution Metrics:
    print(f"Total Agents: {len(metrics['agent_metrics'])}")
    print(f"Total Errors: {metrics['total_errors']}")

    return result


# Main entry point
if __name__ == "__main__":
    user_query = "find a top 3 stocks to trade for short term"
    asyncio.run(run(user_query))
