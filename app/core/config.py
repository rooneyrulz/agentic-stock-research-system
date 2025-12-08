"""
Configuration management for the stock research system.
Centralized configuration with environment variable support and validation.

Uses Pydantic BaseSettings for automatic .env file loading and type-safe configuration.
"""

from typing import Optional
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.model_config import ModelConfig
from app.core.mcp_config import MCPConfig


# MAIN SYSTEM CONFIGURATION
class SystemConfig(BaseSettings):
    """
    Main system configuration with automatic .env loading.

    This is the primary configuration class that loads all settings from:
    1. Environment variables
    2. .env file
    3. Default values (if not provided)

    Environment variables:
        GROQ_API_KEY: Groq API key (required)
        OPENAI_API_KEY: OpenAI API key (optional, for fallback)
        BRIGHT_DATA_API_TOKEN: Bright Data API token (required)
        MAX_RETRIES: Maximum retry attempts for failed operations
        RETRY_DELAY_SECONDS: Delay between retries
        ENABLE_CACHING: Enable response caching
        LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR)
        STOCK_FINDER_TIMEOUT: Timeout for stock finder agent
        MARKET_DATA_TIMEOUT: Timeout for market data agent
        NEWS_ANALYST_TIMEOUT: Timeout for news analyst agent
        RECOMMENDATION_TIMEOUT: Timeout for recommendation agent

    Example .env file:
        # Required
        GROQ_API_KEY=gsk_your_key_here
        BRIGHT_DATA_API_TOKEN=your_token_here

        # Optional
        OPENAI_API_KEY=sk-your_key_here
        MAX_RETRIES=3
        LOG_LEVEL=INFO
    """

    groq_api_key: str = Field(
        ...,  # Required
        description="Groq API key for LLM inference",
    )

    openai_api_key: Optional[str] = Field(
        default=None, description="OpenAI API key (optional, for fallback)"
    )

    bright_data_api_token: str = Field(
        ...,  # Required
        description="Bright Data API token for web scraping",
    )

    # Model configuration - automatically loads MODEL_* env vars
    model: ModelConfig = Field(
        default_factory=ModelConfig, description="LLM model configuration"
    )

    # MCP configuration - automatically loads from env vars
    mcp: MCPConfig = Field(
        default_factory=lambda: MCPConfig(),
        description="MCP (Bright Data) configuration",
    )

    # SYSTEM SETTINGS
    max_retries: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum retry attempts for failed operations",
    )

    retry_delay_seconds: float = Field(
        default=2.0, gt=0, description="Delay between retry attempts (seconds)"
    )

    enable_caching: bool = Field(
        default=True, description="Enable response caching to reduce API calls"
    )

    log_level: str = Field(
        default="INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL",
    )

    # AGENT TIMEOUTS (in seconds)
    stock_finder_timeout: int = Field(
        default=120, gt=0, description="Timeout for stock finder agent (seconds)"
    )

    market_data_timeout: int = Field(
        default=180, gt=0, description="Timeout for market data agent (seconds)"
    )

    news_analyst_timeout: int = Field(
        default=180, gt=0, description="Timeout for news analyst agent (seconds)"
    )

    recommendation_timeout: int = Field(
        default=120, gt=0, description="Timeout for recommendation agent (seconds)"
    )

    # VALIDATORS
    @field_validator("groq_api_key")
    @classmethod
    def validate_groq_key(cls, v: str) -> str:
        """
        Validate Groq API key format and presence.

        Groq API keys typically start with 'gsk_' prefix.
        """
        if not v or v.strip() == "":
            raise ValueError(
                "GROQ_API_KEY environment variable is required. "
                "Get your API key from https://console.groq.com"
            )

        cleaned_key = v.strip()

        # Check if it looks like a Groq key (starts with gsk_)
        if not cleaned_key.startswith("gsk_"):
            raise ValueError(
                "Invalid Groq API key format. "
                "Groq keys should start with 'gsk_'. "
                "Check your GROQ_API_KEY in .env file."
            )

        return cleaned_key

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """
        Validate logging level is one of the standard Python logging levels.
        """
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()

        if v_upper not in valid_levels:
            raise ValueError(
                f"Invalid log level: {v}. Must be one of: {', '.join(valid_levels)}"
            )

        return v_upper

    @model_validator(mode="after")
    def validate_nested_configs(self):
        """
        Ensure nested configurations (model, mcp) have the same Bright Data token.
        This runs after all fields are populated.
        """
        # Sync bright_data_api_token with mcp.api_token
        if self.mcp.api_token != self.bright_data_api_token:
            # Update MCP config to use the main token
            self.mcp.api_token = self.bright_data_api_token

        return self

    # PYDANTIC SETTINGS CONFIGURATION
    model_config = SettingsConfigDict(
        # .env file configuration
        env_file=".env",
        env_file_encoding="utf-8",
        # Allow environment variables to override .env file
        env_nested_delimiter="__",  # Use __ for nested configs (e.g., MODEL__TEMPERATURE)
        # Case sensitivity
        case_sensitive=False,  # GROQ_API_KEY and groq_api_key both work
        # Extra fields
        extra="ignore",  # Ignore unknown environment variables
        # Validation
        validate_default=True,  # Validate default values too
        validate_assignment=True,  # Validate on attribute assignment
    )
