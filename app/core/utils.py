from typing import Optional
from app.core.config import SystemConfig
from app.core.model_config import ModelConfig


def load_config(env_file: str = ".env") -> SystemConfig:
    try:
        config = SystemConfig(_env_file=env_file)
        return config

    except Exception as e:
        error_msg = (
            f"Configuration error: {str(e)}\n\n"
            f"Please check:\n"
            f"1. .env file exists in the project root\n"
            f"2. Required variables are set: GROQ_API_KEY, BRIGHT_DATA_API_TOKEN\n"
            f"3. All values are properly formatted\n\n"
            f"See .env.example for reference."
        )
        raise ValueError(error_msg)


def get_groq_model_info(model_name: str) -> Optional[str]:
    return ModelConfig.GROQ_MODELS.get(model_name)


def print_config_info(config: SystemConfig) -> None:
    print("=" * 60)
    print("SYSTEM CONFIGURATION")
    print("=" * 60)

    # API Keys (redacted)
    print(f"Groq API Key: {'*' * 8}{config.groq_api_key[-4:]}")
    print(f"Bright Data Token: {'*' * 8}{config.bright_data_api_token[-4:]}")

    if config.openai_api_key:
        print(f"OpenAI API Key: {'*' * 8}{config.openai_api_key[-4:]}")

    # Model Config
    print(f"\nModel Provider: {config.model.provider}")
    print(f"Model Name: {config.model.model_name}")
    print(f"Temperature: {config.model.temperature}")
    print(f"Max Tokens: {config.model.max_tokens}")

    # System Settings
    print(f"\nMax Retries: {config.max_retries}")
    print(f"Retry Delay: {config.retry_delay_seconds}s")
    print(f"Caching: {'Enabled' if config.enable_caching else 'Disabled'}")
    print(f"Log Level: {config.log_level}")

    # Timeouts
    print("\nAgent Timeouts:")
    print(f"  Stock Finder: {config.stock_finder_timeout}s")
    print(f"  Market Data: {config.market_data_timeout}s")
    print(f"  News Analyst: {config.news_analyst_timeout}s")
    print(f"  Recommendation: {config.recommendation_timeout}s")

    print("=" * 60)
