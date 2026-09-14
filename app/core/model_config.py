from typing import ClassVar, Dict
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelConfig(BaseSettings):
    provider: str = Field(
        default="groq", description="LLM provider: 'groq' or 'openai'"
    )

    model_name: str = Field(
        default="llama-3.3-70b-versatile",
        description="Model identifier for the chosen provider",
    )

    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (lower = more deterministic)",
    )

    max_tokens: int = Field(
        default=4096, gt=0, description="Maximum tokens in model response"
    )

    timeout: int = Field(default=60, description="Request timeout in seconds")

    # Available Groq models (class variable, not loaded from env)
    GROQ_MODELS: ClassVar[Dict[str, str]] = {
        "llama-3.3-70b-versatile": "Best balance of speed and quality (Recommended)",
        "llama-3.1-70b-versatile": "Fast and reliable alternative",
        "mixtral-8x7b-32768": "Good for long context (32K tokens)",
        "gemma2-9b-it": "Ultra-fast for simple tasks",
        "openai/gpt-oss-20b": "Fast and reliable alternative",
    }

    @field_validator("model_name")
    @classmethod
    def validate_model(cls, v: str, info) -> str:
        """
        Validate that the model is available for the selected provider.

        For Groq, ensures the model name is in the supported models list.
        """
        # Get provider from validation context
        provider = info.data.get("provider", "groq")

        if provider == "groq" and v not in cls.GROQ_MODELS:
            available = ", ".join(cls.GROQ_MODELS.keys())
            raise ValueError(
                f"Model '{v}' not available for Groq. Available models: {available}"
            )

        return v

    # Pydantic Settings configuration
    model_config = SettingsConfigDict(
        env_prefix="MODEL_",  # Environment variables must start with MODEL_
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,  # MODEL_NAME and model_name both work
        extra="ignore",  # Ignore extra environment variables
    )
