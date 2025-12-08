from typing import List
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class MCPConfig(BaseSettings):
    api_token: str = Field(
        ...,
        description="Bright Data API token for web scraping",
    )

    web_unlocker_zone: str = Field(
        default="unblocker", description="Bright Data web unlocker zone"
    )

    browser_zone: str = Field(
        default="scraping_browser", description="Bright Data browser scraping zone"
    )

    # MCP server settings (typically don't need to change these)
    command: str = Field(default="npx", description="Command to run MCP server")

    args: List[str] = Field(
        default=["@brightdata/mcp"], description="Arguments for MCP server command"
    )

    transport: str = Field(default="stdio", description="MCP transport protocol")

    @field_validator("api_token")
    @classmethod
    def validate_token(cls, v: str) -> str:
        """
        Ensure API token is not empty and properly formatted.

        Raises:
            ValueError: If token is empty or invalid
        """
        if not v or v.strip() == "":
            raise ValueError(
                "Bright Data API token is required. "
                "Set BRIGHT_DATA_API_TOKEN in your .env file."
            )

        # Remove any whitespace
        cleaned_token = v.strip()

        # Basic validation - token should be alphanumeric
        if not cleaned_token.replace("-", "").replace("_", "").isalnum():
            raise ValueError(
                "Invalid Bright Data API token format. "
                "Token should only contain letters, numbers, hyphens, and underscores."
            )

        return cleaned_token

    # Pydantic Settings configuration
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )
