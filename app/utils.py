import re
import json
import logging
from typing import Any, Dict, List, Optional, Union, Type
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pydantic import BaseModel, ValidationError

from app.models.schemas import (
    StockAction,
    NewsSentiment,
    ConfidenceLevel,
    AgentError,
    ValidationError as SchemaValidationError,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# VALIDATION UTILITIES


def validate_nse_symbol(symbol: str) -> bool:
    """
    Validate NSE stock symbol format.

    Args:
        symbol: Stock symbol to validate

    Returns:
        bool: True if valid NSE symbol format
    """
    if not symbol:
        return False

    # NSE symbols are typically 1-10 uppercase letters
    pattern = r"^[A-Z]{1,10}$"
    return bool(re.match(pattern, symbol.strip().upper()))


def validate_price(
    price: Union[float, str], allow_negative: bool = False
) -> Optional[float]:
    """
    Validate and parse price values.

    Args:
        price: Price value to validate (can be string with ₹ symbol)
        allow_negative: Whether to allow negative prices

    Returns:
        Optional[float]: Validated price or None if invalid
    """
    try:
        # Remove currency symbols and commas
        if isinstance(price, str):
            price = price.replace("₹", "").replace("Rs.", "").replace(",", "").strip()

        price_float = float(price)

        if not allow_negative and price_float < 0:
            logger.warning(f"Negative price not allowed: {price_float}")
            return None

        return price_float

    except (ValueError, TypeError) as e:
        logger.error(f"Invalid price format: {price}, error: {e}")
        return None


def validate_percentage(pct: Union[float, str]) -> Optional[float]:
    """
    Validate percentage values.

    Args:
        pct: Percentage value to validate (can include % symbol)

    Returns:
        Optional[float]: Validated percentage or None if invalid
    """
    try:
        if isinstance(pct, str):
            pct = pct.replace("%", "").strip()

        pct_float = float(pct)

        # Reasonable range for stock price changes
        if abs(pct_float) > 100:
            logger.warning(f"Unusual percentage value: {pct_float}%")

        return pct_float

    except (ValueError, TypeError) as e:
        logger.error(f"Invalid percentage format: {pct}, error: {e}")
        return None


def validate_volume(volume: Union[int, str]) -> Optional[int]:
    """
    Validate trading volume values.

    Args:
        volume: Volume to validate

    Returns:
        Optional[int]: Validated volume or None if invalid
    """
    try:
        if isinstance(volume, str):
            volume = volume.replace(",", "").strip()

        volume_int = int(volume)

        if volume_int < 0:
            logger.warning(f"Negative volume not allowed: {volume_int}")
            return None

        return volume_int

    except (ValueError, TypeError) as e:
        logger.error(f"Invalid volume format: {volume}, error: {e}")
        return None


# PARSING UTILITIES


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON object from text that may contain additional content.
    Handles cases where LLM includes explanation text around JSON.

    Args:
        text: Text potentially containing JSON

    Returns:
        Optional[Dict]: Parsed JSON object or None
    """
    try:
        # First, try direct JSON parsing
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON in code blocks
    json_patterns = [
        r"```json\s*(\{.*?\})\s*```",  # ```json { } ```
        r"```\s*(\{.*?\})\s*```",  # ``` { } ```
        r"\{.*?\}",  # Any JSON object
    ]

    for pattern in json_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

    logger.error(f"Could not extract valid JSON from text: {text[:200]}...")
    return None


def parse_llm_structured_output(
    output: str, schema: Type[BaseModel], fallback_parse: bool = True
) -> Optional[BaseModel]:
    """
    Parse LLM output into Pydantic schema with error handling.

    Args:
        output: Raw LLM output string
        schema: Pydantic model class to parse into
        fallback_parse: Whether to attempt fallback parsing on failure

    Returns:
        Optional[BaseModel]: Parsed and validated model instance or None
    """
    try:
        # Try direct JSON parsing
        if output.strip().startswith("{"):
            data = json.loads(output)
            return schema(**data)

        # Try extracting JSON from text
        data = extract_json_from_text(output)
        if data:
            return schema(**data)

        if fallback_parse:
            logger.warning("Standard parsing failed, attempting fallback...")
            # Implement custom parsing logic here based on schema
            # This is where you'd add schema-specific text parsing

        return None

    except ValidationError as e:
        logger.error(f"Pydantic validation error: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error parsing LLM output: {e}")
        return None


def clean_symbol(symbol: str) -> str:
    """
    Clean and normalize stock symbol.

    Args:
        symbol: Raw symbol string

    Returns:
        str: Cleaned uppercase symbol
    """
    return symbol.strip().upper().replace(".NS", "").replace(".BO", "")


# ERROR HANDLING UTILITIES


def create_agent_error(
    agent_name: str, error: Exception, recoverable: bool = True
) -> AgentError:
    """
    Create standardized agent error object.

    Args:
        agent_name: Name of the agent that encountered error
        error: The exception that occurred
        recoverable: Whether the system can retry/continue

    Returns:
        AgentError: Structured error information
    """
    return AgentError(
        agent_name=agent_name,
        error_type=type(error).__name__,
        error_message=str(error),
        recoverable=recoverable,
    )


def safe_get(
    data: Dict, key: str, default: Any = None, validator: Optional[callable] = None
) -> Any:
    """
    Safely get value from dictionary with optional validation.

    Args:
        data: Dictionary to get value from
        key: Key to retrieve
        default: Default value if key not found
        validator: Optional validation function

    Returns:
        Any: Retrieved and validated value or default
    """
    value = data.get(key, default)

    if validator and value is not None:
        try:
            return validator(value)
        except Exception as e:
            logger.warning(f"Validation failed for key '{key}': {e}")
            return default

    return value


# FORMATTING UTILITIES


def format_indian_currency(amount: float) -> str:
    """
    Format number in Indian currency format (₹ with commas).

    Args:
        amount: Amount to format

    Returns:
        str: Formatted currency string
    """
    try:
        # Indian number system: 12,34,56,789
        s = f"{amount:,.2f}"

        # Convert to Indian comma system
        parts = s.split(".")
        integer_part = parts[0].replace(",", "")

        # Format with Indian commas (last 3, then every 2)
        if len(integer_part) > 3:
            last_three = integer_part[-3:]
            remaining = integer_part[:-3]

            # Add comma every 2 digits from right
            result = last_three
            for i in range(len(remaining) - 1, -1, -2):
                if i == 0:
                    result = remaining[0] + "," + result
                else:
                    result = remaining[max(0, i - 1) : i + 1] + "," + result

            integer_part = result

        return f"₹{integer_part}.{parts[1]}"

    except Exception as e:
        logger.error(f"Error formatting currency: {e}")
        return f"₹{amount:.2f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format percentage value with + or - sign.

    Args:
        value: Percentage value
        decimals: Number of decimal places

    Returns:
        str: Formatted percentage string
    """
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.{decimals}f}%"


def format_large_number(num: int) -> str:
    """
    Format large numbers with Indian suffixes (L for lakhs, Cr for crores).

    Args:
        num: Number to format

    Returns:
        str: Formatted number string
    """
    if num >= 10_000_000:  # 1 crore
        return f"₹{num / 10_000_000:.2f} Cr"
    elif num >= 100_000:  # 1 lakh
        return f"₹{num / 100_000:.2f} L"
    else:
        return f"₹{num:,}"


# DATA QUALITY UTILITIES


def check_data_completeness(
    data: BaseModel, required_fields: List[str]
) -> tuple[bool, List[str]]:
    """
    Check if required fields in a Pydantic model have non-None values.

    Args:
        data: Pydantic model instance
        required_fields: List of field names to check

    Returns:
        tuple: (is_complete, list of missing fields)
    """
    missing = []

    for field in required_fields:
        if not hasattr(data, field) or getattr(data, field) is None:
            missing.append(field)

    return len(missing) == 0, missing


def validate_price_consistency(
    current: float, day_high: float, day_low: float, previous_close: float
) -> tuple[bool, Optional[str]]:
    """
    Validate price data consistency.

    Args:
        current: Current price
        day_high: Day's high
        day_low: Day's low
        previous_close: Previous close

    Returns:
        tuple: (is_valid, error_message)
    """
    if current < day_low or current > day_high:
        return (
            False,
            f"Current price {current} outside day range [{day_low}, {day_high}]",
        )

    if day_low > day_high:
        return False, f"Day low {day_low} greater than day high {day_high}"

    # Allow up to 20% circuit limit in NSE
    max_change = previous_close * 0.20
    if abs(current - previous_close) > max_change:
        logger.warning(
            f"Price change exceeds 20% circuit: {current} vs {previous_close}"
        )

    return True, None


# TIMING AND RETRY UTILITIES


def is_market_open() -> bool:
    """
    Check if NSE market is currently open.

    Returns:
        bool: True if market is open
    """
    now = datetime.now()

    # Check if it's a weekday (Monday=0, Sunday=6)
    if now.weekday() > 4:  # Saturday or Sunday
        return False

    # Check time (09:15 to 15:30)
    market_time = now.time()
    from datetime import time

    market_open = time(9, 15)
    market_close = time(15, 30)

    return market_open <= market_time <= market_close


def get_market_status() -> str:
    """
    Get current market status description.

    Returns:
        str: Market status description
    """
    now = datetime.now()

    if now.weekday() > 4:
        return "Market Closed (Weekend)"

    current_time = now.time()
    from datetime import time

    if current_time < time(9, 0):
        return "Pre-Market"
    elif time(9, 0) <= current_time < time(9, 15):
        return "Pre-Open Session"
    elif time(9, 15) <= current_time <= time(15, 30):
        return "Market Open"
    else:
        return "Market Closed"


# LOGGING UTILITIES


def log_agent_execution(
    agent_name: str, status: str, duration: float, details: Optional[str] = None
):
    """
    Log agent execution with structured information.

    Args:
        agent_name: Name of the agent
        status: Execution status (success/error/timeout)
        duration: Execution time in seconds
        details: Additional details
    """
    log_msg = f"Agent: {agent_name} | Status: {status} | Duration: {duration:.2f}s"

    if details:
        log_msg += f" | Details: {details}"

    if status == "error":
        logger.error(log_msg)
    elif status == "timeout":
        logger.warning(log_msg)
    else:
        logger.info(log_msg)
