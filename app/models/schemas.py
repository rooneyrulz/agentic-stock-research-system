from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Any
from enum import Enum
from datetime import datetime


# ENUMS - Constrained values for better type safety
class StockAction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class NewsSentiment(str, Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"


class ConfidenceLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class TechnicalSignal(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class TimeHorizon(str, Enum):
    SHORT_TERM = "1-3 days"
    MEDIUM_TERM = "1-2 weeks"


# STOCK FINDER AGENT OUTPUT SCHEMA
class SelectedStock(BaseModel):
    symbol: str = Field(
        ...,
        description="NSE stock symbol (e.g., RELIANCE, TCS)",
        min_length=1,
        max_length=20,
    )
    company_name: str = Field(..., description="Full company name", min_length=2)
    sector: str = Field(..., description="Industry sector")
    market_cap_cr: float = Field(
        ..., description="Market capitalization in crores", gt=0
    )
    avg_daily_volume_cr: float = Field(
        ..., description="Average daily trading volume in crores", gt=0
    )
    selection_reason: str = Field(
        ..., description="2-3 line rationale for selection", min_length=50
    )

    @field_validator("symbol")
    def validate_symbol(cls, v):
        return v.strip().upper()

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "RELIANCE",
                "company_name": "Reliance Industries Limited",
                "sector": "Oil & Gas",
                "market_cap_cr": 1800000.0,
                "avg_daily_volume_cr": 2500.0,
                "selection_reason": "Strong momentum with breakout above 2800 level, backed by refining margin expansion and consistent volume support.",
            }
        }


class StockFinderOutput(BaseModel):
    selected_stocks: List[SelectedStock] = Field(
        ..., description="List of 2-3 selected stocks", min_items=2, max_items=3
    )
    analysis_timestamp: datetime = Field(
        default_factory=datetime.now, description="When the analysis was performed"
    )
    market_context: Optional[str] = Field(
        None, description="Brief market context for selections"
    )


# MARKET DATA AGENT OUTPUT SCHEMA
class PriceData(BaseModel):
    current_price: float = Field(..., gt=0, description="Current market price in ₹")
    previous_close: float = Field(..., gt=0, description="Previous day close in ₹")
    day_high: float = Field(..., gt=0, description="Day's high in ₹")
    day_low: float = Field(..., gt=0, description="Day's low in ₹")
    price_change: float = Field(..., description="Absolute price change in ₹")
    price_change_pct: float = Field(..., description="Percentage price change")
    week_52_high: Optional[float] = Field(None, gt=0, description="52-week high in ₹")
    week_52_low: Optional[float] = Field(None, gt=0, description="52-week low in ₹")


class VolumeData(BaseModel):
    today_volume_shares: int = Field(..., ge=0, description="Today's volume in shares")
    today_volume_cr: float = Field(..., ge=0, description="Today's volume in ₹ crores")
    avg_5day_volume_shares: Optional[int] = Field(None, ge=0)
    volume_status: str = Field(..., description="Above/Below average volume")


class TechnicalIndicators(BaseModel):
    rsi_14: Optional[float] = Field(None, ge=0, le=100, description="14-period RSI")
    rsi_signal: Optional[str] = Field(None, description="Overbought/Oversold/Neutral")
    ma_20: Optional[float] = Field(None, gt=0, description="20-day moving average")
    ma_50: Optional[float] = Field(None, gt=0, description="50-day moving average")
    ma_200: Optional[float] = Field(None, gt=0, description="200-day moving average")
    price_vs_ma50_pct: Optional[float] = Field(
        None, description="Price deviation from MA50"
    )
    macd_signal: Optional[TechnicalSignal] = Field(None, description="MACD signal")
    support_level: Optional[float] = Field(None, gt=0)
    resistance_level: Optional[float] = Field(None, gt=0)


class TrendAnalysis(BaseModel):
    trend_7d_pct: float = Field(..., description="7-day price change %")
    trend_30d_pct: float = Field(..., description="30-day price change %")
    momentum: str = Field(..., description="Strong/Weak/Sideways")
    volatility_assessment: Optional[str] = Field(None)


class MarketDataOutput(BaseModel):
    symbol: str = Field(..., description="Stock symbol")
    company_name: str = Field(..., description="Company name")
    price_data: PriceData
    volume_data: VolumeData
    technical_indicators: TechnicalIndicators
    trend_analysis: TrendAnalysis
    data_timestamp: datetime = Field(default_factory=datetime.now)

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "TCS",
                "company_name": "Tata Consultancy Services",
                "price_data": {
                    "current_price": 3850.50,
                    "previous_close": 3820.00,
                    "day_high": 3865.00,
                    "day_low": 3835.00,
                    "price_change": 30.50,
                    "price_change_pct": 0.80,
                },
                "volume_data": {
                    "today_volume_shares": 2500000,
                    "today_volume_cr": 962.5,
                    "volume_status": "Above Average",
                },
                "technical_indicators": {
                    "rsi_14": 58.5,
                    "rsi_signal": "Neutral",
                    "ma_50": 3780.00,
                    "macd_signal": "BULLISH",
                },
                "trend_analysis": {
                    "trend_7d_pct": 2.3,
                    "trend_30d_pct": 5.8,
                    "momentum": "Strong",
                },
            }
        }


class AllMarketDataOutput(BaseModel):
    stocks_data: List[MarketDataOutput] = Field(..., min_items=1)
    analysis_timestamp: datetime = Field(default_factory=datetime.now)


# NEWS ANALYST AGENT OUTPUT SCHEMA
class NewsItem(BaseModel):
    date: str = Field(..., description="News date")
    headline: str = Field(..., description="News headline or summary", min_length=10)
    impact: NewsSentiment = Field(..., description="Expected impact")
    relevance_score: Optional[int] = Field(
        None, ge=1, le=10, description="Relevance 1-10"
    )


class NewsAnalysisOutput(BaseModel):
    symbol: str = Field(..., description="Stock symbol")
    company_name: str = Field(..., description="Company name")
    recent_news: List[NewsItem] = Field(
        ...,
        description="3-5 most relevant recent news items",
        min_items=0,
        max_items=10,
    )
    overall_sentiment: NewsSentiment = Field(..., description="Aggregated sentiment")
    confidence_level: ConfidenceLevel = Field(..., description="Analysis confidence")
    short_term_impact: str = Field(
        ..., description="Expected 1-3 day impact", min_length=20
    )
    key_catalysts: List[str] = Field(
        default_factory=list, description="Upcoming events/announcements"
    )
    risk_factors: List[str] = Field(
        default_factory=list, description="Potential negative triggers"
    )
    analysis_timestamp: datetime = Field(default_factory=datetime.now)


class AllNewsAnalysisOutput(BaseModel):
    news_analysis: List[NewsAnalysisOutput] = Field(..., min_items=1)
    analysis_timestamp: datetime = Field(default_factory=datetime.now)


# RECOMMENDATION AGENT OUTPUT SCHEMA
class EntryStrategy(BaseModel):
    current_price: float = Field(..., gt=0, description="Current price in ₹")
    suggested_entry_min: float = Field(..., gt=0, description="Entry range minimum")
    suggested_entry_max: float = Field(..., gt=0, description="Entry range maximum")
    stop_loss: float = Field(..., gt=0, description="Stop loss level in ₹")
    stop_loss_pct: float = Field(..., description="Stop loss percentage")
    target_price: float = Field(..., gt=0, description="Target price in ₹")
    target_upside_pct: float = Field(..., description="Expected upside %")
    risk_reward_ratio: str = Field(..., description="e.g., 1:2, 1:3")


class RecommendationRationale(BaseModel):
    technical_factors: str = Field(..., min_length=30)
    fundamental_factors: str = Field(..., min_length=30)
    key_strengths: List[str] = Field(..., min_items=1, max_items=5)
    key_risks: List[str] = Field(..., min_items=1, max_items=5)


class StockRecommendation(BaseModel):
    symbol: str = Field(..., description="Stock symbol")
    company_name: str = Field(..., description="Company name")
    recommendation: StockAction = Field(..., description="BUY/SELL/HOLD")
    confidence: ConfidenceLevel = Field(..., description="Recommendation confidence")
    time_horizon: TimeHorizon = Field(..., description="Expected time frame")
    entry_strategy: EntryStrategy
    rationale: RecommendationRationale
    monitoring_points: List[str] = Field(
        ..., description="Key levels/events to watch", min_items=1, max_items=5
    )
    recommendation_timestamp: datetime = Field(default_factory=datetime.now)

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "INFY",
                "company_name": "Infosys Limited",
                "recommendation": "BUY",
                "confidence": "HIGH",
                "time_horizon": "1-2 weeks",
                "entry_strategy": {
                    "current_price": 1450.00,
                    "suggested_entry_min": 1440.00,
                    "suggested_entry_max": 1460.00,
                    "stop_loss": 1400.00,
                    "stop_loss_pct": -3.45,
                    "target_price": 1550.00,
                    "target_upside_pct": 6.90,
                    "risk_reward_ratio": "1:2",
                },
                "rationale": {
                    "technical_factors": "Breakout above 1440 with volume support",
                    "fundamental_factors": "Strong Q3 results, margin expansion",
                    "key_strengths": ["Technical breakout", "Positive earnings"],
                    "key_risks": ["USD volatility", "Global IT spending slowdown"],
                },
                "monitoring_points": ["Watch 1440 support", "Q4 guidance on Feb 15"],
            }
        }


class FinalRecommendationsOutput(BaseModel):
    recommendations: List[StockRecommendation] = Field(..., min_items=1)
    market_summary: Optional[str] = Field(None, description="Overall market context")
    analysis_timestamp: datetime = Field(default_factory=datetime.now)
    disclaimer: str = Field(
        default="This analysis is for educational purposes only. Please consult a financial advisor before making investment decisions."
    )


# SYSTEM OUTPUT SCHEMA
class StockResearchResult(BaseModel):
    status: str = Field(..., description="completed/error/partial")
    selected_stocks: Optional[StockFinderOutput] = None
    market_data: Optional[AllMarketDataOutput] = None
    news_analysis: Optional[AllNewsAnalysisOutput] = None
    recommendations: Optional[FinalRecommendationsOutput] = None
    execution_time_seconds: Optional[float] = None
    error_message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


# ERROR HANDLING SCHEMAS
class AgentError(BaseModel):
    agent_name: str
    error_type: str
    error_message: str
    timestamp: datetime = Field(default_factory=datetime.now)
    recoverable: bool = Field(default=True)


class ValidationError(BaseModel):
    field: str
    error: str
    provided_value: Any
    timestamp: datetime = Field(default_factory=datetime.now)
