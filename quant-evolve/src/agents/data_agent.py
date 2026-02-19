"""
Data Agent for QuantEvolve.

Implements Section 5.1 initialization logic:
1. Analyze available OHLCV data and build a structured data schema.
2. Recommend strategy categories (C) based on market characteristics.
3. Generate C+1 seed strategy specifications (including buy-and-hold).
4. Build initial island specifications (one seed strategy per island).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


# Keep category order stable so category_bit_index remains deterministic.
STANDARD_CATEGORIES: List[str] = [
    "momentum",
    "mean_reversion",
    "trend_following",
    "breakout",
    "volatility",
    "market_microstructure",
    "fundamental_value",
    "machine_learning",
]


class DataAgent(BaseAgent):
    """
    Data Agent for initializing QuantEvolve from market data.

    The agent supports both:
    - LLM-driven generation via prompt templates.
    - Deterministic local fallback (useful for offline/dev mode).
    """

    def __init__(
        self,
        model_name: str = "Qwen3-30B-A3B-Instruct-2507",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.3,
        top_p: float = 0.9,
        max_tokens: int = 4096,
        **kwargs: Any,
    ):
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            **kwargs,
        )
        self.latest_output: Dict[str, Any] = {}
        logger.info("DataAgent initialized")

    # ---------------------------------------------------------------------
    # Public workflow methods
    # ---------------------------------------------------------------------

    def analyze_data_schema(
        self,
        market_data: Optional[Dict[str, pd.DataFrame]] = None,
        data_description: Optional[str] = None,
        use_llm: bool = True,
    ) -> Dict[str, Any]:
        """
        Analyze OHLCV data and return structured schema + market characteristics.
        """
        description = self._build_data_description(market_data, data_description)

        if use_llm:
            try:
                prompt = self._load_prompt(
                    "DATA_SCHEMA_PROMPT",
                    fallback_template=(
                        "Analyze this market data description and return JSON with "
                        "keys: data_schema, market_characteristics.\n\n{data_description}"
                    ),
                )
                response = self.generate_response(
                    system_prompt=(
                        "You are the Data Agent in QuantEvolve. Return valid JSON only."
                    ),
                    user_prompt=prompt.format(data_description=description),
                    parse_json=True,
                    max_retries=2,
                )
                normalized = self._normalize_schema_response(response, market_data)
                logger.info("Data schema generated via LLM")
                return normalized
            except Exception as exc:
                logger.warning(f"LLM schema generation failed, fallback to local analysis: {exc}")

        return self._infer_schema_locally(market_data, data_description)

    def identify_strategy_categories(
        self,
        schema_result: Dict[str, Any],
        num_categories: int = 8,
        use_llm: bool = True,
    ) -> Dict[str, Any]:
        """
        Identify C strategy categories from data/market characteristics.
        """
        num_categories = int(max(3, min(8, num_categories)))
        market_characteristics = schema_result.get("market_characteristics", {})

        if use_llm:
            try:
                prompt = self._load_prompt(
                    "CATEGORY_IDENTIFICATION_PROMPT",
                    fallback_template=(
                        "Recommend 6-8 categories from: momentum, mean_reversion, "
                        "trend_following, breakout, volatility, market_microstructure, "
                        "fundamental_value, machine_learning. Return valid JSON."
                    ),
                )
                response = self.generate_response(
                    system_prompt=(
                        "You are the Data Agent. Return valid JSON only with "
                        "recommended_categories and market_suitability_analysis."
                    ),
                    user_prompt=prompt.format(
                        market_characteristics=json.dumps(
                            market_characteristics, ensure_ascii=False, indent=2
                        )
                    ),
                    parse_json=True,
                    max_retries=2,
                )
                normalized = self._normalize_category_response(response, num_categories)
                logger.info("Category analysis generated via LLM")
                return normalized
            except Exception as exc:
                logger.warning(f"LLM category generation failed, fallback to local logic: {exc}")

        return self._recommend_categories_locally(
            market_characteristics=market_characteristics,
            num_categories=num_categories,
            data_schema=schema_result.get("data_schema", {}),
        )

    def generate_seed_strategies(
        self,
        category_result: Dict[str, Any],
        schema_result: Dict[str, Any],
        use_llm: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate C+1 seed strategy specifications (C categories + buy-and-hold).
        """
        if use_llm:
            try:
                prompt = self._load_prompt(
                    "SEED_STRATEGY_PROMPT",
                    fallback_template=(
                        "Generate seed strategy specs and return JSON with "
                        "seed_strategies and implementation_notes."
                    ),
                )
                response = self.generate_response(
                    system_prompt=(
                        "You are the Data Agent. Return valid JSON only for seed strategies."
                    ),
                    user_prompt=prompt.format(
                        category_recommendations=json.dumps(
                            category_result, ensure_ascii=False, indent=2
                        ),
                        data_schema=json.dumps(
                            schema_result.get("data_schema", {}),
                            ensure_ascii=False,
                            indent=2,
                        ),
                    ),
                    parse_json=True,
                    max_retries=2,
                )
                normalized = self._normalize_seed_response(
                    response=response,
                    categories=category_result.get("recommended_categories", []),
                    data_schema=schema_result.get("data_schema", {}),
                )
                logger.info("Seed strategies generated via LLM")
                return normalized
            except Exception as exc:
                logger.warning(f"LLM seed generation failed, fallback to local templates: {exc}")

        return self._build_seed_strategies_locally(
            categories=category_result.get("recommended_categories", []),
            data_schema=schema_result.get("data_schema", {}),
        )

    def build_island_initialization(self, seed_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build island initialization plan where each seed strategy forms one island.
        """
        seed_strategies = seed_result.get("seed_strategies", [])
        islands = []
        for island_id, seed in enumerate(seed_strategies):
            islands.append(
                {
                    "island_id": island_id,
                    "category_focus": seed.get("category", "unknown"),
                    "seed_strategy_id": seed.get("strategy_id", f"seed_{island_id}"),
                    "seed_strategy": seed,
                }
            )

        return {
            "num_islands": len(islands),
            "islands": islands,
            "initialization_note": (
                "N = C + 1 islands initialized (C categories + 1 buy_and_hold benchmark)."
            ),
        }

    def run_full_workflow(
        self,
        market_data: Optional[Dict[str, pd.DataFrame]] = None,
        data_description: Optional[str] = None,
        num_categories: int = 8,
        use_llm: bool = True,
    ) -> Dict[str, Any]:
        """
        Full 5.1 initialization workflow:
        data schema -> category analysis -> seed strategies -> island plan.
        """
        schema_result = self.analyze_data_schema(
            market_data=market_data,
            data_description=data_description,
            use_llm=use_llm,
        )
        category_result = self.identify_strategy_categories(
            schema_result=schema_result,
            num_categories=num_categories,
            use_llm=use_llm,
        )
        seed_result = self.generate_seed_strategies(
            category_result=category_result,
            schema_result=schema_result,
            use_llm=use_llm,
        )
        island_init = self.build_island_initialization(seed_result)

        output = {
            "data_schema": schema_result.get("data_schema", {}),
            "market_characteristics": schema_result.get("market_characteristics", {}),
            "category_analysis": category_result,
            "seed_strategies": seed_result,
            "island_initialization": island_init,
            "generated_at": datetime.now().isoformat(),
            "agent": "DataAgent",
        }
        self.latest_output = output
        return output

    def execute(
        self,
        market_data: Optional[Dict[str, pd.DataFrame]] = None,
        data_description: Optional[str] = None,
        num_categories: int = 8,
        use_llm: bool = True,
    ) -> Dict[str, Any]:
        """
        BaseAgent entrypoint.
        """
        return self.run_full_workflow(
            market_data=market_data,
            data_description=data_description,
            num_categories=num_categories,
            use_llm=use_llm,
        )

    # ---------------------------------------------------------------------
    # LLM output normalization
    # ---------------------------------------------------------------------

    def _normalize_schema_response(
        self,
        response: Any,
        market_data: Optional[Dict[str, pd.DataFrame]],
    ) -> Dict[str, Any]:
        if not isinstance(response, dict):
            return self._infer_schema_locally(market_data, data_description=None)

        payload = dict(response)
        if "data_schema" not in payload:
            payload = {"data_schema": payload}

        payload.setdefault("market_characteristics", {})

        data_schema = payload.get("data_schema", {})
        data_schema.setdefault("assets", list(market_data.keys()) if market_data else [])
        data_schema.setdefault("frequency", "daily")
        data_schema.setdefault("columns", {})
        data_schema.setdefault("date_range", {"start": None, "end": None})
        data_schema.setdefault("total_records", 0)
        data_schema.setdefault("missing_data_percentage", 0.0)
        data_schema.setdefault("data_quality_notes", "Generated by DataAgent")

        market = payload.get("market_characteristics", {})
        market.setdefault("average_daily_volume", 0.0)
        market.setdefault("average_price_range_pct", 0.0)
        market.setdefault("volatility_regime", "medium")
        market.setdefault("trend_characteristics", "mixed")
        market.setdefault("liquidity_assessment", "medium")

        return {"data_schema": data_schema, "market_characteristics": market}

    def _normalize_category_response(self, response: Any, num_categories: int) -> Dict[str, Any]:
        if not isinstance(response, dict):
            return self._recommend_categories_locally({}, num_categories, {})

        categories = response.get("recommended_categories", [])
        if not isinstance(categories, list):
            categories = []

        normalized_categories = []
        seen = set()
        for item in categories:
            if not isinstance(item, dict):
                continue
            name = str(item.get("category_name", "")).strip().lower()
            if name not in STANDARD_CATEGORIES or name in seen:
                continue
            seen.add(name)
            normalized_categories.append(
                {
                    "category_id": len(normalized_categories),
                    "category_name": name,
                    "rationale": item.get("rationale", "LLM-generated category."),
                    "expected_effectiveness": item.get("expected_effectiveness", "medium"),
                    "key_indicators": item.get("key_indicators", []),
                    "typical_lookback_periods": item.get("typical_lookback_periods", [20, 60]),
                }
            )
            if len(normalized_categories) >= num_categories:
                break

        if not normalized_categories:
            return self._recommend_categories_locally({}, num_categories, {})

        suitability = response.get("market_suitability_analysis", {})
        if not isinstance(suitability, dict):
            suitability = {}
        suitability.setdefault(
            "best_fit_categories",
            [c["category_name"] for c in normalized_categories[:3]],
        )
        suitability.setdefault("challenging_categories", [])
        suitability.setdefault("overall_market_type", "mixed")
        suitability.setdefault("recommendation", "Prioritize diverse seed strategies.")

        return {
            "recommended_categories": normalized_categories,
            "market_suitability_analysis": suitability,
        }

    def _normalize_seed_response(
        self,
        response: Any,
        categories: List[Dict[str, Any]],
        data_schema: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not isinstance(response, dict):
            return self._build_seed_strategies_locally(categories, data_schema)

        seeds = response.get("seed_strategies", [])
        if not isinstance(seeds, list) or len(seeds) == 0:
            return self._build_seed_strategies_locally(categories, data_schema)

        normalized = []
        seen_ids = set()
        for seed in seeds:
            if not isinstance(seed, dict):
                continue
            sid = str(seed.get("strategy_id", "")).strip()
            if not sid:
                sid = f"seed_{len(normalized)}"
            if sid in seen_ids:
                sid = f"{sid}_{len(normalized)}"
            seen_ids.add(sid)

            normalized.append(
                {
                    "strategy_id": sid,
                    "category": seed.get("category", "unknown"),
                    "category_bit_index": seed.get("category_bit_index"),
                    "name": seed.get("name", sid),
                    "description": seed.get("description", "Seed strategy generated by DataAgent."),
                    "complexity": seed.get("complexity", "low"),
                    "key_parameters": seed.get("key_parameters", {}),
                    "pseudocode": seed.get("pseudocode", ""),
                    "expected_characteristics": seed.get("expected_characteristics", {}),
                }
            )

        if not any(s.get("category") == "benchmark" for s in normalized):
            normalized.append(self._build_benchmark_seed(data_schema))

        implementation_notes = response.get("implementation_notes", {})
        if not isinstance(implementation_notes, dict):
            implementation_notes = {}
        implementation_notes.setdefault("total_strategies", len(normalized))
        implementation_notes.setdefault(
            "categories_covered",
            list({seed.get("category", "unknown") for seed in normalized}),
        )
        implementation_notes.setdefault("benchmark_included", True)
        implementation_notes.setdefault("all_strategies_zipline_compatible", True)

        return {"seed_strategies": normalized, "implementation_notes": implementation_notes}

    # ---------------------------------------------------------------------
    # Local fallback logic
    # ---------------------------------------------------------------------

    def _infer_schema_locally(
        self,
        market_data: Optional[Dict[str, pd.DataFrame]],
        data_description: Optional[str],
    ) -> Dict[str, Any]:
        if not market_data:
            return {
                "data_schema": {
                    "assets": [],
                    "columns": {
                        "open": {"dtype": "float64", "description": "Opening price"},
                        "high": {"dtype": "float64", "description": "Highest price"},
                        "low": {"dtype": "float64", "description": "Lowest price"},
                        "close": {"dtype": "float64", "description": "Closing price"},
                        "volume": {"dtype": "float64", "description": "Trading volume"},
                    },
                    "frequency": "daily",
                    "date_range": {"start": None, "end": None},
                    "total_records": 0,
                    "missing_data_percentage": 0.0,
                    "data_quality_notes": data_description or "No in-memory market data provided.",
                },
                "market_characteristics": {
                    "average_daily_volume": 0.0,
                    "average_price_range_pct": 0.0,
                    "volatility_regime": "medium",
                    "trend_characteristics": "mixed",
                    "liquidity_assessment": "medium",
                },
            }

        assets = list(market_data.keys())
        columns = {}
        total_records = 0
        missing_ratio_values = []
        volume_samples = []
        range_pct_samples = []
        return_samples = []
        starts = []
        ends = []
        inferred_frequency = "daily"

        for symbol, df in market_data.items():
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue

            total_records += len(df)

            if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 1:
                starts.append(df.index.min())
                ends.append(df.index.max())
                inferred_frequency = self._detect_frequency(df.index)

            for col in df.columns:
                if col not in columns:
                    columns[col] = {
                        "dtype": str(df[col].dtype),
                        "description": self._column_description(col),
                    }

            numeric_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
            if numeric_cols:
                missing_ratio_values.append(float(df[numeric_cols].isna().mean().mean()))

            if "volume" in df.columns:
                volume_samples.append(float(pd.to_numeric(df["volume"], errors="coerce").mean()))

            if all(c in df.columns for c in ["high", "low", "close"]):
                high = pd.to_numeric(df["high"], errors="coerce")
                low = pd.to_numeric(df["low"], errors="coerce")
                close = pd.to_numeric(df["close"], errors="coerce").replace(0, np.nan)
                rpct = ((high - low) / close).replace([np.inf, -np.inf], np.nan)
                range_pct_samples.append(float(rpct.mean()))

            if "close" in df.columns:
                close = pd.to_numeric(df["close"], errors="coerce")
                ret = close.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
                if not ret.empty:
                    return_samples.extend(ret.values.tolist())

        start = str(min(starts).date()) if starts else None
        end = str(max(ends).date()) if ends else None
        missing_pct = float(np.nanmean(missing_ratio_values)) if missing_ratio_values else 0.0
        avg_volume = float(np.nanmean(volume_samples)) if volume_samples else 0.0
        avg_range_pct = float(np.nanmean(range_pct_samples)) if range_pct_samples else 0.0

        if return_samples:
            returns_arr = np.asarray(return_samples, dtype=np.float64)
            annualized_vol = float(np.nanstd(returns_arr) * np.sqrt(252))
            avg_return = float(np.nanmean(returns_arr))
        else:
            annualized_vol = 0.0
            avg_return = 0.0

        volatility_regime = self._classify_volatility_regime(annualized_vol)
        trend_characteristics = self._classify_trend(avg_return, annualized_vol)
        liquidity_assessment = self._classify_liquidity(avg_volume)

        return {
            "data_schema": {
                "assets": assets,
                "columns": columns,
                "frequency": inferred_frequency,
                "date_range": {"start": start, "end": end},
                "total_records": int(total_records),
                "missing_data_percentage": round(missing_pct * 100, 4),
                "data_quality_notes": "Local deterministic schema inference by DataAgent.",
            },
            "market_characteristics": {
                "average_daily_volume": round(avg_volume, 4),
                "average_price_range_pct": round(avg_range_pct, 4),
                "volatility_regime": volatility_regime,
                "trend_characteristics": trend_characteristics,
                "liquidity_assessment": liquidity_assessment,
            },
        }

    def _recommend_categories_locally(
        self,
        market_characteristics: Dict[str, Any],
        num_categories: int,
        data_schema: Dict[str, Any],
    ) -> Dict[str, Any]:
        trend = str(market_characteristics.get("trend_characteristics", "mixed")).lower()
        vol = str(market_characteristics.get("volatility_regime", "medium")).lower()
        liq = str(market_characteristics.get("liquidity_assessment", "medium")).lower()
        has_fundamental = any(
            k in str(col).lower()
            for col in data_schema.get("columns", {}).keys()
            for k in ("pe", "pb", "eps", "market_cap")
        )

        ordered = []
        if trend == "trending":
            ordered.extend(["momentum", "trend_following", "breakout"])
        elif trend == "mean-reverting":
            ordered.extend(["mean_reversion", "breakout", "momentum"])
        else:
            ordered.extend(["momentum", "mean_reversion", "trend_following"])

        if vol in {"high", "medium"}:
            ordered.append("volatility")
        if liq == "high":
            ordered.append("market_microstructure")
        if has_fundamental:
            ordered.append("fundamental_value")
        ordered.append("machine_learning")

        # Fill remaining slots with standard categories while preserving order.
        for cat in STANDARD_CATEGORIES:
            if cat not in ordered:
                ordered.append(cat)

        selected = ordered[:num_categories]
        categories = []
        for idx, cat in enumerate(selected):
            categories.append(
                {
                    "category_id": idx,
                    "category_name": cat,
                    "rationale": self._category_rationale(cat, trend, vol, liq),
                    "expected_effectiveness": "medium",
                    "key_indicators": self._category_indicators(cat),
                    "typical_lookback_periods": self._category_lookbacks(cat),
                }
            )

        challenging = []
        if liq == "low":
            challenging.append("market_microstructure")
        if not has_fundamental:
            challenging.append("fundamental_value")

        return {
            "recommended_categories": categories,
            "market_suitability_analysis": {
                "best_fit_categories": [c["category_name"] for c in categories[:3]],
                "challenging_categories": challenging,
                "overall_market_type": trend if trend in {"trending", "mean-reverting"} else "mixed",
                "recommendation": "Initialize one island per category plus a buy-and-hold benchmark.",
            },
        }

    def _build_seed_strategies_locally(
        self,
        categories: List[Dict[str, Any]],
        data_schema: Dict[str, Any],
    ) -> Dict[str, Any]:
        assets = data_schema.get("assets", [])
        seed_strategies = []
        for category in categories:
            cname = category.get("category_name", "momentum")
            seed_strategies.append(self._build_seed_template(cname, assets))

        seed_strategies.append(self._build_benchmark_seed(data_schema))

        return {
            "seed_strategies": seed_strategies,
            "implementation_notes": {
                "total_strategies": len(seed_strategies),
                "categories_covered": [c.get("category_name", "") for c in categories] + ["benchmark"],
                "benchmark_included": True,
                "all_strategies_zipline_compatible": True,
            },
        }

    # ---------------------------------------------------------------------
    # Helper utilities
    # ---------------------------------------------------------------------

    def _build_data_description(
        self,
        market_data: Optional[Dict[str, pd.DataFrame]],
        data_description: Optional[str],
    ) -> str:
        if data_description:
            return data_description
        if not market_data:
            return "No explicit market_data provided."
        summary = []
        for symbol, df in market_data.items():
            if not isinstance(df, pd.DataFrame):
                summary.append(f"{symbol}: non-DataFrame input")
                continue
            idx_desc = "no_datetime_index"
            if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 0:
                idx_desc = f"{df.index.min().date()}->{df.index.max().date()}"
            summary.append(
                f"{symbol}: rows={len(df)}, cols={list(df.columns)}, range={idx_desc}"
            )
        return "\n".join(summary)

    def _load_prompt(self, prompt_name: str, fallback_template: str) -> str:
        try:
            from ..prompts import data_agent_prompts

            return str(getattr(data_agent_prompts, prompt_name))
        except Exception:
            return fallback_template

    def _detect_frequency(self, index: pd.DatetimeIndex) -> str:
        if len(index) < 2:
            return "daily"
        deltas = np.diff(index.values).astype("timedelta64[s]").astype(float)
        median_seconds = float(np.median(deltas))
        day = 24 * 3600
        if median_seconds < day:
            return "intraday"
        if median_seconds <= 3 * day:
            return "daily"
        if median_seconds <= 10 * day:
            return "weekly"
        return "monthly"

    def _column_description(self, column: str) -> str:
        c = column.lower()
        mapping = {
            "open": "Opening price",
            "high": "Highest price during period",
            "low": "Lowest price during period",
            "close": "Closing price",
            "adj_close": "Adjusted close price",
            "volume": "Trading volume",
        }
        return mapping.get(c, f"Column: {column}")

    def _classify_volatility_regime(self, annualized_vol: float) -> str:
        if annualized_vol < 0.15:
            return "low"
        if annualized_vol < 0.35:
            return "medium"
        return "high"

    def _classify_trend(self, avg_return: float, annualized_vol: float) -> str:
        if avg_return > 0.0003 and annualized_vol < 0.5:
            return "trending"
        if abs(avg_return) < 0.0001 and annualized_vol < 0.25:
            return "mean-reverting"
        return "mixed"

    def _classify_liquidity(self, average_volume: float) -> str:
        if average_volume >= 5_000_000:
            return "high"
        if average_volume >= 1_000_000:
            return "medium"
        return "low"

    def _category_rationale(self, cat: str, trend: str, vol: str, liq: str) -> str:
        if cat == "momentum":
            return "Momentum captures continuation when markets exhibit directional persistence."
        if cat == "mean_reversion":
            return "Mean-reversion can exploit short-term overreaction and pullback behavior."
        if cat == "trend_following":
            return "Trend-following benefits from multi-period directional moves."
        if cat == "breakout":
            return "Breakout strategies react to regime transitions and range expansion."
        if cat == "volatility":
            return "Volatility-aware logic helps adapt position sizing and entry timing."
        if cat == "market_microstructure":
            return f"Liquidity={liq} supports volume/flow-based signals and execution-aware logic."
        if cat == "fundamental_value":
            return "Include only when fundamental features are available in the data universe."
        return "Machine-learning style templates can combine multiple weak signals adaptively."

    def _category_indicators(self, cat: str) -> List[str]:
        mapping = {
            "momentum": ["ROC", "RSI", "moving_average_crossover"],
            "mean_reversion": ["Bollinger_Bands", "z_score", "RSI_extremes"],
            "trend_following": ["MA_50_200", "ADX", "donchian_trend"],
            "breakout": ["Donchian_Channel", "ATR_breakout", "volume_confirmation"],
            "volatility": ["realized_volatility", "ATR", "volatility_targeting"],
            "market_microstructure": ["volume_spike", "dollar_volume", "amihud_proxy"],
            "fundamental_value": ["PE", "PB", "earnings_yield"],
            "machine_learning": ["feature_ensemble", "regime_classifier", "meta_signal"],
        }
        return mapping.get(cat, [])

    def _category_lookbacks(self, cat: str) -> List[int]:
        mapping = {
            "momentum": [10, 20, 60],
            "mean_reversion": [5, 20, 60],
            "trend_following": [20, 50, 200],
            "breakout": [20, 55, 100],
            "volatility": [10, 20, 63],
            "market_microstructure": [5, 10, 20],
            "fundamental_value": [63, 126, 252],
            "machine_learning": [20, 63, 126],
        }
        return mapping.get(cat, [20, 60])

    def _build_seed_template(self, category_name: str, assets: List[str]) -> Dict[str, Any]:
        bit_index = STANDARD_CATEGORIES.index(category_name) if category_name in STANDARD_CATEGORIES else None
        strategy_id = f"seed_{category_name}_001"
        name = category_name.replace("_", " ").title()
        lookbacks = self._category_lookbacks(category_name)
        indicators = self._category_indicators(category_name)

        pseudocode = (
            "1. Pull historical prices with data.history()\n"
            "2. Compute category-specific signals\n"
            "3. Rank/select tradable assets\n"
            "4. Apply risk-aware position sizing and rebalance"
        )
        if category_name == "momentum":
            pseudocode = (
                "1. Compute 20-day returns for each asset\n"
                "2. Rank assets by return\n"
                "3. Long top-ranked subset equally\n"
                "4. Rebalance weekly"
            )
        elif category_name == "mean_reversion":
            pseudocode = (
                "1. Compute rolling z-score/Bollinger bands\n"
                "2. Buy oversold assets, reduce overbought\n"
                "3. Exit near mean reversion level\n"
                "4. Rebalance daily or every few bars"
            )
        elif category_name == "breakout":
            pseudocode = (
                "1. Compute Donchian highs/lows\n"
                "2. Enter on breakout with volume confirmation\n"
                "3. Use ATR-based stop logic\n"
                "4. Rebalance weekly"
            )

        return {
            "strategy_id": strategy_id,
            "category": category_name,
            "category_bit_index": bit_index,
            "name": name,
            "description": f"Seed {category_name} strategy for initial island population.",
            "complexity": "low",
            "key_parameters": {
                "assets": assets,
                "lookback_periods": lookbacks,
                "key_indicators": indicators,
                "rebalance_frequency": "weekly",
                "position_sizing": "equal_weight",
            },
            "pseudocode": pseudocode,
            "expected_characteristics": {
                "trading_frequency": "low_to_medium",
                "expected_turnover": "medium",
                "risk_profile": "medium",
            },
        }

    def _build_benchmark_seed(self, data_schema: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "strategy_id": "seed_benchmark_bnh",
            "category": "benchmark",
            "category_bit_index": None,
            "name": "Buy and Hold",
            "description": "Equal-weight buy-and-hold benchmark across available assets.",
            "complexity": "minimal",
            "key_parameters": {
                "assets": data_schema.get("assets", []),
                "rebalance": "never",
                "weights": "equal",
            },
            "pseudocode": (
                "1. On first bar, allocate equally across all assets\n"
                "2. Hold positions through the full backtest horizon"
            ),
            "expected_characteristics": {
                "trading_frequency": "minimal",
                "expected_turnover": "minimal",
                "risk_profile": "market",
            },
        }
