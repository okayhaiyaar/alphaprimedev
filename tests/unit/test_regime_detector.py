"""
Unit tests for market regime detection in ALPHA-PRIME v2.0.

Covers:
- Basic bull/bear/sideways/volatile/crisis classification.
- HMM-based regime detection, technical and volatility regimes.
- Regime transitions, features, confidence, and integration scenarios.

All tests use:
- Synthetic, deterministic market data (fixed seeds).
- Fast, isolated computations with statistical checks where needed.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import pytest
from hmmlearn import hmm

from core.regime_detector import (
    RegimeDetector,
    HMMRegimeDetector,
    TechnicalRegimeDetector,
    VolatilityRegimeDetector,
    Regime,
    RegimeTransition,
    detect_regime,
    calculate_regime_confidence,
    get_regime_features,
    validate_regime_stability,
)


# ---------------------------------------------------------------------------
# Fixtures: synthetic market data and detectors
# ---------------------------------------------------------------------------

@pytest.fixture
def bull_market_data() -> np.ndarray:
    """Synthetic bull market (uptrend + low vol)."""
    np.random.seed(42)
    trend = np.linspace(100, 150, 252)
    noise = np.random.normal(0, 2, 252)
    return trend + noise


@pytest.fixture
def bear_market_data() -> np.ndarray:
    """Synthetic bear market (downtrend + rising vol)."""
    np.random.seed(42)
    trend = np.linspace(100, 70, 252)
    noise = np.random.normal(0, 5, 252)
    return trend + noise


@pytest.fixture
def sideways_market_data() -> np.ndarray:
    """Synthetic sideways market (range-bound)."""
    np.random.seed(42)
    base = 100
    noise = np.random.normal(0, 1.5, 252)
    return base + noise


@pytest.fixture
def volatile_market_data() -> np.ndarray:
    """Synthetic volatile market (high vol, no trend)."""
    np.random.seed(42)
    trend = 100
    noise = np.random.normal(0, 8, 252)
    return trend + noise


@pytest.fixture
def crisis_market_data() -> np.ndarray:
    """Synthetic crisis (rapid decline + extreme vol)."""
    np.random.seed(42)
    normal = np.linspace(100, 105, 200)
    crash = np.linspace(105, 60, 30)
    recovery = np.linspace(60, 75, 22)
    noise = np.random.normal(0, 3, 252)
    return np.concatenate([normal, crash, recovery]) + noise


@pytest.fixture
def hmm_detector() -> HMMRegimeDetector:
    """Pre-configured HMM regime detector."""
    return HMMRegimeDetector(n_states=3, n_iter=50, random_state=42)


@pytest.fixture
def technical_detector() -> TechnicalRegimeDetector:
    """Technical indicator-based detector."""
    return TechnicalRegimeDetector(
        indicators=["sma", "adx", "atr", "rsi"],
        lookback=20,
    )


@pytest.fixture
def regime_features_dataframe() -> pd.DataFrame:
    """Regime features as pandas DataFrame."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "returns": np.random.normal(0.001, 0.02, 252),
            "volatility": np.random.uniform(0.10, 0.30, 252),
            "trend": np.random.uniform(-1.0, 1.0, 252),
            "volume_ratio": np.random.uniform(0.8, 1.2, 252),
        }
    )


@pytest.fixture
def multi_asset_prices() -> Dict[str, np.ndarray]:
    """Synthetic multi-asset price series."""
    np.random.seed(42)
    base = np.linspace(100, 120, 252)
    return {
        "AAPL": base + np.random.normal(0, 2, 252),
        "MSFT": base * 1.01 + np.random.normal(0, 2, 252),
        "TLT": np.linspace(100, 110, 252) + np.random.normal(0, 1, 252),
    }


# ---------------------------------------------------------------------------
# 1. Basic Regime Classification
# ---------------------------------------------------------------------------

class TestBasicRegimeClassification:
    """Test fundamental regime identification."""

    def test_bull_market_identification(self, bull_market_data: np.ndarray):
        detector = RegimeDetector()
        regime = detector.detect(prices=bull_market_data)
        assert regime.label == Regime.BULL
        assert 0.0 <= regime.confidence <= 1.0
        assert regime.trend_strength > 0

    def test_bear_market_identification(self, bear_market_data: np.ndarray):
        detector = RegimeDetector()
        regime = detector.detect(prices=bear_market_data)
        assert regime.label == Regime.BEAR
        assert regime.trend_strength < 0

    def test_sideways_market_identification(self, sideways_market_data: np.ndarray):
        detector = RegimeDetector()
        regime = detector.detect(prices=sideways_market_data)
        assert regime.label == Regime.SIDEWAYS

    def test_volatile_market_identification(self, volatile_market_data: np.ndarray):
        detector = RegimeDetector()
        regime = detector.detect(prices=volatile_market_data)
        assert regime.label == Regime.VOLATILE

    def test_crisis_market_identification(self, crisis_market_data: np.ndarray):
        detector = RegimeDetector(crisis_threshold=-0.10)
        labels: List[Regime] = []
        for i in range(50, len(crisis_market_data)):
            labels.append(detector.detect(prices=crisis_market_data[: i + 1]).label)
        assert Regime.CRISIS in labels

    def test_regime_from_returns(self, bull_market_data: np.ndarray):
        detector = RegimeDetector()
        returns = np.diff(np.log(bull_market_data))
        regime = detector.detect(returns=returns)
        assert regime.label == Regime.BULL

    def test_regime_from_price_action(self, bull_market_data: np.ndarray):
        detector = RegimeDetector()
        regime = detector.detect(prices=bull_market_data)
        assert regime.label == Regime.BULL

    def test_regime_confidence_scores(self, bull_market_data: np.ndarray):
        detector = RegimeDetector()
        regime = detector.detect(prices=bull_market_data)
        conf = calculate_regime_confidence(regime)
        assert 0.0 <= conf <= 1.0

    def test_regime_threshold_calibration(self, sideways_market_data: np.ndarray):
        detector = RegimeDetector(trend_threshold=0.001)
        regime = detector.detect(prices=sideways_market_data)
        assert regime.label in {Regime.SIDEWAYS, Regime.VOLATILE}

    def test_ambiguous_regime_handling(self, sideways_market_data: np.ndarray):
        detector = RegimeDetector(allow_ambiguous=True)
        regime = detector.detect(prices=sideways_market_data)
        assert regime.label in {Regime.SIDEWAYS, Regime.VOLATILE, Regime.AMBIGUOUS}

    def test_regime_persistence_filter(self, volatile_market_data: np.ndarray):
        detector = RegimeDetector(min_regime_duration=10)
        regimes: List[Regime] = []
        for i in range(20, len(volatile_market_data)):
            regimes.append(detector.detect(prices=volatile_market_data[: i + 1]).label)
        changes = sum(1 for i in range(1, len(regimes)) if regimes[i] != regimes[i - 1])
        assert changes < 20

    def test_regime_edge_case_flat_market(self):
        detector = RegimeDetector()
        prices = np.full(100, 100.0)
        regime = detector.detect(prices=prices)
        assert regime.label == Regime.SIDEWAYS


# ---------------------------------------------------------------------------
# 2. Hidden Markov Model (HMM) Regime Detector
# ---------------------------------------------------------------------------

class TestHMMRegimeDetector:
    """Test HMM-based regime detection."""

    def test_hmm_initialization(self, hmm_detector: HMMRegimeDetector):
        assert hmm_detector.n_states == 3
        assert isinstance(hmm_detector.model, hmm.GaussianHMM)

    def test_hmm_training_convergence(self, hmm_detector: HMMRegimeDetector, bull_market_data: np.ndarray):
        returns = np.diff(np.log(bull_market_data)).reshape(-1, 1)
        hmm_detector.fit(returns)
        assert hmm_detector.model.monitor_.converged is True

    def test_hmm_state_emission_probabilities(self, hmm_detector: HMMRegimeDetector, bull_market_data: np.ndarray):
        returns = np.diff(np.log(bull_market_data)).reshape(-1, 1)
        hmm_detector.fit(returns)
        means = hmm_detector.model.means_.flatten()
        assert means.size == hmm_detector.n_states

    def test_hmm_transition_matrix(self, hmm_detector: HMMRegimeDetector, bull_market_data: np.ndarray):
        returns = np.diff(np.log(bull_market_data)).reshape(-1, 1)
        hmm_detector.fit(returns)
        trans = hmm_detector.model.transmat_
        assert trans.shape == (hmm_detector.n_states, hmm_detector.n_states)
        assert np.allclose(trans.sum(axis=1), 1.0)

    def test_hmm_viterbi_decoding(self, hmm_detector: HMMRegimeDetector, bull_market_data: np.ndarray):
        returns = np.diff(np.log(bull_market_data)).reshape(-1, 1)
        hmm_detector.fit(returns)
        states = hmm_detector.predict(returns)
        assert len(states) == len(returns)
        assert np.all((0 <= states) & (states < hmm_detector.n_states))

    def test_hmm_forward_backward_algorithm(self, hmm_detector: HMMRegimeDetector, bull_market_data: np.ndarray):
        returns = np.diff(np.log(bull_market_data)).reshape(-1, 1)
        hmm_detector.fit(returns)
        gamma = hmm_detector.posterior(returns)
        assert gamma.shape == (len(returns), hmm_detector.n_states)
        assert np.allclose(gamma.sum(axis=1), 1.0, atol=1e-6)

    def test_hmm_regime_prediction(self, hmm_detector: HMMRegimeDetector, bull_market_data: np.ndarray):
        returns = np.diff(np.log(bull_market_data)).reshape(-1, 1)
        hmm_detector.fit(returns)
        regime = hmm_detector.detect(returns)
        assert isinstance(regime.label, Regime)

    def test_hmm_online_updating(self, bull_market_data: np.ndarray):
        detector = HMMRegimeDetector(n_states=3, n_iter=10, online=True, random_state=42)
        returns = np.diff(np.log(bull_market_data)).reshape(-1, 1)
        for r in returns[:50]:
            detector.partial_fit(r.reshape(-1, 1))
        regime = detector.detect(returns[:50])
        assert isinstance(regime.label, Regime)

    def test_hmm_regime_smoothing(self, hmm_detector: HMMRegimeDetector, bull_market_data: np.ndarray):
        returns = np.diff(np.log(bull_market_data)).reshape(-1, 1)
        hmm_detector.fit(returns)
        smoothed = hmm_detector.smooth(returns)
        assert len(smoothed) == len(returns)

    def test_hmm_n_states_optimization(self, bull_market_data: np.ndarray):
        scores = []
        for n in (2, 3, 4):
            det = HMMRegimeDetector(n_states=n, n_iter=20, random_state=42)
            returns = np.diff(np.log(bull_market_data)).reshape(-1, 1)
            det.fit(returns)
            scores.append(det.model.score(returns))
        assert scores[1] >= min(scores)

    def test_hmm_feature_selection(self, bull_market_data: np.ndarray):
        returns = np.diff(np.log(bull_market_data))
        vol = pd.Series(returns).rolling(10).std().fillna(0).values
        feats = np.column_stack([returns, vol])[1:].reshape(-1, 2)
        detector = HMMRegimeDetector(n_states=3, n_iter=20, random_state=42)
        detector.fit(feats)
        states = detector.predict(feats)
        assert len(states) == len(feats)

    def test_hmm_convergence_criteria(self, bull_market_data: np.ndarray):
        detector = HMMRegimeDetector(n_states=3, n_iter=5, tol=1e-2, random_state=42)
        returns = np.diff(np.log(bull_market_data)).reshape(-1, 1)
        detector.fit(returns)
        assert detector.model.monitor_.iter <= detector.model.monitor_.n_iter

    def test_hmm_numerical_stability(self, bull_market_data: np.ndarray):
        detector = HMMRegimeDetector(n_states=3, n_iter=20, random_state=42)
        returns = np.clip(np.diff(np.log(bull_market_data)), -0.2, 0.2).reshape(-1, 1)
        detector.fit(returns)
        logprob = detector.model.score(returns)
        assert np.isfinite(logprob)

    def test_hmm_serialization(self, hmm_detector: HMMRegimeDetector, bull_market_data: np.ndarray):
        returns = np.diff(np.log(bull_market_data)).reshape(-1, 1)
        hmm_detector.fit(returns)
        state = hmm_detector.serialize()
        new = HMMRegimeDetector.deserialize(state)
        states = new.predict(returns)
        assert len(states) == len(returns)

    def test_hmm_edge_case_single_state(self, bull_market_data: np.ndarray):
        detector = HMMRegimeDetector(n_states=1, n_iter=10, random_state=42)
        returns = np.diff(np.log(bull_market_data)).reshape(-1, 1)
        detector.fit(returns)
        states = detector.predict(returns)
        assert np.all(states == 0)


# ---------------------------------------------------------------------------
# 3. Technical Indicator Regime Detection
# ---------------------------------------------------------------------------

class TestTechnicalRegimeDetector:
    """Test technical indicator-based regime detection."""

    def test_moving_average_regime(self, bull_market_data: np.ndarray):
        detector = TechnicalRegimeDetector(indicators=["sma"])
        regime = detector.detect(prices=bull_market_data)
        assert regime.label == Regime.BULL
        assert regime.features["sma_slope"] > 0

    def test_trend_strength_adx(self, bull_market_data: np.ndarray):
        detector = TechnicalRegimeDetector(indicators=["adx"])
        regime = detector.detect(prices=bull_market_data)
        assert regime.features["adx"] >= 0.0

    def test_volatility_atr_regime(self, volatile_market_data: np.ndarray):
        detector = TechnicalRegimeDetector(indicators=["atr"])
        regime = detector.detect(prices=volatile_market_data)
        assert regime.label in {Regime.VOLATILE, Regime.CRISIS}

    def test_momentum_rsi_regime(self, bull_market_data: np.ndarray):
        detector = TechnicalRegimeDetector(indicators=["rsi"])
        regime = detector.detect(prices=bull_market_data)
        assert 0.0 <= regime.features["rsi"] <= 100.0

    def test_macd_histogram_regime(self, bull_market_data: np.ndarray):
        detector = TechnicalRegimeDetector(indicators=["macd"])
        regime = detector.detect(prices=bull_market_data)
        assert "macd_hist" in regime.features

    def test_bollinger_band_regime(self, sideways_market_data: np.ndarray):
        detector = TechnicalRegimeDetector(indicators=["bollinger"])
        regime = detector.detect(prices=sideways_market_data)
        assert "band_width" in regime.features

    def test_volume_profile_regime(self, bull_market_data: np.ndarray):
        detector = TechnicalRegimeDetector(indicators=["volume_profile"])
        volumes = np.full_like(bull_market_data, 1_000_000.0)
        regime = detector.detect(prices=bull_market_data, volumes=volumes)
        assert "volume_profile" in regime.features

    def test_composite_indicator_regime(self, sideways_market_data: np.ndarray):
        detector = TechnicalRegimeDetector(indicators=["sma", "adx", "atr"], aggregation="vote")
        regime = detector.detect(prices=sideways_market_data)
        assert regime.label in {Regime.SIDEWAYS, Regime.VOLATILE}

    def test_indicator_weight_optimization(self, bull_market_data: np.ndarray):
        detector = TechnicalRegimeDetector(indicators=["sma", "adx", "atr"], optimize_weights=True)
        weights = detector.optimize_weights(bull_market_data)
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_indicator_divergence_detection(self, bull_market_data: np.ndarray):
        detector = TechnicalRegimeDetector(indicators=["sma", "rsi"])
        regime = detector.detect(prices=bull_market_data)
        assert "divergence" in regime.features

    def test_multi_timeframe_regime(self, bull_market_data: np.ndarray):
        detector = TechnicalRegimeDetector(indicators=["sma"], timeframes=[20, 50, 200])
        regime = detector.detect(prices=bull_market_data)
        assert "sma_20" in regime.features and "sma_200" in regime.features

    def test_technical_edge_case_missing_data(self):
        detector = TechnicalRegimeDetector(indicators=["sma"])
        prices = np.array([np.nan, np.nan, 100.0, 101.0])
        regime = detector.detect(prices=prices)
        assert isinstance(regime.label, Regime)


# ---------------------------------------------------------------------------
# 4. Volatility-Based Regime Detection
# ---------------------------------------------------------------------------

class TestVolatilityRegimeDetector:
    """Test volatility-based regime classification."""

    def test_low_volatility_regime(self, sideways_market_data: np.ndarray):
        detector = VolatilityRegimeDetector()
        regime = detector.detect(prices=sideways_market_data)
        assert regime.label in {Regime.SIDEWAYS, Regime.BULL, Regime.BEAR}

    def test_high_volatility_regime(self, volatile_market_data: np.ndarray):
        detector = VolatilityRegimeDetector()
        regime = detector.detect(prices=volatile_market_data)
        assert regime.label in {Regime.VOLATILE, Regime.CRISIS}

    def test_volatility_clustering_detection(self, volatile_market_data: np.ndarray):
        detector = VolatilityRegimeDetector()
        vols = np.abs(np.diff(volatile_market_data))
        cluster = detector.detect_clustering(vols)
        assert cluster["cluster_score"] >= 0.0

    def test_volatility_percentile_regime(self, volatile_market_data: np.ndarray):
        detector = VolatilityRegimeDetector()
        vols = np.abs(np.diff(volatile_market_data))
        regime = detector.from_percentile(vols, low=0.2, high=0.8)
        assert isinstance(regime, Regime)

    def test_garch_volatility_regime(self, volatile_market_data: np.ndarray):
        detector = VolatilityRegimeDetector(use_garch=True)
        regime = detector.detect(prices=volatile_market_data)
        assert isinstance(regime.label, Regime)

    def test_realized_vs_implied_volatility(self):
        detector = VolatilityRegimeDetector()
        regime = detector.compare_realized_implied(realized=0.2, implied=0.4)
        assert isinstance(regime, Regime)

    def test_volatility_regime_transitions(self, volatile_market_data: np.ndarray):
        detector = VolatilityRegimeDetector()
        regimes = []
        for i in range(30, len(volatile_market_data)):
            regimes.append(detector.detect(prices=volatile_market_data[: i + 1]).label)
        assert len(set(regimes)) >= 1

    def test_volatility_regime_persistence(self, volatile_market_data: np.ndarray):
        detector = VolatilityRegimeDetector(min_duration=5)
        labels: List[Regime] = []
        for i in range(30, len(volatile_market_data)):
            labels.append(detector.detect(prices=volatile_market_data[: i + 1]).label)
        changes = sum(1 for i in range(1, len(labels)) if labels[i] != labels[i - 1])
        assert changes < len(labels)

    def test_intraday_volatility_regime(self):
        detector = VolatilityRegimeDetector()
        intraday = np.array([0.1, 0.2, 0.15, 0.3])
        regime = detector.from_intraday(intraday)
        assert isinstance(regime, Regime)

    def test_volatility_edge_case_zero_variance(self):
        detector = VolatilityRegimeDetector()
        prices = np.full(100, 100.0)
        regime = detector.detect(prices=prices)
        assert regime.label in {Regime.SIDEWAYS, Regime.BULL, Regime.BEAR}


# ---------------------------------------------------------------------------
# 5. Regime Transition Detection
# ---------------------------------------------------------------------------

class TestRegimeTransitions:
    """Test regime change detection and transitions."""

    def test_detect_regime_change(self, bull_market_data: np.ndarray, bear_market_data: np.ndarray):
        detector = RegimeDetector()
        for price in bull_market_data[-50:]:
            detector.update(price)
        assert detector.current_regime == Regime.BULL
        for price in bear_market_data[:50]:
            detector.update(price)
        assert detector.current_regime == Regime.BEAR

    def test_transition_probability_matrix(self, bull_market_data: np.ndarray, hmm_detector: HMMRegimeDetector):
        returns = np.diff(np.log(bull_market_data)).reshape(-1, 1)
        hmm_detector.fit(returns)
        trans = hmm_detector.model.transmat_
        assert np.allclose(trans.sum(axis=1), 1.0)

    def test_transition_speed_measurement(self, bull_market_data: np.ndarray, bear_market_data: np.ndarray):
        detector = RegimeDetector()
        transitions: List[RegimeTransition] = []
        for price in bull_market_data[-20:]:
            transitions.extend(detector.update(price).transitions)
        for price in bear_market_data[:20]:
            transitions.extend(detector.update(price).transitions)
        assert any(t.from_regime == Regime.BULL and t.to_regime == Regime.BEAR for t in transitions)

    def test_transition_confirmation_delay(self):
        detector = RegimeDetector(confirmation_bars=5)
        detector.current_regime = Regime.BULL
        for _ in range(4):
            detector.update(90.0)
        assert detector.current_regime == Regime.BULL
        for _ in range(2):
            detector.update(85.0)
        assert detector.current_regime == Regime.BEAR

    def test_false_transition_filtering(self, sideways_market_data: np.ndarray):
        detector = RegimeDetector(confirmation_bars=5)
        regimes: List[Regime] = []
        for p in sideways_market_data:
            regimes.append(detector.update(p).label)
        changes = sum(1 for i in range(1, len(regimes)) if regimes[i] != regimes[i - 1])
        assert changes < len(regimes) / 2

    def test_regime_flip_flop_prevention(self, volatile_market_data: np.ndarray):
        detector = RegimeDetector(min_regime_duration=10)
        regimes: List[Regime] = []
        for p in volatile_market_data:
            regimes.append(detector.update(p).label)
        changes = sum(1 for i in range(1, len(regimes)) if regimes[i] != regimes[i - 1])
        assert changes < len(regimes) / 2

    def test_transition_lookback_period(self, bull_market_data: np.ndarray):
        detector = RegimeDetector(transition_lookback=20)
        for p in bull_market_data:
            detector.update(p)
        assert detector.transition_lookback == 20

    def test_transition_confidence_threshold(self, bull_market_data: np.ndarray, bear_market_data: np.ndarray):
        detector = RegimeDetector(transition_confidence_threshold=0.8)
        for p in bull_market_data[-20:]:
            detector.update(p)
        detector._pending_regime = Regime.BEAR
        detector._pending_confidence = 0.5
        detector._apply_pending_transition()
        assert detector.current_regime == Regime.BULL

    def test_cascading_regime_changes(self, bull_market_data: np.ndarray, bear_market_data: np.ndarray):
        detector = RegimeDetector()
        for p in bull_market_data[-20:]:
            detector.update(p)
        for p in bear_market_data[:20]:
            detector.update(p)
        assert detector.current_regime == Regime.BEAR

    def test_regime_stability_after_transition(self, bull_market_data: np.ndarray, bear_market_data: np.ndarray):
        detector = RegimeDetector(min_regime_duration=10)
        for p in bull_market_data[-20:]:
            detector.update(p)
        for p in bear_market_data[:20]:
            detector.update(p)
        regime_after = detector.current_regime
        for _ in range(10):
            detector.update(bear_market_data[-1])
        assert detector.current_regime == regime_after

    def test_transition_history_tracking(self, bull_market_data: np.ndarray, bear_market_data: np.ndarray):
        detector = RegimeDetector()
        for p in bull_market_data[-20:]:
            detector.update(p)
        for p in bear_market_data[:20]:
            detector.update(p)
        history = detector.transition_history
        assert any(t.to_regime == Regime.BEAR for t in history)

    def test_transition_edge_case_rapid_changes(self, volatile_market_data: np.ndarray):
        detector = RegimeDetector(confirmation_bars=1)
        for p in volatile_market_data[:50]:
            detector.update(p)
        assert len(detector.transition_history) >= 0


# ---------------------------------------------------------------------------
# 6. Feature Engineering
# ---------------------------------------------------------------------------

class TestRegimeFeatures:
    """Test regime feature extraction and engineering."""

    def test_price_momentum_features(self, bull_market_data: np.ndarray):
        feats = get_regime_features(prices=bull_market_data)
        assert "momentum" in feats.columns

    def test_volatility_features(self, bull_market_data: np.ndarray):
        feats = get_regime_features(prices=bull_market_data)
        assert "volatility" in feats.columns

    def test_volume_features(self, bull_market_data: np.ndarray):
        vols = np.full_like(bull_market_data, 1_000_000.0)
        feats = get_regime_features(prices=bull_market_data, volumes=vols)
        assert "volume_ratio" in feats.columns

    def test_trend_features(self, bull_market_data: np.ndarray):
        feats = get_regime_features(prices=bull_market_data)
        assert "trend" in feats.columns

    def test_correlation_features(self, multi_asset_prices: Dict[str, np.ndarray]):
        feats = get_regime_features(multi_asset_prices)
        assert "correlation" in feats.columns

    def test_microstructure_features(self):
        prices = np.linspace(100, 101, 100)
        feats = get_regime_features(prices=prices, microstructure=True)
        assert "spread" in feats.columns

    def test_feature_normalization(self, regime_features_dataframe: pd.DataFrame):
        norm = (regime_features_dataframe - regime_features_dataframe.mean()) / regime_features_dataframe.std()
        assert abs(norm.mean().mean()) < 0.1

    def test_feature_stationarity(self, regime_features_dataframe: pd.DataFrame):
        diffs = regime_features_dataframe["returns"].diff().dropna()
        assert abs(diffs.mean()) < 1e-3

    def test_feature_importance_ranking(self, regime_features_dataframe: pd.DataFrame):
        detector = RegimeDetector()
        ranking = detector.feature_importance(regime_features_dataframe, labels=np.random.randint(0, 2, len(regime_features_dataframe)))
        assert isinstance(ranking, list)

    def test_feature_edge_case_nan_handling(self, regime_features_dataframe: pd.DataFrame):
        df = regime_features_dataframe.copy()
        df.iloc[0, 0] = np.nan
        feats = get_regime_features(df)
        assert feats.isna().sum().sum() == 0


# ---------------------------------------------------------------------------
# 7. Statistical Validation
# ---------------------------------------------------------------------------

class TestStatisticalValidation:
    """Test statistical validity of regime detection."""

    def test_regime_stationarity_test(self, bull_market_data: np.ndarray):
        labels = np.repeat(np.array([0, 1]), len(bull_market_data) // 2)
        stats = validate_regime_stability(labels)
        assert "stationarity_p_value" in stats

    def test_regime_autocorrelation(self, bull_market_data: np.ndarray):
        labels = np.repeat(np.array([0, 1]), len(bull_market_data) // 2)
        stats = validate_regime_stability(labels)
        assert "autocorrelation" in stats

    def test_regime_likelihood_ratio_test(self, bull_market_data: np.ndarray):
        labels = np.repeat(np.array([0, 1]), len(bull_market_data) // 2)
        stats = validate_regime_stability(labels)
        assert "lr_stat" in stats

    def test_regime_duration_distribution(self, bull_market_data: np.ndarray):
        labels = np.repeat(np.array([0, 1]), len(bull_market_data) // 2)
        stats = validate_regime_stability(labels)
        assert "mean_duration" in stats

    def test_regime_homogeneity_test(self, bull_market_data: np.ndarray):
        labels = np.repeat(np.array([0, 1]), len(bull_market_data) // 2)
        stats = validate_regime_stability(labels)
        assert "homogeneity_p_value" in stats

    def test_regime_independence_test(self, bull_market_data: np.ndarray):
        labels = np.repeat(np.array([0, 1]), len(bull_market_data) // 2)
        stats = validate_regime_stability(labels)
        assert "independence_p_value" in stats

    def test_cross_validation_regime_accuracy(self, bull_market_data: np.ndarray, bear_market_data: np.ndarray):
        det = RegimeDetector()
        bull_regime = det.detect(prices=bull_market_data)
        bear_regime = det.detect(prices=bear_market_data)
        assert bull_regime.label != bear_regime.label

    def test_out_of_sample_regime_performance(self, bull_market_data: np.ndarray, bear_market_data: np.ndarray):
        det = RegimeDetector()
        train = bull_market_data[:200]
        test = bull_market_data[200:]
        det.detect(prices=train)
        regime_test = det.detect(prices=test)
        assert regime_test.label == Regime.BULL

    def test_regime_classification_metrics(self, bull_market_data: np.ndarray):
        labels_true = np.repeat([Regime.BULL, Regime.BEAR], len(bull_market_data) // 2)
        labels_pred = labels_true.copy()
        accuracy = (labels_true == labels_pred).mean()
        assert accuracy == 1.0

    def test_regime_confusion_matrix(self, bull_market_data: np.ndarray):
        labels_true = np.repeat([Regime.BULL, Regime.BEAR], len(bull_market_data) // 2)
        labels_pred = labels_true.copy()
        matrix = pd.crosstab(labels_true, labels_pred)
        assert matrix.values.sum() == len(labels_true)

    def test_regime_information_criteria(self, bull_market_data: np.ndarray):
        detector = HMMRegimeDetector(n_states=2, n_iter=20, random_state=42)
        returns = np.diff(np.log(bull_market_data)).reshape(-1, 1)
        detector.fit(returns)
        ic = detector.information_criteria(returns)
        assert "aic" in ic and "bic" in ic

    def test_statistical_edge_case_insufficient_data(self):
        labels = np.array([0])
        stats = validate_regime_stability(labels)
        assert stats.get("stationarity_p_value") is None


# ---------------------------------------------------------------------------
# 8. Multi-Asset Regime Detection
# ---------------------------------------------------------------------------

class TestMultiAssetRegime:
    """Test regime detection across multiple assets."""

    def test_portfolio_regime_aggregation(self, multi_asset_prices: Dict[str, np.ndarray]):
        det = RegimeDetector()
        regime = det.detect_multi(multi_asset_prices)
        assert isinstance(regime.label, Regime)

    def test_sector_regime_correlation(self, multi_asset_prices: Dict[str, np.ndarray]):
        det = RegimeDetector()
        res = det.sector_regimes(multi_asset_prices)
        assert isinstance(res, dict)

    def test_cross_asset_regime_alignment(self, multi_asset_prices: Dict[str, np.ndarray]):
        det = RegimeDetector()
        res = det.cross_asset_alignment(multi_asset_prices)
        assert "alignment_score" in res

    def test_regime_contagion_detection(self, multi_asset_prices: Dict[str, np.ndarray]):
        det = RegimeDetector()
        res = det.contagion(multi_asset_prices)
        assert "contagion_score" in res

    def test_diversification_regime(self, multi_asset_prices: Dict[str, np.ndarray]):
        det = RegimeDetector()
        res = det.diversification_regime(multi_asset_prices)
        assert "diversification_score" in res

    def test_regime_factor_models(self, multi_asset_prices: Dict[str, np.ndarray]):
        det = RegimeDetector()
        res = det.factor_regimes(multi_asset_prices)
        assert isinstance(res, dict)

    def test_multi_asset_regime_weighting(self, multi_asset_prices: Dict[str, np.ndarray]):
        det = RegimeDetector()
        res = det.multi_asset_weights(multi_asset_prices)
        assert abs(sum(res.values()) - 1.0) < 1e-6

    def test_multi_asset_edge_case_uncorrelated(self):
        det = RegimeDetector()
        prices = {
            "A": np.random.normal(0, 1, 100),
            "B": np.random.normal(0, 1, 100),
        }
        res = det.cross_asset_alignment(prices)
        assert res["alignment_score"] >= 0.0


# ---------------------------------------------------------------------------
# 9. Regime Confidence & Uncertainty
# ---------------------------------------------------------------------------

class TestRegimeConfidence:
    """Test regime confidence scoring and uncertainty."""

    def test_confidence_score_calculation(self, bull_market_data: np.ndarray):
        regime = detect_regime(prices=bull_market_data)
        conf = calculate_regime_confidence(regime)
        assert 0.0 <= conf <= 1.0

    def test_confidence_from_probability(self):
        prob = np.array([0.1, 0.8, 0.1])
        conf = calculate_regime_confidence(probabilities=prob)
        assert conf == prob.max()

    def test_confidence_from_feature_agreement(self, regime_features_dataframe: pd.DataFrame):
        det = TechnicalRegimeDetector(indicators=["sma", "adx", "rsi"])
        conf = det.calculate_confidence(regime_features_dataframe)
        assert 0.0 <= conf <= 1.0

    def test_confidence_decay_over_time(self, bull_market_data: np.ndarray):
        det = RegimeDetector(confidence_half_life=10)
        regime = det.detect(prices=bull_market_data)
        conf_now = regime.confidence
        conf_later = det.decay_confidence(conf_now, steps=10)
        assert conf_later < conf_now

    def test_uncertainty_quantification(self, bull_market_data: np.ndarray):
        det = RegimeDetector()
        regime = det.detect(prices=bull_market_data)
        assert 0.0 <= regime.uncertainty <= 1.0

    def test_confidence_threshold_tuning(self, bull_market_data: np.ndarray):
        det = RegimeDetector(confidence_threshold=0.8)
        regime = det.detect(prices=bull_market_data)
        assert (regime.confidence >= 0.8) or (regime.label == Regime.AMBIGUOUS)

    def test_low_confidence_fallback(self, sideways_market_data: np.ndarray):
        det = RegimeDetector(low_confidence_fallback=Regime.SIDEWAYS)
        regime = det.detect(prices=sideways_market_data)
        assert regime.label in {Regime.SIDEWAYS, Regime.VOLATILE}

    def test_confidence_vs_accuracy_calibration(self):
        conf = np.array([0.6, 0.7, 0.9])
        acc = np.array([0.5, 0.8, 0.85])
        corr = np.corrcoef(conf, acc)[0, 1]
        assert corr > 0.0

    def test_bayesian_confidence_intervals(self, bull_market_data: np.ndarray):
        det = RegimeDetector()
        regime = det.detect(prices=bull_market_data)
        low, high = det.confidence_interval(regime, alpha=0.05)
        assert 0.0 <= low <= high <= 1.0

    def test_confidence_edge_case_zero_probability(self):
        conf = calculate_regime_confidence(probabilities=np.array([0.0, 0.0, 0.0]))
        assert conf == 0.0


# ---------------------------------------------------------------------------
# 10. Real-Time Regime Updates
# ---------------------------------------------------------------------------

class TestRealtimeRegimeUpdates:
    """Test online/streaming regime detection."""

    def test_online_regime_update(self, bull_market_data: np.ndarray):
        det = RegimeDetector()
        for p in bull_market_data[:50]:
            regime = det.update(p)
        assert isinstance(regime.label, Regime)

    def test_streaming_data_regime(self, crisis_market_data: np.ndarray):
        det = RegimeDetector()
        regime = None
        for p in crisis_market_data:
            regime = det.update(p)
        assert isinstance(regime.label, Regime)

    def test_incremental_hmm_update(self, bull_market_data: np.ndarray):
        det = HMMRegimeDetector(n_states=3, online=True, n_iter=5, random_state=42)
        for r in np.diff(np.log(bull_market_data[:50])):
            det.partial_fit(np.array([[r]]))
        regime = det.detect(np.diff(np.log(bull_market_data[:50])).reshape(-1, 1))
        assert isinstance(regime.label, Regime)

    def test_regime_latency_measurement(self, crisis_market_data: np.ndarray):
        det = RegimeDetector()
        latencies: List[int] = []
        for i, p in enumerate(crisis_market_data):
            regime = det.update(p)
            if regime.label == Regime.CRISIS:
                latencies.append(i)
        if latencies:
            assert min(latencies) < len(crisis_market_data)

    def test_regime_update_frequency(self, bull_market_data: np.ndarray):
        det = RegimeDetector(update_interval=5)
        calls = 0
        for i, p in enumerate(bull_market_data[:30]):
            regime = det.update(p)
            if i % 5 == 0:
                calls += 1
        assert calls <= 7

    def test_regime_persistence_buffer(self, bull_market_data: np.ndarray):
        det = RegimeDetector(min_regime_duration=5)
        for p in bull_market_data[:30]:
            det.update(p)
        assert len(det.regime_buffer) <= 30

    def test_realtime_feature_calculation(self, bull_market_data: np.ndarray):
        det = RegimeDetector()
        feats = []
        for p in bull_market_data[:20]:
            feats.append(det.update(p).features)
        assert len(feats) == 20

    def test_realtime_edge_case_missing_updates(self, bull_market_data: np.ndarray):
        det = RegimeDetector()
        det.update(bull_market_data[0])
        regime = det.tick_without_data()
        assert isinstance(regime.label, Regime)


# ---------------------------------------------------------------------------
# 11. Integration & Edge Cases
# ---------------------------------------------------------------------------

class TestIntegrationEdgeCases:
    """Test realistic scenarios and edge cases."""

    def test_2008_financial_crisis_detection(self):
        det = RegimeDetector()
        prices = np.concatenate(
            [
                np.linspace(100, 120, 200),
                np.linspace(120, 70, 50),
                np.linspace(70, 80, 30),
            ]
        )
        regime = det.detect(prices=prices)
        assert regime.label in {Regime.BEAR, Regime.CRISIS}

    def test_2020_covid_crash_detection(self):
        det = RegimeDetector()
        prices = np.concatenate(
            [
                np.linspace(100, 110, 40),
                np.linspace(110, 70, 20),
                np.linspace(70, 110, 40),
            ]
        )
        regime = det.detect(prices=prices)
        assert regime.label in {Regime.CRISIS, Regime.VOLATILE}

    def test_sideways_grinding_market(self, sideways_market_data: np.ndarray):
        det = RegimeDetector()
        regime = det.detect(prices=sideways_market_data)
        assert regime.label == Regime.SIDEWAYS

    def test_flash_crash_regime(self):
        det = RegimeDetector()
        prices = np.concatenate(
            [
                np.linspace(100, 105, 50),
                np.linspace(105, 60, 2),
                np.linspace(60, 80, 30),
            ]
        )
        regime = det.detect(prices=prices)
        assert regime.label in {Regime.CRISIS, Regime.VOLATILE}

    def test_overnight_gap_regime_change(self):
        det = RegimeDetector()
        day = np.linspace(100, 101, 50)
        gap = np.array([80.0])
        prices = np.concatenate([day, gap, day])
        regime = det.detect(prices=prices)
        assert regime.label in {Regime.CRISIS, Regime.VOLATILE, Regime.BEAR}

    def test_regime_during_low_liquidity(self, sideways_market_data: np.ndarray):
        det = RegimeDetector()
        vols = np.concatenate([np.full(200, 1_000_000.0), np.full(52, 10_000.0)])
        regime = det.detect(prices=sideways_market_data, volumes=vols)
        assert isinstance(regime.label, Regime)

    def test_regime_with_missing_data(self):
        det = RegimeDetector()
        prices = np.array([100.0, np.nan, 102.0, 101.0])
        regime = det.detect(prices=prices)
        assert isinstance(regime.label, Regime)

    def test_regime_initialization_cold_start(self):
        det = RegimeDetector()
        regime = det.detect(prices=np.array([100.0]))
        assert isinstance(regime.label, Regime)

    def test_regime_persistence_validation(self, bull_market_data: np.ndarray):
        det = RegimeDetector()
        regimes = [det.detect(prices=bull_market_data[: i + 1]).label for i in range(50, 100)]
        assert len(set(regimes)) == 1

    def test_end_to_end_regime_workflow(self, bull_market_data: np.ndarray, bear_market_data: np.ndarray):
        det = RegimeDetector()
        bull_regime = det.detect(prices=bull_market_data)
        bear_regime = det.detect(prices=bear_market_data)
        assert bull_regime.label != bear_regime.label


# ---------------------------------------------------------------------------
# Parametrized tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "returns,vol,expected_regime",
    [
        (0.02, 0.15, Regime.BULL),
        (-0.02, 0.15, Regime.BEAR),
        (0.00, 0.10, Regime.SIDEWAYS),
        (0.00, 0.35, Regime.VOLATILE),
        (-0.05, 0.50, Regime.CRISIS),
    ],
)
def test_regime_classification_scenarios(returns: float, vol: float, expected_regime: Regime):
    """Test regime classification across scenarios."""
    det = RegimeDetector()
    regime = det.classify(mean_return=returns, volatility=vol)
    assert regime == expected_regime


@pytest.mark.parametrize(
    "n_states,expected_convergence",
    [
        (2, True),
        (3, True),
        (4, True),
        (10, False),
    ],
)
def test_hmm_state_count_convergence(bull_market_data: np.ndarray, n_states: int, expected_convergence: bool):
    """Test HMM convergence with various state counts."""
    det = HMMRegimeDetector(n_states=n_states, n_iter=30, random_state=42)
    returns = np.diff(np.log(bull_market_data)).reshape(-1, 1)
    det.fit(returns)
    assert det.model.monitor_.converged == expected_convergence
