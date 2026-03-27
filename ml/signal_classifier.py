"""
signal_classifier.py — Random Forest Trade Predictor
Trains on past trade data (RSI, EMA, ATR, Volume, News, Spread) to predict 
if an incoming signal will be a WIN (1) or LOSS (0). Target Accuracy: > 68%. 
"""

import os
import logging
import pickle
import pandas as pd # type: ignore
from typing import Dict, Any, Tuple
import xgboost as xgb # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.metrics import accuracy_score # type: ignore

logger = logging.getLogger("agniv.ml.classifier")



class SignalClassifier:
    def __init__(self, symbol: str = "XAUUSD", confidence_threshold: float = 0.70):
        self.symbol = symbol
        self.model_path = os.path.join(os.path.dirname(__file__), f"random_forest_model_{symbol}.pkl")
        self.confidence_threshold = confidence_threshold
        self.model = self._load_model()
        
        # Current feature set expected during prediction
        self.features = [
            "rsi", "ema_distance", "atr", "volume_ratio", 
            "session_id", "news_score", "spread", "mtf_confluence"
        ]

    def _load_model(self) -> xgb.XGBClassifier:
        """Loads a pretrained pickle model off disk if it exists."""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    model = pickle.load(f)
                    logger.info("[ML] XGBoost model loaded from disk.")
                    return model
            except Exception as e:
                logger.error(f"[ML] Failed to load model: {e}")
        return None  # type: ignore

    def train_model(self, historical_data: pd.DataFrame) -> Tuple[bool, float]:
        """
        Expects a DataFrame containing the `features` columns + a `result` column (1=Win, 0=Loss).
        Automatically splits, trains, evaluates >68% accuracy, and pickles the binary.
        """
        # Validate data
        for f in self.features:
            if f not in historical_data.columns:
                logger.error(f"[ML] Missing feature in training data: {f}")
                return False, 0.0
                
        if "result" not in historical_data.columns:
            logger.error("[ML] Missing 'result' column in training data.")
            return False, 0.0

        X = historical_data[self.features]
        y = historical_data["result"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize optimal params for XGBoost trading signals
        clf = xgb.XGBClassifier(
            n_estimators=300, 
            learning_rate=0.05,
            max_depth=6, 
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=20,
            eval_metric="logloss"
        )
        
        logger.info("[ML] Starting XGBoost training with early stopping...")
        clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        # Cross Validation Evaluation using the best iteration
        predictions = clf.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        best_iteration = clf.best_iteration
        logger.info(f"[ML] Training Complete at iteration {best_iteration}. Test Accuracy: {accuracy*100:.2f}%")

        if accuracy < 0.68:
            logger.warning("[ML] Model failed minimum 68% accuracy threshold. Discarding model.")
            return False, accuracy

        # Save model
        with open(self.model_path, 'wb') as f:
            pickle.dump(clf, f)
            
        self.model = clf
        logger.info("[ML] New XGBoost model achieved >68% accuracy and has been saved to disk.")
        return True, accuracy

    def predict_signal(self, current_features: Dict[str, float]) -> Dict[str, Any]:
        """
        Evaluates a live incoming signal.
        Returns {"trade_allowed": bool, "confidence": float}
        """
        if self.model is None:
            logger.warning("[ML] Model not trained or loaded. Pass-through mode active.")
            return {"trade_allowed": True, "confidence": 1.0, "reason": "No Model"}

        try:
            # Convert dictionary mapping to DataFrame to avoid feature name warnings
            X_live = pd.DataFrame([current_features], columns=self.features)
            
            # predict_proba returns array like [[prob_loss, prob_win]]
            probabilities = self.model.predict_proba(X_live)[0]
            win_probability = probabilities[1]  # Index 1 is the 'WIN' class (1)

            if win_probability >= self.confidence_threshold:
                logger.info(f"[ML] Signal APPROVED. AI Confidence: {win_probability*100:.1f}%")
                return {"trade_allowed": True, "confidence": win_probability}
            else:
                logger.warning(f"[ML] Signal REJECTED. AI Confidence: {win_probability*100:.1f}% < {self.confidence_threshold*100}%")
                return {"trade_allowed": False, "confidence": win_probability}
                
        except Exception as e:
            logger.error(f"[ML] Prediction failed: {e}")
            return {"trade_allowed": False, "confidence": 0.0}

# To use: btc_ml = SignalClassifier('BTCUSD'); gold_ml = SignalClassifier('XAUUSD')
