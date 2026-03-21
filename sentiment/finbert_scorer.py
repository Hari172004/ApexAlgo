import logging
from typing import List, Dict, Any

try:
    from transformers import pipeline
    FINBERT_AVAILABLE = True
except ImportError:
    FINBERT_AVAILABLE = False

logger = logging.getLogger("apexalgo.sentiment.finbert")

class FinBERTScorer:
    def __init__(self):
        self.model_name = "ProsusAI/finbert"
        self.nlp = None
        
        if FINBERT_AVAILABLE:
            try:
                logger.info("[SENTIMENT] Halting thread to load FinBERT model...")
                self.nlp = pipeline("text-classification", model=self.model_name)
                logger.info("[SENTIMENT] FinBERT Model successfully initialized.")
            except Exception as e:
                logger.error(f"[SENTIMENT] Could not initialize FinBERT pipeline: {e}")
        else:
            logger.warning("[SENTIMENT] 'transformers' library not installed. FinBERT disabled.")

    def _map_raw_to_spectrum(self, label: str, score: float) -> str:
        """
        FinBERT returns: 'positive', 'neutral', 'negative'.
        We remap this based on high confidence into 5 tiers.
        """
        if label == "positive":
            return "strongly bullish" if score >= 0.85 else "bullish"
        elif label == "negative":
            return "strongly bearish" if score >= 0.85 else "bearish"
        else:
            return "neutral"

    def analyze_headline(self, headline: str) -> Dict[str, Any]:
        """Runs a single headline through the model."""
        nlp_func = self.nlp
        if nlp_func is None:
            return {"sentiment": "neutral", "raw_score": 0.0, "reason": "Model Offline"}

        try:
            results = nlp_func(headline)
            if not results:
                return {"sentiment": "neutral", "raw_score": 0.0}
                
            best = results[0]
            label = str(best.get('label', '')).lower()
            confidence = float(best.get('score', 0.0))
            
            tiered_sentiment = self._map_raw_to_spectrum(label, confidence)
            
            logger.debug(f"[SENTIMENT] Analyzed headline -> {tiered_sentiment.upper()} ({confidence*100:.1f}%)")
            return {"sentiment": tiered_sentiment, "raw_score": confidence, "label": label}
            
        except Exception as e:
            logger.error(f"[SENTIMENT] Headline evaluation failed: {e}")
            return {"sentiment": "neutral", "raw_score": 0.0}

    def aggregate_news_score(self, headlines: List[str]) -> Dict[str, Any]:
        """
        Scans a batch of current news.
        Returns the overall dominant sentiment and applies position sizing modifiers.
        """
        if not headlines:
            return {"dominant": "neutral", "modifier": 1.0, "block_longs": False, "block_shorts": False}

        counts = {
            "strongly bullish": 0,
            "bullish": 0,
            "strongly bearish": 0,
            "bearish": 0
        }

        for text in headlines:
            res: Dict[str, Any] = self.analyze_headline(text)
            sent: str = str(res.get("sentiment", "neutral"))
            if sent in counts:
                counts[sent] = counts[sent] + 1

        # Calculate momentum
        val_sb = counts["strongly bullish"] * 2
        val_b = counts["bullish"]
        val_sbe = counts["strongly bearish"] * 2
        val_be = counts["bearish"]
        
        net_score = (val_sb + val_b) - (val_sbe + val_be)
        
        dominant = "neutral"
        modifier = 1.0
        block_longs = False
        block_shorts = False

        if counts["strongly bearish"] > 0 or net_score <= -3:
            dominant = "strongly bearish"
            block_longs = True
        elif net_score < 0:
            dominant = "bearish"
        elif counts["strongly bullish"] > 0 or net_score >= 3:
            dominant = "strongly bullish"
            block_shorts = True
            modifier = 1.25
        elif net_score > 0:
            dominant = "bullish"

        result = {
            "dominant": dominant,
            "modifier": modifier,
            "block_longs": block_longs,
            "block_shorts": block_shorts,
            "net_score": net_score
        }
        
        logger.info(f"[SENTIMENT] Aggregated Result: {dominant.upper()} | Mod: {modifier}x | Block L:{block_longs} S:{block_shorts}")
        return result

global_finbert_scorer = FinBERTScorer()
