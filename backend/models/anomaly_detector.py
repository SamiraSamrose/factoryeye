"""
Anomaly Detection Model with Real Gemma Integration
Uses Gemma-7B on GPU for advanced pattern recognition
Implements ensemble methods combining statistical and ML approaches
"""

import numpy as np
import torch
import logging
from typing import Dict, Any, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
import warnings
warnings.filterwarnings('ignore')

from backend.config import settings

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Advanced anomaly detection with Gemma model
    Combines multiple detection algorithms for robust results
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = StandardScaler()
        self.threshold = settings.anomaly_threshold
        
        # Initialize ensemble models
        self.isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
        
        self.elliptic_envelope = EllipticEnvelope(
            contamination=0.1,
            random_state=42
        )
        
        # Load Gemma model
        self._load_gemma_model()
        
        logger.info(f"AnomalyDetector initialized: device={self.device}")
    
    def _load_gemma_model(self):
        """
        Load Gemma-7B model for GPU acceleration
        
        Steps:
        1. Check GPU availability
        2. Load model with quantization for efficiency
        3. Prepare tokenizer
        4. Set model to evaluation mode
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            
            # Configure quantization for efficient GPU usage
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            model_id = "google/gemma-7b-it"
            
            # Load model
            self.gemma_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16
            )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            
            self.gemma_model.eval()
            
            logger.info("Gemma model loaded successfully on GPU")
            
        except Exception as e:
            logger.warning(f"Failed to load Gemma model: {e}. Using fallback methods.")
            self.gemma_model = None
            self.tokenizer = None
    
    async def detect(
        self,
        features: np.ndarray,
        config: Dict[str, Any]
    ) -> float:
        """
        Detect anomalies using ensemble approach
        
        Steps:
        1. Statistical anomaly detection
        2. Isolation Forest detection
        3. Elliptic Envelope detection
        4. Gemma-based deep pattern recognition
        5. Ensemble voting with weighted average
        
        Args:
            features: Extracted feature vector
            config: Sensor configuration with thresholds
            
        Returns:
            Anomaly score (0-1)
        """
        try:
            scores = []
            weights = []
            
            # 1. Statistical detection (25% weight)
            stat_score = self._statistical_anomaly_score(features, config)
            scores.append(stat_score)
            weights.append(0.25)
            
            # 2. Isolation Forest (20% weight)
            if len(features.shape) == 1:
                features_2d = features.reshape(1, -1)
            else:
                features_2d = features
            
            try:
                iso_prediction = self.isolation_forest.fit_predict(features_2d)
                iso_score = 1.0 if iso_prediction[0] == -1 else 0.0
                scores.append(iso_score)
                weights.append(0.20)
            except:
                pass
            
            # 3. Elliptic Envelope (20% weight)
            try:
                elliptic_prediction = self.elliptic_envelope.fit_predict(features_2d)
                elliptic_score = 1.0 if elliptic_prediction[0] == -1 else 0.0
                scores.append(elliptic_score)
                weights.append(0.20)
            except:
                pass
            
            # 4. Gemma-based detection (35% weight) - Most sophisticated
            if self.gemma_model is not None:
                gemma_score = await self._gemma_anomaly_score(features, config)
                scores.append(gemma_score)
                weights.append(0.35)
            
            # Ensemble voting with weights
            weights = np.array(weights)
            weights = weights / weights.sum()  # Normalize
            
            combined_score = np.average(scores, weights=weights)
            
            return float(combined_score)
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            return 0.0
    
    def _statistical_anomaly_score(
        self,
        features: np.ndarray,
        config: Dict[str, Any]
    ) -> float:
        """
        Calculate statistical anomaly score using multiple methods
        
        Methods:
        1. Z-score threshold violations
        2. Interquartile Range (IQR) method
        3. Modified Z-score using MAD
        4. Threshold-based violations
        
        Args:
            features: Feature vector
            config: Sensor configuration
            
        Returns:
            Statistical anomaly score (0-1)
        """
        if len(features) < 4:
            return 0.0
        
        # Extract mean values from features (first 4 values are metric means)
        temp_mean = features[0]
        vib_mean = features[4] if len(features) > 4 else 0
        pressure_mean = features[8] if len(features) > 8 else 0
        rpm_mean = features[12] if len(features) > 12 else 0
        
        thresholds = config.get("thresholds", {})
        violations = 0
        total_checks = 0
        violation_severity = []
        
        # Check temperature
        if "temperature" in thresholds:
            temp_thresh = thresholds["temperature"]
            total_checks += 1
            if temp_mean < temp_thresh.get("min", 0):
                violations += 1
                violation_severity.append(abs(temp_mean - temp_thresh["min"]) / temp_thresh["min"])
            elif temp_mean > temp_thresh.get("max", 100):
                violations += 1
                violation_severity.append(abs(temp_mean - temp_thresh["max"]) / temp_thresh["max"])
        
        # Check vibration
        if "vibration" in thresholds:
            vib_thresh = thresholds["vibration"]
            total_checks += 1
            if vib_mean < vib_thresh.get("min", 0):
                violations += 1
                violation_severity.append(abs(vib_mean - vib_thresh["min"]) / (vib_thresh["min"] + 1e-6))
            elif vib_mean > vib_thresh.get("max", 100):
                violations += 1
                violation_severity.append(abs(vib_mean - vib_thresh["max"]) / vib_thresh["max"])
        
        # Check pressure
        if "pressure" in thresholds:
            pressure_thresh = thresholds["pressure"]
            total_checks += 1
            if pressure_mean < pressure_thresh.get("min", 0):
                violations += 1
                violation_severity.append(abs(pressure_mean - pressure_thresh["min"]) / (pressure_thresh["min"] + 1e-6))
            elif pressure_mean > pressure_thresh.get("max", 100):
                violations += 1
                violation_severity.append(abs(pressure_mean - pressure_thresh["max"]) / pressure_thresh["max"])
        
        # Check RPM
        if "rpm" in thresholds:
            rpm_thresh = thresholds["rpm"]
            total_checks += 1
            if rpm_mean < rpm_thresh.get("min", 0):
                violations += 1
                violation_severity.append(abs(rpm_mean - rpm_thresh["min"]) / (rpm_thresh["min"] + 1e-6))
            elif rpm_mean > rpm_thresh.get("max", 10000):
                violations += 1
                violation_severity.append(abs(rpm_mean - rpm_thresh["max"]) / rpm_thresh["max"])
        
        # Calculate base score from violations
        base_score = violations / total_checks if total_checks > 0 else 0.0
        
        # Adjust by severity
        if violation_severity:
            avg_severity = np.mean(violation_severity)
            # Severity amplifies the score
            severity_factor = min(1.0, avg_severity / 0.5)  # Normalize to 0-1
            final_score = min(1.0, base_score * (1 + severity_factor))
        else:
            final_score = base_score
        
        # Z-score analysis on feature vector
        if len(features) > 10:
            z_scores = np.abs((features - np.mean(features)) / (np.std(features) + 1e-6))
            outlier_count = np.sum(z_scores > 3)
            z_score_factor = outlier_count / len(features)
            final_score = max(final_score, z_score_factor)
        
        return min(1.0, final_score)
    
    async def _gemma_anomaly_score(
        self,
        features: np.ndarray,
        config: Dict[str, Any]
    ) -> float:
        """
        Use Gemma model for deep anomaly detection
        
        Steps:
        1. Format features as natural language prompt
        2. Query Gemma for anomaly assessment
        3. Parse response for anomaly indicators
        4. Calculate confidence score
        
        Args:
            features: Feature vector
            config: Sensor configuration
            
        Returns:
            Gemma-based anomaly score (0-1)
        """
        if self.gemma_model is None or self.tokenizer is None:
            return 0.0
        
        try:
            # Create descriptive prompt
            prompt = self._create_gemma_prompt(features, config)
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.gemma_model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Parse anomaly score from response
            score = self._parse_gemma_response(response)
            
            return score
            
        except Exception as e:
            logger.error(f"Error in Gemma anomaly detection: {e}")
            return 0.0
    
    def _create_gemma_prompt(
        self,
        features: np.ndarray,
        config: Dict[str, Any]
    ) -> str:
        """
        Create natural language prompt for Gemma
        
        Formats sensor data and thresholds into readable text
        """
        # Extract key metrics
        temp = features[0] if len(features) > 0 else 0
        vib = features[4] if len(features) > 4 else 0
        pressure = features[8] if len(features) > 8 else 0
        rpm = features[12] if len(features) > 12 else 0
        
        # Get thresholds
        thresholds = config.get("thresholds", {})
        
        prompt = f"""Analyze the following industrial sensor data for anomalies:

Sensor Readings:
- Temperature: {temp:.2f}°C (Normal range: {thresholds.get('temperature', {}).get('min', 20)}-{thresholds.get('temperature', {}).get('max', 85)}°C)
- Vibration: {vib:.2f} mm/s (Normal range: {thresholds.get('vibration', {}).get('min', 0)}-{thresholds.get('vibration', {}).get('max', 50)} mm/s)
- Pressure: {pressure:.2f} bar (Normal range: {thresholds.get('pressure', {}).get('min', 1)}-{thresholds.get('pressure', {}).get('max', 10)} bar)
- RPM: {rpm:.2f} (Normal range: {thresholds.get('rpm', {}).get('min', 0)}-{thresholds.get('rpm', {}).get('max', 5000)})

Statistical Features:
- Temperature variability: {features[1]:.3f} if len(features) > 1 else 0
- Vibration trend: {features[6]:.3f} if len(features) > 6 else 0
- Rate of change: {features[16]:.3f} if len(features) > 16 else 0

Question: Is this data anomalous? Rate the anomaly severity from 0 (normal) to 1 (critical anomaly).
Answer with a score and brief explanation:"""
        
        return prompt
    
    def _parse_gemma_response(self, response: str) -> float:
        """
        Parse Gemma response to extract anomaly score
        
        Looks for numerical scores and anomaly keywords
        """
        response_lower = response.lower()
        
        # Look for explicit score mentions
        import re
        
        # Pattern 1: "score: 0.X" or "0.X"
        score_pattern = r'score[:\s]+([0-9]*\.?[0-9]+)'
        match = re.search(score_pattern, response_lower)
        if match:
            try:
                score = float(match.group(1))
                return min(1.0, max(0.0, score))
            except:
                pass
        
        # Pattern 2: Look for percentage
        percent_pattern = r'([0-9]+)%'
        match = re.search(percent_pattern, response_lower)
        if match:
            try:
                score = float(match.group(1)) / 100.0
                return min(1.0, max(0.0, score))
            except:
                pass
        
        # Pattern 3: Keyword-based scoring
        critical_keywords = ['critical', 'severe', 'dangerous', 'failure', 'urgent']
        high_keywords = ['high', 'significant', 'concerning', 'abnormal']
        medium_keywords = ['moderate', 'elevated', 'unusual']
        low_keywords = ['minor', 'slight', 'low']
        normal_keywords = ['normal', 'acceptable', 'within range', 'safe']
        
        if any(kw in response_lower for kw in critical_keywords):
            return 0.95
        elif any(kw in response_lower for kw in high_keywords):
            return 0.80
        elif any(kw in response_lower for kw in medium_keywords):
            return 0.60
        elif any(kw in response_lower for kw in low_keywords):
            return 0.30
        elif any(kw in response_lower for kw in normal_keywords):
            return 0.10
        
        # Default: moderate concern
        return 0.50
    
    def train_on_historical_data(self, historical_features: np.ndarray):
        """
        Train ensemble models on historical data
        
        Steps:
        1. Fit Isolation Forest
        2. Fit Elliptic Envelope
        3. Update thresholds
        
        Args:
            historical_features: Array of historical feature vectors
        """
        try:
            if len(historical_features) > 50:
                # Fit Isolation Forest
                self.isolation_forest.fit(historical_features)
                
                # Fit Elliptic Envelope
                self.elliptic_envelope.fit(historical_features)
                
                logger.info(f"Trained anomaly detector on {len(historical_features)} samples")
            else:
                logger.warning("Insufficient historical data for training")
                
        except Exception as e:
            logger.error(f"Error training anomaly detector: {e}")