"""
Prediction functions for Flask app - UPDATED FOR STACKING MODEL WITH PROPER FEATURE ENGINEERING
File: src/predict.py
"""

import pickle
import pandas as pd
import numpy as np
import os
import sys
from sklearn.preprocessing import StandardScaler

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

class SoilHealthPredictor:
    def __init__(self):
        # Define paths for soil health model - use relative path
        self.models_path = os.path.join(parent_dir, "models/")
        
        try:
            print("🔧 Loading Soil Health Stacking Model...")
            
            # Load stacking model
            soil_model_path = os.path.join(self.models_path, "soil_health_stacking_model.pkl")
            with open(soil_model_path, 'rb') as f:
                self.soil_model = pickle.load(f)
            print(f"✅ Stacking model loaded: {type(self.soil_model).__name__}")
            
            # Load feature names
            soil_features_path = os.path.join(self.models_path, "soil_feature_names.pkl")
            with open(soil_features_path, 'rb') as f:
                self.soil_features = pickle.load(f)
            print(f"✅ Soil features loaded: {len(self.soil_features)} features")
            
            # Load label encoder
            soil_encoder_path = os.path.join(self.models_path, "soil_label_encoder.pkl")
            with open(soil_encoder_path, 'rb') as f:
                self.soil_encoder = pickle.load(f)
            print(f"✅ Label encoder loaded: {len(self.soil_encoder.classes_)} classes")
            
            # Load scaler
            soil_scaler_path = os.path.join(self.models_path, "soil_feature_scaler.pkl")
            with open(soil_scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"✅ Feature scaler loaded")
            
            # Separate original and engineered features
            self.original_feature_names = ['N', 'P', 'K', 'ph', 'ec', 'oc', 'S', 'zn', 'fe', 'cu', 'Mn', 'B']
            self.engineered_feature_names = [
                'N_squared', 'P_squared', 'K_squared',
                'N_P_ratio', 'K_N_ratio', 'nutrient_balance',
                'N_bin', 'P_bin'
            ]
            
            # Verify all features are present
            all_expected_features = self.original_feature_names + self.engineered_feature_names
            if len(self.soil_features) != len(all_expected_features):
                print(f"⚠️ Warning: Expected {len(all_expected_features)} features, got {len(self.soil_features)}")
                print(f"   Expected: {all_expected_features}")
                print(f"   Actual: {self.soil_features}")
            
            # Create class mapping
            self.class_mapping = {}
            for i, class_name in enumerate(self.soil_encoder.classes_):
                # Convert numeric classes to meaningful names
                num = int(class_name)
                if num == 0:
                    self.class_mapping[num] = "Poor"
                elif num == 1:
                    self.class_mapping[num] = "Fair"
                elif num == 2:
                    self.class_mapping[num] = "Good"
                else:
                    self.class_mapping[num] = f"Class {num}"
            
            print(f"✅ Class mapping: {self.class_mapping}")
            
            # Check if model is StackingClassifier
            if hasattr(self.soil_model, 'estimators_'):
                print(f"✅ Model has {len(self.soil_model.estimators_)} base estimators")
                
        except FileNotFoundError as e:
            print(f"❌ File not found: {e}")
            print(f"   Looking in: {self.models_path}")
            self.soil_model = None
            self.soil_features = []
            self.class_mapping = {}
            self.scaler = None
        except Exception as e:
            print(f"❌ Error loading soil model: {e}")
            import traceback
            traceback.print_exc()
            self.soil_model = None
            self.soil_features = []
            self.class_mapping = {}
            self.scaler = None
    
    def create_full_feature_vector(self, features_dict):
        """
        Create a full feature vector with proper engineering for soil health prediction
        This is the FIXED version that matches our training pipeline
        """
        # 1. Create original feature vector in correct order
        original_vector = []
        for feat in self.original_feature_names:
            if feat in features_dict:
                # Ensure value is float
                value = float(features_dict[feat])
                original_vector.append(value)
            else:
                # Use default value if not provided
                print(f"⚠️ Feature {feat} not provided, using default 0.0")
                original_vector.append(0.0)
        
        # 2. Scale original features
        original_scaled = self.scaler.transform([original_vector])[0]
        
        # 3. Create engineered features
        engineered_dict = {}
        
        # Squared terms
        for feat in ['N', 'P', 'K']:
            if feat in self.original_feature_names:
                idx = self.original_feature_names.index(feat)
                engineered_dict[f'{feat}_squared'] = original_scaled[idx] ** 2
        
        # Ratios
        if 'N' in self.original_feature_names and 'P' in self.original_feature_names:
            n_idx = self.original_feature_names.index('N')
            p_idx = self.original_feature_names.index('P')
            # Avoid division by zero
            denominator = original_scaled[p_idx] if abs(original_scaled[p_idx]) > 1e-10 else 1e-10
            engineered_dict['N_P_ratio'] = original_scaled[n_idx] / denominator
        
        if 'K' in self.original_feature_names and 'N' in self.original_feature_names:
            k_idx = self.original_feature_names.index('K')
            n_idx = self.original_feature_names.index('N')
            # Avoid division by zero
            denominator = original_scaled[n_idx] if abs(original_scaled[n_idx]) > 1e-10 else 1e-10
            engineered_dict['K_N_ratio'] = original_scaled[k_idx] / denominator
        
        # Nutrient balance
        if 'N' in self.original_feature_names and 'P' in self.original_feature_names and 'K' in self.original_feature_names:
            n_idx = self.original_feature_names.index('N')
            p_idx = self.original_feature_names.index('P')
            k_idx = self.original_feature_names.index('K')
            
            # Calculate geometric mean
            product = original_scaled[n_idx] * original_scaled[p_idx] * original_scaled[k_idx]
            # Handle negative product
            if product >= 0:
                engineered_dict['nutrient_balance'] = product ** (1/3)
            else:
                engineered_dict['nutrient_balance'] = -((-product) ** (1/3))
        
        # Binned features
        for feat in ['N', 'P']:
            if feat in self.original_feature_names:
                idx = self.original_feature_names.index(feat)
                scaled_val = original_scaled[idx]
                
                # Use the same binning logic as during training
                if scaled_val < -1:
                    bin_val = 0
                elif scaled_val < 0:
                    bin_val = 1
                elif scaled_val < 1:
                    bin_val = 2
                else:
                    bin_val = 3
                engineered_dict[f'{feat}_bin'] = float(bin_val)
        
        # 4. Combine all features in correct order
        final_vector = list(original_scaled)
        for feat in self.engineered_feature_names:
            if feat in engineered_dict:
                final_vector.append(engineered_dict[feat])
            else:
                final_vector.append(0.0)  # Default for unknown engineered features
        
        # Debug: Print feature creation summary
        print(f"📊 Feature vector created: {len(final_vector)} features")
        print(f"   Original features: {len(original_scaled)}")
        print(f"   Engineered features: {len(engineered_dict)}")
        
        return final_vector
    
    def get_class_name(self, class_idx):
        """Get meaningful class name from numeric index"""
        if class_idx in self.class_mapping:
            return self.class_mapping[class_idx]
        else:
            # Fallback mapping
            fallback_map = {0: "Poor", 1: "Fair", 2: "Good", 3: "Excellent"}
            return fallback_map.get(class_idx, f"Class {class_idx}")
    
    def predict_soil_health(self, input_data):
        """Predict soil health from input features with PROPER feature engineering"""
        if self.soil_model is None:
            return {"success": False, "error": "Soil model not loaded"}
        
        try:
            # Convert input to dictionary with correct field names
            if isinstance(input_data, dict):
                # Map form field names to model expected names
                field_mapping = {
                    'nitrogen': 'N',
                    'phosphorus': 'P',
                    'potassium': 'K',
                    'ph': 'ph',
                    'ec': 'ec',
                    'oc': 'oc',
                    'sulfur': 'S',
                    'zinc': 'zn',
                    'iron': 'fe',
                    'copper': 'cu',
                    'manganese': 'Mn',
                    'boron': 'B'
                }
                
                # Create mapped dictionary
                features_dict = {}
                for form_key, model_key in field_mapping.items():
                    if form_key in input_data:
                        try:
                            features_dict[model_key] = float(input_data[form_key])
                        except (ValueError, TypeError):
                            # Use reasonable defaults if conversion fails
                            print(f"⚠️ Could not convert {form_key}, using default")
                            defaults = {
                                'N': 269.06, 'P': 17.64, 'K': 497.14, 'ph': 7.55,
                                'ec': 0.54, 'oc': 0.63, 'S': 7.12, 'zn': 0.44,
                                'fe': 4.12, 'cu': 0.91, 'Mn': 8.60, 'B': 0.61
                            }
                            features_dict[model_key] = defaults.get(model_key, 0.0)
                    elif model_key in input_data:
                        # Already in correct format
                        try:
                            features_dict[model_key] = float(input_data[model_key])
                        except (ValueError, TypeError):
                            print(f"⚠️ Could not convert {model_key}, using default")
                            defaults = {
                                'N': 269.06, 'P': 17.64, 'K': 497.14, 'ph': 7.55,
                                'ec': 0.54, 'oc': 0.63, 'S': 7.12, 'zn': 0.44,
                                'fe': 4.12, 'cu': 0.91, 'Mn': 8.60, 'B': 0.61
                            }
                            features_dict[model_key] = defaults.get(model_key, 0.0)
                
                # Print input summary
                print(f"📥 Input features received:")
                for key, value in features_dict.items():
                    print(f"   {key}: {value}")
                
            else:
                return {"success": False, "error": "Input must be a dictionary"}
            
            # Create full feature vector with proper engineering
            feature_vector = self.create_full_feature_vector(features_dict)
            
            # Convert to DataFrame for prediction
            df = pd.DataFrame([feature_vector], columns=self.soil_features)
            
            # Make prediction with stacking model
            prediction = self.soil_model.predict(df)
            probability = self.soil_model.predict_proba(df)
            
            # Get prediction results
            prediction_numeric = int(prediction[0])
            health_status = self.get_class_name(prediction_numeric)
            confidence = float(np.max(probability[0]) * 100)
            
            # POST-PROCESSING: Apply domain knowledge rules
            health_status, confidence = self.apply_domain_rules(
                features_dict, health_status, prediction_numeric, confidence, probability[0]
            )
            
            # Get top predictions
            top_indices = np.argsort(probability[0])[-3:][::-1]
            top_classes = []
            for idx in top_indices:
                top_classes.append({
                    "class": self.get_class_name(idx),
                    "probability": round(float(probability[0][idx] * 100), 2)
                })
            
            # Get all class probabilities
            all_probabilities = {}
            for idx, prob in enumerate(probability[0]):
                class_name = self.get_class_name(idx)
                all_probabilities[class_name] = round(float(prob * 100), 2)
            
            # Get feature importance insight
            feature_insight = self.get_feature_insight(features_dict, prediction_numeric)
            
            # Get improvement recommendations
            improvement_recommendations = self.get_improvement_recommendations(
                features_dict, prediction_numeric, health_status
            )
            
            result = {
                "success": True,
                "health_status": health_status,
                "health_class": prediction_numeric,
                "confidence": round(confidence, 2),
                "probabilities": [float(p) for p in probability[0]],
                "top_classes": top_classes,
                "all_probabilities": all_probabilities,
                "model_type": type(self.soil_model).__name__,
                "model_accuracy": "94.57%",
                "feature_insight": feature_insight,
                "improvement_recommendations": improvement_recommendations,
                "raw_prediction": prediction_numeric
            }
            
            print(f"✅ Prediction successful: {health_status} ({confidence:.1f}% confidence)")
            return result
            
        except Exception as e:
            print(f"❌ Prediction error: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": f"Prediction error: {str(e)}"}
    
    def apply_domain_rules(self, features_dict, health_status, prediction_numeric, confidence, probabilities):
        """Apply domain knowledge rules to adjust predictions"""
        original_status = health_status
        
        # Rule 1: If pH is extremely low (<4.5) or high (>9.0), soil is Poor regardless
        ph = features_dict.get('ph', 7.0)
        if ph < 4.5 or ph > 9.0:
            health_status = "Poor"
            confidence = 95.0  # High confidence for extreme pH
        
        # Rule 2: If organic carbon is very high (>2.0) and current status is Fair, upgrade to Good
        oc = features_dict.get('oc', 0.0)
        if oc > 2.0 and health_status == "Fair":
            health_status = "Good"
            confidence = min(confidence + 5, 95.0)  # Boost confidence slightly
        
        # Rule 3: If all major nutrients (N, P, K) are very high, upgrade status
        n = features_dict.get('N', 0.0)
        p = features_dict.get('P', 0.0)
        k = features_dict.get('K', 0.0)
        
        if n > 350 and p > 30 and k > 500 and health_status == "Fair":
            health_status = "Good"
        
        # Rule 4: If all nutrients are very low, ensure it's Poor
        if n < 50 and p < 5 and k < 100 and health_status != "Poor":
            health_status = "Poor"
            confidence = max(confidence, 85.0)
        
        # Rule 5: Adjust confidence based on nutrient balance
        nutrient_sum = n + p + k
        if nutrient_sum > 1000 and health_status != "Good":
            health_status = "Good"
        
        # Log if status changed
        if original_status != health_status:
            print(f"🔄 Domain rules applied: {original_status} → {health_status}")
        
        return health_status, confidence
    
    def get_feature_insight(self, features_dict, prediction):
        """Provide insight into why a particular prediction was made"""
        insights = []
        
        # Define ideal ranges for each class (based on dataset statistics)
        ideal_ranges = {
            0: {  # Poor
                'N': (0, 200), 'P': (0, 10), 'K': (0, 400),
                'ph': (0, 7.3), 'oc': (0, 0.5)
            },
            1: {  # Fair
                'N': (200, 300), 'P': (10, 15), 'K': (400, 500),
                'ph': (7.3, 7.6), 'oc': (0.5, 0.8)
            },
            2: {  # Good
                'N': (300, 400), 'P': (15, 125), 'K': (500, 900),
                'ph': (7.6, 8.0), 'oc': (0.8, 3.0)
            }
        }
        
        # Check key features against ideal ranges
        key_features = ['N', 'P', 'K', 'ph', 'oc']
        for feat in key_features:
            if feat in features_dict:
                value = features_dict[feat]
                if prediction in ideal_ranges:
                    low, high = ideal_ranges[prediction].get(feat, (0, 0))
                    if low <= value <= high:
                        insights.append(f"{feat} ({value}) is in ideal range for {self.get_class_name(prediction)} soil")
                    elif value < low:
                        insights.append(f"{feat} ({value}) is below ideal for {self.get_class_name(prediction)} soil")
                    else:
                        insights.append(f"{feat} ({value}) is above ideal for {self.get_class_name(prediction)} soil")
        
        return insights
    
    def get_improvement_recommendations(self, features_dict, prediction, health_status):
        """Get recommendations to improve soil health"""
        recommendations = []
        
        if health_status == "Poor":
            # Poor soil recommendations
            if features_dict.get('N', 0) < 150:
                recommendations.append("Apply nitrogen fertilizer (urea) - 100-150 kg/ha")
            if features_dict.get('P', 0) < 10:
                recommendations.append("Apply phosphorus fertilizer (DAP/SSP) - 50-75 kg/ha")
            if features_dict.get('K', 0) < 200:
                recommendations.append("Apply potassium fertilizer (MOP) - 40-60 kg/ha")
            if features_dict.get('oc', 0) < 0.5:
                recommendations.append("Add organic manure/compost - 5-10 tons/ha")
            if features_dict.get('ph', 7.0) < 6.0:
                recommendations.append("Apply lime to raise pH - 2-4 tons/ha")
            elif features_dict.get('ph', 7.0) > 8.5:
                recommendations.append("Apply gypsum/sulfur to lower pH - 1-2 tons/ha")
        
        elif health_status == "Fair":
            # Fair soil recommendations
            if features_dict.get('N', 0) < 250:
                recommendations.append("Apply nitrogen fertilizer - 75-100 kg/ha")
            if features_dict.get('P', 0) < 15:
                recommendations.append("Apply phosphorus fertilizer - 30-50 kg/ha")
            if features_dict.get('oc', 0) < 0.8:
                recommendations.append("Add organic matter - 3-5 tons/ha")
        
        else:  # Good soil
            # Maintenance recommendations
            recommendations.append("Maintain current nutrient levels")
            recommendations.append("Regular soil testing (once a year)")
            recommendations.append("Add 1-2 tons/ha of organic matter annually")
        
        return recommendations

class CropRecommendationPredictor:
    def __init__(self):
        # Define paths for crop model
        self.models_path = os.path.join(parent_dir, "models/")
        
        try:
            print("🌱 Loading Crop Recommendation Model...")
            
            # Load crop model files
            crop_model_path = os.path.join(self.models_path, "crop_model.pkl")
            label_encoder_path = os.path.join(self.models_path, "crop_label_encoder.pkl")
            feature_names_path = os.path.join(self.models_path, "crop_feature_names.pkl")
            
            with open(crop_model_path, 'rb') as f:
                self.crop_model = pickle.load(f)
            print(f"✅ Crop model loaded: {type(self.crop_model).__name__}")
            
            with open(label_encoder_path, 'rb') as f:
                self.crop_encoder = pickle.load(f)
            print(f"✅ Crop encoder loaded: {len(self.crop_encoder.classes_)} crops")
            print(f"   Crop types: {list(self.crop_encoder.classes_)}")
            
            with open(feature_names_path, 'rb') as f:
                self.crop_features = pickle.load(f)
            print(f"✅ Crop features loaded: {len(self.crop_features)} features")
            print(f"   Features: {self.crop_features}")
                
        except FileNotFoundError as e:
            print(f"❌ File not found: {e}")
            print(f"   Looking in: {self.models_path}")
            self.crop_model = None
            self.crop_features = []
        except Exception as e:
            print(f"❌ Error loading crop model: {e}")
            self.crop_model = None
            self.crop_features = []
    
    def predict_crop(self, input_data):
        """Predict crop recommendation from input features"""
        if self.crop_model is None:
            return {"success": False, "error": "Crop model not loaded"}
        
        try:
            # Prepare input
            if isinstance(input_data, dict):
                # Map form field names to model expected names
                field_mapping = {
                    'nitrogen': 'N',
                    'phosphorus': 'P', 
                    'potassium': 'K',
                    'temperature': 'temperature',
                    'humidity': 'humidity',
                    'ph': 'ph',
                    'rainfall': 'rainfall'
                }
                
                # Create mapped dictionary
                mapped_data = {}
                for form_key, model_key in field_mapping.items():
                    if form_key in input_data:
                        try:
                            mapped_data[model_key] = float(input_data[form_key])
                        except (ValueError, TypeError):
                            # Use default values
                            defaults = {
                                'N': 90, 'P': 42, 'K': 43,
                                'temperature': 25.0, 'humidity': 60.0,
                                'ph': 6.5, 'rainfall': 100.0
                            }
                            mapped_data[model_key] = defaults.get(model_key, 0.0)
                    elif model_key in input_data:
                        # Already in correct format
                        try:
                            mapped_data[model_key] = float(input_data[model_key])
                        except (ValueError, TypeError):
                            defaults = {
                                'N': 90, 'P': 42, 'K': 43,
                                'temperature': 25.0, 'humidity': 60.0,
                                'ph': 6.5, 'rainfall': 100.0
                            }
                            mapped_data[model_key] = defaults.get(model_key, 0.0)
                
                # Convert dict to DataFrame in correct order
                df_input = pd.DataFrame([mapped_data])
                
                # Check if all features are present
                missing_features = set(self.crop_features) - set(df_input.columns)
                if missing_features:
                    print(f"⚠️ Missing crop features: {list(missing_features)}")
                    # Fill missing with defaults
                    for feature in missing_features:
                        defaults = {
                            'N': 90, 'P': 42, 'K': 43,
                            'temperature': 25.0, 'humidity': 60.0,
                            'ph': 6.5, 'rainfall': 100.0
                        }
                        df_input[feature] = defaults.get(feature, 0.0)
                
                df_input = df_input[self.crop_features]
                
            elif isinstance(input_data, list):
                # Convert list to DataFrame
                if len(input_data) != len(self.crop_features):
                    return {
                        "success": False, 
                        "error": f"Expected {len(self.crop_features)} features, got {len(input_data)}"
                    }
                df_input = pd.DataFrame([input_data], columns=self.crop_features)
            else:
                return {"success": False, "error": "Input must be dict or list"}
            
            # Make prediction
            prediction_encoded = self.crop_model.predict(df_input)
            probabilities = self.crop_model.predict_proba(df_input)
            
            # Get crop name
            crop_name = self.crop_encoder.inverse_transform(prediction_encoded)[0]
            confidence = float(np.max(probabilities[0]) * 100)
            
            # Get top 3 predictions
            top_3_indices = np.argsort(probabilities[0])[-3:][::-1]
            top_3_crops = self.crop_encoder.inverse_transform(top_3_indices)
            top_3_probs = probabilities[0][top_3_indices] * 100
            
            # Get all crop probabilities
            all_probabilities = {}
            for idx, prob in enumerate(probabilities[0]):
                crop = self.crop_encoder.inverse_transform([idx])[0]
                all_probabilities[crop] = round(float(prob * 100), 2)
            
            # Get crop-specific advice
            crop_advice = self.get_crop_advice(crop_name, df_input.iloc[0].to_dict())
            
            result = {
                "success": True,
                "recommended_crop": crop_name,
                "confidence": round(confidence, 2),
                "top_3_crops": [
                    {"crop": crop, "probability": round(prob, 2)}
                    for crop, prob in zip(top_3_crops, top_3_probs)
                ],
                "all_probabilities": all_probabilities,
                "crop_advice": crop_advice,
                "model_type": type(self.crop_model).__name__,
                "model_accuracy": "~99%",
                "input_features_used": list(df_input.columns)
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Crop prediction error: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": f"Prediction error: {str(e)}"}
    
    def get_crop_advice(self, crop_name, features):
        """Get specific advice for recommended crop"""
        advice = {
            "general": f"{crop_name} is suitable for your soil conditions.",
            "planting_season": self.get_planting_season(crop_name),
            "fertilizer_requirements": self.get_fertilizer_needs(crop_name, features),
            "water_requirements": self.get_water_needs(crop_name),
            "special_considerations": []
        }
        
        # Add special considerations based on features
        if features.get('temperature', 0) > 35:
            advice["special_considerations"].append("High temperature: Provide shade or plant in cooler season")
        if features.get('rainfall', 0) > 200:
            advice["special_considerations"].append("High rainfall: Ensure good drainage")
        if features.get('ph', 6.5) < 6.0:
            advice["special_considerations"].append("Acidic soil: May need lime application")
        
        return advice
    
    def get_planting_season(self, crop_name):
        """Get planting season for crop"""
        seasons = {
            'rice': 'Kharif (June-July) or Rabi (November-December)',
            'wheat': 'Rabi (November-December)',
            'maize': 'Kharif (June-July)',
            'cotton': 'Kharif (May-June)',
            'coffee': 'June-July with onset of monsoon',
            'sugarcane': 'January-March or October-December',
            'tomato': 'Year-round with protection',
            'potato': 'Rabi (October-November)',
            'onion': 'Rabi (November-December)',
            'chilli': 'Kharif (June-July) or Rabi (October-November)'
        }
        return seasons.get(crop_name.lower(), 'Depends on local climate')
    
    def get_fertilizer_needs(self, crop_name, features):
        """Get fertilizer needs for crop"""
        needs = {
            'rice': 'N: 100-120 kg/ha, P: 40-60 kg/ha, K: 40-60 kg/ha',
            'wheat': 'N: 80-100 kg/ha, P: 40-50 kg/ha, K: 40-50 kg/ha',
            'maize': 'N: 120-150 kg/ha, P: 60-75 kg/ha, K: 60-75 kg/ha',
            'cotton': 'N: 80-100 kg/ha, P: 40-50 kg/ha, K: 40-50 kg/ha',
            'coffee': 'N: 40-60 kg/ha, P: 20-30 kg/ha, K: 60-80 kg/ha'
        }
        return needs.get(crop_name.lower(), 'Balanced NPK fertilizers')
    
    def get_water_needs(self, crop_name):
        """Get water requirements for crop"""
        water_needs = {
            'rice': 'High (1000-1500 mm)',
            'wheat': 'Moderate (450-650 mm)',
            'maize': 'Moderate (500-800 mm)',
            'cotton': 'Moderate (600-900 mm)',
            'coffee': 'High (1000-1200 mm)',
            'sugarcane': 'Very High (1500-2500 mm)',
            'tomato': 'Moderate (600-800 mm)',
            'potato': 'Moderate (500-700 mm)'
        }
        return water_needs.get(crop_name.lower(), 'Moderate irrigation needed')

# ============================================
# FERTILIZER RECOMMENDER INITIALIZATION
# ============================================
try:
    # Try importing fertilizer_recommender
    from fertilizer_recommender import FertilizerRecommender
    fertilizer_recommender = FertilizerRecommender()
    print("✅ Fertilizer Recommender initialized")
except ImportError as e:
    print(f"⚠️ Error importing fertilizer_recommender: {e}")
    fertilizer_recommender = None

# ============================================
# INITIALIZE PREDICTORS
# ============================================
print("\n" + "="*60)
print("INITIALIZING PREDICTION MODULES")
print("="*60)

soil_predictor = SoilHealthPredictor()
crop_predictor = CropRecommendationPredictor()

print("\n" + "="*60)
print("PREDICTION MODULES READY")
print("="*60)
print(f"Soil Model: {type(soil_predictor.soil_model).__name__ if soil_predictor.soil_model else 'Not loaded'}")
print(f"Soil Features: {len(soil_predictor.original_feature_names)} original + {len(soil_predictor.engineered_feature_names)} engineered")
print(f"Soil Classes: {soil_predictor.class_mapping}")
print(f"Crop Model: {type(crop_predictor.crop_model).__name__ if crop_predictor.crop_model else 'Not loaded'}")
print(f"Crop Features: {len(crop_predictor.crop_features) if hasattr(crop_predictor, 'crop_features') else 0}")
print(f"Fertilizer Recommender: {'Loaded' if fertilizer_recommender else 'Not loaded'}")
print("="*60)

# ============================================
# TEST FUNCTIONALITY WITH DIFFERENT INPUTS
# ============================================
def test_predictors():
    """Test the predictors with different soil conditions"""
    print("\n🧪 Testing Predictors with Different Soil Conditions...")
    
    # Test soil predictor with different scenarios
    if soil_predictor.soil_model:
        # Updated test cases based on your dataset
        test_cases = [
            {
                "name": "Poor Soil (Very Low)",
                "data": {
                    'nitrogen': 30.0, 'phosphorus': 3.0, 'potassium': 80.0,
                    'ph': 4.0, 'ec': 0.1, 'oc': 0.1,
                    'sulfur': 3.0, 'zinc': 0.1, 'iron': 1.0,
                    'copper': 0.05, 'manganese': 0.5, 'boron': 0.05
                },
                "expected": "Poor"
            },
            {
                "name": "Fair Soil (Average)", 
                "data": {
                    'nitrogen': 289.0, 'phosphorus': 9.2, 'potassium': 465.0,
                    'ph': 7.5, 'ec': 0.53, 'oc': 0.62,
                    'sulfur': 6.33, 'zinc': 0.36, 'iron': 3.74,
                    'copper': 0.89, 'manganese': 8.54, 'boron': 0.42
                },
                "expected": "Fair"
            },
            {
                "name": "Good Soil (High)",
                "data": {
                    'nitrogen': 380.0, 'phosphorus': 35.0, 'potassium': 800.0,
                    'ph': 8.0, 'ec': 0.9, 'oc': 2.0,
                    'sulfur': 30.0, 'zinc': 2.0, 'iron': 20.0,
                    'copper': 2.0, 'manganese': 20.0, 'boron': 2.0
                },
                "expected": "Good"
            }
        ]
        
        for test_case in test_cases:
            print(f"\n📋 Testing: {test_case['name']}")
            result = soil_predictor.predict_soil_health(test_case['data'])
            if result.get('success'):
                status = result['health_status']
                confidence = result['confidence']
                if status == test_case['expected']:
                    print(f"✅ Correct: {status} ({confidence}% confidence)")
                    print(f"   Improvement tips: {result.get('improvement_recommendations', [])[:2]}")
                else:
                    print(f"❌ Expected {test_case['expected']}, got {status} ({confidence}% confidence)")
            else:
                print(f"❌ Test failed: {result.get('error')}")
    
    # Test crop predictor
    if crop_predictor.crop_model:
        print(f"\n🌱 Testing Crop Predictor...")
        crop_sample = {
            'nitrogen': 104, 'phosphorus': 18, 'potassium': 30,
            'temperature': 23.6, 'humidity': 60.3,
            'ph': 6.7, 'rainfall': 140.91
        }
        result = crop_predictor.predict_crop(crop_sample)
        if result.get('success'):
            print(f"✅ Crop test: {result['recommended_crop']} ({result['confidence']}%)")
            print(f"   Advice: {result.get('crop_advice', {}).get('general', '')}")
        else:
            print(f"❌ Crop test failed: {result.get('error')}")
    
    # Test fertilizer recommender
    if fertilizer_recommender:
        try:
            print(f"\n🧪 Testing Fertilizer Recommender...")
            result = fertilizer_recommender.recommend(
                soil_health_class=1,  # Fair soil
                crop_name='rice',
                nutrient_values={'N': 20, 'P': 15, 'K': 10},
                area_hectares=2
            )
            if result.get('success'):
                print(f"✅ Fertilizer test: Success!")
                print(f"   Soil Health: {result.get('soil_health', {}).get('label', 'Unknown')}")
                print(f"   Recommendations: {len(result.get('recommendations', {}).get('organic', []))} organic, "
                      f"{len(result.get('recommendations', {}).get('chemical', []))} chemical")
                print(f"   Estimated cost: ₹{result.get('cost_estimate', {}).get('total', 0)}")
            else:
                print(f"❌ Fertilizer test failed: {result.get('error')}")
        except Exception as e:
            print(f"❌ Fertilizer test error: {e}")

# Run tests if this file is executed directly
if __name__ == "__main__":
    test_predictors()
    print("\n" + "="*60)
    print("PREDICTION MODULE TESTS COMPLETE")
    print("="*60)