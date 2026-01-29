"""
Main Flask Application - Soil Health & Crop Recommendation System
Updated with proper routing and session management
"""

from flask import Flask, render_template, request, jsonify, session, redirect, url_for, g
import json
import os
import sys
import traceback
import pandas as pd
import numpy as np
from datetime import timedelta

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import prediction modules
print("🔧 Importing prediction modules...")
try:
    sys.path.insert(0, os.path.join(current_dir, 'src'))
    from src.predict import soil_predictor, crop_predictor
    print("✅ All prediction modules imported successfully")
except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    print("⚠️ Creating dummy predictors for testing...")
    
    # Create dummy predictors
    class DummyPredictor:
        def predict_soil_health(self, data):
            # Simple mock prediction
            n = float(data.get('nitrogen', 0))
            if n < 100:
                return {"success": True, "health_status": "Poor", "health_class": 0, "confidence": 85.0}
            elif n < 200:
                return {"success": True, "health_status": "Fair", "health_class": 1, "confidence": 90.0}
            else:
                return {"success": True, "health_status": "Good", "health_class": 2, "confidence": 95.0}
        
        def predict_crop(self, data):
            # Simple mock prediction
            return {
                "success": True, 
                "recommended_crop": "Rice",
                "confidence": 99.0,
                "top_3_crops": [
                    {"crop": "Rice", "probability": 99.0},
                    {"crop": "Maize", "probability": 85.0},
                    {"crop": "Wheat", "probability": 70.0}
                ]
            }
    
    soil_predictor = DummyPredictor()
    crop_predictor = DummyPredictor()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'soil-health-analytics-secret-key-2024-tamil-nadu-farmers'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=1)

# Store predictors in app config for easy access
app.config['soil_predictor'] = soil_predictor
app.config['crop_predictor'] = crop_predictor

# ============================================
# CONTEXT PROCESSORS & BEFORE REQUEST
# ============================================

@app.before_request
def before_request():
    """Set up language and other request-specific data"""
    # Initialize language if not set
    if 'language' not in session:
        session['language'] = 'en'
    
    # Store language in g for easy access
    g.current_language = session.get('language', 'en')

@app.context_processor
def inject_language():
    """Inject current language into all templates"""
    return {'current_language': session.get('language', 'en')}

# ============================================
# LANGUAGE SUPPORT
# ============================================

@app.route('/switch-language/<lang>')
def switch_language(lang):
    """Switch language between English and Tamil"""
    if lang in ['en', 'ta']:
        session['language'] = lang
        session.permanent = True
        return jsonify({"success": True, "language": lang})
    return jsonify({"success": False, "error": "Invalid language"})

@app.route('/debug-language')
def debug_language():
    """Debug route to check language switching"""
    return jsonify({
        "current_language": session.get('language', 'not set'),
        "session_id": session.get('_id', 'no session'),
        "session_keys": list(session.keys())
    })

# ============================================
# FRONTEND ROUTES
# ============================================

@app.route('/')
def home():
    """Homepage"""
    return render_template('index.html')

@app.route('/soil-analysis')
def soil_analysis():
    """Soil analysis form page"""
    return render_template('soil_analysis.html')

@app.route('/crop-recommendation')
def crop_recommendation():
    """Crop recommendation form page"""
    return render_template('crop_recommendation.html')

@app.route('/crop-recommendation-page')
def crop_recommendation_page():
    """Alias for crop recommendation page"""
    return redirect(url_for('crop_recommendation'))

@app.route('/fertilizer-advice')
def fertilizer_advice():
    """Fertilizer advice page"""
    return render_template('fertilizer_advice.html')

@app.route('/results')
def results():
    """Results display page"""
    # Check if we have results in session
    if 'soil_result' not in session and 'crop_result' not in session:
        return redirect(url_for('soil_analysis'))
    return render_template('results.html')

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

# ============================================
# API ENDPOINTS
# ============================================

@app.route('/api/predict/soil', methods=['POST'])
def predict_soil():
    """Predict soil health from form data"""
    try:
        # Get form data
        if request.is_json:
            data = request.json
        else:
            data = request.form.to_dict()
        
        print(f"📥 Soil prediction data received: {data}")
        
        # Validate required fields
        required_fields = ['nitrogen', 'phosphorus', 'potassium', 'ph']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    "success": False,
                    "error": f"Missing required field: {field}"
                })
        
        # Convert to floats
        soil_data = {}
        for key, value in data.items():
            try:
                soil_data[key] = float(value)
            except (ValueError, TypeError):
                return jsonify({
                    "success": False,
                    "error": f"Invalid value for {key}: {value}"
                })
        
        # Make prediction
        predictor = app.config['soil_predictor']
        result = predictor.predict_soil_health(soil_data)
        
        if result.get('success'):
            # Store in session
            session['soil_result'] = json.dumps(result)
            session['last_analysis_type'] = 'soil'
            
            # Prepare fertilizer recommendation based on soil health
            if 'health_class' in result:
                fertilizer_recommendation = get_fertilizer_recommendation(
                    result['health_class'],
                    result.get('health_status', 'Unknown'),
                    soil_data
                )
                session['fertilizer_recommendation'] = json.dumps(fertilizer_recommendation)
            
            return jsonify(result)
        else:
            return jsonify(result)
            
    except Exception as e:
        print(f"❌ Error in soil prediction: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": f"Prediction error: {str(e)}"
        })

@app.route('/api/predict/crop', methods=['POST'])
def predict_crop():
    """Predict crop recommendation from form data"""
    try:
        # Get form data
        if request.is_json:
            data = request.json
        else:
            data = request.form.to_dict()
        
        print(f"📥 Crop prediction data received: {data}")
        
        # Validate required fields
        required_fields = ['nitrogen', 'phosphorus', 'potassium', 'ph', 'temperature', 'humidity', 'rainfall']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    "success": False,
                    "error": f"Missing required field: {field}"
                })
        
        # Convert to appropriate types
        crop_data = {}
        for key, value in data.items():
            if key in ['nitrogen', 'phosphorus', 'potassium']:
                try:
                    crop_data[key] = int(float(value))
                except (ValueError, TypeError):
                    crop_data[key] = 0
            elif key in ['temperature', 'humidity', 'ph', 'rainfall']:
                try:
                    crop_data[key] = float(value)
                except (ValueError, TypeError):
                    crop_data[key] = 0.0
            else:
                crop_data[key] = value
        
        # Make prediction
        predictor = app.config['crop_predictor']
        result = predictor.predict_crop(crop_data)
        
        if result.get('success'):
            # Store in session
            session['crop_result'] = json.dumps(result)
            session['last_analysis_type'] = 'crop'
            
            # Get fertilizer recommendation for the recommended crop
            if 'recommended_crop' in result:
                fertilizer_recommendation = get_crop_fertilizer_recommendation(
                    result['recommended_crop'],
                    crop_data
                )
                session['fertilizer_recommendation'] = json.dumps(fertilizer_recommendation)
            
            return jsonify(result)
        else:
            return jsonify(result)
            
    except Exception as e:
        print(f"❌ Error in crop prediction: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": f"Prediction error: {str(e)}"
        })

@app.route('/api/get-results', methods=['GET'])
def get_results():
    """Get all analysis results from session"""
    results = {}
    
    if 'soil_result' in session:
        results['soil'] = json.loads(session['soil_result'])
    
    if 'crop_result' in session:
        results['crop'] = json.loads(session['crop_result'])
    
    if 'fertilizer_recommendation' in session:
        results['fertilizer'] = json.loads(session['fertilizer_recommendation'])
    
    return jsonify(results)

@app.route('/api/clear-results', methods=['GET'])
def clear_results():
    """Clear all results from session"""
    session.pop('soil_result', None)
    session.pop('crop_result', None)
    session.pop('fertilizer_recommendation', None)
    session.pop('last_analysis_type', None)
    
    return jsonify({"success": True, "message": "Results cleared"})

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "running",
        "soil_model": hasattr(app.config['soil_predictor'], 'soil_model') and app.config['soil_predictor'].soil_model is not None,
        "crop_model": hasattr(app.config['crop_predictor'], 'crop_model') and app.config['crop_predictor'].crop_model is not None,
        "language": session.get('language', 'en')
    })

# ============================================
# HELPER FUNCTIONS
# ============================================

def get_fertilizer_recommendation(soil_class, soil_status, soil_data):
    """Generate fertilizer recommendations based on soil health"""
    n = soil_data.get('nitrogen', 0)
    p = soil_data.get('phosphorus', 0)
    k = soil_data.get('potassium', 0)
    ph = soil_data.get('ph', 7.0)
    
    recommendations = []
    
    # Nitrogen recommendations
    if n < 100:
        recommendations.append({
            "fertilizer": "Urea",
            "amount": "100-150 kg/acre",
            "purpose": "Increase nitrogen levels",
            "timing": "Basal + Top dressing (split application)"
        })
    elif n < 200:
        recommendations.append({
            "fertilizer": "Urea",
            "amount": "50-100 kg/acre",
            "purpose": "Maintain nitrogen levels",
            "timing": "Top dressing during growth"
        })
    
    # Phosphorus recommendations
    if p < 15:
        recommendations.append({
            "fertilizer": "DAP (Diammonium Phosphate)",
            "amount": "50-75 kg/acre",
            "purpose": "Increase phosphorus levels",
            "timing": "Basal application before sowing"
        })
    
    # Potassium recommendations
    if k < 100:
        recommendations.append({
            "fertilizer": "MOP (Muriate of Potash)",
            "amount": "40-60 kg/acre",
            "purpose": "Increase potassium levels",
            "timing": "Basal application"
        })
    
    # pH adjustment
    if ph < 6.0:
        recommendations.append({
            "fertilizer": "Agricultural Lime",
            "amount": "1-2 tons/acre",
            "purpose": "Raise soil pH",
            "timing": "Apply 2-3 weeks before sowing"
        })
    elif ph > 8.0:
        recommendations.append({
            "fertilizer": "Gypsum",
            "amount": "0.5-1 ton/acre",
            "purpose": "Lower soil pH",
            "timing": "Apply before sowing"
        })
    
    # Organic matter
    if soil_class == 0:  # Poor soil
        recommendations.append({
            "fertilizer": "Farm Yard Manure",
            "amount": "5-10 tons/acre",
            "purpose": "Improve soil structure and organic matter",
            "timing": "Apply 3-4 weeks before sowing"
        })
    
    return {
        "success": True,
        "soil_health_label": soil_status,
        "soil_class": soil_class,
        "recommendations": recommendations,
        "general_advice": get_general_fertilizer_advice(soil_class)
    }

def get_crop_fertilizer_recommendation(crop_name, crop_data):
    """Generate fertilizer recommendations for specific crop"""
    # Standard fertilizer recommendations for common crops
    crop_fertilizers = {
        "rice": [
            {"fertilizer": "Urea", "amount": "120-150 kg/acre", "timing": "Split application"},
            {"fertilizer": "DAP", "amount": "50-60 kg/acre", "timing": "Basal"},
            {"fertilizer": "MOP", "amount": "40-50 kg/acre", "timing": "Basal"}
        ],
        "wheat": [
            {"fertilizer": "Urea", "amount": "100-120 kg/acre", "timing": "Split application"},
            {"fertilizer": "DAP", "amount": "40-50 kg/acre", "timing": "Basal"},
            {"fertilizer": "MOP", "amount": "30-40 kg/acre", "timing": "Basal"}
        ],
        "maize": [
            {"fertilizer": "Urea", "amount": "150-180 kg/acre", "timing": "Split application"},
            {"fertilizer": "DAP", "amount": "60-75 kg/acre", "timing": "Basal"},
            {"fertilizer": "MOP", "amount": "40-50 kg/acre", "timing": "Basal"}
        ],
        "cotton": [
            {"fertilizer": "Urea", "amount": "80-100 kg/acre", "timing": "Split application"},
            {"fertilizer": "DAP", "amount": "40-50 kg/acre", "timing": "Basal"},
            {"fertilizer": "MOP", "amount": "40-50 kg/acre", "timing": "Basal"}
        ],
        "sugarcane": [
            {"fertilizer": "Urea", "amount": "200-250 kg/acre", "timing": "Split application"},
            {"fertilizer": "DAP", "amount": "80-100 kg/acre", "timing": "Basal"},
            {"fertilizer": "MOP", "amount": "80-100 kg/acre", "timing": "Basal"}
        ],
        "apple": [
            {"fertilizer": "NPK 10:10:10", "amount": "500-600 g/tree", "timing": "Before flowering"},
            {"fertilizer": "Farm Yard Manure", "amount": "20-25 kg/tree", "timing": "Annual"}
        ],
        "banana": [
            {"fertilizer": "Urea", "amount": "200-250 g/plant", "timing": "Monthly"},
            {"fertilizer": "MOP", "amount": "300-400 g/plant", "timing": "During flowering"}
        ]
    }
    
    recommendations = crop_fertilizers.get(crop_name.lower(), [
        {"fertilizer": "NPK 10:26:26", "amount": "100-150 kg/acre", "timing": "Basal application"},
        {"fertilizer": "Urea", "amount": "50-100 kg/acre", "timing": "Top dressing"}
    ])
    
    return {
        "success": True,
        "crop_name": crop_name,
        "recommendations": recommendations,
        "general_advice": f"Specific fertilizer schedule for {crop_name}"
    }

def get_general_fertilizer_advice(soil_class):
    """Get general fertilizer advice based on soil health class"""
    advice = {
        0: "Poor soil requires significant soil amendment. Focus on organic matter addition and balanced fertilization.",
        1: "Fair soil needs moderate improvement. Maintain balanced fertilization and add organic matter.",
        2: "Good soil requires maintenance. Continue balanced fertilization and regular soil testing."
    }
    return advice.get(soil_class, "Regular soil testing and balanced fertilization is recommended.")

# ============================================
# ERROR HANDLERS
# ============================================

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors"""
    current_language = session.get('language', 'en')
    error_message = "Page not found" if current_language == 'en' else "பக்கம் கிடைக்கவில்லை"
    return render_template('error.html', 
                         error_code=404,
                         error_message=error_message), 404

@app.errorhandler(500)
def internal_server_error(e):
    """Handle 500 errors"""
    current_language = session.get('language', 'en')
    error_message = "Internal server error" if current_language == 'en' else "சேவையகப் பிழை"
    return render_template('error.html',
                         error_code=500,
                         error_message=error_message), 500

# ============================================
# APPLICATION STARTUP
# ============================================

if __name__ == '__main__':
    print("="*70)
    print("🌱 SOIL HEALTH ANALYTICS SYSTEM - TAMIL NADU")
    print("="*70)
    
    # Print system info
    print(f"✅ Soil Model: {type(app.config['soil_predictor']).__name__}")
    print(f"✅ Crop Model: {type(app.config['crop_predictor']).__name__}")
    
    print("\n🌐 AVAILABLE ENDPOINTS:")
    print("   http://localhost:5000/                  - Homepage")
    print("   http://localhost:5000/soil-analysis     - Soil Analysis Form")
    print("   http://localhost:5000/crop-recommendation - Crop Recommendation Form")
    print("   http://localhost:5000/fertilizer-advice - Fertilizer Advice")
    print("   http://localhost:5000/results           - Results Page")
    print("   http://localhost:5000/about             - About Page")
    
    print("\n📊 API ENDPOINTS:")
    print("   POST /api/predict/soil     - Predict soil health")
    print("   POST /api/predict/crop     - Recommend crop")
    print("   GET  /api/get-results      - Get all results")
    print("   GET  /api/clear-results    - Clear results")
    print("   GET  /api/health           - Health check")
    
    print("\n🚀 Starting Flask server...")
    print("="*70)
    
    # Run the app
    app.run(debug=True, port=5000, host='0.0.0.0')