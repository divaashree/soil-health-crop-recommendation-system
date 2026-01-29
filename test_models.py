"""
Test script to verify all models are working correctly
Run this before submission to ensure everything works
"""

import sys
import os
import json

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

print("="*70)
print("🧪 COMPLETE MODEL TESTING SCRIPT")
print("="*70)

def test_imports():
    """Test if all required packages are installed"""
    print("\n📦 TESTING IMPORTS...")
    
    required_packages = [
        ('flask', 'Flask'),
        ('sklearn', 'scikit-learn'),
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('pickle', 'pickle'),
        ('json', 'json')
    ]
    
    all_ok = True
    for package, display_name in required_packages:
        try:
            __import__(package)
            print(f"✅ {display_name}")
        except ImportError:
            print(f"❌ {display_name} - NOT INSTALLED")
            all_ok = False
    
    return all_ok

def test_model_files():
    """Check if all model files exist"""
    print("\n📁 TESTING MODEL FILES...")
    
    model_files = [
        'models/soil_health_stacking_model.pkl',
        'models/soil_label_encoder.pkl',
        'models/soil_feature_names.pkl',
        'models/soil_feature_scaler.pkl',
        'models/soil_stacking_model_info.json',
        'models/crop_model.pkl',
        'models/crop_label_encoder.pkl',
        'models/crop_feature_names.pkl',
        'models/crop_model_info.json'
    ]
    
    all_exist = True
    for file in model_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - NOT FOUND")
            all_exist = False
    
    return all_exist

def test_soil_predictor():
    """Test soil health predictor"""
    print("\n🌱 TESTING SOIL HEALTH PREDICTOR...")
    
    try:
        from src.predict import soil_predictor
        
        test_data = {
            'nitrogen': 269.06,
            'phosphorus': 17.64,
            'potassium': 497.14,
            'ph': 7.55,
            'ec': 0.54,
            'oc': 0.63,
            'sulfur': 7.12,
            'zinc': 0.44,
            'iron': 4.12,
            'copper': 0.91,
            'manganese': 8.60,
            'boron': 0.61
        }
        
        result = soil_predictor.predict_soil_health(test_data)
        
        if result.get('success'):
            print("✅ Soil Predictor Working")
            print(f"   Health Status: {result['health_status']}")
            print(f"   Confidence: {result['confidence']}%")
            print(f"   Model Type: {result['model_type']}")
            print(f"   Accuracy: {result.get('model_accuracy', 'N/A')}")
            
            print("\n🧪 Testing Different Soil Conditions:")
            
            test_scenarios = [
                {"name": "Poor Soil", "data": {'nitrogen': 50, 'phosphorus': 5, 'potassium': 100, 'ph': 5.0}},
                {"name": "Fair Soil", "data": {'nitrogen': 200, 'phosphorus': 15, 'potassium': 400, 'ph': 7.0}},
                {"name": "Good Soil", "data": {'nitrogen': 400, 'phosphorus': 30, 'potassium': 800, 'ph': 7.8}},
            ]
            
            for scenario in test_scenarios:
                full_data = test_data.copy()
                full_data.update(scenario['data'])
                result = soil_predictor.predict_soil_health(full_data)
                if result.get('success'):
                    print(f"   {scenario['name']}: {result['health_status']} ({result['confidence']:.1f}%)")
            
            return True
        else:
            print(f"❌ Soil Predictor Error: {result.get('error')}")
            return False
            
    except Exception as e:
        print(f"❌ Soil Predictor Exception: {e}")
        return False

def test_crop_predictor():
    """Test crop recommendation predictor"""
    print("\n🌾 TESTING CROP RECOMMENDATION PREDICTOR...")
    
    try:
        from src.predict import crop_predictor
        
        test_data = {
            'nitrogen': 90,
            'phosphorus': 42,
            'potassium': 43,
            'temperature': 25.0,
            'humidity': 60.0,
            'ph': 6.5,
            'rainfall': 100.0
        }
        
        result = crop_predictor.predict_crop(test_data)
        
        if result.get('success'):
            print("✅ Crop Predictor Working")
            print(f"   Recommended Crop: {result['recommended_crop']}")
            print(f"   Confidence: {result['confidence']}%")
            print(f"   Model Type: {result['model_type']}")
            print(f"   Accuracy: {result.get('model_accuracy', 'N/A')}")
            
            print("\n🏆 Top 3 Recommendations:")
            for i, crop_info in enumerate(result.get('top_3_crops', []), 1):
                print(f"   {i}. {crop_info['crop']} ({crop_info['probability']:.1f}%)")
            
            return True
        else:
            print(f"❌ Crop Predictor Error: {result.get('error')}")
            return False
            
    except Exception as e:
        print(f"❌ Crop Predictor Exception: {e}")
        return False

def test_flask_app():
    """Test Flask application startup"""
    print("\n🚀 TESTING FLASK APPLICATION...")
    
    try:
        import importlib.util
        
        spec = importlib.util.spec_from_file_location("app", "app.py")
        app_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(app_module)
        
        print("✅ Flask app module loaded successfully")
        
        if hasattr(app_module, 'app'):
            print("✅ Flask application object found")
            
            with app_module.app.test_client() as client:
                response = client.get('/')
                if response.status_code == 200:
                    print("✅ Homepage accessible (status 200)")
                else:
                    print(f"⚠️ Homepage status: {response.status_code}")
            
            return True
        else:
            print("❌ No Flask app object found in app.py")
            return False
            
    except Exception as e:
        print(f"❌ Flask App Exception: {e}")
        return False

def test_data_files():
    """Test data files exist and are readable"""
    print("\n📊 TESTING DATA FILES...")
    
    try:
        import pandas as pd
        
        data_files = [
            ('data/soil_health.csv', 'Soil Health'),
            ('data/Crop_recommendation.csv', 'Crop Recommendation')
        ]
        
        all_ok = True
        for file_path, file_name in data_files:
            try:
                df = pd.read_csv(file_path)
                print(f"✅ {file_name}: {len(df)} rows, {len(df.columns)} columns")
            except Exception as e:
                print(f"❌ {file_name}: Error reading - {e}")
                all_ok = False
        
        return all_ok
        
    except ImportError:
        print("❌ pandas not installed")
        return False

def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("🚀 STARTING COMPREHENSIVE TEST SUITE")
    print("="*70)
    
    results = {
        'imports': test_imports(),
        'model_files': test_model_files(),
        'data_files': test_data_files(),
        'soil_predictor': test_soil_predictor(),
        'crop_predictor': test_crop_predictor(),
        'flask_app': test_flask_app()
    }
    
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n🎯 Total Tests: {total_tests}")
    print(f"✅ Passed: {passed_tests}")
    print(f"❌ Failed: {total_tests - passed_tests}")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED! Project is ready for submission.")
    else:
        print("\n⚠️ Some tests failed. Please fix before submission.")
    
    print("="*70)

if __name__ == "__main__":
    main()
