const translations = {
    'en': {
        // Navigation
        'home': 'Home',
        'soil_analysis': 'Soil Analysis',
        'crop_recommendation': 'Crop Recommendation',
        'fertilizer_guide': 'Fertilizer Guide',
        'about': 'About',
        
        // Common
        'submit': 'Submit',
        'analyze': 'Analyze',
        'recommend': 'Recommend',
        'loading': 'Loading...',
        'success': 'Success',
        'error': 'Error',
        
        // Soil Analysis
        'nitrogen': 'Nitrogen (N)',
        'phosphorus': 'Phosphorus (P)',
        'potassium': 'Potassium (K)',
        'ph_level': 'pH Level',
        
        // Results
        'soil_health': 'Soil Health',
        'recommended_crop': 'Recommended Crop',
        'fertilizer_recommendation': 'Fertilizer Recommendation'
    },
    'ta': {
        // Navigation
        'home': 'முகப்பு',
        'soil_analysis': 'மண் பரிசோதனை',
        'crop_recommendation': 'பயிர் பரிந்துரை',
        'fertilizer_guide': 'உர வழிகாட்டி',
        'about': 'தகவல்',
        
        // Common
        'submit': 'சமர்ப்பிக்கவும்',
        'analyze': 'பகுப்பாய்வு செய்',
        'recommend': 'பரிந்துரை',
        'loading': 'ஏற்றப்படுகிறது...',
        'success': 'வெற்றி',
        'error': 'பிழை',
        
        // Soil Analysis
        'nitrogen': 'நைட்ரஜன் (N)',
        'phosphorus': 'பாஸ்பரஸ் (P)',
        'potassium': 'பொட்டாசியம் (K)',
        'ph_level': 'pH மட்டம்',
        
        // Results
        'soil_health': 'மண் ஆரோக்கியம்',
        'recommended_crop': 'பரிந்துரைக்கப்பட்ட பயிர்',
        'fertilizer_recommendation': 'உரம் பரிந்துரை'
    }
};

function getTranslation(key) {
    const lang = localStorage.getItem('language') || 'en';
    return translations[lang][key] || key;
}