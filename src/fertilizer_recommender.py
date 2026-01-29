"""
Fertilizer Recommendation System - UPDATED FOR SOIL HEALTH MODEL
File: src/fertilizer_recommender.py
"""

import numpy as np

class FertilizerRecommender:
    def __init__(self):
        # Fertilizer database with more options
        self.fertilizers = {
            'urea': {
                'type': 'Nitrogen',
                'npk_ratio': '46-0-0',  # 46% N, 0% P, 0% K
                'common_names': ['Urea', 'Carbamide'],
                'uses': ['General nitrogen deficiency', 'Vegetative growth'],
                'price_per_kg': 30,  # ₹ per kg
                'application_rate': '50-100 kg/acre'
            },
            'dap': {
                'type': 'Phosphorus',
                'npk_ratio': '18-46-0',  # 18% N, 46% P, 0% K
                'common_names': ['DAP', 'Diammonium Phosphate'],
                'uses': ['Root development', 'Flowering', 'Fruit setting'],
                'price_per_kg': 35,
                'application_rate': '50-80 kg/acre'
            },
            'mop': {
                'type': 'Potassium', 
                'npk_ratio': '0-0-60',  # 0% N, 0% P, 60% K
                'common_names': ['MOP', 'Muriate of Potash'],
                'uses': ['Disease resistance', 'Fruit quality', 'Drought tolerance'],
                'price_per_kg': 40,
                'application_rate': '30-60 kg/acre'
            },
            'ssp': {
                'type': 'Phosphorus',
                'npk_ratio': '0-16-0',  # 0% N, 16% P, 0% K
                'common_names': ['SSP', 'Single Super Phosphate'],
                'uses': ['Acidic soils', 'Sulfur deficiency'],
                'price_per_kg': 25,
                'application_rate': '100-200 kg/acre'
            },
            'npk_10_26_26': {
                'type': 'Complex',
                'npk_ratio': '10-26-26',
                'common_names': ['NPK Complex', '10-26-26'],
                'uses': ['Balanced nutrition', 'General soil improvement'],
                'price_per_kg': 45,
                'application_rate': '75-150 kg/acre'
            },
            'npk_12_32_16': {
                'type': 'Complex',
                'npk_ratio': '12-32-16',
                'common_names': ['NPK Complex', '12-32-16'],
                'uses': ['Fruit crops', 'Flowering plants'],
                'price_per_kg': 48,
                'application_rate': '60-120 kg/acre'
            },
            'npk_20_20_20': {
                'type': 'Complex',
                'npk_ratio': '20-20-20',
                'common_names': ['Balanced NPK', '20-20-20'],
                'uses': ['General purpose', 'All crops'],
                'price_per_kg': 50,
                'application_rate': '40-80 kg/acre'
            },
            'organic_manure': {
                'type': 'Organic',
                'npk_ratio': '0.5-0.3-0.5',  # Approximate
                'common_names': ['Farm Yard Manure', 'Compost', 'Vermicompost'],
                'uses': ['Soil structure improvement', 'Long-term fertility', 'Microbial activity'],
                'price_per_kg': 2,
                'application_rate': '2-5 tons/acre'
            },
            'vermicompost': {
                'type': 'Organic',
                'npk_ratio': '1-0.5-0.7',  # Approximate
                'common_names': ['Vermicompost', 'Worm castings'],
                'uses': ['Organic farming', 'Seedlings', 'Potting mix'],
                'price_per_kg': 5,
                'application_rate': '1-2 tons/acre'
            },
            'bone_meal': {
                'type': 'Organic',
                'npk_ratio': '4-20-0',  # Approximate
                'common_names': ['Bone Meal'],
                'uses': ['Phosphorus source', 'Flowering plants', 'Root crops'],
                'price_per_kg': 40,
                'application_rate': '50-100 kg/acre'
            }
        }
        
        # Crop-specific fertilizer requirements (kg/acre)
        self.crop_requirements = {
            'rice': {'N': 80, 'P': 40, 'K': 40, 'remarks': 'Split N application - 50% basal, 25% tillering, 25% panicle'},
            'wheat': {'N': 60, 'P': 30, 'K': 30, 'remarks': 'Apply full P and K at sowing, N split application'},
            'maize': {'N': 100, 'P': 50, 'K': 50, 'remarks': 'N in 2-3 splits, P and K as basal'},
            'cotton': {'N': 80, 'P': 40, 'K': 40, 'remarks': 'N in 3 splits, K for boll development'},
            'coffee': {'N': 40, 'P': 20, 'K': 60, 'remarks': 'Apply during monsoon in 2 splits'},
            'sugarcane': {'N': 120, 'P': 60, 'K': 80, 'remarks': 'N in 3 splits, high K requirement'},
            'tomato': {'N': 60, 'P': 40, 'K': 60, 'remarks': 'High K for fruit quality'},
            'potato': {'N': 80, 'P': 50, 'K': 100, 'remarks': 'High K requirement, avoid excess N'},
            'onion': {'N': 40, 'P': 30, 'K': 40, 'remarks': 'Balanced nutrition, moderate N'},
            'chilli': {'N': 50, 'P': 30, 'K': 50, 'remarks': 'N in splits, K for fruit setting'},
            'brinjal': {'N': 60, 'P': 40, 'K': 60, 'remarks': 'Continuous feeder, regular fertilization'},
            'cabbage': {'N': 70, 'P': 35, 'K': 70, 'remarks': 'Heavy feeder, high N and K'},
            'cauliflower': {'N': 80, 'P': 40, 'K': 80, 'remarks': 'High N for curd development'},
            'apple': {'N': 40, 'P': 20, 'K': 60, 'remarks': 'Fruit trees need balanced nutrition'},
            'banana': {'N': 100, 'P': 50, 'K': 200, 'remarks': 'Very high K requirement'},
            'mango': {'N': 60, 'P': 30, 'K': 80, 'remarks': 'Fruit quality depends on K'},
            'citrus': {'N': 40, 'P': 20, 'K': 50, 'remarks': 'Micronutrients important'},
            'grapes': {'N': 50, 'P': 30, 'K': 80, 'remarks': 'High K for fruit quality'},
            'default': {'N': 60, 'P': 30, 'K': 40, 'remarks': 'General recommendation'}
        }
        
        # Ideal soil nutrient ranges (kg/ha)
        self.ideal_ranges = {
            'N': {'low': 0, 'medium': 25, 'high': 50, 'very_high': 75},
            'P': {'low': 0, 'medium': 15, 'high': 30, 'very_high': 45},
            'K': {'low': 0, 'medium': 20, 'high': 40, 'very_high': 60}
        }
        
        # Soil health class based recommendations
        self.soil_health_recommendations = {
            0: {  # Poor soil
                'priority': 'Soil improvement',
                'organic_matter': 'High (5-10 tons/acre)',
                'chemical_fertilizers': 'Start with 50% of recommended dose',
                'focus': 'Build soil structure first',
                'duration': '1-2 seasons for improvement'
            },
            1: {  # Fair soil
                'priority': 'Balanced nutrition',
                'organic_matter': 'Moderate (3-5 tons/acre)',
                'chemical_fertilizers': '75-100% of recommended dose',
                'focus': 'Maintain and improve fertility',
                'duration': 'Regular maintenance'
            },
            2: {  # Good soil
                'priority': 'Maintenance',
                'organic_matter': 'Low (1-2 tons/acre)',
                'chemical_fertilizers': '50-75% of recommended dose',
                'focus': 'Prevent depletion, target specific deficiencies',
                'duration': 'Sustainable management'
            }
        }
        
        print("✅ Fertilizer Recommender initialized")
    
    def get_crop_requirement(self, crop_name):
        """Get fertilizer requirements for specific crop"""
        if not crop_name:
            return self.crop_requirements['default']
        
        crop_lower = crop_name.lower()
        
        # Try exact match first
        if crop_lower in self.crop_requirements:
            return self.crop_requirements[crop_lower]
        
        # Try partial match
        for crop_key in self.crop_requirements:
            if crop_key in crop_lower:
                return self.crop_requirements[crop_key]
        
        return self.crop_requirements['default']
    
    def analyze_nutrient_status(self, nutrient_values):
        """Analyze nutrient status and classify"""
        status = {'deficient': [], 'adequate': [], 'excessive': []}
        details = {}
        
        for nutrient, value in nutrient_values.items():
            if nutrient in self.ideal_ranges:
                ranges = self.ideal_ranges[nutrient]
                
                if value < ranges['medium']:
                    status['deficient'].append(nutrient)
                    level = 'low'
                elif value < ranges['high']:
                    status['adequate'].append(nutrient)
                    level = 'medium'
                elif value < ranges['very_high']:
                    status['adequate'].append(nutrient)
                    level = 'high'
                else:
                    status['excessive'].append(nutrient)
                    level = 'very_high'
                
                details[nutrient] = {
                    'value': value,
                    'level': level,
                    'ideal_range': f"{ranges['medium']}-{ranges['high']} kg/ha",
                    'recommendation': self.get_nutrient_recommendation(nutrient, value)
                }
        
        return status, details
    
    def get_nutrient_recommendation(self, nutrient, value):
        """Get specific recommendation for nutrient level"""
        if nutrient == 'N':
            if value < 25:
                return "Apply nitrogen fertilizer immediately"
            elif value < 50:
                return "Apply nitrogen at 50-75% of recommended dose"
            else:
                return "Nitrogen sufficient, no application needed"
        
        elif nutrient == 'P':
            if value < 15:
                return "Apply phosphorus fertilizer before planting"
            elif value < 30:
                return "Apply phosphorus at maintenance dose"
            else:
                return "Phosphorus sufficient"
        
        elif nutrient == 'K':
            if value < 20:
                return "Apply potassium fertilizer in split doses"
            elif value < 40:
                return "Apply potassium at maintenance dose"
            else:
                return "Potassium sufficient"
        
        return "Monitor regularly"
    
    def calculate_fertilizer_for_deficiency(self, nutrient, current_value, target_value, area_hectares):
        """Calculate fertilizer amount for specific deficiency"""
        recommendations = []
        
        # Convert to kg/hectare (simplified)
        deficit = max(0, target_value - current_value)
        
        if deficit <= 0:
            return recommendations
        
        # Select appropriate fertilizers
        if nutrient == 'N':
            fertilizers = ['urea', 'npk_20_20_20']
            efficiency = [0.46, 0.20]  # N content
        elif nutrient == 'P':
            fertilizers = ['dap', 'ssp', 'bone_meal']
            efficiency = [0.46, 0.16, 0.20]
        elif nutrient == 'K':
            fertilizers = ['mop', 'npk_20_20_20']
            efficiency = [0.60, 0.20]
        else:
            fertilizers = ['npk_20_20_20']
            efficiency = [0.20]
        
        for fert, eff in zip(fertilizers, efficiency):
            amount_kg = (deficit / eff) * area_hectares
            
            if amount_kg > 0:
                recommendations.append({
                    'fertilizer': fert,
                    'fertilizer_info': self.fertilizers[fert],
                    'amount_kg': round(amount_kg, 2),
                    'amount_per_acre': round(amount_kg * 0.4, 2),  # Convert to acres
                    'nutrient_target': f"Increase {nutrient} by {deficit:.1f} kg/ha",
                    'application_timing': self.get_application_timing(nutrient, amount_kg),
                    'estimated_cost': round(amount_kg * self.fertilizers[fert].get('price_per_kg', 30), 2)
                })
        
        return recommendations
    
    def get_application_timing(self, nutrient, amount_kg):
        """Get application timing based on nutrient and amount"""
        if nutrient == 'N':
            if amount_kg > 50:
                return "Split application: 50% basal, 25% at 30 days, 25% at 60 days"
            else:
                return "Apply 2/3 basal, 1/3 top dressing at 30 days"
        
        elif nutrient == 'P':
            return "Apply full dose as basal before planting, incorporate into soil"
        
        elif nutrient == 'K':
            if amount_kg > 40:
                return "Split: 50% basal, 50% at flowering/fruit setting"
            else:
                return "Apply full dose as basal"
        
        return "Apply as per crop growth stage"
    
    def get_soil_health_advice(self, soil_health_class):
        """Get advice based on soil health class"""
        if soil_health_class in self.soil_health_recommendations:
            advice = self.soil_health_recommendations[soil_health_class]
            
            return {
                'priority': advice['priority'],
                'organic_matter_recommendation': advice['organic_matter'],
                'chemical_fertilizer_strategy': advice['chemical_fertilizers'],
                'key_focus': advice['focus'],
                'improvement_duration': advice['duration'],
                'specific_actions': self.get_specific_actions(soil_health_class)
            }
        
        return {
            'priority': 'General improvement',
            'organic_matter_recommendation': '3-5 tons/acre',
            'chemical_fertilizer_strategy': '75% of recommended dose',
            'key_focus': 'Balanced nutrition',
            'improvement_duration': 'Regular monitoring'
        }
    
    def get_specific_actions(self, soil_health_class):
        """Get specific actions for soil health class"""
        actions = []
        
        if soil_health_class == 0:  # Poor
            actions = [
                "Add 5-10 tons/acre of well-decomposed organic manure",
                "Use green manure crops (sunhemp, dhaincha)",
                "Apply lime if pH < 6.0",
                "Start with 50% of recommended fertilizer dose",
                "Practice crop rotation with legumes",
                "Add bio-fertilizers (Rhizobium, Azotobacter)"
            ]
        elif soil_health_class == 1:  # Fair
            actions = [
                "Add 3-5 tons/acre of organic manure",
                "Apply 75-100% of recommended fertilizer dose",
                "Practice balanced fertilization",
                "Monitor soil nutrients regularly",
                "Use crop residues as mulch",
                "Consider cover cropping"
            ]
        elif soil_health_class == 2:  # Good
            actions = [
                "Add 1-2 tons/acre of organic manure for maintenance",
                "Apply 50-75% of recommended fertilizer dose",
                "Practice precision farming",
                "Regular soil testing (once a year)",
                "Maintain soil organic matter",
                "Avoid over-fertilization"
            ]
        
        return actions
    
    def recommend_organic_amendments(self, soil_health_class, area_hectares):
        """Recommend organic amendments"""
        recommendations = []
        
        if soil_health_class == 0:  # Poor
            recommendations.append({
                'amendment': 'organic_manure',
                'amount_kg': round(10000 * area_hectares, 2),  # 10 tons/ha
                'purpose': 'Improve soil structure and organic matter',
                'timing': 'Apply 2-3 weeks before planting',
                'method': 'Spread evenly and incorporate into top 15-20 cm soil'
            })
            recommendations.append({
                'amendment': 'vermicompost',
                'amount_kg': round(2000 * area_hectares, 2),  # 2 tons/ha
                'purpose': 'Add beneficial microbes',
                'timing': 'Apply at planting time',
                'method': 'Mix with soil in planting holes or rows'
            })
        
        elif soil_health_class == 1:  # Fair
            recommendations.append({
                'amendment': 'organic_manure',
                'amount_kg': round(5000 * area_hectares, 2),  # 5 tons/ha
                'purpose': 'Maintain soil organic matter',
                'timing': 'Apply before planting season',
                'method': 'Spread and incorporate into soil'
            })
        
        else:  # Good
            recommendations.append({
                'amendment': 'organic_manure',
                'amount_kg': round(2000 * area_hectares, 2),  # 2 tons/ha
                'purpose': 'Maintenance application',
                'timing': 'Once a year',
                'method': 'Top dressing or incorporation'
            })
        
        return recommendations
    
    def recommend(self, soil_health_class, crop_name=None, nutrient_values=None, area_hectares=1):
        """
        Main recommendation function
        
        Parameters:
        -----------
        soil_health_class: int (0=Poor, 1=Fair, 2=Good)
        crop_name: str (optional) - Name of crop to be grown
        nutrient_values: dict (optional) - {'N': value, 'P': value, 'K': value}
        area_hectares: float - Area in hectares (default: 1)
        
        Returns:
        --------
        dict: Complete fertilizer recommendation
        """
        try:
            # Convert area to acres for Indian context
            area_acres = area_hectares * 2.471
            
            # Initialize result structure
            result = {
                'success': True,
                'soil_health': {
                    'class': soil_health_class,
                    'label': ['Poor', 'Fair', 'Good'][soil_health_class] if soil_health_class in [0,1,2] else 'Unknown',
                    'advice': self.get_soil_health_advice(soil_health_class)
                },
                'crop_info': {
                    'name': crop_name,
                    'requirements': {}
                },
                'nutrient_analysis': {
                    'status': {},
                    'details': {}
                },
                'recommendations': {
                    'organic': [],
                    'chemical': [],
                    'general': []
                },
                'application_schedule': [],
                'cost_estimate': {
                    'organic': 0,
                    'chemical': 0,
                    'total': 0
                },
                'area': {
                    'hectares': area_hectares,
                    'acres': round(area_acres, 2)
                },
                'important_notes': []
            }
            
            # 1. Get crop requirements if crop specified
            if crop_name:
                crop_req = self.get_crop_requirement(crop_name)
                result['crop_info']['requirements'] = crop_req
                result['crop_info']['remarks'] = crop_req.get('remarks', 'General cultivation practices')
            
            # 2. Analyze nutrient status if values provided
            if nutrient_values and all(n in nutrient_values for n in ['N', 'P', 'K']):
                status, details = self.analyze_nutrient_status(nutrient_values)
                result['nutrient_analysis']['status'] = status
                result['nutrient_analysis']['details'] = details
                
                # Calculate fertilizer for deficiencies
                if crop_name and 'deficient' in status:
                    for nutrient in status['deficient']:
                        current_val = nutrient_values[nutrient]
                        target_val = crop_req.get(nutrient, self.crop_requirements['default'][nutrient])
                        
                        fert_recs = self.calculate_fertilizer_for_deficiency(
                            nutrient, current_val, target_val, area_hectares
                        )
                        result['recommendations']['chemical'].extend(fert_recs)
            
            # 3. Add organic amendments based on soil health
            organic_recs = self.recommend_organic_amendments(soil_health_class, area_hectares)
            result['recommendations']['organic'] = organic_recs
            
            # 4. General recommendations based on soil health
            if soil_health_class == 0:  # Poor
                result['recommendations']['general'].append({
                    'type': 'complex_fertilizer',
                    'product': 'npk_10_26_26',
                    'amount_kg': round(150 * area_hectares, 2),
                    'purpose': 'Balanced nutrition for soil improvement',
                    'timing': 'Apply as basal dose'
                })
            elif soil_health_class == 1:  # Fair
                result['recommendations']['general'].append({
                    'type': 'complex_fertilizer',
                    'product': 'npk_12_32_16',
                    'amount_kg': round(100 * area_hectares, 2),
                    'purpose': 'Maintain soil fertility',
                    'timing': 'Apply as basal dose'
                })
            
            # 5. Create application schedule
            schedule = self.create_application_schedule(result)
            result['application_schedule'] = schedule
            
            # 6. Calculate cost estimates
            total_cost = 0
            
            # Organic cost
            org_cost = 0
            for org in result['recommendations']['organic']:
                amount = org.get('amount_kg', 0)
                price = self.fertilizers.get(org['amendment'], {}).get('price_per_kg', 0)
                org_cost += amount * price
            
            # Chemical cost
            chem_cost = 0
            for chem in result['recommendations']['chemical']:
                chem_cost += chem.get('estimated_cost', 0)
            
            total_cost = org_cost + chem_cost
            
            result['cost_estimate'] = {
                'organic': round(org_cost, 2),
                'chemical': round(chem_cost, 2),
                'total': round(total_cost, 2),
                'currency': 'INR (₹)'
            }
            
            # 7. Add important notes
            result['important_notes'] = [
                "Always conduct soil test before major fertilizer application",
                "Adjust recommendations based on local conditions and weather",
                "Split nitrogen applications for better efficiency",
                "Incorporate organic matter regularly for long-term soil health",
                "Monitor crop response and adjust accordingly"
            ]
            
            # Add success message
            result['message'] = self.get_success_message(soil_health_class, crop_name)
            
            return result
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': f"Fertilizer recommendation failed: {str(e)}"
            }
    
    def create_application_schedule(self, result):
        """Create detailed application schedule"""
        schedule = []
        area_hectares = result['area']['hectares']
        
        # Week 1: Organic amendments
        if result['recommendations']['organic']:
            org_names = []
            for org in result['recommendations']['organic']:
                org_names.append(self.fertilizers[org['amendment']]['common_names'][0])
            
            schedule.append({
                'week': 1,
                'activity': f"Apply {' and '.join(org_names)}",
                'details': "Incorporate into top 15-20 cm of soil",
                'purpose': "Improve soil structure and organic matter"
            })
        
        # Week 2-3: Basal fertilizers
        basal_ferts = []
        for chem in result['recommendations']['chemical']:
            if 'basal' in chem['application_timing'].lower():
                basal_ferts.append(chem['fertilizer_info']['common_names'][0])
        
        if basal_ferts or result['recommendations']['general']:
            week = 2 if result['recommendations']['organic'] else 1
            fert_list = basal_ferts + [g['product'] for g in result['recommendations']['general']]
            schedule.append({
                'week': week,
                'activity': f"Apply {' and '.join(fert_list)} as basal dose",
                'details': "Broadcast and incorporate before planting",
                'purpose': "Provide balanced nutrition"
            })
        
        # Week 4-6: Top dressing if needed
        top_dress = []
        for chem in result['recommendations']['chemical']:
            if 'top dressing' in chem['application_timing'].lower() or 'split' in chem['application_timing'].lower():
                top_dress.append(chem['fertilizer_info']['common_names'][0])
        
        if top_dress:
            schedule.append({
                'week': 4,
                'activity': f"Apply {' and '.join(top_dress)} as top dressing",
                'details': "Apply near root zone and irrigate",
                'purpose': "Supplement nutrition during growth"
            })
        
        # Add crop-specific timing if crop specified
        if result['crop_info']['name']:
            crop = result['crop_info']['name'].lower()
            if crop == 'rice':
                schedule.append({
                    'week': 8,
                    'activity': "Second top dressing of nitrogen",
                    'details': "Apply at panicle initiation stage",
                    'purpose': "Enhance grain formation"
                })
            elif crop == 'wheat':
                schedule.append({
                    'week': 6,
                    'activity': "Top dressing of nitrogen",
                    'details': "Apply at crown root initiation stage",
                    'purpose': "Promote tillering"
                })
        
        return schedule
    
    def get_success_message(self, soil_health_class, crop_name):
        """Generate success message"""
        messages = {
            0: f"⚠️ Soil needs significant improvement. Focus on building soil health first.",
            1: f"✅ Soil condition is fair. With proper management, good yields can be achieved.",
            2: f"🎉 Excellent! Soil is in good condition. Maintain with minimal inputs."
        }
        
        base_msg = messages.get(soil_health_class, "Follow recommendations for optimal results.")
        
        if crop_name:
            base_msg += f" For {crop_name}, adjust based on specific crop requirements."
        
        return base_msg

# Initialize fertilizer recommender
fertilizer_recommender = FertilizerRecommender()

print("\n" + "="*60)
print("FERTILIZER RECOMMENDATION MODULE READY")
print("="*60)
print(f"✓ {len(fertilizer_recommender.fertilizers)} fertilizers in database")
print(f"✓ {len(fertilizer_recommender.crop_requirements)} crop-specific recommendations")
print(f"✓ Integrated with soil health classes (Poor, Fair, Good)")
print("="*60)