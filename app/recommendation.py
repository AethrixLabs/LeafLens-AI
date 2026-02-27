from datetime import datetime, timedelta

eco_recommendations = {

    # ---------------- RICE ----------------
    "blast": {
        "crop": "Rice",
        "remedy": "Neem oil spray 5ml per liter of water",
        "frequency": "Every 7 days",
        "preventive": "Maintain proper drainage and avoid excess nitrogen fertilizer",
        "alert_days": 7,
        "sustainability_score": 9.5
    },

    "bacterial_blight": {
        "crop": "Rice",
        "remedy": "Apply Pseudomonas fluorescens bio-agent",
        "frequency": "Every 10 days",
        "preventive": "Use disease-free seeds and avoid water stagnation",
        "alert_days": 10,
        "sustainability_score": 9.3
    },

    "brown_spot": {
        "crop": "Rice",
        "remedy": "Spray compost tea or neem-based fungicide",
        "frequency": "Every 8 days",
        "preventive": "Balanced fertilization and proper spacing",
        "alert_days": 8,
        "sustainability_score": 9.0
    },

    "tungro": {
        "crop": "Rice",
        "remedy": "Control leafhoppers using neem-based insecticide",
        "frequency": "Every 6 days",
        "preventive": "Remove infected plants and control vector population",
        "alert_days": 6,
        "sustainability_score": 8.8
    },

    # ---------------- WHEAT ----------------
    "black_rust": {
        "crop": "Wheat",
        "remedy": "Apply Trichoderma bio-fungicide",
        "frequency": "Every 10 days",
        "preventive": "Remove infected debris and ensure proper air circulation",
        "alert_days": 10,
        "sustainability_score": 9.2
    },

    "brown_rust": {
        "crop": "Wheat",
        "remedy": "Use Bacillus subtilis bio-fungicide",
        "frequency": "Every 9 days",
        "preventive": "Plant resistant varieties and monitor humidity",
        "alert_days": 9,
        "sustainability_score": 9.1
    },

    "yellow_rust": {
        "crop": "Wheat",
        "remedy": "Neem oil spray 4ml per liter",
        "frequency": "Every 7 days",
        "preventive": "Avoid dense planting and remove infected leaves",
        "alert_days": 7,
        "sustainability_score": 9.0
    },

    "healthy": {
        "crop": "Wheat",
        "remedy": "No treatment required",
        "frequency": "Monitor weekly",
        "preventive": "Maintain balanced fertilization and irrigation",
        "alert_days": 14,
        "sustainability_score": 10
    },

    # ---------------- CORN ----------------
    "cercospora_leaf_spot": {
        "crop": "Corn",
        "remedy": "Spray neem extract or copper-based bio-fungicide",
        "frequency": "Every 8 days",
        "preventive": "Crop rotation and removal of infected leaves",
        "alert_days": 8,
        "sustainability_score": 9.1
    },

    "common_rust": {
        "crop": "Corn",
        "remedy": "Apply Bacillus-based bio-fungicide",
        "frequency": "Every 9 days",
        "preventive": "Ensure proper plant spacing and field sanitation",
        "alert_days": 9,
        "sustainability_score": 9.0
    },

    "northern_leaf_blight": {
        "crop": "Corn",
        "remedy": "Use Trichoderma-based organic fungicide",
        "frequency": "Every 7 days",
        "preventive": "Avoid overhead irrigation and rotate crops",
        "alert_days": 7,
        "sustainability_score": 9.2
    },

    "healthy": {
        "crop": "Corn",
        "remedy": "No treatment required",
        "frequency": "Monitor every 14 days",
        "preventive": "Maintain soil health and proper irrigation",
        "alert_days": 14,
        "sustainability_score": 10
    }
}


def get_recommendation(disease: str):
    disease = disease.lower()

    if disease not in eco_recommendations:
        return {
            "status": "error",
            "message": "No eco-friendly recommendation found."
        }

    data = eco_recommendations[disease]
    next_alert = datetime.now() + timedelta(days=data["alert_days"])

    return {
        "status": "success",
        "crop": data["crop"],
        "disease": disease,
        "eco_remedy": data["remedy"],
        "application_frequency": data["frequency"],
        "preventive_measures": data["preventive"],
        "sustainability_score": data["sustainability_score"],
        "next_alert_date": next_alert.strftime("%Y-%m-%d")
    }