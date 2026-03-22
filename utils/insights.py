"""
============================================
Antigravity - Health Insights Module
============================================
Provides:
  1. BMI calculation + WHO classification
  2. Personalized health insights based on BMI/age/gender
  3. Ideal weight range calculator
  4. Mental health scoring (PHQ-4 style)
============================================
"""


# ============================================
# 1. BMI CALCULATOR
# ============================================

def calculate_bmi(weight_kg: float, height_cm: float) -> float:
    """
    Calculate Body Mass Index (BMI).
    
    Formula: BMI = weight(kg) / height(m)^2
    
    Args:
        weight_kg: Weight in kilograms
        height_cm: Height in centimeters
        
    Returns:
        float: BMI value rounded to 1 decimal
    """
    height_m = height_cm / 100.0
    if height_m <= 0:
        return 0.0
    return round(weight_kg / (height_m ** 2), 1)


def get_bmi_category(bmi: float) -> str:
    """
    Classify BMI according to WHO standards.
    
    Args:
        bmi: BMI value
        
    Returns:
        str: WHO category string
    """
    if bmi < 16:
        return "Severe Thinness"
    elif bmi < 17:
        return "Moderate Thinness"
    elif bmi < 18.5:
        return "Mild Thinness"
    elif bmi < 25:
        return "Normal weight"
    elif bmi < 30:
        return "Overweight"
    elif bmi < 35:
        return "Obese Class I"
    elif bmi < 40:
        return "Obese Class II"
    else:
        return "Obese Class III"


def get_bmi_color(category: str) -> str:
    """
    Return a color hex code for the BMI category.
    
    Args:
        category: WHO BMI category string
        
    Returns:
        str: Hex color code
    """
    color_map = {
        "Severe Thinness": "#e74c3c",
        "Moderate Thinness": "#e67e22",
        "Mild Thinness": "#f1c40f",
        "Normal weight": "#2ecc71",
        "Overweight": "#f39c12",
        "Obese Class I": "#e74c3c",
        "Obese Class II": "#c0392b",
        "Obese Class III": "#8e44ad",
    }
    return color_map.get(category, "#95a5a6")


def calculate_ideal_weight_range(height_cm: float) -> tuple:
    """
    Calculate the ideal weight range (for BMI 18.5–24.9).
    
    Args:
        height_cm: Height in centimeters
        
    Returns:
        tuple: (min_weight_kg, max_weight_kg) rounded to 1 decimal
    """
    height_m = height_cm / 100.0
    min_weight = 18.5 * (height_m ** 2)
    max_weight = 24.9 * (height_m ** 2)
    return (round(min_weight, 1), round(max_weight, 1))


def get_bmi_insights(bmi: float, category: str, age: int, gender: str) -> list:
    """
    Generate personalized health insights based on BMI, age, and gender.
    
    Args:
        bmi: Calculated BMI value
        category: WHO BMI category
        age: Patient age in years
        gender: "Male" or "Female"
        
    Returns:
        list: List of insight strings (3-5 insights)
    """
    insights = []
    
    # ── BMI-based insights ──
    if "Thinness" in category:
        insights.append(
            f"Your BMI of {bmi} indicates you are underweight. "
            "Consider consulting a nutritionist to ensure adequate calorie and nutrient intake."
        )
        insights.append(
            "Underweight individuals may be at higher risk for weakened immune function, "
            "osteoporosis, and fertility issues."
        )
    elif category == "Normal weight":
        insights.append(
            f"Great news! Your BMI of {bmi} is within the healthy range (18.5–24.9). "
            "Maintain your current lifestyle with balanced nutrition and regular exercise."
        )
    elif category == "Overweight":
        insights.append(
            f"Your BMI of {bmi} puts you in the overweight category. "
            "Even a 5–10% reduction in body weight can significantly reduce health risks."
        )
        insights.append(
            "Consider 150 minutes of moderate-intensity aerobic activity per week, "
            "such as brisk walking, swimming, or cycling."
        )
    elif "Obese" in category:
        insights.append(
            f"Your BMI of {bmi} indicates obesity, which increases risk of heart disease, "
            "type 2 diabetes, and certain cancers. Please consult a healthcare professional."
        )
        insights.append(
            "A combination of dietary changes, regular physical activity, and behavioral "
            "strategies is recommended. Medical supervision is advised for significant weight loss."
        )
    
    # ── Age-based insights ──
    if age >= 50:
        insights.append(
            "At your age, regular health screenings (cholesterol, blood pressure, blood sugar) "
            "are especially important. Consider a comprehensive health check-up annually."
        )
    elif age < 25:
        insights.append(
            "Building healthy habits now can have lifelong benefits. Focus on regular exercise, "
            "balanced nutrition, adequate sleep (7–9 hours), and stress management."
        )
    
    # ── Gender-based insights ──
    if gender == "Female" and age >= 40:
        insights.append(
            "Women over 40 should pay extra attention to bone health (calcium + vitamin D), "
            "cardiovascular health, and breast cancer screening."
        )
    elif gender == "Male" and age >= 45:
        insights.append(
            "Men over 45 have increased cardiovascular risk. Regular monitoring of blood pressure, "
            "cholesterol, and blood sugar levels is strongly recommended."
        )
    
    # ── Universal insight ──
    insights.append(
        "Stay hydrated (aim for 2–3 liters of water daily), prioritize mental well-being, "
        "and ensure you're getting 7–9 hours of quality sleep."
    )
    
    return insights


# ============================================
# 2. MENTAL HEALTH SCORING (PHQ-4 STYLE)
# ============================================

# PHQ-4 Questions (2 GAD-2 for anxiety + 2 PHQ-2 for depression + 4 additional)
PHQ4_QUESTIONS = [
    # GAD-2 (Anxiety)
    "Over the last 2 weeks, how often have you been bothered by feeling nervous, anxious, or on edge?",
    "Over the last 2 weeks, how often have you been unable to stop or control worrying?",
    "Over the last 2 weeks, how often have you felt restless or found it hard to sit still?",
    "Over the last 2 weeks, how often have you become easily annoyed or irritable?",
    # PHQ-2 (Depression)
    "Over the last 2 weeks, how often have you had little interest or pleasure in doing things?",
    "Over the last 2 weeks, how often have you been feeling down, depressed, or hopeless?",
    "Over the last 2 weeks, how often have you had trouble falling asleep, staying asleep, or sleeping too much?",
    "Over the last 2 weeks, how often have you felt tired or had little energy?",
]

# Response options and scores
PHQ4_OPTIONS = {
    "Not at all": 0,
    "Several days": 1,
    "More than half the days": 2,
    "Nearly every day": 3,
}


def calculate_mental_health_score(answers: list) -> dict:
    """
    Calculate mental health scores from PHQ-4 style questionnaire.
    
    Args:
        answers: List of 8 answer strings (from PHQ4_OPTIONS keys)
        
    Returns:
        dict with keys:
            - total_score (0-24 mapped to 0-12 display)
            - anxiety_score (0-12 mapped to 0-6 display)
            - depression_score (0-12 mapped to 0-6 display)
            - category: "Normal" / "Mild" / "Moderate" / "Severe"
            - recommendations: list of recommendation strings
    """
    # Calculate raw scores
    scores = [PHQ4_OPTIONS.get(a, 0) for a in answers]
    
    anxiety_raw = sum(scores[0:4])     # Questions 1-4 (anxiety)
    depression_raw = sum(scores[4:8])  # Questions 5-8 (depression)
    total_raw = anxiety_raw + depression_raw
    
    # Scale to standard PHQ-4 ranges (0-6 per subscale, 0-12 total)
    anxiety_score = min(round(anxiety_raw / 2), 6)
    depression_score = min(round(depression_raw / 2), 6)
    total_score = anxiety_score + depression_score
    
    # Determine severity category
    if total_score <= 2:
        category = "Normal"
    elif total_score <= 5:
        category = "Mild"
    elif total_score <= 8:
        category = "Moderate"
    else:
        category = "Severe"
    
    # Generate recommendations based on severity
    recommendations = _get_mental_health_recommendations(
        category, anxiety_score, depression_score
    )
    
    return {
        "total_score": total_score,
        "anxiety_score": anxiety_score,
        "depression_score": depression_score,
        "category": category,
        "recommendations": recommendations,
    }


def _get_mental_health_recommendations(category: str, anxiety: int, depression: int) -> list:
    """
    Generate recommendations based on mental health screening results.
    
    Args:
        category: Severity category
        anxiety: Anxiety subscale score
        depression: Depression subscale score
        
    Returns:
        list: Personalized recommendation strings
    """
    recs = []
    
    if category == "Normal":
        recs.append(
            "Your screening results suggest no significant symptoms. "
            "Continue maintaining your mental wellness through regular self-care."
        )
        recs.append(
            "Practice stress management: mindfulness, deep breathing, or journaling "
            "can help maintain good mental health."
        )
    
    elif category == "Mild":
        recs.append(
            "Your results suggest mild symptoms. While these may be situational, "
            "consider monitoring how you feel over the next few weeks."
        )
        recs.append(
            "Self-care strategies such as regular exercise, maintaining social connections, "
            "and adequate sleep can be very effective."
        )
        if anxiety > depression:
            recs.append(
                "Your anxiety symptoms are slightly elevated. Try relaxation techniques "
                "like progressive muscle relaxation (PMR) or guided meditation."
            )
        else:
            recs.append(
                "Your mood symptoms are slightly elevated. Engaging in activities you "
                "enjoy and staying physically active can help improve your mood."
            )
    
    elif category == "Moderate":
        recs.append(
            "Your results indicate moderate symptoms. Consider talking to a mental "
            "health professional — a counselor, therapist, or your primary care doctor."
        )
        recs.append(
            "Evidence-based treatments like Cognitive Behavioral Therapy (CBT) "
            "are highly effective for these symptoms."
        )
        recs.append(
            "In the meantime, maintain a routine, stay connected with people you trust, "
            "and limit alcohol or substance use."
        )
    
    elif category == "Severe":
        recs.append(
            "Your results suggest significant symptoms. Please consider reaching out to "
            "a mental health professional or your doctor as soon as possible."
        )
        recs.append(
            "If you are in crisis or having thoughts of self-harm, please contact "
            "your local emergency services or a crisis helpline immediately."
        )
        recs.append(
            "You are not alone. Effective treatments are available, and reaching out "
            "for help is a sign of strength, not weakness."
        )
    
    # Universal recommendation
    recs.append(
        "Remember: This is a brief screening tool, not a clinical diagnosis. "
        "Only a qualified professional can provide a proper assessment."
    )
    
    return recs


def get_mental_health_color(category: str) -> str:
    """Return a color hex code for the mental health category."""
    colors = {
        "Normal": "#2ecc71",
        "Mild": "#3498db",
        "Moderate": "#f39c12",
        "Severe": "#e74c3c",
    }
    return colors.get(category, "#95a5a6")
