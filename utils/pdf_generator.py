"""
============================================
VitaScan AI - PDF Report Generator
============================================
Generates downloadable PDF reports for:
  - Heart Disease Risk Assessment
  - BMI & Health Insights
  - Mental Health Screening
============================================
"""

import io
from datetime import datetime
from fpdf import FPDF


class VitaScanPDF(FPDF):
    """Custom PDF class with VitaScan AI branding."""

    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=20)

    @staticmethod
    def sanitize(text: str) -> str:
        """Replace non-latin1 characters with ASCII equivalents for FPDF."""
        replacements = {
            '\u2014': '-', '\u2013': '-',   # em-dash, en-dash
            '\u2018': "'", '\u2019': "'",   # smart quotes
            '\u201c': '"', '\u201d': '"',   # smart double quotes
            '\u2026': '...', '\u2022': '-', # ellipsis, bullet
            '\u00b7': '-',                   # middle dot
            '\u2192': '->', '\u2190': '<-',  # arrows
            '\u2265': '>=', '\u2264': '<=',  # >= <=
            '\u00b0': ' deg',                # degree sign
        }
        for char, replacement in replacements.items():
            text = text.replace(char, replacement)
        # Strip any remaining non-latin1 characters
        return text.encode('latin-1', errors='replace').decode('latin-1')

    def header(self):
        """Branded header on every page."""
        self.set_fill_color(15, 76, 117)
        self.rect(0, 0, 210, 22, "F")

        self.set_font("Helvetica", "B", 14)
        self.set_text_color(255, 255, 255)
        self.cell(0, 12, "VITASCAN AI - Smart Health Screening", align="C",
                  new_x="LMARGIN", new_y="NEXT")

        self.set_font("Helvetica", "I", 8)
        self.set_text_color(200, 220, 240)
        self.cell(0, 6, "Smart health screening, accessible to all.", align="C",
                  new_x="LMARGIN", new_y="NEXT")

        self.ln(8)

    def footer(self):
        """Footer with disclaimer and page number."""
        self.set_y(-20)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(150, 150, 150)
        self.cell(0, 5,
                  "DISCLAIMER: This is a screening tool for educational purposes only. "
                  "Consult a licensed healthcare professional.",
                  align="C", new_x="LMARGIN", new_y="NEXT")
        self.cell(0, 5,
                  f"Page {self.page_no()} | Generated on "
                  f"{datetime.now().strftime('%Y-%m-%d %H:%M')}",
                  align="C")

    def section_title(self, title):
        """Add a styled section title."""
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(15, 76, 117)
        self.cell(0, 10, self.sanitize(title), new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(15, 76, 117)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def key_value(self, key, value):
        """Add a key-value pair."""
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(60, 60, 60)
        self.cell(60, 7, self.sanitize(f"{key}:"), new_x="RIGHT")
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.cell(0, 7, self.sanitize(str(value)), new_x="LMARGIN", new_y="NEXT")

    def body_text(self, text):
        """Add a paragraph of body text."""
        self.set_font("Helvetica", "", 10)
        self.set_text_color(50, 50, 50)
        self.multi_cell(0, 6, self.sanitize(text))
        self.ln(3)

    def highlight_box(self, text, color="blue"):
        """Add a highlighted result box."""
        colors = {
            "blue": (15, 76, 117),
            "green": (46, 204, 113),
            "orange": (243, 156, 18),
            "red": (231, 76, 60),
        }
        r, g, b = colors.get(color, (15, 76, 117))

        self.set_fill_color(r, g, b)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 12, f"  {self.sanitize(text)}", fill=True, new_x="LMARGIN", new_y="NEXT")
        self.ln(5)
        self.set_text_color(0, 0, 0)


def generate_heart_report(patient_data: dict, risk_probability: float,
                          risk_category: str, top_factors: list) -> bytes:
    """Generate a PDF report for Heart Disease Risk Assessment."""
    pdf = VitaScanPDF()
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(15, 76, 117)
    pdf.cell(0, 15, "Heart Disease Risk Assessment Report", align="C",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    color_map = {"Low Risk": "green", "Medium Risk": "orange", "High Risk": "red"}
    pdf.highlight_box(
        f"RESULT: {risk_category} ({risk_probability:.1%} probability)",
        color=color_map.get(risk_category, "blue")
    )

    pdf.section_title("Patient Information")
    label_map = {
        "age": "Age", "sex": "Sex", "cp": "Chest Pain Type",
        "trestbps": "Resting BP (mmHg)", "chol": "Cholesterol (mg/dl)",
        "fbs": "Fasting Blood Sugar > 120", "restecg": "Resting ECG",
        "thalach": "Max Heart Rate", "exang": "Exercise Angina",
        "oldpeak": "ST Depression", "slope": "ST Slope",
        "ca": "Major Vessels", "thal": "Thalassemia",
    }
    for key, label in label_map.items():
        if key in patient_data:
            val = patient_data[key]
            if key == "sex":
                val = "Male" if val == 1 else "Female"
            elif key == "fbs":
                val = "Yes" if val == 1 else "No"
            elif key == "exang":
                val = "Yes" if val == 1 else "No"
            pdf.key_value(label, val)

    pdf.ln(5)

    pdf.section_title("Top Contributing Factors")
    for i, (feature, importance) in enumerate(top_factors[:5], 1):
        label = label_map.get(feature, feature)
        pdf.body_text(f"{i}. {label} - Importance: {importance:.3f}")

    pdf.section_title("General Recommendations")
    if risk_category == "High Risk":
        pdf.body_text("- Please consult a cardiologist as soon as possible.")
        pdf.body_text("- Regular monitoring of blood pressure and cholesterol is recommended.")
        pdf.body_text("- Consider a cardiac stress test under medical supervision.")
    elif risk_category == "Medium Risk":
        pdf.body_text("- Schedule a check-up with your primary care physician.")
        pdf.body_text("- Maintain a heart-healthy diet low in saturated fats and sodium.")
        pdf.body_text("- Aim for 150 minutes of moderate exercise per week.")
    else:
        pdf.body_text("- Continue maintaining a healthy lifestyle.")
        pdf.body_text("- Regular check-ups are still recommended.")
        pdf.body_text("- Stay physically active and manage stress levels.")

    return pdf.output()


def generate_bmi_report(height_cm: float, weight_kg: float, age: int,
                        gender: str, bmi: float, category: str,
                        insights: list, ideal_range: tuple) -> bytes:
    """Generate a PDF report for BMI & Health Insights."""
    pdf = VitaScanPDF()
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(15, 76, 117)
    pdf.cell(0, 15, "BMI & Health Insights Report", align="C",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    color = "green" if category == "Normal weight" else (
        "orange" if "Overweight" in category else (
            "red" if "Obese" in category else "blue"))
    pdf.highlight_box(f"BMI: {bmi:.1f} - {category}", color=color)

    pdf.section_title("Your Information")
    pdf.key_value("Height", f"{height_cm} cm")
    pdf.key_value("Weight", f"{weight_kg} kg")
    pdf.key_value("Age", str(age))
    pdf.key_value("Gender", gender)
    pdf.key_value("Ideal Weight Range", f"{ideal_range[0]:.1f} - {ideal_range[1]:.1f} kg")
    pdf.ln(5)

    pdf.section_title("Personalized Health Insights")
    for insight in insights:
        pdf.body_text(f"- {insight}")

    return pdf.output()


def generate_mental_health_report(answers: dict, total_score: int,
                                  anxiety_score: int, depression_score: int,
                                  category: str, recommendations: list) -> bytes:
    """Generate a PDF report for Mental Health Screening (PHQ-4)."""
    pdf = VitaScanPDF()
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(15, 76, 117)
    pdf.cell(0, 15, "Mental Health Screening Report", align="C",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    color_map = {"Normal": "green", "Mild": "blue", "Moderate": "orange", "Severe": "red"}
    pdf.highlight_box(f"Score: {total_score}/12 - {category}",
                      color=color_map.get(category, "blue"))

    pdf.section_title("Subscale Breakdown")
    pdf.key_value("Anxiety Score (GAD-2)", f"{anxiety_score}/6")
    pdf.key_value("Depression Score (PHQ-2)", f"{depression_score}/6")
    pdf.ln(5)

    pdf.section_title("Recommended Next Steps")
    for rec in recommendations:
        pdf.body_text(f"- {rec}")

    pdf.ln(10)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(150, 150, 150)
    pdf.multi_cell(0, 5,
                   "PRIVACY NOTE: No data from this screening has been stored. "
                   "This report was generated locally on your device. "
                   "If you are in crisis, please contact your local emergency services "
                   "or a mental health helpline immediately.")

    return pdf.output()
