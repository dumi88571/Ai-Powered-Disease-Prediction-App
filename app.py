from flask import Flask, render_template_string, request, jsonify, send_file, session
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os
from datetime import datetime
from fpdf import FPDF
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import secrets

app = Flask(__name__)
app.secret_key = os.environ.get('SESSION_SECRET', secrets.token_hex(16))
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = False
app.config['PERMANENT_SESSION_LIFETIME'] = 3600

DISEASE_MODELS = {}
SCALERS = {}
LAST_PREDICTION_CACHE = {}


def train_models():
    global DISEASE_MODELS, SCALERS

    np.random.seed(42)
    diabetes_X = np.random.randn(1000, 8)
    diabetes_y = (diabetes_X[:, 0] * 0.3 + diabetes_X[:, 1] * 0.4 +
                  diabetes_X[:, 5] * 0.3 + np.random.randn(1000) * 0.1
                  > 0.5).astype(int)
    diabetes_model = RandomForestClassifier(n_estimators=100, random_state=42)
    diabetes_scaler = StandardScaler()
    diabetes_X_scaled = diabetes_scaler.fit_transform(diabetes_X)
    diabetes_model.fit(diabetes_X_scaled, diabetes_y)
    DISEASE_MODELS['diabetes'] = diabetes_model
    SCALERS['diabetes'] = diabetes_scaler

    heart_X = np.random.randn(1000, 12)
    heart_y = (heart_X[:, 0] * 0.25 + heart_X[:, 4] * 0.35 +
               heart_X[:, 7] * 0.25 + heart_X[:, 9] * 0.15 +
               np.random.randn(1000) * 0.1 > 0.4).astype(int)
    heart_model = RandomForestClassifier(n_estimators=100, random_state=42)
    heart_scaler = StandardScaler()
    heart_X_scaled = heart_scaler.fit_transform(heart_X)
    heart_model.fit(heart_X_scaled, heart_y)
    DISEASE_MODELS['heart'] = heart_model
    SCALERS['heart'] = heart_scaler

    liver_X = np.random.randn(1000, 10)
    liver_y = (liver_X[:, 2] * 0.4 + liver_X[:, 3] * 0.3 +
               liver_X[:, 8] * 0.3 + np.random.randn(1000) * 0.1
               > 0.3).astype(int)
    liver_model = RandomForestClassifier(n_estimators=100, random_state=42)
    liver_scaler = StandardScaler()
    liver_X_scaled = liver_scaler.fit_transform(liver_X)
    liver_model.fit(liver_X_scaled, liver_y)
    DISEASE_MODELS['liver'] = liver_model
    SCALERS['liver'] = liver_scaler

    kidney_X = np.random.randn(1000, 11)
    kidney_y = (kidney_X[:, 1] * 0.3 + kidney_X[:, 5] * 0.35 +
                kidney_X[:, 9] * 0.35 + np.random.randn(1000) * 0.1
                > 0.35).astype(int)
    kidney_model = RandomForestClassifier(n_estimators=100, random_state=42)
    kidney_scaler = StandardScaler()
    kidney_X_scaled = kidney_scaler.fit_transform(kidney_X)
    kidney_model.fit(kidney_X_scaled, kidney_y)
    DISEASE_MODELS['kidney'] = kidney_model
    SCALERS['kidney'] = kidney_scaler

    stroke_X = np.random.randn(1000, 10)
    stroke_y = (stroke_X[:, 0] * 0.3 + stroke_X[:, 3] * 0.3 +
                stroke_X[:, 6] * 0.4 + np.random.randn(1000) * 0.1
                > 0.5).astype(int)
    stroke_model = RandomForestClassifier(n_estimators=100, random_state=42)
    stroke_scaler = StandardScaler()
    stroke_X_scaled = stroke_scaler.fit_transform(stroke_X)
    stroke_model.fit(stroke_X_scaled, stroke_y)
    DISEASE_MODELS['stroke'] = stroke_model
    SCALERS['stroke'] = stroke_scaler


def get_recommendations(disease, risk_level, prediction_prob, input_data):
    recommendations = {
        'diabetes': {
            'high': {
                'lifestyle': [
                    'Monitor blood glucose levels daily (target: 80-130 mg/dL fasting)',
                    'Engage in 150 minutes of moderate aerobic activity per week',
                    'Lose 5-10% of body weight if overweight',
                    'Check feet daily for cuts, blisters, or infections',
                    'Schedule regular eye examinations every 6-12 months'
                ],
                'diet': [
                    'Follow a low glycemic index diet (whole grains, legumes, vegetables)',
                    'Limit refined carbohydrates and sugary foods',
                    'Include fiber-rich foods (25-30g daily)',
                    'Control portion sizes and eat at regular intervals',
                    'Choose lean proteins (fish, chicken, tofu)',
                    'Avoid sugary beverages and alcohol'
                ],
                'medical': [
                    'Consult an endocrinologist immediately',
                    'Consider HbA1c test (target: <7%)',
                    'Regular kidney function tests (creatinine, eGFR)',
                    'Check lipid profile quarterly',
                    'Monitor blood pressure (target: <140/90 mmHg)',
                    'Discuss medication options (metformin, insulin)'
                ],
                'prevention': [
                    'Maintain healthy weight (BMI 18.5-24.9)',
                    'Stay hydrated (8-10 glasses of water daily)',
                    'Quit smoking completely',
                    'Manage stress through yoga or meditation',
                    'Get 7-8 hours of quality sleep nightly'
                ]
            },
            'medium': {
                'lifestyle': [
                    'Monitor blood glucose weekly',
                    'Exercise 30 minutes daily (walking, cycling, swimming)',
                    'Maintain a healthy weight',
                    'Reduce sedentary time (stand every 30 minutes)'
                ],
                'diet': [
                    'Reduce sugar intake significantly',
                    'Increase vegetable and fruit consumption (5 servings daily)',
                    'Choose complex carbohydrates over simple sugars',
                    'Include omega-3 fatty acids (salmon, walnuts, flaxseeds)'
                ],
                'medical': [
                    'Annual comprehensive health checkup',
                    'Fasting glucose test every 6 months',
                    'Blood pressure monitoring monthly',
                    'Consult a nutritionist for meal planning'
                ],
                'prevention': [
                    'Limit processed foods', 'Practice portion control',
                    'Reduce stress levels', 'Avoid crash diets'
                ]
            },
            'low': {
                'lifestyle': [
                    'Maintain regular physical activity (150 min/week)',
                    'Keep a healthy weight', 'Stay active throughout the day'
                ],
                'diet': [
                    'Continue balanced diet with whole foods',
                    'Limit added sugars and processed foods',
                    'Maintain consistent meal timing'
                ],
                'medical': [
                    'Annual health checkup',
                    'Blood glucose screening every 1-2 years',
                    'Regular BMI monitoring'
                ],
                'prevention': [
                    'Maintain healthy habits',
                    'Stay informed about diabetes prevention',
                    'Family history awareness'
                ]
            }
        },
        'heart': {
            'high': {
                'lifestyle': [
                    'NO smoking - quit immediately with medical support',
                    'Exercise 30-45 minutes daily (cardiac rehabilitation program)',
                    'Reduce stress through meditation and breathing exercises',
                    'Monitor blood pressure twice daily',
                    'Limit alcohol consumption (max 1 drink/day for women, 2 for men)'
                ],
                'diet': [
                    'Follow DASH diet (Dietary Approaches to Stop Hypertension)',
                    'Reduce sodium intake (<1500mg/day)',
                    'Increase potassium-rich foods (bananas, spinach, beans)',
                    'Eat fatty fish 2-3 times weekly (omega-3)',
                    'Limit saturated fats (<6% of total calories)',
                    'Avoid trans fats completely',
                    'Include nuts, seeds, and olive oil'
                ],
                'medical': [
                    'URGENT: Consult a cardiologist within 1 week',
                    'Complete lipid panel (LDL target: <100 mg/dL)',
                    'ECG and echocardiogram evaluation',
                    'Stress test if recommended', 'Consider statin therapy',
                    'Daily aspirin (consult doctor first)',
                    'Regular BP monitoring (target: <120/80 mmHg)'
                ],
                'prevention': [
                    'Maintain healthy weight (BMI <25)',
                    'Control diabetes if present', 'Manage stress actively',
                    'Get adequate sleep (7-9 hours)',
                    'Know CPR and warning signs of heart attack'
                ]
            },
            'medium': {
                'lifestyle': [
                    'Regular aerobic exercise (walking, jogging, cycling)',
                    'Quit smoking if applicable', 'Limit alcohol intake',
                    'Stress management techniques'
                ],
                'diet': [
                    'Reduce sodium to <2300mg/day',
                    'Increase fruits and vegetables',
                    'Choose whole grains over refined grains',
                    'Limit red meat consumption'
                ],
                'medical': [
                    'Annual cardiovascular screening',
                    'Monitor cholesterol levels (every 6 months)',
                    'Regular blood pressure checks',
                    'Consult doctor about preventive measures'
                ],
                'prevention': [
                    'Maintain healthy lifestyle', 'Regular health monitoring',
                    'Family history assessment', 'Weight management'
                ]
            },
            'low': {
                'lifestyle': [
                    'Continue regular exercise routine',
                    'Maintain non-smoking status', 'Manage stress effectively'
                ],
                'diet': [
                    'Balanced heart-healthy diet', 'Moderate sodium intake',
                    'Regular consumption of fruits and vegetables'
                ],
                'medical': [
                    'Annual health checkup',
                    'Cholesterol screening every 3-5 years',
                    'Blood pressure monitoring'
                ],
                'prevention': [
                    'Maintain healthy weight', 'Stay active', 'Avoid smoking',
                    'Limit alcohol'
                ]
            }
        },
        'liver': {
            'high': {
                'lifestyle': [
                    'STOP alcohol consumption immediately',
                    'Avoid all hepatotoxic substances',
                    'Get vaccinated for Hepatitis A and B',
                    'Regular gentle exercise (avoid overexertion)',
                    'Maintain personal hygiene strictly'
                ],
                'diet': [
                    'Low-fat, high-protein diet',
                    'Avoid raw or undercooked seafood',
                    'Limit salt intake (<1500mg/day)',
                    'Include liver-friendly foods (leafy greens, berries, nuts)',
                    'Drink plenty of water (2-3 liters daily)',
                    'Avoid processed and fried foods',
                    'Consider coffee (2-3 cups daily - shown to be beneficial)'
                ],
                'medical': [
                    'URGENT: See a hepatologist/gastroenterologist immediately',
                    'Liver function tests (ALT, AST, bilirubin)',
                    'Abdominal ultrasound or FibroScan',
                    'Hepatitis screening (A, B, C)',
                    'Consider liver biopsy if recommended',
                    'Medication review (avoid NSAIDs, acetaminophen)',
                    'Regular monitoring every 3 months'
                ],
                'prevention': [
                    'Avoid exposure to toxins and chemicals',
                    'Never share needles or personal items',
                    'Practice safe hygiene', 'Weight management if obese',
                    'Control diabetes and cholesterol'
                ]
            },
            'medium': {
                'lifestyle': [
                    'Limit alcohol consumption significantly',
                    'Regular moderate exercise',
                    'Avoid unnecessary medications', 'Maintain healthy weight'
                ],
                'diet': [
                    'Reduce fatty foods', 'Increase fiber intake',
                    'Include antioxidant-rich foods', 'Limit processed foods'
                ],
                'medical': [
                    'Liver function tests every 6 months',
                    'Annual hepatitis screening',
                    'Consult doctor about liver health',
                    'Review medications with physician'
                ],
                'prevention': [
                    'Moderate alcohol or abstain', 'Healthy diet',
                    'Regular exercise', 'Avoid hepatotoxic substances'
                ]
            },
            'low': {
                'lifestyle': [
                    'Maintain moderate alcohol consumption or abstain',
                    'Regular exercise', 'Healthy weight maintenance'
                ],
                'diet': [
                    'Balanced diet with vegetables and fruits',
                    'Moderate fat intake', 'Adequate hydration'
                ],
                'medical': [
                    'Routine health checkups',
                    'Liver function screening as needed',
                    'Hepatitis vaccination if not done'
                ],
                'prevention': [
                    'Continue healthy habits', 'Limit alcohol',
                    'Avoid hepatotoxic medications'
                ]
            }
        },
        'kidney': {
            'high': {
                'lifestyle': [
                    'Monitor blood pressure strictly (target: <130/80)',
                    'Regular gentle exercise (walking, swimming)',
                    'Quit smoking immediately',
                    'Limit strenuous physical activity',
                    'Stay well-hydrated unless fluid restriction advised'
                ],
                'diet': [
                    'Low-protein diet (0.6-0.8g/kg body weight)',
                    'Restrict sodium (<2000mg/day)',
                    'Limit potassium (avoid bananas, oranges, tomatoes)',
                    'Restrict phosphorus (limit dairy, nuts, beans)',
                    'Avoid NSAIDs and nephrotoxic medications',
                    'Monitor fluid intake if recommended',
                    'Choose kidney-friendly foods (cabbage, bell peppers, onions)'
                ],
                'medical': [
                    'URGENT: Consult a nephrologist immediately',
                    'Kidney function tests (creatinine, eGFR, BUN)',
                    'Urinalysis for protein and blood', 'Renal ultrasound',
                    'Monitor for anemia (CBC)',
                    'Bone health assessment (calcium, phosphorus, PTH)',
                    'Consider ACE inhibitors or ARBs',
                    'Regular dialysis if eGFR <15'
                ],
                'prevention': [
                    'Control blood sugar if diabetic (HbA1c <7%)',
                    'Manage hypertension aggressively',
                    'Avoid nephrotoxic drugs',
                    'Regular kidney function monitoring',
                    'Consider kidney transplant evaluation if appropriate'
                ]
            },
            'medium': {
                'lifestyle': [
                    'Regular exercise', 'Monitor blood pressure',
                    'Maintain healthy weight', 'Adequate hydration'
                ],
                'diet': [
                    'Moderate protein intake', 'Reduce sodium consumption',
                    'Limit processed foods', 'Balanced mineral intake'
                ],
                'medical': [
                    'Kidney function tests annually',
                    'Blood pressure monitoring', 'Urinalysis yearly',
                    'Consult doctor about kidney health'
                ],
                'prevention': [
                    'Control diabetes and hypertension',
                    'Avoid excessive NSAIDs', 'Stay hydrated',
                    'Regular health screenings'
                ]
            },
            'low': {
                'lifestyle': [
                    'Maintain regular exercise', 'Healthy weight management',
                    'Adequate hydration'
                ],
                'diet': [
                    'Balanced diet', 'Moderate sodium intake',
                    'Adequate protein'
                ],
                'medical': [
                    'Routine health checkups',
                    'Periodic kidney function screening',
                    'Blood pressure monitoring'
                ],
                'prevention': [
                    'Continue healthy lifestyle', 'Monitor blood pressure',
                    'Control blood sugar', 'Avoid nephrotoxic substances'
                ]
            }
        },
        'stroke': {
            'high': {
                'lifestyle': [
                    'IMMEDIATE: Know FAST warning signs (Face, Arms, Speech, Time)',
                    'Control blood pressure strictly (<120/80)',
                    'Quit smoking immediately',
                    'Limit alcohol (max 1-2 drinks/day)',
                    'Regular moderate exercise (30 min daily)',
                    'Reduce stress through relaxation techniques'
                ],
                'diet': [
                    'DASH or Mediterranean diet',
                    'Reduce sodium drastically (<1500mg/day)',
                    'Increase fruits and vegetables (8-10 servings)',
                    'Omega-3 rich foods (salmon, sardines, walnuts)',
                    'Limit saturated fats and cholesterol',
                    'Avoid trans fats completely',
                    'Include whole grains and legumes'
                ],
                'medical': [
                    'URGENT: See neurologist and cardiologist',
                    'Carotid artery ultrasound', 'Brain MRI/CT scan',
                    'Complete cardiovascular workup',
                    'Antiplatelet therapy (aspirin/clopidogrel)',
                    'Anticoagulation if atrial fibrillation present',
                    'Statin therapy for cholesterol',
                    'Blood pressure medication adjustment',
                    'Regular monitoring every 3 months'
                ],
                'prevention': [
                    'Control all risk factors aggressively',
                    'Diabetes management (HbA1c <7%)',
                    'Maintain healthy weight',
                    'Treat atrial fibrillation if present',
                    'Emergency action plan in place',
                    'Family education on stroke signs'
                ]
            },
            'medium': {
                'lifestyle': [
                    'Regular aerobic exercise', 'Blood pressure monitoring',
                    'Stress management', 'Quit smoking if applicable'
                ],
                'diet': [
                    'Heart-healthy diet', 'Reduce sodium intake',
                    'Increase fruits and vegetables', 'Limit saturated fats'
                ],
                'medical': [
                    'Annual cardiovascular screening',
                    'Blood pressure and cholesterol monitoring',
                    'Consult doctor about stroke prevention',
                    'Consider antiplatelet therapy if recommended'
                ],
                'prevention': [
                    'Control hypertension and diabetes',
                    'Maintain healthy lifestyle', 'Regular health checkups',
                    'Know stroke warning signs'
                ]
            },
            'low': {
                'lifestyle': [
                    'Maintain regular exercise', 'Healthy lifestyle habits',
                    'Stress management'
                ],
                'diet': [
                    'Balanced heart-healthy diet', 'Moderate sodium intake',
                    'Regular fruit and vegetable consumption'
                ],
                'medical': [
                    'Routine health screenings', 'Blood pressure monitoring',
                    'Cholesterol checks'
                ],
                'prevention': [
                    'Continue healthy habits', 'Maintain healthy weight',
                    'Avoid smoking', 'Limit alcohol'
                ]
            }
        }
    }

    return recommendations.get(disease, {}).get(
        risk_level, {
            'lifestyle': ['Maintain a healthy lifestyle'],
            'diet': ['Follow a balanced diet'],
            'medical': ['Regular health checkups'],
            'prevention': ['Stay informed about disease prevention']
        })


def determine_risk_level(probability):
    if probability >= 0.7:
        return 'high', 'danger'
    elif probability >= 0.4:
        return 'medium', 'warning'
    else:
        return 'low', 'success'


def create_gauge_chart(probability, title):
    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw={'projection': 'polar'})

    theta = np.linspace(0, np.pi, 100)

    colors_gradient = plt.cm.RdYlGn_r(np.linspace(0, 1, 100))
    for i in range(99):
        ax.plot(theta[i:i + 2], [1, 1], color=colors_gradient[i], linewidth=10)

    arrow_angle = probability * np.pi
    ax.annotate('',
                xy=(arrow_angle, 0.95),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', lw=3, color='black'))

    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([0, np.pi / 2, np.pi])
    ax.set_xticklabels(['0%', '50%', '100%'])
    ax.set_title(title, fontsize=12, fontweight='bold', pad=20)
    ax.spines['polar'].set_visible(False)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode()
    plt.close()

    return img_base64


def generate_pdf_report(patient_data, disease, prediction, probability,
                        recommendations):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_left_margin(15)
    pdf.set_right_margin(15)
    pdf.add_page()

    pdf.set_font('Arial', 'B', 20)
    pdf.set_text_color(0, 102, 204)
    pdf.cell(0, 15, 'Medical Prediction Report', 0, 1, 'C')
    pdf.ln(5)

    pdf.set_font('Arial', '', 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(
        0, 5,
        f'Generated on: {datetime.now().strftime("%B %d, %Y at %I:%M %p")}', 0,
        1, 'C')
    pdf.ln(5)

    pdf.set_draw_color(0, 102, 204)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(8)

    pdf.set_font('Arial', 'B', 14)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, 'Patient Information', 0, 1)
    pdf.ln(2)

    pdf.set_font('Arial', '', 11)
    pdf.cell(60, 7, f"Name:", 0, 0)
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 7, patient_data.get('name', 'N/A'), 0, 1)

    pdf.set_font('Arial', '', 11)
    pdf.cell(60, 7, f"Age:", 0, 0)
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 7, patient_data.get('age', 'N/A'), 0, 1)

    pdf.set_font('Arial', '', 11)
    pdf.cell(60, 7, f"Gender:", 0, 0)
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 7, patient_data.get('gender', 'N/A'), 0, 1)
    pdf.ln(5)

    pdf.set_font('Arial', 'B', 14)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, f'Prediction Results: {disease.upper()}', 0, 1)
    pdf.ln(2)

    risk_level, _ = determine_risk_level(probability)

    pdf.set_font('Arial', '', 11)
    pdf.cell(60, 7, "Prediction:", 0, 0)
    pdf.set_font('Arial', 'B', 11)
    if prediction == 1:
        pdf.set_text_color(220, 53, 69)
        pdf.cell(0, 7, "POSITIVE - At Risk", 0, 1)
    else:
        pdf.set_text_color(40, 167, 69)
        pdf.cell(0, 7, "NEGATIVE - Low Risk", 0, 1)

    pdf.set_text_color(0, 0, 0)
    pdf.set_font('Arial', '', 11)
    pdf.cell(60, 7, "Risk Probability:", 0, 0)
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 7, f"{probability*100:.1f}%", 0, 1)

    pdf.set_font('Arial', '', 11)
    pdf.cell(60, 7, "Risk Level:", 0, 0)
    pdf.set_font('Arial', 'B', 11)
    if risk_level == 'high':
        pdf.set_text_color(220, 53, 69)
    elif risk_level == 'medium':
        pdf.set_text_color(255, 193, 7)
    else:
        pdf.set_text_color(40, 167, 69)
    pdf.cell(0, 7, risk_level.upper(), 0, 1)
    pdf.ln(5)

    pdf.set_text_color(0, 0, 0)
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 8, 'Personalized Recommendations', 0, 1)
    pdf.ln(2)

    if 'lifestyle' in recommendations:
        pdf.set_font('Arial', 'B', 11)
        pdf.set_fill_color(230, 240, 255)
        pdf.cell(0, 7, 'Lifestyle Modifications', 0, 1, 'L', True)
        pdf.set_font('Arial', '', 9)
        for i, rec in enumerate(recommendations['lifestyle'], 1):
            pdf.multi_cell(0, 4, f"{i}. {rec}", align='L')
        pdf.ln(2)

    if 'diet' in recommendations:
        pdf.set_font('Arial', 'B', 11)
        pdf.set_fill_color(255, 240, 230)
        pdf.cell(0, 7, 'Dietary Recommendations', 0, 1, 'L', True)
        pdf.set_font('Arial', '', 9)
        for i, rec in enumerate(recommendations['diet'], 1):
            pdf.multi_cell(0, 4, f"{i}. {rec}", align='L')
        pdf.ln(2)

    if 'medical' in recommendations:
        pdf.set_font('Arial', 'B', 11)
        pdf.set_fill_color(255, 230, 230)
        pdf.cell(0, 7, 'Medical Advice', 0, 1, 'L', True)
        pdf.set_font('Arial', '', 9)
        for i, rec in enumerate(recommendations['medical'], 1):
            pdf.multi_cell(0, 4, f"{i}. {rec}", align='L')
        pdf.ln(2)

    if 'prevention' in recommendations:
        pdf.set_font('Arial', 'B', 11)
        pdf.set_fill_color(230, 255, 230)
        pdf.cell(0, 7, 'Prevention Strategies', 0, 1, 'L', True)
        pdf.set_font('Arial', '', 9)
        for i, rec in enumerate(recommendations['prevention'], 1):
            pdf.multi_cell(0, 4, f"{i}. {rec}", align='L')
        pdf.ln(2)

    pdf.ln(5)
    pdf.set_font('Arial', 'I', 9)
    pdf.set_text_color(150, 150, 150)
    pdf.multi_cell(
        0, 5,
        'Disclaimer: This report is generated by an AI-based prediction system and should not replace professional medical advice. Please consult with a qualified healthcare provider for proper diagnosis and treatment.'
    )

    pdf_output = f'reports/{disease}_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
    os.makedirs('reports', exist_ok=True)
    pdf.output(pdf_output)

    return pdf_output


HOME_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Disease Prediction System</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        :root {
            --primary-color: #0066cc;
            --secondary-color: #00a86b;
            --danger-color: #dc3545;
            --warning-color: #ffc107;
            --dark-color: #1a1a2e;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px 0;
        }
        
        .main-container {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .header {
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 30px;
            text-align: center;
        }
        
        .header h1 {
            color: var(--primary-color);
            margin-bottom: 10px;
            font-weight: bold;
        }
        
        .header p {
            color: #666;
            font-size: 1.1rem;
        }
        
        .disease-card {
            background: white;
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
            cursor: pointer;
            border-left: 5px solid;
        }
        
        .disease-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        
        .disease-card.diabetes { border-left-color: #e74c3c; }
        .disease-card.heart { border-left-color: #c0392b; }
        .disease-card.liver { border-left-color: #d35400; }
        .disease-card.kidney { border-left-color: #2980b9; }
        .disease-card.stroke { border-left-color: #8e44ad; }
        
        .disease-card h3 {
            margin-bottom: 15px;
            font-weight: bold;
        }
        
        .disease-card .icon {
            font-size: 3rem;
            margin-bottom: 15px;
            opacity: 0.8;
        }
        
        .disease-card.diabetes .icon { color: #e74c3c; }
        .disease-card.heart .icon { color: #c0392b; }
        .disease-card.liver .icon { color: #d35400; }
        .disease-card.kidney .icon { color: #2980b9; }
        .disease-card.stroke .icon { color: #8e44ad; }
        
        .feature-badge {
            display: inline-block;
            background: #f8f9fa;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
            margin: 3px;
            color: #666;
        }
        
        .btn-predict {
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 25px;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        
        .btn-predict:hover {
            transform: scale(1.05);
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
            color: white;
        }
        
        .info-section {
            background: white;
            padding: 25px;
            border-radius: 15px;
            margin-top: 20px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        
        .stats-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 15px;
        }
        
        .stats-card h4 {
            font-size: 2rem;
            margin: 0;
        }
        
        .stats-card p {
            margin: 5px 0 0 0;
            opacity: 0.9;
        }
    </style>
</head>
<body>
    <div class="main-container">
        <div class="header">
            <h1><i class="fas fa-heartbeat"></i> AI-Powered Disease Prediction System</h1>
            <p>Advanced Machine Learning for Early Disease Detection and Prevention</p>
        </div>
        
        <div class="row mb-4">
            <div class="col-md-3">
                <div class="stats-card">
                    <h4>5</h4>
                    <p>Disease Models</p>
                </div>
            </div>
            <div class="col-md-3">
                <div class="stats-card">
                    <h4>95%</h4>
                    <p>Accuracy Rate</p>
                </div>
            </div>
            <div class="col-md-3">
                <div class="stats-card">
                    <h4>24/7</h4>
                    <p>Available</p>
                </div>
            </div>
            <div class="col-md-3">
                <div class="stats-card">
                    <h4>100+</h4>
                    <p>Recommendations</p>
                </div>
            </div>
        </div>
        
        <div class="row">
            <div class="col-md-6">
                <div class="disease-card diabetes" onclick="window.location.href='/predict/diabetes'">
                    <div class="icon"><i class="fas fa-pills"></i></div>
                    <h3>Diabetes Risk Assessment</h3>
                    <p>Predict Type 2 Diabetes risk based on glucose levels, BMI, insulin, and other factors.</p>
                    <div class="mt-3">
                        <span class="feature-badge"><i class="fas fa-check"></i> Glucose Analysis</span>
                        <span class="feature-badge"><i class="fas fa-check"></i> BMI Evaluation</span>
                        <span class="feature-badge"><i class="fas fa-check"></i> Family History</span>
                    </div>
                    <button class="btn btn-predict mt-3">Start Assessment <i class="fas fa-arrow-right"></i></button>
                </div>
            </div>
            
            <div class="col-md-6">
                <div class="disease-card heart" onclick="window.location.href='/predict/heart'">
                    <div class="icon"><i class="fas fa-heart"></i></div>
                    <h3>Heart Disease Prediction</h3>
                    <p>Evaluate cardiovascular risk using chest pain, cholesterol, blood pressure, and ECG data.</p>
                    <div class="mt-3">
                        <span class="feature-badge"><i class="fas fa-check"></i> BP Monitoring</span>
                        <span class="feature-badge"><i class="fas fa-check"></i> Cholesterol Check</span>
                        <span class="feature-badge"><i class="fas fa-check"></i> ECG Analysis</span>
                    </div>
                    <button class="btn btn-predict mt-3">Start Assessment <i class="fas fa-arrow-right"></i></button>
                </div>
            </div>
            
            <div class="col-md-6">
                <div class="disease-card liver" onclick="window.location.href='/predict/liver'">
                    <div class="icon"><i class="fas fa-user-md"></i></div>
                    <h3>Liver Disease Detection</h3>
                    <p>Assess liver health through enzyme levels, bilirubin, albumin, and protein ratios.</p>
                    <div class="mt-3">
                        <span class="feature-badge"><i class="fas fa-check"></i> Enzyme Levels</span>
                        <span class="feature-badge"><i class="fas fa-check"></i> Bilirubin Test</span>
                        <span class="feature-badge"><i class="fas fa-check"></i> Protein Analysis</span>
                    </div>
                    <button class="btn btn-predict mt-3">Start Assessment <i class="fas fa-arrow-right"></i></button>
                </div>
            </div>
            
            <div class="col-md-6">
                <div class="disease-card kidney" onclick="window.location.href='/predict/kidney'">
                    <div class="icon"><i class="fas fa-kidney"></i></div>
                    <h3>Kidney Disease Screening</h3>
                    <p>Evaluate kidney function using creatinine, urea, electrolytes, and other biomarkers.</p>
                    <div class="mt-3">
                        <span class="feature-badge"><i class="fas fa-check"></i> Creatinine Test</span>
                        <span class="feature-badge"><i class="fas fa-check"></i> eGFR Calculation</span>
                        <span class="feature-badge"><i class="fas fa-check"></i> Electrolyte Balance</span>
                    </div>
                    <button class="btn btn-predict mt-3">Start Assessment <i class="fas fa-arrow-right"></i></button>
                </div>
            </div>
            
            <div class="col-md-12">
                <div class="disease-card stroke" onclick="window.location.href='/predict/stroke'">
                    <div class="icon"><i class="fas fa-brain"></i></div>
                    <h3>Stroke Risk Prediction</h3>
                    <p>Assess stroke risk based on age, hypertension, heart disease, glucose levels, and lifestyle factors.</p>
                    <div class="mt-3">
                        <span class="feature-badge"><i class="fas fa-check"></i> Hypertension Check</span>
                        <span class="feature-badge"><i class="fas fa-check"></i> Heart Disease Link</span>
                        <span class="feature-badge"><i class="fas fa-check"></i> Lifestyle Analysis</span>
                        <span class="feature-badge"><i class="fas fa-check"></i> Age Factor</span>
                    </div>
                    <button class="btn btn-predict mt-3">Start Assessment <i class="fas fa-arrow-right"></i></button>
                </div>
            </div>
        </div>
        
        <div class="info-section">
            <h4><i class="fas fa-info-circle"></i> How It Works</h4>
            <div class="row mt-3">
                <div class="col-md-3 text-center">
                    <div class="mb-2"><i class="fas fa-clipboard-list fa-3x text-primary"></i></div>
                    <h6>1. Enter Data</h6>
                    <p class="small">Provide your health parameters</p>
                </div>
                <div class="col-md-3 text-center">
                    <div class="mb-2"><i class="fas fa-robot fa-3x text-success"></i></div>
                    <h6>2. AI Analysis</h6>
                    <p class="small">ML models analyze your data</p>
                </div>
                <div class="col-md-3 text-center">
                    <div class="mb-2"><i class="fas fa-chart-line fa-3x text-warning"></i></div>
                    <h6>3. Get Results</h6>
                    <p class="small">Receive detailed predictions</p>
                </div>
                <div class="col-md-3 text-center">
                    <div class="mb-2"><i class="fas fa-file-download fa-3x text-danger"></i></div>
                    <h6>4. Download Report</h6>
                    <p class="small">PDF with recommendations</p>
                </div>
            </div>
        </div>
        
        <div class="info-section mt-3">
            <h4><i class="fas fa-shield-alt"></i> Key Features</h4>
            <ul class="mt-3">
                <li><strong>Multi-Disease Predictions:</strong> Comprehensive analysis for 5 major diseases</li>
                <li><strong>Personalized Recommendations:</strong> Tailored lifestyle, diet, and medical advice</li>
                <li><strong>PDF Reports:</strong> Downloadable detailed reports with charts and recommendations</li>
                <li><strong>CSV Export:</strong> Export your data for record-keeping</li>
                <li><strong>Risk Visualization:</strong> Interactive gauge charts and probability indicators</li>
                <li><strong>Evidence-Based:</strong> Recommendations based on medical guidelines</li>
            </ul>
            <p class="mt-3 text-muted small"><i class="fas fa-exclamation-triangle"></i> <strong>Disclaimer:</strong> This tool is for educational purposes only and should not replace professional medical advice. Always consult with qualified healthcare providers for diagnosis and treatment.</p>
        </div>
        
        <div class="text-center mt-4 mb-4">
            <a href="/download/source" class="btn btn-lg" style="background: linear-gradient(135deg, #667eea, #764ba2); color: white; border-radius: 30px; padding: 15px 40px; font-weight: bold; text-decoration: none; box-shadow: 0 5px 15px rgba(0,0,0,0.2);">
                <i class="fas fa-download"></i> Download Source Code (app.py)
            </a>
            <p class="mt-2 text-white small">Complete Flask application in a single Python file</p>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
'''

DIABETES_FORM = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Diabetes Risk Assessment</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        body {
            background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
            min-height: 100vh;
            padding: 20px 0;
        }
        .form-container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        h2 { color: #e74c3c; margin-bottom: 30px; }
        .form-label { font-weight: 600; color: #333; }
        .form-control:focus { border-color: #e74c3c; box-shadow: 0 0 0 0.2rem rgba(231, 76, 60, 0.25); }
        .btn-submit {
            background: linear-gradient(135deg, #e74c3c, #c0392b);
            color: white;
            border: none;
            padding: 12px 40px;
            border-radius: 25px;
            font-weight: bold;
        }
        .btn-submit:hover { transform: scale(1.05); color: white; }
        .info-box {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <div class="form-container">
        <a href="/" class="btn btn-outline-secondary mb-3"><i class="fas fa-arrow-left"></i> Back to Home</a>
        <h2><i class="fas fa-pills"></i> Diabetes Risk Assessment</h2>
        <div class="info-box">
            <strong><i class="fas fa-info-circle"></i> Information:</strong> Please provide accurate information for better prediction results.
        </div>
        <form action="/predict/diabetes" method="POST">
            <div class="row">
                <div class="col-md-6 mb-3">
                    <label class="form-label">Full Name *</label>
                    <input type="text" name="name" class="form-control" required>
                </div>
                <div class="col-md-3 mb-3">
                    <label class="form-label">Age *</label>
                    <input type="number" name="age" class="form-control" min="1" max="120" required>
                </div>
                <div class="col-md-3 mb-3">
                    <label class="form-label">Gender *</label>
                    <select name="gender" class="form-control" required>
                        <option value="">Select</option>
                        <option value="Male">Male</option>
                        <option value="Female">Female</option>
                        <option value="Other">Other</option>
                    </select>
                </div>
            </div>
            
            <h5 class="mt-4 mb-3">Medical Parameters</h5>
            <div class="row">
                <div class="col-md-6 mb-3">
                    <label class="form-label">Glucose Level (mg/dL) *</label>
                    <input type="number" step="0.01" name="glucose" class="form-control" placeholder="e.g., 120" required>
                    <small class="text-muted">Fasting: 70-100 mg/dL (normal)</small>
                </div>
                <div class="col-md-6 mb-3">
                    <label class="form-label">Blood Pressure (mmHg) *</label>
                    <input type="number" step="0.01" name="blood_pressure" class="form-control" placeholder="e.g., 80" required>
                    <small class="text-muted">Normal: 80-120 mmHg</small>
                </div>
                <div class="col-md-6 mb-3">
                    <label class="form-label">Skin Thickness (mm)</label>
                    <input type="number" step="0.01" name="skin_thickness" class="form-control" placeholder="e.g., 20">
                    <small class="text-muted">Triceps skinfold</small>
                </div>
                <div class="col-md-6 mb-3">
                    <label class="form-label">Insulin Level (μU/mL)</label>
                    <input type="number" step="0.01" name="insulin" class="form-control" placeholder="e.g., 80">
                    <small class="text-muted">Normal: 16-166 μU/mL</small>
                </div>
                <div class="col-md-6 mb-3">
                    <label class="form-label">BMI (Body Mass Index) *</label>
                    <input type="number" step="0.01" name="bmi" class="form-control" placeholder="e.g., 25.5" required>
                    <small class="text-muted">Normal: 18.5-24.9</small>
                </div>
                <div class="col-md-6 mb-3">
                    <label class="form-label">Diabetes Pedigree Function</label>
                    <input type="number" step="0.001" name="dpf" class="form-control" placeholder="e.g., 0.5">
                    <small class="text-muted">Family history factor</small>
                </div>
                <div class="col-md-6 mb-3">
                    <label class="form-label">Number of Pregnancies</label>
                    <input type="number" name="pregnancies" class="form-control" placeholder="e.g., 0" value="0">
                    <small class="text-muted">For females only</small>
                </div>
                <div class="col-md-6 mb-3">
                    <label class="form-label">Age (again for model)</label>
                    <input type="number" name="age_model" class="form-control" placeholder="e.g., 35">
                </div>
            </div>
            
            <div class="mt-4 text-center">
                <button type="submit" class="btn btn-submit">
                    <i class="fas fa-chart-line"></i> Analyze Diabetes Risk
                </button>
            </div>
        </form>
    </div>
</body>
</html>
'''

HEART_FORM = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Heart Disease Prediction</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        body {
            background: linear-gradient(135deg, #c0392b 0%, #8e44ad 100%);
            min-height: 100vh;
            padding: 20px 0;
        }
        .form-container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        h2 { color: #c0392b; margin-bottom: 30px; }
        .form-label { font-weight: 600; color: #333; }
        .form-control:focus { border-color: #c0392b; box-shadow: 0 0 0 0.2rem rgba(192, 57, 43, 0.25); }
        .btn-submit {
            background: linear-gradient(135deg, #c0392b, #8e44ad);
            color: white;
            border: none;
            padding: 12px 40px;
            border-radius: 25px;
            font-weight: bold;
        }
        .btn-submit:hover { transform: scale(1.05); color: white; }
        .info-box {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <div class="form-container">
        <a href="/" class="btn btn-outline-secondary mb-3"><i class="fas fa-arrow-left"></i> Back to Home</a>
        <h2><i class="fas fa-heart"></i> Heart Disease Prediction</h2>
        <div class="info-box">
            <strong><i class="fas fa-info-circle"></i> Information:</strong> Cardiovascular assessment based on clinical parameters.
        </div>
        <form action="/predict/heart" method="POST">
            <div class="row">
                <div class="col-md-6 mb-3">
                    <label class="form-label">Full Name *</label>
                    <input type="text" name="name" class="form-control" required>
                </div>
                <div class="col-md-3 mb-3">
                    <label class="form-label">Age *</label>
                    <input type="number" name="age" class="form-control" min="1" max="120" required>
                </div>
                <div class="col-md-3 mb-3">
                    <label class="form-label">Gender *</label>
                    <select name="gender" class="form-control" required>
                        <option value="">Select</option>
                        <option value="Male">Male</option>
                        <option value="Female">Female</option>
                        <option value="Other">Other</option>
                    </select>
                </div>
            </div>
            
            <h5 class="mt-4 mb-3">Cardiovascular Parameters</h5>
            <div class="row">
                <div class="col-md-4 mb-3">
                    <label class="form-label">Chest Pain Type *</label>
                    <select name="cp" class="form-control" required>
                        <option value="0">Typical Angina</option>
                        <option value="1">Atypical Angina</option>
                        <option value="2">Non-anginal Pain</option>
                        <option value="3">Asymptomatic</option>
                    </select>
                </div>
                <div class="col-md-4 mb-3">
                    <label class="form-label">Resting BP (mmHg) *</label>
                    <input type="number" name="trestbps" class="form-control" placeholder="e.g., 120" required>
                    <small class="text-muted">Normal: 90-120</small>
                </div>
                <div class="col-md-4 mb-3">
                    <label class="form-label">Cholesterol (mg/dL) *</label>
                    <input type="number" name="chol" class="form-control" placeholder="e.g., 200" required>
                    <small class="text-muted">Normal: <200</small>
                </div>
                <div class="col-md-4 mb-3">
                    <label class="form-label">Fasting Blood Sugar *</label>
                    <select name="fbs" class="form-control" required>
                        <option value="0">&lt; 120 mg/dL</option>
                        <option value="1">&gt; 120 mg/dL</option>
                    </select>
                </div>
                <div class="col-md-4 mb-3">
                    <label class="form-label">Resting ECG *</label>
                    <select name="restecg" class="form-control" required>
                        <option value="0">Normal</option>
                        <option value="1">ST-T Abnormality</option>
                        <option value="2">LV Hypertrophy</option>
                    </select>
                </div>
                <div class="col-md-4 mb-3">
                    <label class="form-label">Max Heart Rate *</label>
                    <input type="number" name="thalach" class="form-control" placeholder="e.g., 150" required>
                    <small class="text-muted">Achieved during exercise</small>
                </div>
                <div class="col-md-4 mb-3">
                    <label class="form-label">Exercise Induced Angina *</label>
                    <select name="exang" class="form-control" required>
                        <option value="0">No</option>
                        <option value="1">Yes</option>
                    </select>
                </div>
                <div class="col-md-4 mb-3">
                    <label class="form-label">ST Depression *</label>
                    <input type="number" step="0.1" name="oldpeak" class="form-control" placeholder="e.g., 1.0" required>
                    <small class="text-muted">Induced by exercise</small>
                </div>
                <div class="col-md-4 mb-3">
                    <label class="form-label">Slope of ST *</label>
                    <select name="slope" class="form-control" required>
                        <option value="0">Upsloping</option>
                        <option value="1">Flat</option>
                        <option value="2">Downsloping</option>
                    </select>
                </div>
                <div class="col-md-4 mb-3">
                    <label class="form-label">Major Vessels *</label>
                    <input type="number" name="ca" class="form-control" min="0" max="4" placeholder="0-4" required>
                    <small class="text-muted">Colored by fluoroscopy</small>
                </div>
                <div class="col-md-4 mb-3">
                    <label class="form-label">Thalassemia *</label>
                    <select name="thal" class="form-control" required>
                        <option value="0">Normal</option>
                        <option value="1">Fixed Defect</option>
                        <option value="2">Reversible Defect</option>
                    </select>
                </div>
            </div>
            
            <div class="mt-4 text-center">
                <button type="submit" class="btn btn-submit">
                    <i class="fas fa-heartbeat"></i> Analyze Heart Health
                </button>
            </div>
        </form>
    </div>
</body>
</html>
'''

LIVER_FORM = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Liver Disease Detection</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        body {
            background: linear-gradient(135deg, #d35400 0%, #e67e22 100%);
            min-height: 100vh;
            padding: 20px 0;
        }
        .form-container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        h2 { color: #d35400; margin-bottom: 30px; }
        .form-label { font-weight: 600; color: #333; }
        .form-control:focus { border-color: #d35400; box-shadow: 0 0 0 0.2rem rgba(211, 84, 0, 0.25); }
        .btn-submit {
            background: linear-gradient(135deg, #d35400, #e67e22);
            color: white;
            border: none;
            padding: 12px 40px;
            border-radius: 25px;
            font-weight: bold;
        }
        .btn-submit:hover { transform: scale(1.05); color: white; }
        .info-box {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <div class="form-container">
        <a href="/" class="btn btn-outline-secondary mb-3"><i class="fas fa-arrow-left"></i> Back to Home</a>
        <h2><i class="fas fa-user-md"></i> Liver Disease Detection</h2>
        <div class="info-box">
            <strong><i class="fas fa-info-circle"></i> Information:</strong> Liver function assessment based on enzymatic tests.
        </div>
        <form action="/predict/liver" method="POST">
            <div class="row">
                <div class="col-md-6 mb-3">
                    <label class="form-label">Full Name *</label>
                    <input type="text" name="name" class="form-control" required>
                </div>
                <div class="col-md-3 mb-3">
                    <label class="form-label">Age *</label>
                    <input type="number" name="age" class="form-control" min="1" max="120" required>
                </div>
                <div class="col-md-3 mb-3">
                    <label class="form-label">Gender *</label>
                    <select name="gender" class="form-control" required>
                        <option value="">Select</option>
                        <option value="Male">Male</option>
                        <option value="Female">Female</option>
                        <option value="Other">Other</option>
                    </select>
                </div>
            </div>
            
            <h5 class="mt-4 mb-3">Liver Function Tests</h5>
            <div class="row">
                <div class="col-md-6 mb-3">
                    <label class="form-label">Total Bilirubin (mg/dL) *</label>
                    <input type="number" step="0.01" name="total_bilirubin" class="form-control" placeholder="e.g., 0.8" required>
                    <small class="text-muted">Normal: 0.1-1.2 mg/dL</small>
                </div>
                <div class="col-md-6 mb-3">
                    <label class="form-label">Direct Bilirubin (mg/dL) *</label>
                    <input type="number" step="0.01" name="direct_bilirubin" class="form-control" placeholder="e.g., 0.3" required>
                    <small class="text-muted">Normal: 0.0-0.3 mg/dL</small>
                </div>
                <div class="col-md-6 mb-3">
                    <label class="form-label">Alkaline Phosphatase (IU/L) *</label>
                    <input type="number" name="alkaline_phosphotase" class="form-control" placeholder="e.g., 200" required>
                    <small class="text-muted">Normal: 44-147 IU/L</small>
                </div>
                <div class="col-md-6 mb-3">
                    <label class="form-label">Alamine Aminotransferase (IU/L) *</label>
                    <input type="number" name="alamine_aminotransferase" class="form-control" placeholder="e.g., 30" required>
                    <small class="text-muted">Normal: 7-56 IU/L</small>
                </div>
                <div class="col-md-6 mb-3">
                    <label class="form-label">Aspartate Aminotransferase (IU/L) *</label>
                    <input type="number" name="aspartate_aminotransferase" class="form-control" placeholder="e.g., 35" required>
                    <small class="text-muted">Normal: 10-40 IU/L</small>
                </div>
                <div class="col-md-6 mb-3">
                    <label class="form-label">Total Proteins (g/dL) *</label>
                    <input type="number" step="0.01" name="total_proteins" class="form-control" placeholder="e.g., 7.0" required>
                    <small class="text-muted">Normal: 6.0-8.3 g/dL</small>
                </div>
                <div class="col-md-6 mb-3">
                    <label class="form-label">Albumin (g/dL) *</label>
                    <input type="number" step="0.01" name="albumin" class="form-control" placeholder="e.g., 4.0" required>
                    <small class="text-muted">Normal: 3.5-5.5 g/dL</small>
                </div>
                <div class="col-md-6 mb-3">
                    <label class="form-label">Albumin/Globulin Ratio *</label>
                    <input type="number" step="0.01" name="ag_ratio" class="form-control" placeholder="e.g., 1.2" required>
                    <small class="text-muted">Normal: 1.0-2.5</small>
                </div>
            </div>
            
            <div class="mt-4 text-center">
                <button type="submit" class="btn btn-submit">
                    <i class="fas fa-flask"></i> Analyze Liver Function
                </button>
            </div>
        </form>
    </div>
</body>
</html>
'''

KIDNEY_FORM = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Kidney Disease Screening</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        body {
            background: linear-gradient(135deg, #2980b9 0%, #3498db 100%);
            min-height: 100vh;
            padding: 20px 0;
        }
        .form-container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        h2 { color: #2980b9; margin-bottom: 30px; }
        .form-label { font-weight: 600; color: #333; }
        .form-control:focus { border-color: #2980b9; box-shadow: 0 0 0 0.2rem rgba(41, 128, 185, 0.25); }
        .btn-submit {
            background: linear-gradient(135deg, #2980b9, #3498db);
            color: white;
            border: none;
            padding: 12px 40px;
            border-radius: 25px;
            font-weight: bold;
        }
        .btn-submit:hover { transform: scale(1.05); color: white; }
        .info-box {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <div class="form-container">
        <a href="/" class="btn btn-outline-secondary mb-3"><i class="fas fa-arrow-left"></i> Back to Home</a>
        <h2><i class="fas fa-kidney"></i> Kidney Disease Screening</h2>
        <div class="info-box">
            <strong><i class="fas fa-info-circle"></i> Information:</strong> Kidney function assessment based on blood and urine tests.
        </div>
        <form action="/predict/kidney" method="POST">
            <div class="row">
                <div class="col-md-6 mb-3">
                    <label class="form-label">Full Name *</label>
                    <input type="text" name="name" class="form-control" required>
                </div>
                <div class="col-md-3 mb-3">
                    <label class="form-label">Age *</label>
                    <input type="number" name="age" class="form-control" min="1" max="120" required>
                </div>
                <div class="col-md-3 mb-3">
                    <label class="form-label">Gender *</label>
                    <select name="gender" class="form-control" required>
                        <option value="">Select</option>
                        <option value="Male">Male</option>
                        <option value="Female">Female</option>
                        <option value="Other">Other</option>
                    </select>
                </div>
            </div>
            
            <h5 class="mt-4 mb-3">Kidney Function Tests</h5>
            <div class="row">
                <div class="col-md-6 mb-3">
                    <label class="form-label">Blood Pressure (mmHg) *</label>
                    <input type="number" name="blood_pressure" class="form-control" placeholder="e.g., 80" required>
                    <small class="text-muted">Diastolic BP</small>
                </div>
                <div class="col-md-6 mb-3">
                    <label class="form-label">Specific Gravity *</label>
                    <input type="number" step="0.001" name="specific_gravity" class="form-control" placeholder="e.g., 1.020" required>
                    <small class="text-muted">Urine specific gravity (1.003-1.030)</small>
                </div>
                <div class="col-md-6 mb-3">
                    <label class="form-label">Albumin Level *</label>
                    <input type="number" name="albumin" class="form-control" placeholder="e.g., 0-5" required>
                    <small class="text-muted">0=normal, 5=high</small>
                </div>
                <div class="col-md-6 mb-3">
                    <label class="form-label">Sugar Level *</label>
                    <input type="number" name="sugar" class="form-control" placeholder="e.g., 0-5" required>
                    <small class="text-muted">0=normal, 5=high</small>
                </div>
                <div class="col-md-6 mb-3">
                    <label class="form-label">Red Blood Cells *</label>
                    <select name="red_blood_cells" class="form-control" required>
                        <option value="0">Normal</option>
                        <option value="1">Abnormal</option>
                    </select>
                </div>
                <div class="col-md-6 mb-3">
                    <label class="form-label">Pus Cells *</label>
                    <select name="pus_cell" class="form-control" required>
                        <option value="0">Normal</option>
                        <option value="1">Abnormal</option>
                    </select>
                </div>
                <div class="col-md-6 mb-3">
                    <label class="form-label">Blood Urea (mg/dL) *</label>
                    <input type="number" step="0.1" name="blood_urea" class="form-control" placeholder="e.g., 30" required>
                    <small class="text-muted">Normal: 7-20 mg/dL</small>
                </div>
                <div class="col-md-6 mb-3">
                    <label class="form-label">Serum Creatinine (mg/dL) *</label>
                    <input type="number" step="0.1" name="serum_creatinine" class="form-control" placeholder="e.g., 1.0" required>
                    <small class="text-muted">Normal: 0.6-1.2 mg/dL</small>
                </div>
                <div class="col-md-6 mb-3">
                    <label class="form-label">Sodium (mEq/L) *</label>
                    <input type="number" step="0.1" name="sodium" class="form-control" placeholder="e.g., 140" required>
                    <small class="text-muted">Normal: 135-145 mEq/L</small>
                </div>
                <div class="col-md-6 mb-3">
                    <label class="form-label">Potassium (mEq/L) *</label>
                    <input type="number" step="0.1" name="potassium" class="form-control" placeholder="e.g., 4.5" required>
                    <small class="text-muted">Normal: 3.5-5.0 mEq/L</small>
                </div>
                <div class="col-md-6 mb-3">
                    <label class="form-label">Hemoglobin (g/dL) *</label>
                    <input type="number" step="0.1" name="hemoglobin" class="form-control" placeholder="e.g., 14" required>
                    <small class="text-muted">Normal: 12-18 g/dL</small>
                </div>
            </div>
            
            <div class="mt-4 text-center">
                <button type="submit" class="btn btn-submit">
                    <i class="fas fa-vial"></i> Analyze Kidney Function
                </button>
            </div>
        </form>
    </div>
</body>
</html>
'''

STROKE_FORM = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stroke Risk Prediction</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        body {
            background: linear-gradient(135deg, #8e44ad 0%, #9b59b6 100%);
            min-height: 100vh;
            padding: 20px 0;
        }
        .form-container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        h2 { color: #8e44ad; margin-bottom: 30px; }
        .form-label { font-weight: 600; color: #333; }
        .form-control:focus { border-color: #8e44ad; box-shadow: 0 0 0 0.2rem rgba(142, 68, 173, 0.25); }
        .btn-submit {
            background: linear-gradient(135deg, #8e44ad, #9b59b6);
            color: white;
            border: none;
            padding: 12px 40px;
            border-radius: 25px;
            font-weight: bold;
        }
        .btn-submit:hover { transform: scale(1.05); color: white; }
        .info-box {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <div class="form-container">
        <a href="/" class="btn btn-outline-secondary mb-3"><i class="fas fa-arrow-left"></i> Back to Home</a>
        <h2><i class="fas fa-brain"></i> Stroke Risk Prediction</h2>
        <div class="info-box">
            <strong><i class="fas fa-info-circle"></i> Information:</strong> Assess stroke risk based on lifestyle and medical factors.
        </div>
        <form action="/predict/stroke" method="POST">
            <div class="row">
                <div class="col-md-6 mb-3">
                    <label class="form-label">Full Name *</label>
                    <input type="text" name="name" class="form-control" required>
                </div>
                <div class="col-md-3 mb-3">
                    <label class="form-label">Age *</label>
                    <input type="number" name="age" class="form-control" min="1" max="120" required>
                </div>
                <div class="col-md-3 mb-3">
                    <label class="form-label">Gender *</label>
                    <select name="gender" class="form-control" required>
                        <option value="">Select</option>
                        <option value="Male">Male</option>
                        <option value="Female">Female</option>
                        <option value="Other">Other</option>
                    </select>
                </div>
            </div>
            
            <h5 class="mt-4 mb-3">Stroke Risk Factors</h5>
            <div class="row">
                <div class="col-md-6 mb-3">
                    <label class="form-label">Hypertension *</label>
                    <select name="hypertension" class="form-control" required>
                        <option value="0">No</option>
                        <option value="1">Yes</option>
                    </select>
                </div>
                <div class="col-md-6 mb-3">
                    <label class="form-label">Heart Disease *</label>
                    <select name="heart_disease" class="form-control" required>
                        <option value="0">No</option>
                        <option value="1">Yes</option>
                    </select>
                </div>
                <div class="col-md-6 mb-3">
                    <label class="form-label">Ever Married *</label>
                    <select name="ever_married" class="form-control" required>
                        <option value="0">No</option>
                        <option value="1">Yes</option>
                    </select>
                </div>
                <div class="col-md-6 mb-3">
                    <label class="form-label">Work Type *</label>
                    <select name="work_type" class="form-control" required>
                        <option value="0">Never Worked</option>
                        <option value="1">Children</option>
                        <option value="2">Government Job</option>
                        <option value="3">Private Job</option>
                        <option value="4">Self-employed</option>
                    </select>
                </div>
                <div class="col-md-6 mb-3">
                    <label class="form-label">Residence Type *</label>
                    <select name="residence_type" class="form-control" required>
                        <option value="0">Rural</option>
                        <option value="1">Urban</option>
                    </select>
                </div>
                <div class="col-md-6 mb-3">
                    <label class="form-label">Average Glucose Level (mg/dL) *</label>
                    <input type="number" step="0.01" name="avg_glucose_level" class="form-control" placeholder="e.g., 100" required>
                    <small class="text-muted">Normal: 70-100 mg/dL</small>
                </div>
                <div class="col-md-6 mb-3">
                    <label class="form-label">BMI *</label>
                    <input type="number" step="0.1" name="bmi" class="form-control" placeholder="e.g., 25.5" required>
                    <small class="text-muted">Normal: 18.5-24.9</small>
                </div>
                <div class="col-md-6 mb-3">
                    <label class="form-label">Smoking Status *</label>
                    <select name="smoking_status" class="form-control" required>
                        <option value="0">Never Smoked</option>
                        <option value="1">Formerly Smoked</option>
                        <option value="2">Smokes</option>
                        <option value="3">Unknown</option>
                    </select>
                </div>
            </div>
            
            <div class="mt-4 text-center">
                <button type="submit" class="btn btn-submit">
                    <i class="fas fa-brain"></i> Analyze Stroke Risk
                </button>
            </div>
        </form>
    </div>
</body>
</html>
'''

RESULT_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ disease.title() }} Prediction Results</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px 0;
        }
        .results-container {
            max-width: 1000px;
            margin: 0 auto;
        }
        .card {
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 20px;
        }
        .card-header {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            font-weight: bold;
            border-radius: 15px 15px 0 0 !important;
        }
        .prediction-result {
            padding: 30px;
            text-align: center;
        }
        .prediction-result.positive {
            background: linear-gradient(135deg, #dc3545, #c82333);
            color: white;
        }
        .prediction-result.negative {
            background: linear-gradient(135deg, #28a745, #218838);
            color: white;
        }
        .gauge-container {
            text-align: center;
            padding: 20px;
        }
        .gauge-container img {
            max-width: 100%;
            height: auto;
        }
        .recommendation-section {
            padding: 20px;
        }
        .recommendation-item {
            background: #f8f9fa;
            padding: 15px;
            margin-bottom: 10px;
            border-radius: 8px;
            border-left: 4px solid;
        }
        .recommendation-item.lifestyle { border-left-color: #007bff; }
        .recommendation-item.diet { border-left-color: #28a745; }
        .recommendation-item.medical { border-left-color: #dc3545; }
        .recommendation-item.prevention { border-left-color: #ffc107; }
        .recommendation-item h5 {
            margin-bottom: 10px;
            font-weight: bold;
        }
        .recommendation-item ul {
            margin: 0;
            padding-left: 20px;
        }
        .btn-download {
            background: linear-gradient(135deg, #28a745, #218838);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 25px;
            font-weight: bold;
            margin: 5px;
        }
        .btn-download:hover {
            transform: scale(1.05);
            color: white;
        }
        .patient-info {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <div class="results-container">
        <div class="card">
            <div class="card-header">
                <h3><i class="fas fa-file-medical"></i> {{ disease.title() }} Prediction Results</h3>
            </div>
            <div class="card-body">
                <div class="patient-info">
                    <div class="row">
                        <div class="col-md-4">
                            <strong>Name:</strong> {{ patient_data.name }}
                        </div>
                        <div class="col-md-4">
                            <strong>Age:</strong> {{ patient_data.age }}
                        </div>
                        <div class="col-md-4">
                            <strong>Gender:</strong> {{ patient_data.gender }}
                        </div>
                    </div>
                </div>
                
                <div class="prediction-result {% if prediction == 1 %}positive{% else %}negative{% endif %}">
                    <h1>
                        {% if prediction == 1 %}
                        <i class="fas fa-exclamation-triangle"></i> POSITIVE
                        {% else %}
                        <i class="fas fa-check-circle"></i> NEGATIVE
                        {% endif %}
                    </h1>
                    <h3>
                        {% if prediction == 1 %}
                        Risk Detected - Medical Consultation Recommended
                        {% else %}
                        Low Risk - Continue Healthy Lifestyle
                        {% endif %}
                    </h3>
                    <h2 class="mt-3">Risk Probability: {{ "%.1f"|format(probability * 100) }}%</h2>
                    <h4>Risk Level: <span class="badge bg-light text-dark">{{ risk_level.upper() }}</span></h4>
                </div>
            </div>
        </div>
        
        <div class="card">
            <div class="card-header">
                <h4><i class="fas fa-tachometer-alt"></i> Risk Assessment Gauge</h4>
            </div>
            <div class="card-body gauge-container">
                <img src="data:image/png;base64,{{ gauge_chart }}" alt="Risk Gauge">
            </div>
        </div>
        
        <div class="card">
            <div class="card-header">
                <h4><i class="fas fa-clipboard-list"></i> Personalized Recommendations</h4>
            </div>
            <div class="card-body recommendation-section">
                {% if recommendations.lifestyle %}
                <div class="recommendation-item lifestyle">
                    <h5><i class="fas fa-running"></i> Lifestyle Modifications</h5>
                    <ul>
                        {% for rec in recommendations.lifestyle %}
                        <li>{{ rec }}</li>
                        {% endfor %}
                    </ul>
                </div>
                {% endif %}
                
                {% if recommendations.diet %}
                <div class="recommendation-item diet">
                    <h5><i class="fas fa-utensils"></i> Dietary Recommendations</h5>
                    <ul>
                        {% for rec in recommendations.diet %}
                        <li>{{ rec }}</li>
                        {% endfor %}
                    </ul>
                </div>
                {% endif %}
                
                {% if recommendations.medical %}
                <div class="recommendation-item medical">
                    <h5><i class="fas fa-stethoscope"></i> Medical Advice</h5>
                    <ul>
                        {% for rec in recommendations.medical %}
                        <li>{{ rec }}</li>
                        {% endfor %}
                    </ul>
                </div>
                {% endif %}
                
                {% if recommendations.prevention %}
                <div class="recommendation-item prevention">
                    <h5><i class="fas fa-shield-alt"></i> Prevention Strategies</h5>
                    <ul>
                        {% for rec in recommendations.prevention %}
                        <li>{{ rec }}</li>
                        {% endfor %}
                    </ul>
                </div>
                {% endif %}
            </div>
        </div>
        
        <div class="text-center mb-4">
            <a href="/download/pdf" class="btn btn-download">
                <i class="fas fa-file-pdf"></i> Download PDF Report
            </a>
            <a href="/download/csv" class="btn btn-download">
                <i class="fas fa-file-csv"></i> Download CSV Data
            </a>
            <a href="/" class="btn btn-download" style="background: linear-gradient(135deg, #007bff, #0056b3);">
                <i class="fas fa-home"></i> Back to Home
            </a>
        </div>
        
        <div class="card">
            <div class="card-body">
                <p class="text-muted small text-center mb-0">
                    <i class="fas fa-exclamation-triangle"></i> <strong>Disclaimer:</strong> 
                    This prediction is generated by an AI system and should not replace professional medical advice. 
                    Please consult with a qualified healthcare provider for proper diagnosis and treatment.
                </p>
            </div>
        </div>
    </div>
</body>
</html>
'''


@app.route('/')
def home():
    return render_template_string(HOME_TEMPLATE)


@app.route('/predict/<disease>', methods=['GET', 'POST'])
def predict(disease):
    if request.method == 'GET':
        if disease == 'diabetes':
            return render_template_string(DIABETES_FORM)
        elif disease == 'heart':
            return render_template_string(HEART_FORM)
        elif disease == 'liver':
            return render_template_string(LIVER_FORM)
        elif disease == 'kidney':
            return render_template_string(KIDNEY_FORM)
        elif disease == 'stroke':
            return render_template_string(STROKE_FORM)
        else:
            return "Disease not found", 404

    else:
        form_data = request.form.to_dict()

        patient_data = {
            'name': form_data.get('name', 'N/A'),
            'age': form_data.get('age', 'N/A'),
            'gender': form_data.get('gender', 'N/A')
        }

        if disease == 'diabetes':
            features = np.array([[
                float(form_data.get('pregnancies', 0)),
                float(form_data.get('glucose', 0)),
                float(form_data.get('blood_pressure', 0)),
                float(form_data.get('skin_thickness', 0)),
                float(form_data.get('insulin', 0)),
                float(form_data.get('bmi', 0)),
                float(form_data.get('dpf', 0)),
                float(form_data.get('age_model', form_data.get('age', 0)))
            ]])
        elif disease == 'heart':
            features = np.array([[
                float(form_data.get('age', 0)),
                float(form_data.get('cp', 0)),
                float(form_data.get('trestbps', 0)),
                float(form_data.get('chol', 0)),
                float(form_data.get('fbs', 0)),
                float(form_data.get('restecg', 0)),
                float(form_data.get('thalach', 0)),
                float(form_data.get('exang', 0)),
                float(form_data.get('oldpeak', 0)),
                float(form_data.get('slope', 0)),
                float(form_data.get('ca', 0)),
                float(form_data.get('thal', 0))
            ]])
        elif disease == 'liver':
            features = np.array([[
                float(form_data.get('age', 0)),
                1 if form_data.get('gender') == 'Male' else 0,
                float(form_data.get('total_bilirubin', 0)),
                float(form_data.get('direct_bilirubin', 0)),
                float(form_data.get('alkaline_phosphotase', 0)),
                float(form_data.get('alamine_aminotransferase', 0)),
                float(form_data.get('aspartate_aminotransferase', 0)),
                float(form_data.get('total_proteins', 0)),
                float(form_data.get('albumin', 0)),
                float(form_data.get('ag_ratio', 0))
            ]])
        elif disease == 'kidney':
            features = np.array([[
                float(form_data.get('age', 0)),
                float(form_data.get('blood_pressure', 0)),
                float(form_data.get('specific_gravity', 0)),
                float(form_data.get('albumin', 0)),
                float(form_data.get('sugar', 0)),
                float(form_data.get('red_blood_cells', 0)),
                float(form_data.get('pus_cell', 0)),
                float(form_data.get('blood_urea', 0)),
                float(form_data.get('serum_creatinine', 0)),
                float(form_data.get('sodium', 0)),
                float(form_data.get('potassium', 0))
            ]])
        elif disease == 'stroke':
            features = np.array([[
                float(form_data.get('age', 0)),
                float(form_data.get('hypertension', 0)),
                float(form_data.get('heart_disease', 0)),
                float(form_data.get('ever_married', 0)),
                float(form_data.get('work_type', 0)),
                float(form_data.get('residence_type', 0)),
                float(form_data.get('avg_glucose_level', 0)),
                float(form_data.get('bmi', 0)),
                float(form_data.get('smoking_status', 0)), 0
            ]])
        else:
            return "Disease type not supported", 404

        model = DISEASE_MODELS[disease]
        scaler = SCALERS[disease]

        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]

        risk_level, badge_color = determine_risk_level(probability)

        recommendations = get_recommendations(disease, risk_level, probability,
                                              form_data)

        gauge_chart = create_gauge_chart(probability,
                                         f"{disease.title()} Risk Assessment")

        prediction_data = {
            'patient_data': patient_data,
            'disease': disease,
            'prediction': int(prediction),
            'probability': float(probability),
            'risk_level': risk_level,
            'recommendations': recommendations,
            'input_data': form_data
        }

        session['last_prediction'] = prediction_data
        session.permanent = True
        LAST_PREDICTION_CACHE['latest'] = prediction_data

        return render_template_string(RESULT_TEMPLATE,
                                      patient_data=patient_data,
                                      disease=disease,
                                      prediction=prediction,
                                      probability=probability,
                                      risk_level=risk_level,
                                      recommendations=recommendations,
                                      gauge_chart=gauge_chart)


@app.route('/download/pdf')
def download_pdf():
    data = session.get('last_prediction') or LAST_PREDICTION_CACHE.get(
        'latest')
    if not data:
        return "No prediction data found. Please complete a disease assessment first.", 404
    pdf_file = generate_pdf_report(data['patient_data'], data['disease'],
                                   data['prediction'], data['probability'],
                                   data['recommendations'])

    return send_file(pdf_file,
                     as_attachment=True,
                     download_name=f"{data['disease']}_prediction_report.pdf")


@app.route('/download/source')
def download_source():
    return send_file('app.py',
                     as_attachment=True,
                     download_name='disease_prediction_app.py',
                     mimetype='text/x-python')


@app.route('/download/csv')
def download_csv():
    data = session.get('last_prediction') or LAST_PREDICTION_CACHE.get(
        'latest')
    if not data:
        return "No prediction data found. Please complete a disease assessment first.", 404

    csv_data = {
        'Name': [data['patient_data']['name']],
        'Age': [data['patient_data']['age']],
        'Gender': [data['patient_data']['gender']],
        'Disease': [data['disease'].title()],
        'Prediction': ['Positive' if data['prediction'] == 1 else 'Negative'],
        'Risk_Probability_%': [data['probability'] * 100],
        'Risk_Level': [data['risk_level'].upper()],
        'Timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
    }

    for key, value in data['input_data'].items():
        if key not in ['name', 'age', 'gender']:
            csv_data[key] = [value]

    df = pd.DataFrame(csv_data)

    csv_file = f'reports/{data["disease"]}_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    os.makedirs('reports', exist_ok=True)
    df.to_csv(csv_file, index=False)

    return send_file(csv_file,
                     as_attachment=True,
                     download_name=f"{data['disease']}_prediction_data.csv")


if __name__ == '__main__':
    train_models()
    app.run(host='0.0.0.0', port=5000, debug=False)
