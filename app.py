import os
import sqlite3
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, Response

app = Flask(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'customer_satisfaction.db')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'final_rf_under.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')
FEATURES_PATH = os.path.join(BASE_DIR, 'models', 'features.pkl')

# Real training dataset label encoding mappings
CUSTOMER_STATE_MAPPING = {
    "SP": 25, "BA": 4, "GO": 8, "RN": 19, "PR": 17, "RJ": 18, "RS": 22, "MG": 10, "SC": 23, 
    "RR": 21, "PE": 15, "TO": 26, "CE": 5, "DF": 6, "SE": 24, "MT": 12, "PB": 14, "PA": 13, 
    "RO": 20, "ES": 7, "AP": 3, "MS": 11, "MA": 9, "PI": 16, "AL": 1, "AC": 0, "AM": 2
}

PRODUCT_CATEGORY_MAPPING = {
    "housewares": 49, "perfumery": 59, "auto": 5, "pet_shop": 60, "stationery": 66, "furniture_decor": 39, 
    "office_furniture": 57, "garden_tools": 42, "computers_accessories": 15, "bed_bath_table": 7, "toys": 69, 
    "telephony": 68, "health_beauty": 43, "electronics": 26, "baby": 6, "cool_stuff": 20, "watches_gifts": 71, 
    "air_conditioning": 1, "sports_leisure": 65, "books_general_interest": 8, "construction_tools_construction": 17, 
    "small_appliances": 63, "food": 36, "fashion_underwear_beach": 33, "unknown": 70, "fashion_bags_accessories": 28, 
    "musical_instruments": 56, "luggage_accessories": 53, "construction_tools_lights": 18, "books_technical": 10, 
    "home_appliances": 44, "market_place": 54, "agro_industry_and_commerce": 0, "party_supplies": 58, "home_confort": 47, 
    "cds_dvds_musicals": 11, "consoles_games": 16, "furniture_bedroom": 38, "construction_tools_safety": 19, 
    "fixed_telephony": 34, "drinks": 24, "kitchen_dining_laundry_garden_furniture": 51, "fashion_shoes": 31, 
    "home_construction": 48, "audio": 4, "home_appliances_2": 45, "cine_photo": 13, "furniture_living_room": 40, 
    "industry_commerce_and_business": 50, "art": 2, "fashion_male_clothing": 30, "costruction_tools_garden": 21, 
    "christmas_supplies": 12, "food_drink": 37, "tablets_printing_image": 67, "fashion_sport": 32, "la_cuisine": 52, 
    "flowers": 35, "computers": 14, "home_comfort_2": 46, "small_appliances_home_oven_and_coffee": 64, 
    "dvds_blu_ray": 25, "costruction_tools_tools": 22, "furniture_mattress_and_upholstery": 41, 
    "signaling_and_security": 62, "fashio_female_clothing": 27, "diapers_and_hygiene": 23, "books_imported": 9, 
    "music": 55, "arts_and_craftmanship": 3, "fashion_childrens_clothes": 29, "security_and_services": 61
}

# Global SHAP Importances (pre-calculated from training dataset)
SHAP_IMPORTANCES = {
    'delivery_delay_days': 0.049605,
    'total_fulfillment_days': 0.042859,
    'total_items': 0.041069,
    'is_late': 0.032470,
    'total_freight': 0.016629,
    'product_category_name_english_encoded': 0.015895,
    'dispatch_delay_days': 0.015309,
    'avg_item_price': 0.011806,
    'total_payment_value': 0.010848,
    'total_price': 0.008812,
    'estimated_delivery_window': 0.008549,
    'order_month': 0.008244,
    'order_confirmation_delay_hrs': 0.008037,
    'payment_installments': 0.007598,
    'seller_state_encoded': 0.007240,
    'freight_ratio': 0.007202,
    'order_hour': 0.006318,
    'customer_state_encoded': 0.005314,
    'order_dayofweek': 0.004625,
    'payment_type_encoded': 0.002235
}

# Database Helpers
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS prediction_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            total_price REAL NOT NULL,
            total_items INTEGER NOT NULL,
            prediction TEXT NOT NULL,
            probability REAL NOT NULL,
            risk_level TEXT NOT NULL
        )
    ''')
    conn.commit()
    
    # Check if we need to seed the database with mock records
    cursor.execute("SELECT COUNT(*) FROM prediction_logs")
    count = cursor.fetchone()[0]
    if count == 0:
        mock_records = [
            ('2026-06-23 10:45:00', 120.50, 3, 'DISSATISFIED', 82.40, 'High Risk'),
            ('2026-06-23 10:30:00', 89.90, 1, 'SATISFIED', 23.15, 'Low Risk'),
            ('2026-06-23 10:12:00', 250.00, 5, 'DISSATISFIED', 67.80, 'Medium Risk'),
            ('2026-06-23 09:55:00', 45.30, 2, 'SATISFIED', 18.60, 'Low Risk'),
            ('2026-06-23 09:40:00', 150.75, 4, 'SATISFIED', 49.20, 'Medium Risk'),
            ('2026-06-23 09:20:00', 99.99, 1, 'SATISFIED', 21.30, 'Low Risk'),
            ('2026-06-23 09:05:00', 310.00, 6, 'DISSATISFIED', 79.10, 'High Risk')
        ]
        cursor.executemany('''
            INSERT INTO prediction_logs (timestamp, total_price, total_items, prediction, probability, risk_level)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', mock_records)
        conn.commit()
    conn.close()

# Initialize DB on startup
init_db()

# Load ML components
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    features_list = joblib.load(FEATURES_PATH)
    model_loaded = True
except Exception as e:
    print(f"Error loading model components: {e}")
    model_loaded = False

@app.route('/')
def home():
    states = sorted(list(CUSTOMER_STATE_MAPPING.keys()))
    categories = sorted(list(PRODUCT_CATEGORY_MAPPING.keys()))
    return render_template('predict_input.html', states=states, categories=categories)

@app.route('/predict', methods=['POST'])
def predict():
    if not model_loaded:
        return "Model components not loaded. Please ensure the models/ folder contains the required pickle files.", 500
        
    try:
        # 1. Read 8 raw inputs from form
        total_price = float(request.form.get('total_price', 0))
        total_freight = float(request.form.get('total_freight', 0))
        total_items = int(request.form.get('total_items', 1))
        payment_installments = int(request.form.get('payment_installments', 1))
        customer_state = request.form.get('customer_state', 'SP')
        product_category = request.form.get('product_category', 'health_beauty')
        dispatch_delay_days = float(request.form.get('dispatch_delay_days', 0))
        delivery_delay_days = float(request.form.get('delivery_delay_days', 0))
        
        # 2. Backend Feature Engineering (Automatic)
        total_payment_value = total_price + total_freight
        freight_ratio = total_freight / total_price if total_price > 0 else 0
        avg_item_price = total_price / total_items if total_items > 0 else 0
        
        # Operational baseline variables
        estimated_delivery_window = 23 # Mean from the training dataset
        order_confirmation_delay_hrs = 0.5
        
        # Calculate total fulfillment days and late status
        total_fulfillment_days = estimated_delivery_window + delivery_delay_days
        is_late = 1 if delivery_delay_days > 0 else 0
        
        # Target map categoricals using training mappings
        customer_state_encoded = CUSTOMER_STATE_MAPPING.get(customer_state, 25)
        product_category_name_english_encoded = PRODUCT_CATEGORY_MAPPING.get(product_category, 43)
        
        payment_type_encoded = 1  # default: credit_card
        seller_state_encoded = 21  # default: SP
        
        # Datetime features
        now = datetime.now()
        order_month = now.month
        order_dayofweek = now.weekday()
        order_hour = now.hour
        
        # 3. Scaling numerical features
        num_cols_data = [
            total_price,
            total_freight,
            total_payment_value,
            freight_ratio,
            estimated_delivery_window,
            order_confirmation_delay_hrs,
            dispatch_delay_days,
            delivery_delay_days,
            total_fulfillment_days,
            payment_installments,
            total_items
        ]
        
        num_cols = [
            'total_price', 'total_freight', 'total_payment_value', 'freight_ratio', 
            'estimated_delivery_window', 'order_confirmation_delay_hrs', 'dispatch_delay_days', 
            'delivery_delay_days', 'total_fulfillment_days', 'payment_installments', 'total_items'
        ]
        num_df = pd.DataFrame([num_cols_data], columns=num_cols)
        scaled_vals = scaler.transform(num_df)[0]
        
        # 4. Construct 20-dimensional feature vector in exact order
        feat_dict = {
            'total_price': scaled_vals[0],
            'total_freight': scaled_vals[1],
            'total_items': scaled_vals[10],
            'total_payment_value': scaled_vals[2],
            'payment_installments': scaled_vals[9],
            'freight_ratio': scaled_vals[3],
            'avg_item_price': avg_item_price,
            'estimated_delivery_window': scaled_vals[4],
            'order_month': order_month,
            'order_dayofweek': order_dayofweek,
            'order_hour': order_hour,
            'order_confirmation_delay_hrs': scaled_vals[5],
            'dispatch_delay_days': scaled_vals[6],
            'delivery_delay_days': scaled_vals[7],
            'total_fulfillment_days': scaled_vals[8],
            'is_late': is_late,
            'payment_type_encoded': payment_type_encoded,
            'customer_state_encoded': customer_state_encoded,
            'product_category_name_english_encoded': product_category_name_english_encoded,
            'seller_state_encoded': seller_state_encoded
        }
        
        feat_df = pd.DataFrame([[feat_dict[f] for f in features_list]], columns=features_list)
        
        # 5. Run ML Model Prediction
        prob_dissatisfied = model.predict_proba(feat_df)[0][1]
        prob_percentage = prob_dissatisfied * 100.0
        
        # Prediction Output (0 = SATISFIED, 1 = DISSATISFIED)
        prediction_class = int(prob_dissatisfied >= 0.60)
        prediction_text = 'DISSATISFIED' if prediction_class == 1 else 'SATISFIED'
        
        # Risk Levels:
        if prob_percentage < 35.0:
            risk_level = 'Low Risk'
        elif prob_percentage < 65.0:
            risk_level = 'Medium Risk'
        else:
            risk_level = 'High Risk'
            
        # 6. SHAP-based local feature contribution analysis
        local_contributions = []
        
        num_feat_mappings = {
            'total_price': (scaled_vals[0], 'Total Price'),
            'total_freight': (scaled_vals[1], 'Freight Value'),
            'total_items': (scaled_vals[10], 'Total Items'),
            'payment_installments': (scaled_vals[9], 'Payment Installments'),
            'dispatch_delay_days': (scaled_vals[6], 'Dispatch Delay'),
            'delivery_delay_days': (scaled_vals[7], 'Delivery Delay'),
            'total_fulfillment_days': (scaled_vals[8], 'Total Fulfillment Time'),
        }
        
        for feat_name, (z_score, display_name) in num_feat_mappings.items():
            global_importance = SHAP_IMPORTANCES.get(feat_name, 0.01)
            contribution = z_score * global_importance
            local_contributions.append((display_name, contribution))
            
        is_late_z = (is_late - 0.10) / 0.30
        local_contributions.append(('Late Delivery Status', is_late_z * SHAP_IMPORTANCES['is_late']))
        
        # Sort contributions descending
        local_contributions.sort(key=lambda x: x[1], reverse=True)
        
        top_factors = []
        for name, contr in local_contributions[:3]:
            if contr >= 0.05:
                impact = 'High Impact'
            elif contr >= 0.01:
                impact = 'Medium Impact'
            else:
                impact = 'Low Impact'
            top_factors.append({'name': name, 'impact': impact})
            
        # 7. Generate Actionable Business Recommendations
        recommendations = []
        if is_late or delivery_delay_days > 0:
            recommendations.append("Ensure timely delivery by reviewing logistic partner performance.")
        if dispatch_delay_days > 2:
            recommendations.append("Reduce order processing and dispatch delay at the warehouse.")
        if total_freight > 30:
            recommendations.append("Optimize freight packaging and shipping profiles for high-freight orders.")
        if total_items > 3:
            recommendations.append("Streamline inventory aggregation checks for multi-item orders.")
        
        if not recommendations:
            recommendations.append("Monitor customer experience metrics and maintain communication.")
            recommendations.append("Perform regular checks on carrier transit times.")
            
        # 8. Save to SQLite
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO prediction_logs (total_price, total_items, prediction, probability, risk_level)
            VALUES (?, ?, ?, ?, ?)
        ''', (total_price, total_items, prediction_text, round(prob_percentage, 2), risk_level))
        conn.commit()
        conn.close()
        
        return render_template(
            'predict_result.html',
            prediction=prediction_text,
            probability=round(prob_percentage, 2),
            risk_level=risk_level,
            top_factors=top_factors,
            recommendations=recommendations
        )
        
    except Exception as e:
        return f"An error occurred during prediction: {e}", 400

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/history')
def history():
    page = request.args.get('page', 1, type=int)
    per_page = 5
    offset = (page - 1) * per_page
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM prediction_logs")
    total_records = cursor.fetchone()[0]
    total_pages = (total_records + per_page - 1) // per_page
    
    cursor.execute('''
        SELECT * FROM prediction_logs
        ORDER BY timestamp DESC
        LIMIT ? OFFSET ?
    ''', (per_page, offset))
    logs = cursor.fetchall()
    conn.close()
    
    return render_template(
        'history.html',
        logs=logs,
        page=page,
        total_pages=total_pages,
        total_records=total_records
    )

@app.route('/export_csv')
def export_csv():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, timestamp, total_price, total_items, prediction, probability, risk_level 
        FROM prediction_logs 
        ORDER BY timestamp DESC
    ''')
    logs = cursor.fetchall()
    conn.close()
    
    csv_data = "ID,Date & Time,Total Price (R$),Total Items,Prediction,Probability (%),Risk Level\n"
    for log in logs:
        dt = datetime.strptime(log['timestamp'], '%Y-%m-%d %H:%M:%S')
        formatted_date = dt.strftime('%d-%m-%Y %H:%M %p')
        csv_data += f"{log['id']},{formatted_date},{log['total_price']:.2f},{log['total_items']},{log['prediction']},{log['probability']:.2f},{log['risk_level']}\n"
        
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=prediction_history.csv"}
    )

if __name__ == '__main__':
    app.run(debug=True)
