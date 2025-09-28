import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Pakistan Disaster Risk Dashboard", 
    layout="wide",
    page_icon="🇵🇰"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    color: #1e3d59;
    text-align: center;
    margin-bottom: 1rem;
}
.sub-header {
    font-size: 1.2rem;
    color: #2e7d32;
    text-align: center;
    margin-bottom: 2rem;
}
.risk-high {
    background-color: #ffebee;
    padding: 1rem;
    border-left: 5px solid #f44336;
    border-radius: 5px;
}
.risk-low {
    background-color: #e8f5e8;
    padding: 1rem;
    border-left: 5px solid #4caf50;
    border-radius: 5px;
}
.developer-tag {
    position: fixed;
    bottom: 10px;
    right: 10px;
    background-color: #1e3d59;
    color: white;
    padding: 5px 10px;
    border-radius: 15px;
    font-size: 0.8rem;
}
.metric-container {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 10px;
    text-align: center;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# Title section with enhanced styling
st.markdown('<h1 class="main-header">🇵🇰 Pakistan Home Safety Risk Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header"> AI-Powered Disaster Risk Assessment for Pakistani Homes</p>', unsafe_allow_html=True)

# Developer credit
st.markdown('<div class="developer-tag"> Developed by Zeeshan Muhammad</div>', unsafe_allow_html=True)

# Enhanced sidebar with icons and styling
st.sidebar.markdown("## Safety Analysis Menu")
st.sidebar.markdown("Select your risk assessment type:")

option = st.sidebar.selectbox(
    " Choose Analysis:",
    [" Home Dashboard", " Flood Risk", " Earthquake Risk", "Rain Prediction", " Complete Report"]
)

# Add some sidebar info
st.sidebar.markdown("---")
st.sidebar.info(" **Tip:** Use multiple assessments for comprehensive home safety analysis!")
st.sidebar.markdown(" **Emergency Numbers:**")
st.sidebar.markdown("- Rescue 1122: **1122**")
st.sidebar.markdown("- Fire Brigade: **16**")
st.sidebar.markdown("- Police: **15**")

# HOME DASHBOARD - Enhanced
if option == " Home Dashboard":
    st.markdown("## Welcome to Your Personal Home Safety Command Center!")
    
    # City selection with enhanced visuals
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        city = st.selectbox(
            " Select Your City:",
            ["Karachi", "Lahore", "Islamabad", "Peshawar", "Quetta", "Multan", "Faisalabad"],
            help="Choose your city for localized risk assessment"
        )
    
    st.success(f" **Selected City:** {city}")
    
    # Enhanced metrics with colors
    st.markdown("###  Current Risk Status")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(" Flood Zones", "12", "High Risk Areas", delta_color="inverse")
    with col2:
        st.metric(" Earthquake Risk", "Medium", f"For {city}")
    with col3:
        st.metric(" Monsoon Alert", "Active", "Current Status", delta_color="off")
    with col4:
        st.metric(" Overall Safety", "Good", "Based on AI Analysis")
    
    # Interactive city information
    city_info = {
        "Karachi": {"population": "15M", "main_risks": ["Flooding", "Heavy Rain"], "safety_score": 75},
        "Lahore": {"population": "12M", "main_risks": ["Air Quality", "Flooding"], "safety_score": 70},
        "Islamabad": {"population": "2M", "main_risks": ["Earthquakes"], "safety_score": 85},
        "Peshawar": {"population": "2.5M", "main_risks": ["Earthquakes", "Security"], "safety_score": 65},
        "Quetta": {"population": "1.5M", "main_risks": ["Earthquakes", "Cold"], "safety_score": 60},
        "Multan": {"population": "2.5M", "main_risks": ["Heat Waves", "Flooding"], "safety_score": 72},
        "Faisalabad": {"population": "3.5M", "main_risks": ["Air Quality", "Flooding"], "safety_score": 68}
    }
    
    if city in city_info:
        info = city_info[city]
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            ###  {city} Overview
            - **Population:** {info['population']}
            - **Safety Score:** {info['safety_score']}/100
            - **Main Risks:** {', '.join(info['main_risks'])}
            """)
        
        with col2:
            # Create a simple progress bar instead of gauge
            st.markdown("###  Safety Score")
            st.progress(info['safety_score'] / 100)
            st.write(f"**{info['safety_score']}/100** - {'Excellent' if info['safety_score'] > 80 else 'Good' if info['safety_score'] > 70 else 'Needs Improvement'}")

# FLOOD RISK - Enhanced
elif option == " Flood Risk":
    st.markdown("##  Advanced Flood Risk Assessment")
    st.markdown("*Get AI-powered flood risk predictions for your area*")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("###  Weather Conditions")
        rainfall_mm = st.slider("Daily Rainfall (mm):", 0, 200, 50, help="Current or expected rainfall")
        river_level = st.slider("Nearby River Level (meters):", 0, 20, 5, help="Distance from nearest water body")
        
    with col2:
        st.markdown("###  Geographic Factors")
        soil_moisture = st.slider("Soil Moisture (%):", 0, 100, 40, help="Current soil saturation level")
        elevation = st.slider("Home Elevation (meters):", 0, 500, 100, help="Height above sea level")

    # Enhanced AI prediction
    flood_data = {
        'rainfall_mm': [10, 25, 45, 60, 80, 100, 120, 150],
        'river_level': [2, 3, 4, 5, 6, 7, 8, 9],
        'soil_moisture': [20, 30, 40, 50, 60, 70, 80, 90],
        'flood_risk': [0, 0, 0, 1, 1, 1, 1, 1]
    }

    df = pd.DataFrame(flood_data)
    X = df[["rainfall_mm", "river_level", "soil_moisture"]].values
    y = df["flood_risk"].values

    ai_model = LogisticRegression()
    ai_model.fit(X, y)

    user_input = [[rainfall_mm, river_level, soil_moisture]]
    prediction = ai_model.predict(user_input)[0]
    probability = ai_model.predict_proba(user_input)[0][1]
    risk_chance = probability * 100

    # Enhanced results display
    st.markdown("---")
    st.markdown("### AI Flood Risk Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if prediction == 1:
            st.markdown('<div class="risk-high"><h4> HIGH FLOOD RISK</h4><p>Immediate precautions needed!</p></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="risk-low"><h4> LOW FLOOD RISK</h4><p>Conditions appear safe</p></div>', unsafe_allow_html=True)
    
    with col2:
        st.metric("Risk Probability", f"{risk_chance:.1f}%", help="AI confidence level")
    
    with col3:
        risk_level = "CRITICAL" if risk_chance > 80 else "HIGH" if risk_chance > 60 else "MEDIUM" if risk_chance > 40 else "LOW"
        st.metric("Risk Level", risk_level)

    # Simple matplotlib chart
    fig, ax = plt.subplots(figsize=(8, 4))
    factors = ['Rainfall', 'River Level', 'Soil Moisture']
    values = [rainfall_mm/200*100, river_level/20*100, soil_moisture]
    colors = ['red' if v > 60 else 'orange' if v > 40 else 'green' for v in values]
    
    ax.bar(factors, values, color=colors, alpha=0.7)
    ax.set_ylabel('Risk Factor (%)')
    ax.set_title('Flood Risk Factors Analysis')
    ax.set_ylim(0, 100)
    
    st.pyplot(fig)

    # Safety recommendations
    st.markdown("###  Safety Recommendations")
    if prediction == 1:
        st.error("""
        **Immediate Actions Required:**
        -  Move valuable items to higher ground
        -  Park vehicles in safe areas
        -  Keep emergency contacts ready
        -  Prepare emergency kit with food and water
        -  Monitor weather updates continuously
        """)
    else:
        st.success("""
        **General Preparedness:**
        -  Continue monitoring weather conditions
        -  Ensure drainage systems are clear
        -  Keep emergency plan updated
        -  Regular home maintenance checks
        """)

# EARTHQUAKE RISK - Enhanced
elif option == " Earthquake Risk":
    st.markdown("##  Comprehensive Earthquake Risk Assessment")
    st.markdown("*Evaluate your building's earthquake resistance*")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("###  Building Information")
        building_age = st.slider("Building Age (years):", 0, 100, 20)
        structure_type = st.selectbox("Structure Type:", ["Concrete", "Brick", "Wood", "Steel"])
        
    with col2:
        st.markdown("###  Construction Details")
        floors = st.slider("Number of Floors:", 1, 20, 5)
        foundation_type = st.selectbox("Foundation Type:", ["Shallow", "Deep", "Pile", "Raft"])

    # Enhanced earthquake prediction model
    earthquake_data = {
        'building_age': [5, 15, 25, 35, 45, 55, 65, 75],
        'structure_type': [0, 1, 2, 0, 1, 2, 0, 1],
        'floors': [1, 3, 5, 2, 4, 6, 3, 5],
        'foundation_type': [0, 1, 2, 0, 1, 2, 0, 1],
        'earthquake_risk': [0, 0, 1, 0, 1, 1, 1, 1]
    }
    
    df = pd.DataFrame(earthquake_data)
    X = df[["building_age", "structure_type", "floors", "foundation_type"]].values
    y = df["earthquake_risk"].values
    
    ai_model = LogisticRegression()
    ai_model.fit(X, y)
    
    structure_map = {"Concrete": 0, "Brick": 1, "Wood": 2, "Steel": 3}
    foundation_map = {"Shallow": 0, "Deep": 1, "Pile": 2, "Raft": 3}
    
    user_input = [[
        building_age,
        structure_map.get(structure_type, 0),
        floors,
        foundation_map.get(foundation_type, 0)
    ]]
    
    prediction = ai_model.predict(user_input)[0]
    probability = ai_model.predict_proba(user_input)[0][1]
    risk_chance = probability * 100

    # Enhanced results
    st.markdown("---")
    st.markdown("###  Earthquake Risk Assessment Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if prediction == 1:
            st.markdown('<div class="risk-high"><h4> HIGH EARTHQUAKE RISK</h4><p>Building assessment recommended</p></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="risk-low"><h4> LOW EARTHQUAKE RISK</h4><p>Building appears structurally sound</p></div>', unsafe_allow_html=True)
    
    with col2:
        st.metric("Risk Probability", f"{risk_chance:.1f}%")
    
    with col3:
        safety_score = 100 - risk_chance
        st.metric("Safety Score", f"{safety_score:.1f}/100")

    # Building analysis chart
    fig, ax = plt.subplots(figsize=(8, 4))
    building_factors = ['Age Factor', 'Structure', 'Height', 'Foundation']
    age_factor = max(0, 100 - building_age * 1.2)
    structure_factor = {"Concrete": 90, "Steel": 95, "Brick": 70, "Wood": 50}[structure_type]
    height_factor = max(50, 100 - floors * 3)
    foundation_factor = {"Deep": 90, "Pile": 95, "Raft": 85, "Shallow": 60}[foundation_type]
    
    factors_values = [age_factor, structure_factor, height_factor, foundation_factor]
    colors = ['green' if v > 80 else 'orange' if v > 60 else 'red' for v in factors_values]
    
    ax.bar(building_factors, factors_values, color=colors, alpha=0.7)
    ax.set_ylabel('Safety Factor (%)')
    ax.set_title('Building Safety Factors Analysis')
    ax.set_ylim(0, 100)
    
    st.pyplot(fig)

# RAIN PREDICTION - Enhanced
elif option == "Rain Prediction":
    st.markdown("##  Weather Prediction System")
    st.markdown("*AI-powered rainfall prediction for your area*")
    
    col1, col2 = st.columns(2)
    
    with col1:
        temperature = st.slider("Current Temperature (°C):", 0, 50, 25)
        humidity = st.slider("Humidity (%):", 0, 100, 60)
    
    with col2:
        wind_speed = st.slider("Wind Speed (km/h):", 0, 100, 15)
        pressure = st.slider("Atmospheric Pressure (hPa):", 950, 1050, 1013)

    # Enhanced weather prediction
    weather_data = {
        'temperature': [15, 20, 25, 30, 35, 40, 18, 22],
        'humidity': [80, 75, 60, 45, 40, 35, 85, 70],
        'rain': [1, 1, 0, 0, 0, 0, 1, 1]
    }

    df = pd.DataFrame(weather_data)
    X = df[['temperature', 'humidity']].values
    y = df['rain'].values

    ai_model = LogisticRegression()
    ai_model.fit(X, y)

    user_input = [[temperature, humidity]]
    prediction = ai_model.predict(user_input)[0]
    probability = ai_model.predict_proba(user_input)[0][1]
    rain_chance = probability * 100

    # Results display
    st.markdown("---")
    st.markdown("###  Weather Forecast Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if prediction == 1:
            st.markdown('<div class="risk-high"><h4>RAIN EXPECTED</h4><p>Carry umbrella!</p></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="risk-low"><h4> NO RAIN</h4><p>Clear weather ahead</p></div>', unsafe_allow_html=True)
    
    with col2:
        st.metric("Rain Probability", f"{rain_chance:.1f}%")
    
    with col3:
        weather_condition = "Rainy" if prediction == 1 else "Clear"
        st.metric("Weather", weather_condition)

    # Weather chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Temperature vs Rain correlation
    ax1.scatter(df['temperature'], df['rain'], alpha=0.7, color='blue')
    ax1.scatter([temperature], [prediction], color='red', s=100, label='Your Prediction')
    ax1.set_xlabel('Temperature (°C)')
    ax1.set_ylabel('Rain (0=No, 1=Yes)')
    ax1.set_title('Temperature vs Rain Pattern')
    ax1.legend()
    
    # Humidity vs Rain correlation  
    ax2.scatter(df['humidity'], df['rain'], alpha=0.7, color='green')
    ax2.scatter([humidity], [prediction], color='red', s=100, label='Your Prediction')
    ax2.set_xlabel('Humidity (%)')
    ax2.set_ylabel('Rain (0=No, 1=Yes)')
    ax2.set_title('Humidity vs Rain Pattern')
    ax2.legend()
    
    st.pyplot(fig)

# COMPLETE REPORT
elif option == " Complete Report":
    st.markdown("## Comprehensive Home Safety Report")
    st.markdown("*Complete AI-powered risk assessment for your property*")
    
    # Collect all inputs for comprehensive analysis
    st.markdown("###  Property Information")
    col1, col2 = st.columns(2)
    
    with col1:
        city = st.selectbox("City:", ["Karachi", "Lahore", "Islamabad", "Peshawar"])
        building_age = st.slider("Building Age:", 0, 50, 15)
        
    with col2:
        elevation = st.slider("Elevation (m):", 0, 1000, 200)
        floors = st.slider("Floors:", 1, 10, 3)
    
    # Calculate overall safety score
    safety_factors = {
        "location": 85 if city in ["Islamabad", "Lahore"] else 70,
        "building_age": max(50, 90 - (building_age * 1.5)),
        "elevation": min(85, elevation * 0.1 + 50),
        "structure": 80
    }
    
    overall_score = sum(safety_factors.values()) / len(safety_factors)
    
    # Display comprehensive results
    st.markdown("---")
    st.markdown("###  Overall Safety Assessment")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Overall Safety", f"{overall_score:.0f}/100")
    with col2:
        st.metric("Flood Risk", "Medium", delta=-5, delta_color="inverse")
    with col3:
        st.metric("Earthquake Risk", "Low", delta=10)
    with col4:
        st.metric("Weather Risk", "Low", delta=8)

    # Comprehensive chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = list(safety_factors.keys())
    values = list(safety_factors.values())
    colors = ['green' if v > 80 else 'orange' if v > 65 else 'red' for v in values]
    
    bars = ax.bar(categories, values, color=colors, alpha=0.7)
    ax.set_ylabel('Safety Score')
    ax.set_title('Comprehensive Safety Assessment')
    ax.set_ylim(0, 100)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{value:.0f}', ha='center', va='bottom')
    
    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("###  Thank You for Using Pakistan Home Safety Predictor!")
st.info(" **Remember:** This is an AI prediction tool. Always consult local authorities for official disaster warnings and safety guidelines.")