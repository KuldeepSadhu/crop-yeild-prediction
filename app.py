import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os

# Set Streamlit Page Config
st.set_page_config(
    page_title="AgriYield AI | Smart Crop Production Predictor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Main Background Header Accent */
    .hero-header {
        background: linear-gradient(135deg, #0d3b2e 0%, #165b4c 50%, #208b6e 100%);
        padding: 2rem 2.5rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(13, 59, 46, 0.25);
        position: relative;
        overflow: hidden;
    }
    
    .hero-header h1 {
        font-size: 2.3rem;
        font-weight: 800;
        margin-bottom: 0.4rem;
        color: #ffffff;
    }
    
    .hero-header p {
        font-size: 1.05rem;
        color: #b2e3d3;
        margin-bottom: 0;
    }
    
    /* Card Component Styles */
    .metric-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        border: 1px solid #eef2f5;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
    }
    
    .metric-title {
        font-size: 0.88rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 2.1rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.2rem;
    }
    
    .metric-unit {
        font-size: 0.9rem;
        color: #10b981;
        font-weight: 600;
    }

    /* Section Cards */
    .section-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }

    .stButton>button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        font-weight: 700;
        font-size: 1.05rem;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        border: none;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.35);
        transition: all 0.3s ease;
        width: 100%;
    }

    .stButton>button:hover {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.5);
        transform: translateY(-1px);
        color: white;
    }
    
    /* Result Badge */
    .result-container {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        border: 1.5px solid #a7f3d0;
        border-radius: 18px;
        padding: 1.8rem;
        text-align: center;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
    }

    .result-val {
        font-size: 2.8rem;
        font-weight: 800;
        color: #047857;
    }

    .result-sub {
        font-size: 1.1rem;
        color: #065f46;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_assets():
    base_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '.'
    model_path = os.path.join(base_dir, 'model.pkl')
    prep_path = os.path.join(base_dir, 'preprocessor.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(prep_path):
        model_path = 'model.pkl'
        prep_path = 'preprocessor.pkl'
        
    if not os.path.exists(model_path) or not os.path.exists(prep_path):
        st.error(f"Required model files ('{model_path}' or '{prep_path}') not found.")
        st.stop()
        
    try:
        model = joblib.load(model_path)
        preprocessor = joblib.load(prep_path)
        return model, preprocessor
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

model, preprocessor = load_assets()

# Extract Categorical & Numerical Feature Information dynamically
@st.cache_data
def get_feature_metadata():
    categories_dict = {}
    try:
        # Look for transformers in preprocessor
        for name, trans, cols in preprocessor.transformers_:
            if name == 'cat' and hasattr(trans, 'categories_'):
                for col_name, cat_list in zip(cols, trans.categories_):
                    categories_dict[col_name] = sorted(list(cat_list))
    except Exception as e:
        st.warning(f"Could not extract dynamic categories: {e}")
        
    return categories_dict

cat_options = get_feature_metadata()

# Fallback values if extraction returns empty
states = cat_options.get('State_Name', ['Andhra Pradesh', 'Assam', 'Bihar', 'Chhattisgarh'])
districts = cat_options.get('District_Name', ['GUNTUR', 'PATNA', 'AURANGABAD'])

# Clean display version for season
raw_seasons = cat_options.get('Season', ['Kharif     ', 'Rabi       ', 'Whole Year '])
season_map = {s.strip(): s for s in raw_seasons}
seasons_display = list(season_map.keys())

crops = cat_options.get('Crop', ['Rice', 'Maize', 'Sugarcane', 'Wheat', 'Cotton(lint)'])

# Header Banner
st.markdown("""
<div class="hero-header">
    <h1>🌾 AgriYield AI — Crop Production Predictor</h1>
    <p>Predict agricultural production & optimize crop yield using machine learning trained on historical yield & climatic metrics.</p>
</div>
""", unsafe_allow_html=True)


# Sidebar Configuration
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1500382017468-9049fed747ef?w=600&auto=format&fit=crop&q=80", use_container_width=True)
    st.markdown("### ⚙️ App Controls")
    
    app_mode = st.radio(
        "Select Mode:",
        ["🎯 Single Field Prediction", "📂 Batch CSV Prediction", "📈 Model Insights & Analytics"],
        index=0
    )
    
    # st.divider()
    # st.markdown("#### 💡 Quick Tip")
    # st.info("Input weather and soil metrics along with field area to calculate expected crop production in Tonnes.")
    
    # st.caption("Powered by Scikit-Learn RandomForest Pipeline")


# Mode 1: Single Field Prediction
if app_mode == "🎯 Single Field Prediction":
    st.markdown("### 📋 Enter Field & Environmental Parameters")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.markdown("#### 📍 Location & Crop Information")
            
            selected_state = st.selectbox("State Name", options=states, index=0)
            
            # Filter districts if possible, or show all districts
            selected_district = st.selectbox("District Name", options=districts, index=0)
            
            selected_season_disp = st.selectbox("Season", options=seasons_display, index=0)
            selected_season = season_map.get(selected_season_disp, selected_season_disp)
            
            selected_crop = st.selectbox("Crop Type", options=crops, index=0)
            
            crop_year = st.number_input("Crop Year", min_value=1990, max_value=2035, value=2024, step=1)
            
        with col2:
            st.markdown("#### 🌡️ Climatic & Land Conditions")
            
            temperature = st.slider("Temperature (°C)", min_value=5.0, max_value=50.0, value=28.5, step=0.5,
                                    help="Average temperature during growing season")
            
            humidity = st.slider("Relative Humidity (%)", min_value=10.0, max_value=100.0, value=70.0, step=1.0,
                                 help="Average relative humidity percentage")
            
            soil_moisture = st.slider("Soil Moisture (%)", min_value=5.0, max_value=100.0, value=55.0, step=1.0,
                                      help="Percentage soil moisture content")
            
            area = st.number_input("Cultivated Area (Hectares)", min_value=0.1, max_value=50000.0, value=100.0, step=10.0,
                                  help="Total field area dedicated to crop cultivation")
        
        st.markdown("<br>", unsafe_allow_html=True)
        submit_btn = st.form_submit_button("🚀 Predict Crop Production", use_container_width=True)

    if submit_btn:
        input_data = pd.DataFrame([{
            'State_Name': selected_state,
            'District_Name': selected_district,
            'Crop_Year': int(crop_year),
            'Season': selected_season,
            'Crop': selected_crop,
            'Temperature': float(temperature),
            'Humidity': float(humidity),
            'Soil_Moisture': float(soil_moisture),
            'Area': float(area)
        }])
        
        try:
            # Model prediction
            prediction = model.predict(input_data)[0]
            predicted_production = max(0.0, float(prediction))
            yield_per_hectare = predicted_production / float(area) if float(area) > 0 else 0.0
            
            st.markdown("### 📊 Prediction Summary")
            
            # Highlight Cards
            res_col1, res_col2, res_col3 = st.columns(3)
            
            with res_col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Estimated Production</div>
                    <div class="metric-value">{predicted_production:,.2f}</div>
                    <div class="metric-unit">Tonnes</div>
                </div>
                """, unsafe_allow_html=True)
                
            with res_col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Estimated Yield Intensity</div>
                    <div class="metric-value">{yield_per_hectare:,.2f}</div>
                    <div class="metric-unit">Tonnes / Hectare</div>
                </div>
                """, unsafe_allow_html=True)
                
            with res_col3:
                # Soil moisture visual indicator
                status_color = "#10b981" if soil_moisture >= 40 else "#f59e0b"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Soil Hydration Status</div>
                    <div class="metric-value" style="color: {status_color};">{'Optimal' if soil_moisture >= 40 else 'Moderate'}</div>
                    <div class="metric-unit">{soil_moisture}% moisture level</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            
            # Interactive Visual Breakdown
            ch_col1, ch_col2 = st.columns(2, gap="large")
            
            with ch_col1:
                st.markdown("#### 📈 Area Sensitivity Simulation")
                area_range = np.linspace(max(1.0, area * 0.2), area * 2.0, 20)
                sim_dfs = []
                for a in area_range:
                    df_temp = input_data.copy()
                    df_temp['Area'] = a
                    sim_dfs.append(df_temp)
                
                sim_df_all = pd.concat(sim_dfs, ignore_index=True)
                sim_preds = model.predict(sim_df_all)
                
                chart_df = pd.DataFrame({
                    'Area (Ha)': area_range,
                    'Predicted Production (Tonnes)': sim_preds
                })
                
                fig_area = px.line(
                    chart_df, x='Area (Ha)', y='Predicted Production (Tonnes)',
                    title=f"Predicted Yield vs Cultivated Area ({selected_crop})",
                    markers=True,
                    color_discrete_sequence=['#10b981']
                )
                fig_area.add_vline(x=area, line_dash="dash", line_color="#ef4444", annotation_text="Selected Area")
                fig_area.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_area, use_container_width=True)

            with ch_col2:
                st.markdown("#### 🌡️ Climate Index Radar")
                categories = ['Temperature Norm', 'Humidity Level', 'Soil Moisture', 'Land Scale Index']
                # Normalized values for visual comparison
                norm_temp = min(100, (temperature / 45.0) * 100)
                norm_hum = humidity
                norm_soil = soil_moisture
                norm_area = min(100, (area / 500.0) * 100)
                
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=[norm_temp, norm_hum, norm_soil, norm_area],
                    theta=categories,
                    fill='toself',
                    name='Input Field Profile',
                    fillcolor='rgba(16, 185, 129, 0.3)',
                    line_color='#059669'
                ))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                    showlegend=False,
                    template="plotly_white",
                    margin=dict(l=30, r=30, t=40, b=20)
                )
                st.plotly_chart(fig_radar, use_container_width=True)

        except Exception as e:
            st.error(f"Error executing prediction: {e}")


# Mode 2: Batch CSV Prediction
elif app_mode == "📂 Batch CSV Prediction":
    st.markdown("### 📂 Upload CSV File for Batch Predictions")
    st.write("Upload a `.csv` file containing the required column names to predict crop production for multiple fields at once.")
    
    # Template Download
    template_df = pd.DataFrame([{
        'State_Name': states[0] if states else 'Andhra Pradesh',
        'District_Name': districts[0] if districts else 'GUNTUR',
        'Crop_Year': 2024,
        'Season': raw_seasons[0] if raw_seasons else 'Kharif     ',
        'Crop': crops[0] if crops else 'Rice',
        'Temperature': 28.0,
        'Humidity': 75.0,
        'Soil_Moisture': 60.0,
        'Area': 100.0
    }])
    
    st.download_button(
        label="📥 Download Sample Batch Template CSV",
        data=template_df.to_csv(index=False),
        file_name="sample_crop_batch_input.csv",
        mime="text/csv"
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.markdown("#### 📄 Uploaded Data Preview")
            st.dataframe(df_upload.head(10), use_container_width=True)
            
            required_cols = ['State_Name', 'District_Name', 'Crop_Year', 'Season', 'Crop', 'Temperature', 'Humidity', 'Soil_Moisture', 'Area']
            missing_cols = [c for c in required_cols if c not in df_upload.columns]
            
            if missing_cols:
                st.error(f"Missing required columns in uploaded CSV: {missing_cols}")
            else:
                if st.button("🚀 Run Batch Prediction", use_container_width=True):
                    preds = model.predict(df_upload[required_cols])
                    df_upload['Predicted_Production_Tonnes'] = np.maximum(0.0, preds)
                    df_upload['Yield_Tonnes_Per_Ha'] = df_upload['Predicted_Production_Tonnes'] / df_upload['Area'].replace(0, np.nan)
                    
                    st.success("Batch Prediction Completed Successfully!")
                    st.markdown("#### 📊 Prediction Results Table")
                    st.dataframe(df_upload, use_container_width=True)
                    
                    b_col1, b_col2, b_col3 = st.columns(3)
                    b_col1.metric("Total Records Processed", len(df_upload))
                    b_col2.metric("Cumulative Production", f"{df_upload['Predicted_Production_Tonnes'].sum():,.2f} Tonnes")
                    b_col3.metric("Average Yield / Ha", f"{df_upload['Yield_Tonnes_Per_Ha'].mean():,.2f} Tonnes/Ha")
                    
                    # Download predicted results
                    result_csv = df_upload.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results CSV",
                        data=result_csv,
                        file_name="crop_yield_predictions.csv",
                        mime="text/csv",
                        use_container_width=True    
                    )
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")


# Mode 3: Model Insights & Analytics
elif app_mode == "📈 Model Insights & Analytics":
    st.markdown("### 🧠 Machine Learning Architecture & Feature Metadata")
    
    st.markdown("""
    This application utilizes an end-to-end **Scikit-Learn Machine Learning Pipeline** trained to estimate agricultural yield based on geographical, seasonal, and environmental variables.
    """)
    
    col_info1, col_info2 = st.columns(2, gap="large")
    
    with col_info1:
        st.markdown("<div class='section-box'>", unsafe_allow_html=True)
        st.markdown("#### 🛠️ Pipeline Architecture")
        st.write("**Model Type:** Scikit-Learn Pipeline (`sklearn.pipeline.Pipeline`)")
        st.write("**Regressor:** `RandomForestRegressor` (Ensemble Tree Regressor)")
        st.write("**Preprocessor:** `ColumnTransformer`")
        st.write("- **Categorical Features (OneHotEncoder):** `State_Name`, `District_Name`, `Season`, `Crop`")
        st.write("- **Numerical Features (StandardScaler):** `Crop_Year`, `Temperature`, `Humidity`, `Soil_Moisture`, `Area`")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_info2:
        st.markdown("<div class='section-box'>", unsafe_allow_html=True)
        st.markdown("#### 📊 Categorical Domain Coverage")
        st.write(f"• **Supported States:** {len(states)}")
        st.write(f"• **Supported Districts:** {len(districts)}")
        st.write(f"• **Seasons Covered:** {len(raw_seasons)}")
        st.write(f"• **Crop Varieties:** {len(crops)}")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("#### 📜 Model Details Summary")
    st.code(str(model), language="text")

# Footer
st.markdown("<hr style='margin-top: 3rem; margin-bottom: 1rem;'>", unsafe_allow_html=True)
st.caption("🌾 AgriYield AI | Developed for Machine Learning Project | Built with Streamlit & Scikit-Learn by Sadhu Kuldeep")
