import streamlit as st
import pandas as pd
import numpy as np
import joblib
import rasterio
from rasterio.plot import show
from io import BytesIO
import matplotlib.pyplot as plt
from PIL import Image
import tempfile
import os
import seaborn as sns
from matplotlib.colors import ListedColormap

# Set page configuration
st.set_page_config(page_title="LULC Classification Tool", layout="wide")

# Function to calculate all indices
def calculate_indices(df):
    df['NDVI'] = (df['nir'] - df['red']) / (df['nir'] + df['red'] + 1e-10)
    df['NDWI'] = (df['green'] - df['nir']) / (df['green'] + df['nir'] + 1e-10)
    df['NDBI'] = (df['swir'] - df['nir']) / (df['swir'] + df['nir'] + 1e-10)
    df['NDSI'] = (df['green'] - df['swir']) / (df['green'] + df['swir'] + 1e-10)
    df['BSI'] = ((df['swir'] + df['red']) - (df['nir'] + df['blue'])) / ((df['swir'] + df['red']) + (df['nir'] + df['blue']) + 1e-10)
    df['DBSI'] = (df['swir'] - df['green']) / (df['swir'] + df['green'] + 1e-10)
    df['NDBaI'] = (df['swir'] - df['nir']) / (df['swir'] + df['nir'] + 1e-10)
    df['IBI'] = (df['NDBI'] - (df['NDVI'] + df['NDWI'])/2) / (df['NDBI'] + (df['NDVI'] - df['NDWI'])/2 + 1e-10)
    df['UI'] = (df['swir'] - df['nir']) / (df['swir'] + df['nir'] + 1e-10)
    return df

# Function to detect if the area has snow
def detect_snow_presence(bands):
    """
    Detect if the area has snow by checking if any pixel meets snow criteria
    """
    # Create a sample of pixels to check (every 100th pixel to save time)
    sample_size = min(10000, bands['blue'].size)
    sample_indices = np.random.choice(bands['blue'].size, sample_size, replace=False)
    
    blue_sample = bands['blue'].flatten()[sample_indices]
    green_sample = bands['green'].flatten()[sample_indices]
    red_sample = bands['red'].flatten()[sample_indices]
    nir_sample = bands['nir'].flatten()[sample_indices]
    swir_sample = bands['swir'].flatten()[sample_indices]
    
    # Calculate NDSI for the sample
    ndsi_sample = (green_sample - swir_sample) / (green_sample + swir_sample + 1e-10)
    
    # Check if any pixel meets snow criteria
    snow_pixels = np.where(
        (blue_sample > 0.7) & 
        (green_sample > 0.7) & 
        (red_sample > 0.7) & 
        (nir_sample < blue_sample) & 
        (nir_sample < green_sample) & 
        (nir_sample < red_sample) & 
        (swir_sample < blue_sample) & 
        (swir_sample < green_sample) & 
        (swir_sample < red_sample) & 
        (ndsi_sample > 0)
    )[0]
    
    # If more than 1% of sampled pixels are snow, consider the area as having snow
    snow_ratio = len(snow_pixels) / sample_size
    return snow_ratio > 0.01

# Function to calculate rule similarity scores
def calculate_rule_scores(row, has_snow):
    scores = {}
    
    # Bare land score
    bare_score = 0
    if row['swir'] > row['nir']: bare_score += 1
    if row['nir'] > row['red']: bare_score += 1
    if row['red'] > row['green']: bare_score += 1
    if row['green'] > row['blue']: bare_score += 1
    if row['BSI'] > 0.1: bare_score += 1
    if row['DBSI'] > row['NDVI']: bare_score += 1
    if 0 <= row['NDVI'] <= 0.1: bare_score += 1
    scores[1] = bare_score
    
    # Built-up score
    built_score = 0
    if row['swir'] > row['nir']: built_score += 1
    if row['nir'] > row['red']: built_score += 1
    if row['red'] > row['green']: built_score += 1
    if row['green'] > row['blue']: built_score += 1
    if row['NDBI'] > 0.1: built_score += 1
    if row['UI'] > 0: built_score += 1
    if row['IBI'] > 0: built_score += 1
    scores[2] = built_score
    
    # Dense vegetation score
    dense_veg_score = 0
    if row['blue'] < row['green']: dense_veg_score += 1
    if row['red'] < row['green']: dense_veg_score += 1
    if row['nir'] > row['blue']: dense_veg_score += 1
    if row['nir'] > row['green']: dense_veg_score += 1
    if row['nir'] > row['red']: dense_veg_score += 1
    if row['nir'] > row['swir']: dense_veg_score += 1
    if row['NDVI'] > 0.4: dense_veg_score += 1
    scores[3] = dense_veg_score
    
    # Light vegetation score
    light_veg_score = 0
    if row['blue'] < row['green']: light_veg_score += 1
    if row['red'] < row['green']: light_veg_score += 1
    if row['nir'] > row['blue']: light_veg_score += 1
    if row['nir'] > row['green']: light_veg_score += 1
    if row['nir'] > row['red']: light_veg_score += 1
    if row['nir'] > row['swir']: light_veg_score += 1
    if 0.15 < row['NDVI'] <= 0.4: light_veg_score += 1
    scores[4] = light_veg_score
    
    # Water score
    water_score = 0
    if row['nir'] < row['blue']: water_score += 1
    if row['nir'] < row['green']: water_score += 1
    if row['nir'] < row['red']: water_score += 1
    if row['swir'] < row['blue']: water_score += 1
    if row['swir'] < row['green']: water_score += 1
    if row['swir'] < row['red']: water_score += 1
    if row['NDWI'] > 0: water_score += 1
    if row['NDSI'] > 0: water_score += 1
    scores[5] = water_score
    
    # Snow score (only if area has snow)
    if has_snow:
        snow_score = 0
        if row['blue'] > 0.7: snow_score += 1
        if row['green'] > 0.7: snow_score += 1
        if row['red'] > 0.7: snow_score += 1
        if row['nir'] < row['blue']: snow_score += 1
        if row['nir'] < row['green']: snow_score += 1
        if row['nir'] < row['red']: snow_score += 1
        if row['swir'] < row['blue']: snow_score += 1
        if row['swir'] < row['green']: snow_score += 1
        if row['swir'] < row['red']: snow_score += 1
        if row['NDSI'] > 0: snow_score += 1
        scores[6] = snow_score
    
    return scores

# Enhanced rule-based classification function with fallback
def rule_based_classification(row, has_snow):
    # Bare land rules
    if (row['swir'] > row['nir'] and 
        row['nir'] > row['red'] and 
        row['red'] > row['green'] and 
        row['green'] > row['blue'] and
        row['BSI'] > 0.1 and
        row['DBSI'] > row['NDVI'] and
        row['DBSI'] > row['NDWI'] and
        row['DBSI'] > row['NDBI'] and
        row['DBSI'] > row['NDSI'] and
        row['DBSI'] > row['BSI'] and
        row['DBSI'] > row['NDBaI'] and
        row['DBSI'] > row['IBI'] and
        row['DBSI'] > row['UI'] and
        0 <= row['NDVI'] <= 0.1):
        return 1  # Bare Land
    
    # Built-up rules
    elif (row['swir'] > row['nir'] and 
          row['nir'] > row['red'] and 
          row['red'] > row['green'] and 
          row['green'] > row['blue'] and
          row['NDBI'] > 0.1 and
          row['UI'] > 0.1):
        return 2  # Built-up
    
    # Dense vegetation rules
    elif (row['blue'] < row['green'] and 
        row['red'] < row['green'] and
        row['nir'] > row['blue'] and
        row['nir'] > row['green'] and
        row['nir'] > row['red'] and
        row['nir'] > row['swir'] and
        row['NDVI'] > 0.5): 
        return 3  # Dense Vegetation
    
    # Light vegetation rules
    elif (row['blue'] < row['green'] and 
        row['red'] < row['green'] and
        row['nir'] > row['blue'] and
        row['nir'] > row['green'] and
        row['nir'] > row['red'] and
        row['nir'] > row['swir'] and
        0.1 < row['NDVI'] <= 0.5): 
        return 4  # Light Vegetation
    
    # Water rules
    elif (row['nir'] < row['blue'] and 
          row['nir'] < row['green'] and 
          row['nir'] < row['red'] and
          row['swir'] < row['blue'] and 
          row['swir'] < row['green'] and 
          row['swir'] < row['red'] and
          row['NDWI'] > 0 and
          row['NDSI'] > 0):
        return 5  # Water
    
    # Snow rules (only if area has snow)
    elif has_snow and (row['blue'] > 0.7 and  # High reflectance in visible bands
          row['green'] > 0.7 and
          row['red'] > 0.7 and
          row['nir'] < row['blue'] and 
          row['nir'] < row['green'] and 
          row['nir'] < row['red'] and
          row['swir'] < row['blue'] and 
          row['swir'] < row['green'] and 
          row['swir'] < row['red'] and
          row['NDSI'] > 0):
        return 6  # Snow
    
    else:
        # If no rule matches perfectly, use the rule with highest similarity score
        scores = calculate_rule_scores(row, has_snow)
        # Remove snow from consideration if area doesn't have snow
        if not has_snow and 6 in scores:
            del scores[6]
        return max(scores, key=scores.get) if scores else 5  # Default to water if no scores

# Function to process raster data with ML model
def process_raster_ml(band_files, model_package, has_snow):
    """Process raster using the trained ML model with snow handling"""
    # Read all bands
    bands = {}
    band_names = ['blue', 'green', 'red', 'nir', 'swir']
    
    for i, band_name in enumerate(band_names):
        with rasterio.open(band_files[i]) as src:
            bands[band_name] = src.read(1)
            profile = src.profile
    
    # Get the shape of the raster
    rows, cols = bands['blue'].shape
    
    # Flatten the arrays for processing
    flat_arrays = {}
    for band_name in band_names:
        flat_arrays[band_name] = bands[band_name].flatten()
    
    # Create a DataFrame
    df = pd.DataFrame(flat_arrays)
    
    # Calculate indices
    df = calculate_indices(df)
    
    # Prepare features for ML model
    feature_columns = model_package['feature_columns']
    X = df[feature_columns]
    
    # Handle infinite values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    
    # Scale features using the saved scaler
    scaler = model_package['scaler']
    X_scaled = scaler.transform(X)
    
    # Predict using the model
    model = model_package['model']
    
    if has_snow:
        # Use model predictions as-is
        predictions = model.predict(X_scaled)
    else:
        # Get probabilities and remove snow class
        probabilities = model.predict_proba(X_scaled)
        
        # Find the index of snow class (class 6)
        snow_class_index = None
        for i, class_id in enumerate(model.classes_):
            if class_id == 6:
                snow_class_index = i
                break
        
        if snow_class_index is not None:
            # Set snow probabilities to 0
            probabilities[:, snow_class_index] = 0
            
            # Re-normalize probabilities
            row_sums = probabilities.sum(axis=1)
            probabilities = probabilities / row_sums[:, np.newaxis]
            
            # Get new predictions (excluding snow)
            predictions = model.classes_[np.argmax(probabilities, axis=1)]
        else:
            # If snow class not found, use normal predictions
            predictions = model.predict(X_scaled)
    
    # Reshape the classification result back to original shape
    classification_result = predictions.reshape(rows, cols)
    
    return classification_result, profile

# Function to process raster data
def process_raster(band_files, classification_method, has_snow, model_package=None):
    # Read all bands
    bands = {}
    band_names = ['blue', 'green', 'red', 'nir', 'swir']
    
    for i, band_name in enumerate(band_names):
        with rasterio.open(band_files[i]) as src:
            bands[band_name] = src.read(1)
            profile = src.profile
    
    # Get the shape of the raster
    rows, cols = bands['blue'].shape
    
    # Flatten the arrays for processing
    flat_arrays = {}
    for band_name in band_names:
        flat_arrays[band_name] = bands[band_name].flatten()
    
    # Create a DataFrame
    df = pd.DataFrame(flat_arrays)
    
    # Calculate indices
    df = calculate_indices(df)
    
    # Apply classification
    if classification_method == "Rule-Based":
        # Apply rule-based classification with progress bar
        progress_bar = st.progress(0)
        classifications = []
        total_rows = len(df)
        
        for i, (_, row) in enumerate(df.iterrows()):
            classifications.append(rule_based_classification(row, has_snow))
            if i % 1000 == 0:  # Update progress every 1000 rows
                progress_bar.progress(min(i / total_rows, 1.0))
        
        df['class'] = classifications
        progress_bar.progress(1.0)
        
    else:  # Model-Based
        if model_package is not None:
            # Use the enhanced ML processing with snow handling
            classification_result, profile = process_raster_ml(band_files, model_package, has_snow)
            return classification_result, profile
        else:
            st.error("Model not available. Using rule-based classification instead.")
            # Apply rule-based classification with progress bar
            progress_bar = st.progress(0)
            classifications = []
            total_rows = len(df)
            
            for i, (_, row) in enumerate(df.iterrows()):
                classifications.append(rule_based_classification(row, has_snow))
                if i % 1000 == 0:  # Update progress every 1000 rows
                    progress_bar.progress(min(i / total_rows, 1.0))
            
            df['class'] = classifications
            progress_bar.progress(1.0)
    
    # Reshape the classification result back to original shape
    classification_result = df['class'].values.reshape(rows, cols)
    
    return classification_result, profile

# Load the trained model
@st.cache_resource
def load_model():
    try:
        # Try to load the full model package first
        model_package = joblib.load('lulc_random_forest_model.pkl')
        st.success("✅ ML Model loaded successfully!")
        
        # Display model info
        if 'feature_importance' in model_package:
            st.sidebar.info(f"Model Features: {len(model_package['feature_columns'])}")
            
            # Show top features
            feature_importance = model_package['feature_importance']
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            st.sidebar.write("**Top 5 Features:**")
            for feature, importance in top_features:
                st.sidebar.write(f"• {feature}: {importance:.3f}")
        
        return model_package
        
    except FileNotFoundError:
        st.error("❌ Model file 'lulc_random_forest_model.pkl' not found.")
        st.info("Please train the model first using the training script.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

# Function to save classification result as GeoTIFF
def save_geotiff(data, profile, filename):
    # Update profile for classification output
    profile.update(
        dtype=rasterio.uint8,
        count=1,
        compress='lzw'
    )
    
    with rasterio.open(filename, 'w', **profile) as dst:
        dst.write(data.astype(rasterio.uint8), 1)

# Function for ML model prediction on single pixel
def predict_single_pixel_ml(model_package, blue, green, red, nir, swir, has_snow):
    """Predict single pixel using ML model with snow handling"""
    # Create input data
    input_data = pd.DataFrame({
        'blue': [blue],
        'green': [green],
        'red': [red],
        'nir': [nir],
        'swir': [swir]
    })
    
    # Calculate indices
    input_data = calculate_indices(input_data)
    
    # Prepare features
    feature_columns = model_package['feature_columns']
    X = input_data[feature_columns]
    
    # Handle any potential infinite values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    
    # Scale features
    scaler = model_package['scaler']
    X_scaled = scaler.transform(X)
    
    # Predict
    model = model_package['model']
    
    if has_snow:
        # Use normal prediction
        prediction = model.predict(X_scaled)[0]
        probabilities = model.predict_proba(X_scaled)[0]
    else:
        # Get probabilities and handle snow class
        probabilities = model.predict_proba(X_scaled)[0]
        
        # Find the index of snow class (class 6)
        snow_class_index = None
        for i, class_id in enumerate(model.classes_):
            if class_id == 6:
                snow_class_index = i
                break
        
        if snow_class_index is not None:
            # Set snow probability to 0
            probabilities[snow_class_index] = 0
            
            # Re-normalize probabilities
            probabilities = probabilities / probabilities.sum()
            
            # Get new prediction (excluding snow)
            prediction = model.classes_[np.argmax(probabilities)]
        else:
            # If snow class not found, use normal prediction
            prediction = model.predict(X_scaled)[0]
    
    return prediction, probabilities

# Main function
def main():
    st.title("🌍 Land Use Land Cover Classification Tool")
    st.write("Upload Sentinel-2 bands to generate a LULC classification map")
    
    # Load model at startup
    model_package = load_model()
    
    # Create tabs for different functionality
    tab1, tab2, tab3 = st.tabs(["Raster Classification", "Single Pixel Classification", "Model Information"])
    
    with tab1:
        st.header("Raster Classification")
        
        # File upload for raster bands
        st.subheader("Upload Sentinel-2 Bands")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            blue_band = st.file_uploader("Blue Band (B2)", type=['tif', 'tiff'])
        with col2:
            green_band = st.file_uploader("Green Band (B3)", type=['tif', 'tiff'])
        with col3:
            red_band = st.file_uploader("Red Band (B4)", type=['tif', 'tiff'])
        with col4:
            nir_band = st.file_uploader("NIR Band (B8)", type=['tif', 'tiff'])
        with col5:
            swir_band = st.file_uploader("SWIR Band (B11)", type=['tif', 'tiff'])
        
        # Snow detection override option
        snow_override = st.radio(
            "Snow detection:",
            ["Auto-detect", "Force snow classification", "Disable snow classification"],
            horizontal=True,
            help="Auto-detect: Automatically detect if area has snow. Force: Always classify snow. Disable: Never classify snow."
        )
        
        # Classification method selection
        classification_method = st.radio(
            "Classification Method:",
            ["Rule-Based", "Model-Based"],
            horizontal=True
        )
        
        if classification_method == "Model-Based" and model_package is None:
            st.warning("⚠️ ML model not available. Please use Rule-Based classification or train the model first.")
        
        if st.button("Generate LULC Map", type="primary"):
            # Check if all bands are uploaded
            band_files = [blue_band, green_band, red_band, nir_band, swir_band]
            if all(band is not None for band in band_files):
                with st.spinner("Processing raster data..."):
                    # Save uploaded files temporarily
                    temp_files = []
                    for band_file in band_files:
                        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.tif')
                        temp_file.write(band_file.read())
                        temp_files.append(temp_file.name)
                    
                    # Determine snow setting
                    if snow_override == "Force snow classification":
                        has_snow = True
                        st.info("Snow classification forced by user")
                    elif snow_override == "Disable snow classification":
                        has_snow = False
                        st.info("Snow classification disabled by user")
                    else:  # Auto-detect
                        # Read bands for snow detection
                        bands = {}
                        band_names = ['blue', 'green', 'red', 'nir', 'swir']
                        
                        for i, band_name in enumerate(band_names):
                            with rasterio.open(temp_files[i]) as src:
                                bands[band_name] = src.read(1)
                        
                        has_snow = detect_snow_presence(bands)
                        st.info(f"Snow detection: {'Snow detected in area' if has_snow else 'No snow detected in area'}")
                    
                    # Process the raster with the correct snow setting
                    if classification_method == "Model-Based" and model_package is not None:
                        classification_result, profile = process_raster_ml(temp_files, model_package, has_snow)
                    else:
                        classification_result, profile = process_raster(temp_files, classification_method, has_snow, model_package)
                    
                    # Clean up temporary files
                    for temp_file in temp_files:
                        os.unlink(temp_file)
                    
                    # Display the result
                    st.subheader("Classification Result")
                    
                    # Create a figure
                    fig, ax = plt.subplots(figsize=(10, 8))
                    
                    # Define colors for each class
                    class_colors = {
                        1: 'sandybrown', # Bare Land
                        2: 'red',       # Built-up
                        3: 'darkgreen', # Dense Vegetation
                        4: 'limegreen', # Light Vegetation
                        5: 'blue',      # Water
                        6: 'white'      # Snow
                    }
                    
                    # Create a colormap
                    colors = [class_colors[i] for i in range(1, 7)]
                    cmap = ListedColormap(colors)
                    
                    # Plot the classification
                    im = ax.imshow(classification_result, cmap=cmap, vmin=1, vmax=6)
                    
                    # Add a colorbar
                    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    cbar.set_ticks([1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
                    cbar.set_ticklabels(['Bare Land', 'Built-up', 'Dense Vegetation', 
                                        'Light Vegetation', 'Water', 'Snow'])
                    
                    ax.set_title('Land Use Land Cover Classification')
                    ax.axis('off')
                    
                    st.pyplot(fig)
                    
                    # Calculate class distribution
                    unique, counts = np.unique(classification_result, return_counts=True)
                    class_distribution = dict(zip(unique, counts))
                    
                    # Display class distribution
                    st.subheader("Class Distribution")
                    class_names = {
                        1: "Bare Land",
                        2: "Built-up",
                        3: "Dense Vegetation",
                        4: "Light Vegetation",
                        5: "Water",
                        6: "Snow"
                    }
                    
                    dist_cols = st.columns(6)
                    for i, (class_id, count) in enumerate(class_distribution.items()):
                        with dist_cols[i]:
                            st.metric(class_names[class_id], f"{count} pixels")
                    
                    # Save the result to a temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp:
                        save_geotiff(classification_result, profile, tmp.name)
                        
                        # Download button
                        with open(tmp.name, 'rb') as f:
                            geotiff_data = f.read()
                        
                        st.download_button(
                            label="Download LULC Map as GeoTIFF",
                            data=geotiff_data,
                            file_name="lulc_classification.tif",
                            mime="image/tiff"
                        )
                    
                    # Clean up
                    os.unlink(tmp.name)
            else:
                st.error("Please upload all five band files to proceed.")
    
    with tab2:
        st.header("Single Pixel Classification")
        
        # Manual input
        st.subheader("Enter Band Values")
        col1, col2 = st.columns(2)
        
        with col1:
            blue = st.number_input("Blue band value", value=808.0, min_value=0.0, max_value=10000.0, step=1.0)
            green = st.number_input("Green band value", value=1042.0, min_value=0.0, max_value=10000.0, step=1.0)
            red = st.number_input("Red band value", value=842.0, min_value=0.0, max_value=10000.0, step=1.0)
        
        with col2:
            nir = st.number_input("NIR band value", value=3003.0, min_value=0.0, max_value=10000.0, step=1.0)
            swir = st.number_input("SWIR band value", value=1932.0, min_value=0.0, max_value=10000.0, step=1.0)
        
        # Snow option for single pixel
        snow_option = st.radio(
            "Consider snow classification:",
            ["Yes", "No"],
            horizontal=True,
            key="single_pixel_snow"
        )
        has_snow = snow_option == "Yes"
        
        if st.button("Classify Pixel"):
            # Create a dataframe with the input values
            data = {
                'blue': [blue],
                'green': [green],
                'red': [red],
                'nir': [nir],
                'swir': [swir]
            }
            df = pd.DataFrame(data)
            
            # Calculate indices
            df_with_indices = calculate_indices(df.copy())
            
            # Rule-based classification
            rule_based_class = rule_based_classification(df_with_indices.iloc[0], has_snow)
            
            # Calculate rule scores for display
            rule_scores = calculate_rule_scores(df_with_indices.iloc[0], has_snow)
            
            # Model-based classification
            model_class = None
            probabilities = None
            
            if model_package is not None:
                model_class, probabilities = predict_single_pixel_ml(model_package, blue, green, red, nir, swir, has_snow)
            
            # Display results
            st.subheader("Results")
            
            # Class names mapping
            class_names = {
                1: "Bare Land",
                2: "Built-up",
                3: "Dense Vegetation",
                4: "Light Vegetation",
                5: "Water",
                6: "Snow"
            }
            
            st.write("Calculated Indices:")
            st.dataframe(df_with_indices.style.format("{:.4f}"))
            
            # Display rule scores
            st.write("Rule Similarity Scores:")
            score_df = pd.DataFrame.from_dict(rule_scores, orient='index', columns=['Score'])
            score_df['Class'] = score_df.index.map(class_names)
            st.dataframe(score_df[['Class', 'Score']])
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Rule-based Classification", class_names.get(rule_based_class, "Unknown"))
            with col2:
                if model_class is not None:
                    st.metric("ML Model Classification", class_names.get(model_class, "Unknown"))
            
            # Display probabilities if available
            if probabilities is not None and model_package is not None:
                st.subheader("ML Model Confidence Scores")
                prob_df = pd.DataFrame({
                    'Class': [class_names.get(cls, "Unknown") for cls in model_package['model'].classes_],
                    'Probability': probabilities
                }).sort_values('Probability', ascending=False)
                
                # Create a bar chart of probabilities
                fig, ax = plt.subplots(figsize=(10, 4))
                bars = ax.bar(prob_df['Class'], prob_df['Probability'], color='skyblue')
                ax.set_ylabel('Probability')
                ax.set_title('Classification Probabilities')
                ax.set_ylim(0, 1)
                
                # Add value labels on bars
                for bar, prob in zip(bars, prob_df['Probability']):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{prob:.3f}', ha='center', va='bottom')
                
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
    
    with tab3:
        st.header("Model Information")
        
        if model_package is not None:
            st.success("✅ ML Model is loaded and ready for use!")
            
            # Display model details
            st.subheader("Model Details")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Model Type:** Random Forest")
                st.write(f"**Number of Features:** {len(model_package['feature_columns'])}")
                st.write(f"**Number of Classes:** {len(model_package['class_names'])}")
            
            with col2:
                if 'feature_importance' in model_package:
                    st.write("**Model Features:**")
                    for feature in model_package['feature_columns']:
                        st.write(f"• {feature}")
            
            # Feature importance visualization
            if 'feature_importance' in model_package:
                st.subheader("Feature Importance")
                feature_importance = model_package['feature_importance']
                importance_df = pd.DataFrame({
                    'Feature': list(feature_importance.keys()),
                    'Importance': list(feature_importance.values())
                }).sort_values('Importance', ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(data=importance_df.head(10), x='Importance', y='Feature', ax=ax)
                ax.set_title('Top 10 Most Important Features')
                st.pyplot(fig)
            
            # Class information
            st.subheader("Class Information")
            class_info = model_package['class_names']
            for class_id, class_name in class_info.items():
                st.write(f"**Class {class_id}:** {class_name}")
        
        else:
            st.error("❌ No ML model loaded.")
            st.info("""
            To use the ML model:
            1. Run the training script to generate 'lulc_random_forest_model.pkl'
            2. Ensure the model file is in the same directory as this app
            3. Restart the app
            """)

if __name__ == "__main__":
    main()