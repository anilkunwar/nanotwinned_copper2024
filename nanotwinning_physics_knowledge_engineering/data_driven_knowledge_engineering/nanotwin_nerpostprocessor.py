import pandas as pd
import streamlit as st
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sqlite3
import numpy as np
import logging
import io
import tempfile
import time
import psutil  # For memory usage monitoring

# Define the directory containing the database (same as script directory)
DB_DIR = os.path.dirname(__file__)
DB_FILE = os.path.join(DB_DIR, "nanotwin_knowledge.db")
# If using a subdirectory like pinn_solutions, uncomment below instead:
# DB_DIR = os.path.join(os.path.dirname(__file__), "pinn_solutions")
# DB_FILE = os.path.join(DB_DIR, "nanotwin_knowledge.db")

# Initialize logging
logging.basicConfig(filename='nanotwin_ner.log', level=logging.ERROR)

# Initialize Streamlit app
st.set_page_config(page_title="Nanotwin NER Analysis Tool", layout="wide")
st.title("Nanotwin Electrodeposited FCC Copper NER Analysis Tool")
st.markdown("""
This tool analyzes electrodeposition parameters (e.g., current density, nanotwin spacing, grain size) stored in a SQLite database (`nanotwin_knowledge.db` or an uploaded `.db` file). The database should contain metadata and pre-extracted parameters from a prior arXiv query on nanotwinned copper. Results are visualized and exportable as CSV or JSON for further analysis or cloud-based studies.
""")

# Dependency check
st.sidebar.header("Setup and Dependencies")
st.sidebar.markdown("""
**Required Dependencies**:
- `pandas`, `streamlit`, `matplotlib`, `numpy`, `sqlite3`, `io`, `tempfile`, `psutil`
- Install with: `pip install pandas streamlit matplotlib numpy psutil`
- Ensure Streamlit >= 1.39.0: `pip install --upgrade streamlit`
""")

# Parameter types for filtering and visualization
param_types = ["CURRENT_DENSITY", "VOLTAGE", "ELECTROLYTE_CONC", "PH", "TEMPERATURE", "NANOTWIN_SPACING", "GRAIN_SIZE", "ADDITIVE_CONC", "DEPOSITION_TIME"]

# Color map for parameter histograms
param_colors = {param: cm.tab10(i / len(param_types)) for i, param in enumerate(param_types)}

# Maximum rows for CSV downloads to prevent crashes
MAX_ROWS = 1000

# Validate database
def validate_db(db_file):
    try:
        conn = sqlite3.connect(db_file)
        # Check papers table
        df_papers = pd.read_sql_query("SELECT * FROM papers LIMIT 1", conn)
        required_columns = ["id", "title", "year", "abstract"]
        missing_columns = [col for col in required_columns if col not in df_papers.columns]
        if missing_columns:
            conn.close()
            return False, f"Database 'papers' table missing required columns: {', '.join(missing_columns)}"
        # Check parameters table
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='parameters'")
        if not cursor.fetchone():
            conn.close()
            return False, "Database missing 'parameters' table."
        df_params = pd.read_sql_query("SELECT * FROM parameters LIMIT 1", conn)
        required_param_columns = ["paper_id", "entity_text", "entity_label", "value", "unit", "outcome", "context"]
        missing_param_columns = [col for col in required_param_columns if col not in df_params.columns]
        if missing_param_columns:
            conn.close()
            return False, f"Database 'parameters' table missing required columns: {', '.join(missing_param_columns)}"
        conn.close()
        return True, "Database format is valid."
    except Exception as e:
        return False, f"Error reading database: {str(e)}"

# Process parameters from database
@st.cache_data
def process_params_from_db(db_file):
    # Resolve relative path to absolute path if not already absolute
    if not os.path.isabs(db_file):
        db_file = os.path.join(DB_DIR, db_file)
    
    if not os.path.exists(db_file):
        st.error(f"Database file {db_file} not found. Ensure it exists and contains metadata and parameters from a prior arXiv query.")
        return None
    
    is_valid, validation_message = validate_db(db_file)
    if not is_valid:
        st.error(validation_message)
        return None
    st.info(validation_message)
    
    try:
        conn = sqlite3.connect(db_file)
        # Load parameters
        params_df = pd.read_sql_query("SELECT * FROM parameters", conn)
        # Load papers metadata for title and year
        papers_df = pd.read_sql_query("SELECT id, title, year, abstract FROM papers", conn)
        conn.close()
        
        if params_df.empty:
            st.warning("No parameters found in the database.")
            return None
        
        # Filter papers with relevant abstracts
        papers_df['abstract_lower'] = papers_df['abstract'].str.lower()
        relevant_papers = papers_df[papers_df['abstract_lower'].str.contains('electrodeposition|nanotwinning|nanotwin|fcc copper', na=False)]
        
        # Merge parameters with relevant papers
        df = params_df.merge(relevant_papers[['id', 'title', 'year']], left_on="paper_id", right_on="id", how="inner")
        df = df[["paper_id", "title", "year", "entity_text", "entity_label", "value", "unit", "outcome", "context"]]
        
        if df.empty:
            st.warning("No parameters found for papers with relevant content (electrodeposition, nanotwinning, etc.).")
            return None
        
        # Log dataset size and memory usage
        memory_usage = psutil.virtual_memory().used / (1024 ** 2)  # MB
        logging.info(f"Dataset size: {len(df)} rows, Memory usage: {memory_usage:.2f} MB")
        if len(df) > MAX_ROWS:
            st.warning(f"Dataset exceeds {MAX_ROWS} rows. Limiting to {MAX_ROWS} rows for downloads to prevent crashes.")
            df = df.head(MAX_ROWS)
        
        relevant_papers_count = len(df["paper_id"].unique())
        st.info(f"Found parameters from {relevant_papers_count} relevant papers.")
        return df
    except Exception as e:
        st.error(f"Error reading database: {str(e)}")
        logging.error(f"Database processing failed: {str(e)}")
        return None

# Save histogram data to CSV (in-memory)
@st.cache_data
def save_histogram_to_csv(param_type, values, unit, bins=10):
    try:
        # Convert values to numpy array if not already
        values = np.array(values)
        # Compute histogram
        counts, bin_edges = np.histogram(values, bins=bins)
        # Create DataFrame with bin ranges and counts
        histogram_data = pd.DataFrame({
            'bin_start': bin_edges[:-1],
            'bin_end': bin_edges[1:],
            'count': counts
        })
        # Add parameter type and unit as metadata
        histogram_data['parameter_type'] = param_type
        histogram_data['unit'] = unit if unit else "None"
        # Save to in-memory CSV
        csv_buffer = io.StringIO()
        histogram_data.to_csv(csv_buffer, index=False)
        return csv_buffer.getvalue(), f"histogram_{param_type.lower()}.csv"
    except Exception as e:
        logging.error(f"Failed to save histogram CSV for {param_type}: {str(e)}")
        return None, None

# Initialize session state for download button control
if 'download_clicked' not in st.session_state:
    st.session_state.download_clicked = False

# Sidebar for NER inputs
st.sidebar.header("NER Analysis Parameters")
st.sidebar.markdown("Configure the analysis to extract parameters from the SQLite database.")

# File uploader for .db files
uploaded_db = st.sidebar.file_uploader("Upload SQLite Database (.db)", type=["db", "sqlite", "sqlite3"], key="db_uploader")
db_file_input = st.sidebar.text_input("Or Specify SQLite Database Path", value=DB_FILE, key="ner_db_file")
entity_types = st.multiselect(
    "Parameter Types to Display",
    ["MATERIAL", "CURRENT_DENSITY", "VOLTAGE", "ELECTROLYTE_CONC", "PH", "TEMPERATURE", "NANOTWIN_SPACING", "GRAIN_SIZE", "ADDITIVE_CONC", "DEPOSITION_TIME"],
    default=["MATERIAL", "CURRENT_DENSITY", "ELECTROLYTE_CONC", "NANOTWIN_SPACING", "ADDITIVE_CONC"],
    help="Select parameter types to filter results."
)
sort_by = st.selectbox("Sort By", ["entity_label", "value"], help="Sort by parameter type or value.")
analyze_button = st.sidebar.button("Run NER Analysis")

if analyze_button:
    if not uploaded_db and not db_file_input:
        st.error("Please upload a SQLite database file or specify a database path.")
    else:
        # Handle uploaded database
        db_path = None
        temp_file = None
        if uploaded_db:
            try:
                # Save uploaded file to temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as temp_file:
                    temp_file.write(uploaded_db.read())
                    db_path = temp_file.name
                st.info(f"Uploaded database saved temporarily at: {db_path}")
            except Exception as e:
                st.error(f"Failed to process uploaded database: {str(e)}")
                logging.error(f"Uploaded database processing failed: {str(e)}")
                db_path = None
        else:
            db_path = db_file_input

        if db_path:
            with st.spinner("Processing parameters from database..."):
                df = process_params_from_db(db_path)
            
            # Clean up temporary file
            if temp_file and os.path.exists(temp_file.name):
                try:
                    os.unlink(temp_file.name)
                    st.info("Temporary database file cleaned up.")
                except Exception as e:
                    logging.error(f"Failed to clean up temporary file {temp_file.name}: {str(e)}")

            if df is None or df.empty:
                st.warning("No parameters extracted. Ensure the database contains parameters and relevant papers from a prior arXiv query.")
            else:
                st.success(f"Retrieved **{len(df)}** entities from **{len(df['paper_id'].unique())}** papers!")
                
                if entity_types:
                    df = df[df["entity_label"].isin(entity_types)]
                
                if sort_by == "entity_label":
                    df = df.sort_values(["entity_label", "value"])
                else:
                    df = df.sort_values(["value", "entity_label"], na_position="last")
                
                st.subheader("Extracted Parameters")
                st.dataframe(
                    df[["paper_id", "title", "year", "entity_text", "entity_label", "value", "unit", "outcome", "context"]],
                    use_container_width=True,
                    column_config={
                        "context": st.column_config.TextColumn("Context", help="Surrounding text for the parameter."),
                        "value": st.column_config.NumberColumn("Value", help="Numerical value of the parameter."),
                        "outcome": st.column_config.TextColumn("Outcome", help="Related outcome (e.g., nanotwin density).")
                    }
                )
                
                # Main parameters CSV download (in-memory)
                if not st.session_state.download_clicked:
                    try:
                        with st.spinner("Generating main parameters CSV..."):
                            csv_buffer = io.StringIO()
                            df.to_csv(csv_buffer, index=False)
                            csv_data = csv_buffer.getvalue().encode('utf-8')
                            if st.download_button(
                                "Download Nanotwin Parameters CSV",
                                csv_data,
                                "nanotwin_copper_params.csv",
                                "text/csv",
                                key=f"main_csv_download_{time.time()}",
                                disabled=st.session_state.download_clicked
                            ):
                                st.session_state.download_clicked = True
                                time.sleep(1)  # Prevent rapid clicks
                                st.session_state.download_clicked = False
                    except Exception as e:
                        st.error(f"Failed to generate main CSV download: {str(e)}")
                        logging.error(f"Main CSV download failed: {str(e)}")
                
                # Main parameters JSON download (in-memory)
                if not st.session_state.download_clicked:
                    try:
                        with st.spinner("Generating main parameters JSON..."):
                            json_buffer = io.StringIO()
                            df.to_json(json_buffer, orient="records", lines=True)
                            json_data = json_buffer.getvalue().encode('utf-8')
                            if st.download_button(
                                "Download Nanotwin Parameters JSON",
                                json_data,
                                "nanotwin_copper_params.json",
                                "application/json",
                                key=f"main_json_download_{time.time()}",
                                disabled=st.session_state.download_clicked
                            ):
                                st.session_state.download_clicked = True
                                time.sleep(1)  # Prevent rapid clicks
                                st.session_state.download_clicked = False
                    except Exception as e:
                        st.error(f"Failed to generate main JSON download: {str(e)}")
                        logging.error(f"Main JSON download failed: {str(e)}")
                
                st.subheader("Parameter Distribution Analysis")
                for param_type in entity_types:
                    if param_type in param_types:
                        param_df = df[df["entity_label"] == param_type]
                        if not param_df.empty:
                            values = param_df["value"].dropna()
                            if not values.empty:
                                fig, ax = plt.subplots()
                                ax.hist(values, bins=10, edgecolor="black", color=param_colors[param_type])
                                unit = param_df["unit"].iloc[0] if not param_df["unit"].empty else ""
                                ax.set_xlabel(f"{param_type} ({unit})")
                                ax.set_ylabel("Count")
                                ax.set_title(f"Distribution of {param_type}")
                                st.pyplot(fig)
                                
                                # Histogram CSV download (in-memory)
                                if not st.session_state.download_clicked:
                                    try:
                                        with st.spinner(f"Generating {param_type} histogram CSV..."):
                                            csv_data, csv_filename = save_histogram_to_csv(param_type, values, unit)
                                            if csv_data:
                                                if st.download_button(
                                                    label=f"Download {param_type} Histogram CSV",
                                                    data=csv_data.encode('utf-8'),
                                                    file_name=csv_filename,
                                                    mime="text/csv",
                                                    key=f"histogram_download_{param_type.lower()}_{time.time()}",
                                                    disabled=st.session_state.download_clicked
                                                ):
                                                    st.session_state.download_clicked = True
                                                    time.sleep(1)  # Prevent rapid clicks
                                                    st.session_state.download_clicked = False
                                    except Exception as e:
                                        st.error(f"Failed to generate histogram CSV for {param_type}: {str(e)}")
                                        logging.error(f"Histogram CSV download failed for {param_type}: {str(e)}")
                
                st.write(f"**Summary**: {len(df)} parameters retrieved, including {len(df[df['entity_label'] == 'CURRENT_DENSITY'])} current density, {len(df[df['entity_label'] == 'ELECTROLYTE_CONC'])} electrolyte concentration, {len(df[df['entity_label'] == 'NANOTWIN_SPACING'])} nanotwin spacing, and {len(df[df['entity_label'] == 'ADDITIVE_CONC'])} additive concentration parameters.")
                st.markdown("""
                **Next Steps**:
                - Filter by parameter types to focus on specific settings.
                - Review outcomes to link parameters to properties.
                - Use CSV/JSON for further analysis or cloud studies.
                """)

# Footer
st.markdown("---")
st.write("Developed for nanotwinned FCC copper research in electronic packaging and lithium-ion batteries.")
