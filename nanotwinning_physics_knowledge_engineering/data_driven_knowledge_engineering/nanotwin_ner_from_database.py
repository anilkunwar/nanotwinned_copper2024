import fitz  # PyMuPDF
import spacy
from spacy.matcher import Matcher
import pandas as pd
import streamlit as st
import os
import re
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sqlite3
import logging

# Initialize logging
logging.basicConfig(filename='nanotwin_ner.log', level=logging.ERROR)

# Initialize Streamlit app
st.set_page_config(page_title="Nanotwin NER Analysis Tool", layout="wide")
st.title("Nanotwin Electrodeposited FCC Copper NER Analysis Tool")
st.markdown("""
This tool extracts electrodeposition parameters (e.g., current density, nanotwin spacing, grain size) from PDFs referenced in a preexisting SQLite database (`nanotwin_knowledge.db`). The database should contain metadata and PDF paths from a prior arXiv query on nanotwinned copper. Results are visualized and exportable as CSV or JSON for further analysis or cloud-based studies.

**Note**: The spaCy model may miss complex chemical terms. For better accuracy, consider training a custom spaCy model: [spaCy Training Guide](https://spacy.io/usage/training).
""")

# Dependency check
st.sidebar.header("Setup and Dependencies")
st.sidebar.markdown("""
**Required Dependencies**:
- `pymupdf`, `spacy`, `pandas`, `streamlit`, `matplotlib`
- Install with: `pip install pymupdf spacy pandas streamlit matplotlib`
- For optimal NER, install: `python -m spacy download en_core_web_lg`
""")

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_lg")
except Exception as e:
    st.warning(f"Failed to load 'en_core_web_lg': {e}. Falling back to 'en_core_web_sm'.")
    try:
        nlp = spacy.load("en_core_web_sm")
        st.info("Using en_core_web_sm (less accurate). Install en_core_web_lg: `python -m spacy download en_core_web_lg`")
    except Exception as e2:
        st.error(f"Failed to load spaCy model: {e2}. Install with: `python -m spacy download en_core_web_sm`")
        st.stop()

# Custom NER patterns
matcher = Matcher(nlp.vocab)
patterns = [
    [{"TEXT": {"REGEX": r"^\d+(\.\d+)?$"}}, {"LOWER": {"IN": ["ma/cm²", "ma/cm2", "a/cm²", "a/cm2", "a/m²", "a/m2", "a/dm²", "a/dm2"]}}],  # CURRENT_DENSITY
    [{"TEXT": {"REGEX": r"^\d+(\.\d+)?$"}}, {"LOWER": {"IN": ["v", "volt", "volts"]}}],  # VOLTAGE
    [{"TEXT": {"REGEX": r"^\d+(\.\d+)?$"}}, {"LOWER": {"IN": ["mol/l", "m"]}}, {"LOWER": {"IN": ["cuso4", "cuso₄"]}, "OP": "?" }],  # ELECTROLYTE_CONC
    [{"LOWER": "ph"}, {"TEXT": {"REGEX": r"^\d+(\.\d+)?$"}}],  # PH
    [{"TEXT": {"REGEX": r"^\d+(\.\d+)?$"}}, {"LOWER": {"IN": ["°c", "celsius"]}}],  # TEMPERATURE
    [{"TEXT": {"REGEX": r"^\d+(\.\d+)?$"}}, {"LOWER": {"IN": ["nm"]}}, {"LOWER": {"IN": ["twin", "nanotwin", "spacing"]}, "OP": "?" }],  # NANOTWIN_SPACING
    [{"TEXT": {"REGEX": r"^\d+(\.\d+)?$"}}, {"LOWER": {"IN": ["µm", "micrometer", "micrometers"]}}, {"LOWER": {"IN": ["grain", "size"]}, "OP": "?" }],  # GRAIN_SIZE (µm only)
    [{"TEXT": {"REGEX": r"^\d+(\.\d+)?$"}}, {"LOWER": {"IN": ["g/l", "mg/l"]}}, {"LOWER": {"IN": ["saccharin", "chloride", "chloride ions"]}}],  # ADDITIVE_CONC
    [{"TEXT": {"REGEX": r"^\d+(\.\d+)?$"}}, {"LOWER": {"IN": ["min", "minute", "minutes", "h", "hour", "hours"]}}]  # DEPOSITION_TIME
]
param_types = ["CURRENT_DENSITY", "VOLTAGE", "ELECTROLYTE_CONC", "PH", "TEMPERATURE", "NANOTWIN_SPACING", "GRAIN_SIZE", "ADDITIVE_CONC", "DEPOSITION_TIME"]
for i, pattern in enumerate(patterns):
    matcher.add(f"ELECTRODEPOSITION_PARAM_{param_types[i]}", [pattern])

# Parameter validation ranges
valid_ranges = {
    "CURRENT_DENSITY": (0.1, 1000, "mA/cm²"),
    "VOLTAGE": (0.1, 10, "V"),  # Broader range for flexibility
    "ELECTROLYTE_CONC": (0.01, 5, "mol/L"),
    "PH": (0, 14, ""),
    "TEMPERATURE": (0, 100, "°C"),  # Broader range for flexibility
    "NANOTWIN_SPACING": (1, 90, "nm"),  # Broader range for flexibility
    "GRAIN_SIZE": (0.2, 50, "µm"),  # Standardized to µm
    "ADDITIVE_CONC": (0.001, 10, "g/L"),
    "DEPOSITION_TIME": (1, 1440, "min")
}

# Color map for parameter histograms
param_colors = {param: cm.tab10(i / len(param_types)) for i, param in enumerate(param_types)}

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        logging.error(f"PDF extraction failed for {pdf_path}: {str(e)}")
        return f"Error: {str(e)}"

# Perform NER
def extract_electrodeposition_parameters(text):
    try:
        doc = nlp(text)
        entities = []
        
        for ent in doc.ents:
            if ent.label_ in ["MATERIAL", "ORG", "PRODUCT"] or "copper" in ent.text.lower():
                entities.append({
                    "text": ent.text,
                    "label": "MATERIAL",
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "value": None,
                    "unit": None,
                    "outcome": None
                })
        
        matches = matcher(doc)
        for match_id, start, end in matches:
            span = doc[start:end]
            label = nlp.vocab.strings[match_id].replace("ELECTRODEPOSITION_PARAM_", "")
            match_text = span.text
            value_match = re.match(r"(\d+\.?\d*)", match_text)
            value = float(value_match.group(1)) if value_match else None
            unit = match_text[value_match.end():].strip() if value_match else None
            
            if unit and value is not None:
                if label == "CURRENT_DENSITY":
                    if unit.lower() in ["a/cm²", "a/cm2"]:
                        value *= 1000  # Convert A/cm² to mA/cm²
                        unit = "mA/cm²"
                    elif unit.lower() in ["a/m²", "a/m2"]:
                        value *= 0.1  # Convert A/m² to mA/cm²
                        unit = "mA/cm²"
                    elif unit.lower() in ["a/dm²", "a/dm2"]:
                        value *= 10  # Convert A/dm² to mA/cm²
                        unit = "mA/cm²"
                elif label == "ELECTROLYTE_CONC":
                    if unit.lower() == "m":
                        unit = "mol/L"
                elif label == "ADDITIVE_CONC":
                    if unit.lower() == "mg/l":
                        value /= 1000  # Convert mg/L to g/L
                        unit = "g/L"
                elif label == "DEPOSITION_TIME":
                    if unit.lower() in ["h", "hour", "hours"]:
                        value *= 60  # Convert hours to minutes
                        unit = "min"
                elif label == "GRAIN_SIZE":
                    if unit.lower() in ["µm", "micrometer", "micrometers"]:
                        unit = "µm"  # Standardize to µm
            
            if label in valid_ranges and value is not None:
                min_val, max_val, expected_unit = valid_ranges[label]
                if not (min_val <= value <= max_val and (unit == expected_unit or unit is None or (label == "PH" and unit == ""))):
                    continue
            
            outcome = None
            context_start = max(0, start - 100)
            context_end = min(len(text), end + 100)
            context_text = text[context_start:context_end].lower()
            outcome_terms = ["nanotwin density", "twin boundary", "hardness", "grain size", "corrosion resistance", "strength"]
            for term in outcome_terms:
                if term in context_text:
                    outcome = term
                    break
            
            entities.append({
                "text": span.text,
                "label": label,
                "start": start,
                "end": end,
                "value": value,
                "unit": unit,
                "outcome": outcome
            })
        
        return entities
    except Exception as e:
        logging.error(f"NER failed: {str(e)}")
        return [{"text": f"Error: {str(e)}", "label": "ERROR", "start": 0, "end": 0, "value": None, "unit": None, "outcome": None}]

# Validate metadata from SQLite database
def validate_metadata_db(db_file):
    try:
        conn = sqlite3.connect(db_file)
        df = pd.read_sql_query("SELECT * FROM papers", conn)
        conn.close()
        required_columns = ["id", "title", "year", "abstract", "pdf_path"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Database missing required columns: {', '.join(missing_columns)}"
        return True, "Database metadata format is valid."
    except Exception as e:
        return False, f"Error reading database: {str(e)}"

# Process PDFs for NER using SQLite database
def process_pdfs_from_db(db_file):
    if not os.path.exists(db_file):
        st.error(f"Database file {db_file} not found. Ensure it exists and contains metadata from a prior arXiv query.")
        return None
    
    is_valid, validation_message = validate_metadata_db(db_file)
    if not is_valid:
        st.error(validation_message)
        return None
    st.info(validation_message)
    
    try:
        conn = sqlite3.connect(db_file)
        metadata = pd.read_sql_query("SELECT * FROM papers", conn)
        conn.close()
    except Exception as e:
        st.error(f"Error reading database: {str(e)}")
        return None
    
    results = []
    pdf_files = metadata["pdf_path"].dropna().unique()
    relevant_pdfs = 0
    
    progress_bar = st.progress(0)
    for i, pdf_path in enumerate(pdf_files):
        if not os.path.exists(pdf_path):
            results.append({
                "paper_id": pdf_path.split("/")[-1].replace(".pdf", ""),
                "title": "Unknown",
                "year": None,
                "entity_text": f"PDF not found at {pdf_path}",
                "entity_label": "ERROR",
                "value": None,
                "unit": None,
                "outcome": None,
                "context": ""
            })
            continue
        
        paper_id = pdf_path.split("/")[-1].replace(".pdf", "")
        paper_meta = metadata[metadata["id"] == paper_id]
        if paper_meta.empty:
            continue
        
        abstract = paper_meta["abstract"].iloc[0].lower()
        if not any(term in abstract for term in ["electrodeposition", "nanotwinning", "nanotwin", "fcc copper"]):
            continue
        relevant_pdfs += 1
        
        text = extract_text_from_pdf(pdf_path)
        if text.startswith("Error"):
            results.append({
                "paper_id": paper_id,
                "title": paper_meta["title"].iloc[0],
                "year": paper_meta["year"].iloc[0],
                "entity_text": "Error extracting text",
                "entity_label": "ERROR",
                "value": None,
                "unit": None,
                "outcome": None,
                "context": ""
            })
            continue
        
        entities = extract_electrodeposition_parameters(text)
        for entity in entities:
            start = max(0, entity["start"] - 50)
            end = min(len(text), entity["end"] + 50)
            context = text[start:end].replace("\n", " ")
            
            results.append({
                "paper_id": paper_id,
                "title": paper_meta["title"].iloc[0],
                "year": paper_meta["year"].iloc[0],
                "entity_text": entity["text"],
                "entity_label": entity["label"],
                "value": entity["value"],
                "unit": entity["unit"],
                "outcome": entity["outcome"],
                "context": context
            })
        
        progress_bar.progress((i + 1) / len(pdf_files))
    
    st.info(f"Found {relevant_pdfs} PDFs with relevant content.")
    return pd.DataFrame(results)

# Sidebar for NER inputs
st.sidebar.header("NER Analysis Parameters")
st.sidebar.markdown("Configure the analysis to extract parameters from the SQLite database.")

db_file_input = st.text_input("SQLite Database File", value="nanotwin_knowledge.db", key="ner_db_file")
entity_types = st.multiselect(
    "Parameter Types to Display",
    ["MATERIAL", "CURRENT_DENSITY", "VOLTAGE", "ELECTROLYTE_CONC", "PH", "TEMPERATURE", "NANOTWIN_SPACING", "GRAIN_SIZE", "ADDITIVE_CONC", "DEPOSITION_TIME"],
    default=["MATERIAL", "CURRENT_DENSITY", "ELECTROLYTE_CONC", "NANOTWIN_SPACING", "ADDITIVE_CONC"],
    help="Select parameter types to filter results."
)
sort_by = st.selectbox("Sort By", ["entity_label", "value"], help="Sort by parameter type or value.")
analyze_button = st.button("Run NER Analysis")

if analyze_button:
    if not db_file_input:
        st.error("Please specify the SQLite database file.")
    else:
        with st.spinner("Processing PDFs from database..."):
            df = process_pdfs_from_db(db_file_input)
        
        if df is None or df.empty:
            st.warning("No parameters extracted. Check the database file, ensure it contains valid PDF paths, and verify papers mention electrodeposition and nanotwinning.")
        else:
            st.success(f"Extracted **{len(df)}** entities from **{len(df['paper_id'].unique())}** papers!")
            
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
            
            csv = df.to_csv(index=False)
            st.download_button(
                "Download Nanotwin Parameters CSV",
                csv,
                "nanotwin_copper_params.csv",
                "text/csv"
            )
            
            json_data = df.to_json("nanotwin_copper_params.json", orient="records", lines=True)
            with open("nanotwin_copper_params.json", "rb") as f:
                st.download_button(
                    "Download Nanotwin Parameters JSON",
                    f,
                    "nanotwin_copper_params.json",
                    "application/json"
                )
            
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
            
            st.write(f"**Summary**: {len(df)} parameters extracted, including {len(df[df['entity_label'] == 'CURRENT_DENSITY'])} current density, {len(df[df['entity_label'] == 'ELECTROLYTE_CONC'])} electrolyte concentration, {len(df[df['entity_label'] == 'NANOTWIN_SPACING'])} nanotwin spacing, and {len(df[df['entity_label'] == 'ADDITIVE_CONC'])} additive concentration parameters.")
            st.markdown("""
            **Next Steps**:
            - Filter by parameter types to focus on specific settings.
            - Review outcomes to link parameters to properties.
            - Use CSV/JSON for further analysis or cloud studies.
            """)

# Footer
st.markdown("---")
st.write("Developed for nanotwinned FCC copper research in electronic packaging and lithium-ion batteries.")
