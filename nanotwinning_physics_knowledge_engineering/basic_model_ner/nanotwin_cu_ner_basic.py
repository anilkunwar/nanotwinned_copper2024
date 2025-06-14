import arxiv
import fitz  # PyMuPDF
import spacy
from spacy.matcher import Matcher
import pandas as pd
import streamlit as st
import urllib.request
import os
import re
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import json
from collections import Counter
from datetime import datetime
import numpy as np
import logging
import time

# Initialize logging
logging.basicConfig(filename='nanotwin_combined.log', level=logging.ERROR)

# Initialize Streamlit app
st.set_page_config(page_title="Nanotwin Copper Research Tool", layout="wide")
st.title("Nanotwin Electrodeposited FCC Copper Research Tool")
st.markdown("""
This tool supports research on nanotwinned electrodeposited FCC copper for electronic packaging and lithium-ion batteries. Use the **arXiv Query** tab to search for relevant papers and download PDFs. Then, use the **NER Analysis** tab to extract parameters (e.g., current density, nanotwin spacing, additives) from the PDFs. Results can be visualized and exported as CSV or JSON.

**Note**: The spaCy model may miss complex chemical terms. For better accuracy, consider training a custom spaCy model: [spaCy Training Guide](https://spacy.io/usage/training).
""")

# Dependency check
st.sidebar.header("Setup and Dependencies")
st.sidebar.markdown("""
**Required Dependencies**:
- `arxiv`, `pymupdf`, `spacy`, `pandas`, `streamlit`, `matplotlib`, `numpy`
- Install with: `pip install arxiv pymupdf spacy pandas streamlit matplotlib numpy`
- For optimal NER, install: `python -m spacy download en_core_web_lg`
""")

# Create PDFs directory
pdf_dir = "pdfs"
if not os.path.exists(pdf_dir):
    os.makedirs(pdf_dir)
    st.info(f"Created directory: {pdf_dir} for storing PDFs.")

# Tabs for arXiv Query and NER Analysis
tab1, tab2 = st.tabs(["arXiv Query", "NER Analysis"])

# --- arXiv Query Tab ---
with tab1:
    st.header("arXiv Query for Nanotwin Copper Papers")
    st.markdown("Search arXiv for papers on nanotwinned electrodeposited FCC copper. Customize the query, download PDFs, and generate a metadata CSV for NER analysis.")

    # Query arXiv function
    def query_arxiv(query, categories, max_results, start_year, end_year, exact_phrases=[]):
        try:
            query_terms = query.strip().split()
            formatted_terms = []
            synonyms = {
                "nanotwin": ["nanotwinned", "twin boundary"],
                "electrodeposition": ["electroplating", "deposition"],
                "copper": ["cu"],
                "lithium ion battery": ["li-ion battery", "lithium battery"]
            }
            for term in query_terms:
                if term.startswith('"') and term.endswith('"'):
                    formatted_terms.append(term.strip('"').replace(" ", "+"))
                else:
                    formatted_terms.append(term)
                    for key, syn_list in synonyms.items():
                        if term.lower() == key:
                            formatted_terms.extend(syn_list)
            api_query = " ".join(formatted_terms)
            for phrase in exact_phrases:
                api_query += f' "{phrase.replace(" ", "+")}"'
            
            client = arxiv.Client()
            search = arxiv.Search(
                query=api_query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance,
                sort_order=arxiv.SortOrder.Descending
            )
            papers = []
            for result in client.results(search):
                if any(cat in result.categories for cat in categories) and start_year <= result.published.year <= end_year:
                    abstract = result.summary.lower()
                    title = result.title.lower()
                    query_words = set(word.lower() for word in re.split(r'\s+|\".*?\"', query) if word and not word.startswith('"'))
                    for key, syn_list in synonyms.items():
                        if key in query_words:
                            query_words.update(syn_list)
                    matched_terms = [word for word in query_words if word in abstract or word in title]
                    match_score = len(matched_terms) / max(1, len(query_words))
                    
                    abstract_highlighted = abstract
                    for term in matched_terms:
                        abstract_highlighted = re.sub(r'\b{}\b'.format(term), f'<b style="color: orange">{term}</b>', abstract_highlighted, flags=re.IGNORECASE)
                    
                    papers.append({
                        "id": result.entry_id.split('/')[-1],
                        "title": result.title,
                        "year": result.published.year,
                        "categories": ", ".join(result.categories),
                        "abstract": abstract[:200] + "..." if len(abstract) > 200 else abstract,
                        "abstract_highlighted": abstract_highlighted[:200] + "..." if len(abstract_highlighted) > 200 else abstract_highlighted,
                        "pdf_url": result.pdf_url,
                        "download_status": "Not downloaded",
                        "matched_terms": ", ".join(matched_terms) if matched_terms else "None",
                        "match_score": round(match_score * 100)
                    })
                if len(papers) >= max_results:
                    break
            print(f"Found {len(papers)} papers for query: {api_query}")
            return papers
        except Exception as e:
            logging.error(f"arXiv query failed: {str(e)}")
            st.error(f"Error querying arXiv: {str(e)}")
            return []

    # Download PDF function
    def download_pdf(pdf_url, paper_id):
        pdf_path = os.path.join(pdf_dir, f"{paper_id}.pdf")
        try:
            urllib.request.urlretrieve(pdf_url, pdf_path)
            file_size = os.path.getsize(pdf_path) / 1024  # Size in KB
            return f"Downloaded ({file_size:.2f} KB)"
        except Exception as e:
            logging.error(f"PDF download failed for {paper_id}: {str(e)}")
            return f"Failed: {str(e)}"

    # Sidebar for search inputs
    with st.sidebar:
        st.subheader("arXiv Search Parameters")
        st.markdown("Customize your search. Use exact phrases (e.g., `\"nanotwinned copper\"`) for precision.")
        
        query_option = st.radio(
            "Select Query Type",
            ["Default Query", "Custom Query", "Suggested Queries"],
            help="Choose how to specify the search query."
        )
        exact_phrases = []
        if query_option == "Default Query":
            query = "nanotwin electrodeposition FCC copper electronic packaging lithium ion battery"
            st.write("Using default query: **" + query + "**")
        elif query_option == "Custom Query":
            query = st.text_input("Enter Custom Query", value="nanotwinned copper electrodeposition lithium battery")
            exact_phrases_input = st.text_input("Exact Phrases (comma-separated, e.g., \"nanotwinned copper\")", value="")
            exact_phrases = [p.strip().strip('"') for p in exact_phrases_input.split(",") if p.strip()]
            st.write("Custom query: **" + query + "**")
            if exact_phrases:
                st.write("Exact phrases: **" + ", ".join(f'"{p}"' for p in exact_phrases) + "**")
        else:
            suggested_queries = [
                "nanotwinned copper electrodeposition",
                "FCC copper lithium ion battery",
                "nanotwin copper electronic packaging",
                "electrodeposition Cu lithium battery"
            ]
            query = st.selectbox("Choose Suggested Query", suggested_queries)
            exact_phrases_input = st.text_input("Exact Phrases (comma-separated, e.g., \"electrodeposition FCC copper\")", value="")
            exact_phrases = [p.strip().strip('"') for p in exact_phrases_input.split(",") if p.strip()]
            st.write("Selected query: **" + query + "**")
            if exact_phrases:
                st.write("Exact phrases: **" + ", ".join(f'"{p}"' for p in exact_phrases) + "**")
        
        default_categories = ["cond-mat.mtrl-sci", "physics.app-ph", "physics.chem-ph", "cond-mat.mes-hall"]
        extra_categories = ["physics.optics", "eess.sy", "cond-mat.other"]
        categories = st.multiselect(
            "Select arXiv Categories",
            default_categories + extra_categories,
            default=default_categories,
            help="Filter papers by categories."
        )
        
        max_results = st.slider(
            "Maximum Number of Papers",
            min_value=1,
            max_value=200, #100,
            value=50,
            help="Set the maximum number of papers to retrieve."
        )
        
        current_year = datetime.now().year
        col1, col2 = st.columns(2)
        with col1:
            start_year = st.number_input(
                "Start Year",
                min_value=1900,
                max_value=current_year,
                value=2000,
                help="Earliest publication year."
            )
        with col2:
            end_year = st.number_input(
                "End Year",
                min_value=start_year,
                max_value=current_year,
                value=current_year,
                help="Latest publication year."
            )
        
        search_button = st.button("Search arXiv")

    if search_button:
        if not query.strip():
            st.error("Please enter a valid query.")
        elif not categories:
            st.error("Please select at least one category.")
        elif start_year > end_year:
            st.error("Start year must be less than or equal to end year.")
        else:
            with st.spinner("Querying arXiv..."):
                papers = query_arxiv(query, categories, max_results, start_year, end_year, exact_phrases)
            
            if not papers:
                st.warning("No papers found matching your criteria.")
                st.markdown("""
                **Suggestions to find more papers:**
                - Broaden the query (e.g., 'nanotwin copper electrodeposition').
                - Use fewer terms (e.g., 'nanotwin copper').
                - Add exact phrases (e.g., "nanotwinned copper").
                - Include more categories (e.g., 'cond-mat.other').
                - Expand the year range (e.g., 1990–2025).
                - Increase the maximum number of papers.
                """)
            else:
                st.success(f"Found **{len(papers)}** papers matching your query!")
                exact_display = ', '.join(f'"{p}"' for p in exact_phrases) if exact_phrases else 'None'
                st.write(f"Query: **{query}** | Exact Phrases: **{exact_display}**")
                st.write(f"Categories: **{', '.join(categories)}** | Years: **{start_year}–{end_year}**")
                
                st.subheader("Downloading PDFs")
                progress_bar = st.progress(0)
                for i, paper in enumerate(papers):
                    if paper["pdf_url"]:
                        status = download_pdf(paper["pdf_url"], paper["id"])
                        paper["download_status"] = status
                    else:
                        paper["download_status"] = "No PDF URL"
                    progress_bar.progress((i + 1) / len(papers))
                    time.sleep(0.1)
                
                df = pd.DataFrame(papers)
                st.subheader("Paper Details")
                st.dataframe(
                    df[["id", "title", "year", "categories", "abstract_highlighted", "matched_terms", "match_score", "download_status"]],
                    use_container_width=True,
                    column_config={
                        "abstract_highlighted": st.column_config.TextColumn("Abstract (Highlighted)", help="Matched terms in bold orange.")
                    }
                )
                
                # Save and automatically download CSV
                csv = df.drop(columns=["abstract_highlighted"]).to_csv(index=False)
                csv_path = "copper_papers_metadata.csv"
                with open(csv_path, "w") as f:
                    f.write(csv)
                st.info(f"Metadata CSV saved as {csv_path}. Automatic download starting...")
                
                with open(csv_path, "rb") as f:
                    st.download_button(
                        label="Download Paper Metadata CSV (Automatic)",
                        data=f,
                        file_name="copper_papers_metadata.csv",
                        mime="text/csv",
                        key=f"auto_download_{time.time()}"  # Unique key to trigger download
                    )
                
                # Manual download button
                st.download_button(
                    label="Download Paper Metadata CSV (Manual)",
                    data=csv,
                    file_name="copper_papers_metadata.csv",
                    mime="text/csv",
                    key="manual_download"
                )
                
                downloaded = sum(1 for p in papers if "Downloaded" in p["download_status"])
                st.write(f"**Summary**: {len(papers)} papers found, {downloaded} PDFs downloaded successfully.")
                if downloaded < len(papers):
                    st.warning("Some PDFs failed to download. Check 'download_status' for details.")
                common_terms = set()
                for terms in df["matched_terms"]:
                    if terms and terms != "None":
                        common_terms.update(terms.split(", "))
                if common_terms:
                    st.markdown(f"**Query Refinement Tip**: Common matched terms: {', '.join(common_terms)}. Try focusing on these (e.g., '{' '.join(list(common_terms)[:3])}').")

# --- NER Analysis Tab ---
with tab2:
    st.header("NER Analysis for Nanotwin Parameters")
    st.markdown("Extract electrodeposition parameters from PDFs. Ensure you have run the arXiv query first to download PDFs and generate the metadata CSV.")

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
        [{"TEXT": {"REGEX": r"^\d+(\.\d+)?$"}}, {"LOWER": {"IN": ["ma/cm²", "ma/cm2", "a/cm²", "a/cm2"]}}],
        [{"TEXT": {"REGEX": r"^\d+(\.\d+)?$"}}, {"LOWER": {"IN": ["v", "volt", "volts"]}}],
        [{"TEXT": {"REGEX": r"^\d+(\.\d+)?$"}}, {"LOWER": {"IN": ["mol/l", "m"]}}, {"LOWER": {"IN": ["cuso4", "cuso₄"]}, "OP": "?" }],
        [{"LOWER": "ph"}, {"TEXT": {"REGEX": r"^\d+(\.\d+)?$"}}],
        [{"TEXT": {"REGEX": r"^\d+(\.\d+)?$"}}, {"LOWER": {"IN": ["°c", "celsius"]}}],
        [{"TEXT": {"REGEX": r"^\d+(\.\d+)?$"}}, {"LOWER": {"IN": ["nm"]}}, {"LOWER": {"IN": ["twin", "nanotwin", "spacing"]}, "OP": "?" }],
        [{"TEXT": {"REGEX": r"^\d+(\.\d+)?$"}}, {"LOWER": {"IN": ["nm"]}}, {"LOWER": {"IN": ["grain", "size"]}, "OP": "?" }],
        [{"TEXT": {"REGEX": r"^\d+(\.\d+)?$"}}, {"LOWER": {"IN": ["g/l", "mg/l"]}}, {"LOWER": {"IN": ["saccharin", "chloride", "chloride ions"]}}],
        [{"TEXT": {"REGEX": r"^\d+(\.\d+)?$"}}, {"LOWER": {"IN": ["min", "minute", "minutes", "h", "hour", "hours"]}}]
    ]
    param_types = ["CURRENT_DENSITY", "VOLTAGE", "ELECTROLYTE_CONC", "PH", "TEMPERATURE", "NANOTWIN_SPACING", "GRAIN_SIZE", "ADDITIVE_CONC", "DEPOSITION_TIME"]
    for i, pattern in enumerate(patterns):
        matcher.add(f"ELECTRODEPOSITION_PARAM_{param_types[i]}", [pattern])

    # Parameter validation ranges
    valid_ranges = {
        "CURRENT_DENSITY": (0.1, 1000, "mA/cm²"),
        "VOLTAGE": (0.1, 10, "V"),
        "ELECTROLYTE_CONC": (0.01, 5, "mol/L"),
        "PH": (0, 14, ""),
        "TEMPERATURE": (0, 100, "°C"),
        "NANOTWIN_SPACING": (1, 1000, "nm"),
        "GRAIN_SIZE": (1, 10000, "nm"),
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
                    if unit.lower() in ["a/cm²", "a/cm2"]:
                        value *= 1000
                        unit = "mA/cm²"
                    elif unit.lower() == "m":
                        unit = "mol/L"
                    elif unit.lower() == "mg/l":
                        value /= 1000
                        unit = "g/L"
                    elif unit.lower() in ["h", "hour", "hours"]:
                        value *= 60
                        unit = "min"
                
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

    # Validate metadata CSV
    def validate_metadata_csv(metadata_csv):
        try:
            df = pd.read_csv(metadata_csv)
            required_columns = ["id", "title", "year", "abstract"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                return False, f"Metadata CSV missing required columns: {', '.join(missing_columns)}"
            return True, "Metadata CSV format is valid."
        except Exception as e:
            return False, f"Error reading metadata CSV: {str(e)}"

    # Process PDFs for NER
    def process_pdfs(pdf_dir, metadata_csv):
        if not os.path.exists(pdf_dir):
            st.error(f"PDF directory {pdf_dir} not found. Run the arXiv query first.")
            return None
        
        is_valid, validation_message = validate_metadata_csv(metadata_csv)
        if not is_valid:
            st.error(validation_message)
            return None
        st.info(validation_message)
        
        try:
            metadata = pd.read_csv(metadata_csv)
        except FileNotFoundError:
            st.error(f"Metadata CSV {metadata_csv} not found. Run the arXiv query first.")
            return None
        
        results = []
        pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
        relevant_pdfs = 0
        
        progress_bar = st.progress(0)
        for i, pdf_file in enumerate(pdf_files):
            paper_id = pdf_file.replace(".pdf", "")
            pdf_path = os.path.join(pdf_dir, pdf_file)
            
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
    with st.sidebar:
        st.subheader("NER Analysis Parameters")
        st.markdown("Configure the analysis to extract parameters.")
        
        pdf_dir_input = st.text_input("PDF Directory", value="pdfs", key="ner_pdf_dir")
        metadata_csv = st.text_input("Metadata CSV", value="copper_papers_metadata.csv", key="ner_metadata_csv")
        entity_types = st.multiselect(
            "Parameter Types to Display",
            ["MATERIAL", "CURRENT_DENSITY", "VOLTAGE", "ELECTROLYTE_CONC", "PH", "TEMPERATURE", "NANOTWIN_SPACING", "GRAIN_SIZE", "ADDITIVE_CONC", "DEPOSITION_TIME"],
            default=["MATERIAL", "CURRENT_DENSITY", "ELECTROLYTE_CONC", "NANOTWIN_SPACING", "ADDITIVE_CONC"],
            help="Select parameter types to filter results."
        )
        sort_by = st.selectbox("Sort By", ["entity_label", "value"], help="Sort by parameter type or value.")
        analyze_button = st.button("Run NER Analysis")

    if analyze_button:
        if not pdf_dir_input or not metadata_csv:
            st.error("Please specify PDF directory and metadata CSV.")
        else:
            with st.spinner("Processing PDFs..."):
                df = process_pdfs(pdf_dir_input, metadata_csv)
            
            if df is None or df.empty:
                st.warning("No parameters extracted. Check PDF directory, metadata CSV, or ensure papers mention electrodeposition and nanotwinning.")
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
                - Use CSV/JSON for further analysis.
                """)

# Footer
st.markdown("---")
st.write("Developed for nanotwinned FCC copper electrodeposition research in electronic packaging and lithium-ion batteries.")
