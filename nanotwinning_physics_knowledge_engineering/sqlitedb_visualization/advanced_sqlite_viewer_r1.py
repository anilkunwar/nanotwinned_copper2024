import streamlit as st
import sqlite3
import pandas as pd
import os
import glob
import tempfile  # Added for handling uploaded files
from datetime import datetime
from pathlib import Path

# Page Configuration
st.set_page_config(
    page_title="Advanced SQLite Viewer",
    layout="wide",
    page_icon="🗄️"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: bold; color: #1E88E5; margin-bottom: 20px;}
    .stat-box {background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin: 10px 0;}
    .db-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin: 10px 0;}
    .success-box {background-color: #d4edda; padding: 10px; border-radius: 5px; border-left: 4px solid #28a745;}
    .error-box {background-color: #f8d7da; padding: 10px; border-radius: 5px; border-left: 4px solid #dc3545;}
    .sidebar-db-info {background-color: #1e1e1e; color: #ffffff; padding: 15px; border-radius: 8px; margin: 10px 0;}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🗄️ Advanced SQLite Database Viewer</p>', unsafe_allow_html=True)

# Initialize Session State
if 'selected_db' not in st.session_state:
    st.session_state.selected_db = None
if 'conn' not in st.session_state:
    st.session_state.conn = None
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'current_table' not in st.session_state:
    st.session_state.current_table = None

# ==================== HELPER FUNCTIONS ====================

def find_database_files(search_paths=None):
    """Search for .db files in specified directories"""
    if search_paths is None:
        search_paths = [
            '.',
            './databases',
            './db',
            './data',
            '../databases',
            '../db'
        ]
    
    db_files = []
    for path in search_paths:
        if os.path.exists(path):
            patterns = ['*.db', '*.sqlite', '*.db3', '*.sqlite3']
            for pattern in patterns:
                found = glob.glob(os.path.join(path, pattern))
                for file in found:
                    if file not in db_files:
                        db_files.append(file)
    return sorted(db_files)

def get_db_info(db_path):
    """Get basic information about a database file"""
    try:
        file_size = os.path.getsize(db_path)
        modified_time = datetime.fromtimestamp(os.path.getmtime(db_path))
        conn = sqlite3.connect(db_path)
        table_count = len(get_tables(conn))
        conn.close()
        
        return {
            'size': file_size,
            'size_formatted': format_size(file_size),
            'modified': modified_time,
            'table_count': table_count,
            'valid': True
        }
    except Exception as e:
        return {
            'size': 0,
            'size_formatted': 'Unknown',
            'modified': None,
            'table_count': 0,
            'valid': False,
            'error': str(e)
        }

def format_size(size_bytes):
    """Format file size in human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

def connect_to_db(db_path):
    """Establish database connection"""
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        st.error(f"❌ Connection error: {e}")
        return None

def get_tables(conn):
    """Get all table names from database"""
    query = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
    return [row[0] for row in conn.execute(query).fetchall()]

def get_table_schema(conn, table_name):
    """Get column information for a table"""
    query = f"PRAGMA table_info({table_name})"
    return conn.execute(query).fetchall()

def get_foreign_keys(conn, table_name):
    """Get foreign key relationships for a table"""
    query = f"PRAGMA foreign_key_list({table_name})"
    return conn.execute(query).fetchall()

def get_table_stats(conn, table_name):
    """Get statistics for a table"""
    try:
        count_query = f"SELECT COUNT(*) as total FROM {table_name}"
        total = conn.execute(count_query).fetchone()[0]
        schema = get_table_schema(conn, table_name)
        columns = len(schema)
        return {"rows": total, "columns": columns}
    except:
        return {"rows": 0, "columns": 0}

def run_query(conn, query):
    """Execute SQL query and return results"""
    try:
        df = pd.read_sql_query(query, conn)
        return df, None
    except Exception as e:
        return None, str(e)

def export_data(df, format_type='csv'):
    """Export dataframe to different formats"""
    if format_type == 'csv':
        return df.to_csv(index=False).encode('utf-8')
    elif format_type == 'json':
        return df.to_json(orient='records').encode('utf-8')
    elif format_type == 'excel':
        from io import BytesIO
        output = BytesIO()
        df.to_excel(output, index=False, engine='openpyxl')
        return output.getvalue()

# ==================== SIDEBAR ====================

st.sidebar.title("🔧 Database Controls")
st.sidebar.markdown("---")

# --- NEW: File Uploader Logic ---
uploaded_file = st.sidebar.file_uploader("📤 Upload a Database File", type=['db', 'sqlite', 'sqlite3', 'db3'])

if uploaded_file is not None:
    # Save uploaded file to a temporary location so sqlite3 can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        temp_db_path = tmp_file.name
    
    # Update session state if the uploaded file is new
    if st.session_state.selected_db != temp_db_path:
        st.session_state.selected_db = temp_db_path
        st.session_state.conn = connect_to_db(temp_db_path)
        st.session_state.current_table = None
        st.rerun()
    
    # Show info for uploaded file
    st.sidebar.success(f"✅ Connected to: {uploaded_file.name}")
    navigation = st.sidebar.radio(
        "🧭 Navigation",
        ["📊 Tables", "🔍 Schema", "⌨️ SQL Query", "📈 Statistics"],
        index=0
    )

else:
    # --- Existing: Local File Search Logic ---
    db_files = find_database_files()

    if db_files:
        db_options = {}
        for db_path in db_files:
            db_name = os.path.basename(db_path)
            db_info = get_db_info(db_path)
            if db_info['valid']:
                db_options[f"{db_name} ({db_info['table_count']} tables)"] = db_path
        
        if db_options:
            selected_option = st.sidebar.selectbox(
                "📁 Select Database",
                options=list(db_options.keys()),
                index=0 if st.session_state.selected_db is None else list(db_options.keys()).index(
                    next((k for k, v in db_options.items() if v == st.session_state.selected_db), list(db_options.keys())[0])
                ) if st.session_state.selected_db in db_options.values() else 0
            )
            
            selected_db_path = db_options[selected_option]
            
            if st.session_state.selected_db != selected_db_path:
                st.session_state.selected_db = selected_db_path
                st.session_state.conn = connect_to_db(selected_db_path)
                st.session_state.current_table = None
                st.rerun()
            
            db_info = get_db_info(selected_db_path)
            st.sidebar.markdown(f"""
            <div class="sidebar-db-info">
                <strong>📊 Database Info</strong><br>
                📁 File: {os.path.basename(selected_db_path)}<br>
                📏 Size: {db_info['size_formatted']}<br>
                📋 Tables: {db_info['table_count']}<br>
                🕒 Modified: {db_info['modified'].strftime('%Y-%m-%d %H:%M') if db_info['modified'] else 'N/A'}
            </div>
            """, unsafe_allow_html=True)
            
            navigation = st.sidebar.radio(
                "🧭 Navigation",
                ["📊 Tables", "🔍 Schema", "⌨️ SQL Query", "📈 Statistics"],
                index=0
            )
            st.sidebar.markdown("---")
            st.sidebar.success("✅ Database Connected")
        else:
            st.sidebar.error("❌ No valid database files found")
            navigation = None
    else:
        st.sidebar.warning("⚠️ No database files found in search paths")
        st.sidebar.info("""
        **Search Paths:**
        - ./
        - ./databases
        - ./db
        - ./data
        - ../databases
        - ../db
        
        **Tip:** Use the uploader above to load a file directly!
        """)
        navigation = None

# ==================== MAIN CONTENT ====================

conn = st.session_state.conn

if conn and navigation:
    tables = get_tables(conn)
    
    # ==================== TABLES VIEW ====================
    if navigation == "📊 Tables":
        st.header("📊 Table Browser")
        
        if tables:
            col1, col2 = st.columns([3, 1])
            with col1:
                selected_table = st.selectbox(
                    "Select Table",
                    tables,
                    index=tables.index(st.session_state.current_table) if st.session_state.current_table in tables else 0
                )
            st.session_state.current_table = selected_table
            
            if selected_table:
                stats = get_table_stats(conn, selected_table)
                col1, col2, col3 = st.columns(3)
                col1.metric("📊 Total Rows", f"{stats['rows']:,}")
                col2.metric("📋 Columns", stats['columns'])
                col3.metric("🗃️ Table Name", selected_table)
                
                st.markdown("---")
                st.subheader(f"Data from `{selected_table}`")
                
                # Load data
                df = pd.read_sql_query(f"SELECT * FROM {selected_table}", conn)
                
                st.dataframe(df, use_container_width=True, height=400)
                
                st.markdown("---")
                st.subheader("📥 Export Data")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    csv_data = export_data(df, 'csv')
                    st.download_button(
                        label="📄 Download CSV",
                        data=csv_data,
                        file_name=f"{selected_table}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col2:
                    json_data = export_data(df, 'json')
                    st.download_button(
                        label="📋 Download JSON",
                        data=json_data,
                        file_name=f"{selected_table}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                
                with col3:
                    try:
                        excel_data = export_data(df, 'excel')
                        st.download_button(
                            label="📊 Download Excel",
                            data=excel_data,
                            file_name=f"{selected_table}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
                    except:
                        st.info("Excel export requires openpyxl")
        else:
            st.warning("⚠️ No tables found in this database")
    
    # ==================== SCHEMA VIEW ====================
    elif navigation == "🔍 Schema":
        st.header("🔍 Database Schema Explorer")
        
        if tables:
            selected_table = st.selectbox("Select Table to View Schema", tables)
            
            if selected_table:
                st.subheader(f"📋 Schema for `{selected_table}`")
                schema = get_table_schema(conn, selected_table)
                
                schema_df = pd.DataFrame(
                    schema,
                    columns=["Column ID", "Name", "Type", "Not Null", "Default Value", "Primary Key"]
                )
                st.dataframe(schema_df, use_container_width=True)
                
                st.markdown("---")
                st.subheader("🔗 Foreign Key Relationships")
                fks = get_foreign_keys(conn, selected_table)
                
                if fks:
                    fk_df = pd.DataFrame(
                        fks,
                        columns=["ID", "Sequence", "Referenced Table", "From Column", "To Column", "On Update", "On Delete", "Match"]
                    )
                    st.dataframe(fk_df, use_container_width=True)
                else:
                    st.info("ℹ️ No foreign keys defined for this table")
                
                st.markdown("---")
                st.subheader("🕸️ Table Relationships")
                
                all_relationships = []
                for table in tables:
                    fks = get_foreign_keys(conn, table)
                    for fk in fks:
                        all_relationships.append({
                            "From Table": table,
                            "From Column": fk[3],
                            "To Table": fk[2],
                            "To Column": fk[4]
                        })
                
                if all_relationships:
                    rel_df = pd.DataFrame(all_relationships)
                    st.dataframe(rel_df, use_container_width=True)
                else:
                    st.info("ℹ️ No relationships found in database")
        else:
            st.warning("⚠️ No tables available")
    
    # ==================== SQL QUERY VIEW ====================
    elif navigation == "⌨️ SQL Query":
        st.header("⌨️ SQL Query Editor")
        
        default_query = "SELECT * FROM " + (tables[0] if tables else "")
        query = st.text_area(
            "Enter SQL Query",
            value=default_query,
            height=200,
            help="Write your SQL query here. Supports SELECT, JOIN, GROUP BY, etc."
        )
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            run_button = st.button("▶️ Run Query", type="primary", use_container_width=True)
        with col2:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.query_history = []
                st.rerun()
        with col3:
            show_history = st.checkbox("Show History", value=False)
        
        if run_button:
            df, error = run_query(conn, query)
            
            if error:
                st.error(f"❌ SQL Error: {error}")
            else:
                st.success(f"✅ Query executed successfully! ({len(df)} rows returned)")
                st.dataframe(df, use_container_width=True, height=400)
                
                st.session_state.query_history.append({
                    'query': query,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'rows': len(df)
                })
                
                st.markdown("---")
                st.subheader("📥 Export Query Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    csv_data = export_data(df, 'csv')
                    st.download_button(
                        label="📄 Download CSV",
                        data=csv_data,
                        file_name="query_result.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col2:
                    json_data = export_data(df, 'json')
                    st.download_button(
                        label="📋 Download JSON",
                        data=json_data,
                        file_name="query_result.json",
                        mime="application/json",
                        use_container_width=True
                    )
        
        if show_history and st.session_state.query_history:
            st.markdown("---")
            st.subheader("🕒 Recent Query History")
            
            for i, item in enumerate(reversed(st.session_state.query_history[-10:]), 1):
                with st.expander(f"Query #{len(st.session_state.query_history) - i + 1} - {item['timestamp']} ({item['rows']} rows)"):
                    st.code(item['query'], language="sql")
    
    # ==================== STATISTICS VIEW ====================
    elif navigation == "📈 Statistics":
        st.header("📈 Database Statistics")
        
        if tables:
            col1, col2, col3, col4 = st.columns(4)
            
            total_tables = len(tables)
            total_rows = 0
            total_columns = 0
            
            table_stats = []
            for table in tables:
                stats = get_table_stats(conn, table)
                total_rows += stats['rows']
                total_columns += stats['columns']
                table_stats.append({
                    "Table": table,
                    "Rows": stats['rows'],
                    "Columns": stats['columns']
                })
            
            col1.metric("🗃️ Total Tables", total_tables)
            col2.metric("📊 Total Rows", f"{total_rows:,}")
            col3.metric("📋 Total Columns", total_columns)
            
            db_size = os.path.getsize(st.session_state.selected_db)
            col4.metric("💾 Database Size", format_size(db_size))
            
            st.markdown("---")
            
            st.subheader("📊 Table-by-Table Statistics")
            stats_df = pd.DataFrame(table_stats)
            st.dataframe(stats_df, use_container_width=True)
            
            if not stats_df.empty:
                st.markdown("---")
                st.subheader("🏆 Top 5 Tables by Row Count")
                top_tables = stats_df.nlargest(5, 'Rows')
                st.dataframe(top_tables, use_container_width=True)
            
            st.markdown("---")
            st.subheader("📁 Database File Information")
            db_info = get_db_info(st.session_state.selected_db)
            
            info_col1, info_col2 = st.columns(2)
            with info_col1:
                st.info(f"""
                **File Path:** {st.session_state.selected_db}
                
                **File Name:** {os.path.basename(st.session_state.selected_db)}
                
                **Size:** {db_info['size_formatted']}
                """)
            
            with info_col2:
                st.info(f"""
                **Last Modified:** {db_info['modified'].strftime('%Y-%m-%d %H:%M:%S') if db_info['modified'] else 'N/A'}
                
                **SQLite Version:** {sqlite3.sqlite_version}
                
                **Tables:** {db_info['table_count']}
                """)
        else:
            st.warning("⚠️ No tables to analyze")
    
    st.markdown("---")
    st.caption(f"Session started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Built with Streamlit & SQLite")

else:
    # No database selected
    st.info("""
    ## 👋 Welcome to Advanced SQLite Viewer
    
    ### Getting Started:
    1. **Upload a file** using the uploader in the sidebar, OR
    2. **Add database files** to one of these directories:
       - `./databases/`
       - `./db/`
       - `./data/`
       - `./` (root directory)
    
    3. **Supported formats:** `.db`, `.sqlite`, `.db3`, `.sqlite3`
    
    4. **Explore your data** using Tables, Schema, SQL Query, or Statistics views
    
    ### Features:
    - 📊 Browse all tables with pagination
    - 🔍 View complete schema and relationships
    - ⌨️ Run custom SQL queries
    - 📈 View database statistics
    - 📥 Export data to CSV, JSON, or Excel
    """)

# ==================== CREATE SAMPLE DATABASE BUTTON ====================

# Only show sample DB creation if no file is uploaded and no local files found
if not uploaded_file and not find_database_files():
    st.markdown("---")
    st.subheader("🔧 Quick Setup")
    
    if st.button("Create Sample Database"):
        sample_db_path = "./databases/sample.db"
        os.makedirs("./databases", exist_ok=True)
        
        conn_sample = sqlite3.connect(sample_db_path)
        cursor = conn_sample.cursor()
        
        cursor.execute('''CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            name TEXT,
            email TEXT UNIQUE,
            age INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')
        
        cursor.execute('''CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            product TEXT,
            amount REAL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )''')
        
        cursor.executemany('INSERT INTO users (name, email, age) VALUES (?, ?, ?)', [
            ('Alice Johnson', 'alice@example.com', 25),
            ('Bob Smith', 'bob@example.com', 30),
            ('Charlie Brown', 'charlie@example.com', 22),
            ('Diana Prince', 'diana@example.com', 28)
        ])
        
        cursor.executemany('INSERT INTO orders (user_id, product, amount) VALUES (?, ?, ?)', [
            (1, 'Laptop', 1200.00),
            (1, 'Mouse', 25.50),
            (2, 'Keyboard', 45.00),
            (3, 'Monitor', 300.00),
            (4, 'Headphones', 150.00)
        ])
        
        conn_sample.commit()
        conn_sample.close()
        
        st.success(f"✅ Sample database created at `{sample_db_path}`")
        st.info("🔄 Please refresh the page to see the new database in the dropdown")
