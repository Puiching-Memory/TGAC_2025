import json
import os
import sys
from collections import defaultdict

def analyze_schema():
    print("Starting schema analysis...")
    # Define paths using relative paths to make it more portable
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.abspath(os.path.join(script_dir, '..', '..')) # Go up two levels from T3/script to TGAC_2025

    mschema_path = os.path.join(workspace_root, "T3", "data", "mschema_database_main.json")
    schema_desc_path = os.path.join(workspace_root, "T3", "data", "schema.json")
    
    final_dataset_path = os.path.join(workspace_root, "T3", "data", "final_dataset.json")
    
    # The output_dir for .txt files is no longer needed, but we keep the chroma_db path
    chroma_db_path = os.path.join(workspace_root, "T3", "chroma_db")

    # Define a path for the temporary output file
    temp_output_dir = os.path.join(workspace_root, "T3", "temp")
    os.makedirs(temp_output_dir, exist_ok=True)
    temp_output_path = os.path.join(temp_output_dir, "chroma_insert_docs.json")

    mschema_data = None
    schema_desc_data = None
    final_dataset = None

    # Load all JSON files in a single block
    try:
        print(f"Loading mschema file: {mschema_path}")
        with open(mschema_path, 'r', encoding='utf-8') as f:
            mschema_data = json.load(f)
        print("Successfully loaded mschema file.")

        print(f"Loading schema description file: {schema_desc_path}")
        with open(schema_desc_path, 'r', encoding='utf-8') as f:
            schema_desc_data = json.load(f)
        print("Successfully loaded schema description file.")

        print(f"Loading final dataset file: {final_dataset_path}")
        with open(final_dataset_path, 'r', encoding='utf-8') as f:
            final_dataset = json.load(f)
        print("Successfully loaded final dataset file.")

    except FileNotFoundError as e:
        print(f"Error: A required file was not found. Details: {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode a JSON file. Please check for syntax errors. Details: {e}")
        sys.exit(1)

    # Create a lookup for table and column descriptions
    print("Creating descriptions lookup...")
    descriptions = {}
    for table_info in schema_desc_data:
        table_name = table_info.get('table_name')
        if table_name:
            descriptions[table_name] = {
                "table_description": table_info.get('table_description', ''),
                "columns": {col.get('col'): col.get('description', '') for col in table_info.get('columns', []) if col.get('col')}
            }
    print("Descriptions lookup created.")

    # --- Relationship Analysis ---
    print("Analyzing relationships based on final_dataset.json...")
    table_relationships = defaultdict(set)
    for item in final_dataset:
        tables = item.get("table_list", [])
        if len(tables) > 1:
            for i in range(len(tables)):
                for j in range(i + 1, len(tables)):
                    table1 = tables[i]
                    table2 = tables[j]
                    table_relationships[table1].add(table2)
                    table_relationships[table2].add(table1)
    
    all_tables = mschema_data.get('tables', {})
    if not all_tables:
        print("Warning: No tables found in mschema_database_main.json.")
    
    print("Relationship analysis complete.")

    # --- Store in ChromaDB ---
    print("Storing schema information in ChromaDB...")
    try:
        import chromadb
        from chromadb.utils import embedding_functions
    except ImportError:
        print("Error: 'chromadb' library not found. Please install it using 'pip install chromadb'")
        sys.exit(1)

    try:
        print(f"Connecting to ChromaDB at: {chroma_db_path}")
        chroma_client = chromadb.PersistentClient(path=chroma_db_path)
        
        collection_name = "schema_info"
        collection = chroma_client.get_or_create_collection(
            name=collection_name,
        )
        print(f"Connected to ChromaDB collection '{collection_name}'.")

        documents = []
        metadatas = []
        ids = []
        
        table_count = len(all_tables)
        print(f"Processing {table_count} tables...")

        for i, (table_name, table_data) in enumerate(all_tables.items()):
            # Create a single string document for each table
            doc_lines = []
            
            # Table Info
            table_desc = descriptions.get(table_name, {}).get('table_description', table_data.get('comment', 'No description available.'))
            doc_lines.append(f"Table: {table_name}")
            doc_lines.append(f"Description: {table_desc}")
            
            # Columns Info
            doc_lines.append("\n--- Columns ---")
            table_fields = table_data.get('fields', {})
            if not table_fields:
                doc_lines.append("  - No columns found.")
            else:
                for field_name, field_data in table_fields.items():
                    col_desc = descriptions.get(table_name, {}).get('columns', {}).get(field_name, field_data.get('comment', '')) or "No description."
                    props = [f"Type: {field_data.get('type', 'N/A')}"]
                    if field_data.get('primary_key'): props.append("PRIMARY KEY")
                    if not field_data.get('nullable'): props.append("NOT NULL")
                    
                    doc_lines.append(f"\n- Column: {field_name}")
                    doc_lines.append(f"  - Description: {col_desc}")
                    doc_lines.append(f"  - Properties: {', '.join(props)}")
                    
                    examples = field_data.get('examples', [])
                    if examples:
                        display_examples = [str(e) for e in examples[:5]]
                        doc_lines.append(f"  - Examples: {', '.join(display_examples)}")

            # Relationships Info
            doc_lines.append("\n--- Potential Relationships ---")
            related_tables = sorted(list(table_relationships.get(table_name, set())))
            if related_tables:
                doc_lines.append(f"  - This table is related to: {', '.join(related_tables)}")
                
                # Find specific join columns for related tables
                for related_table in related_tables:
                    join_cols = []
                    related_table_fields = all_tables.get(related_table, {}).get('fields', {}).keys()
                    for col in table_fields.keys():
                        if col in related_table_fields:
                            # This simple check just looks for common columns.
                            join_cols.append(col)
                    if join_cols:
                        doc_lines.append(f"    - Join with '{related_table}' on column(s): {', '.join(join_cols)}")
            else:
                doc_lines.append("  - No relationships found based on usage data.")

            # Assemble the document and add to lists
            document_content = "\n".join(doc_lines)
            documents.append(document_content)
            metadatas.append({"table_name": table_name, "source": "schema_analysis"})
            ids.append(table_name)

        # Upsert data into ChromaDB collection
        if documents:
            # Save the documents to be inserted into a temporary file for inspection
            print(f"Saving documents to temporary file: {temp_output_path}")
            with open(temp_output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "documents": documents,
                    "metadatas": metadatas,
                    "ids": ids
                }, f, ensure_ascii=False, indent=4)
            print("Temporary file saved.")

            print(f"Upserting {len(documents)} documents into ChromaDB. This may take a moment...")
            collection.upsert(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            print("Upsert operation completed.")

    except Exception as e:
        print(f"An error occurred during ChromaDB processing: {e}")
        sys.exit(1)

    print(f"Analysis complete. {len(all_tables)} tables processed and stored in ChromaDB collection '{collection_name}'.")

if __name__ == "__main__":
    analyze_schema()
