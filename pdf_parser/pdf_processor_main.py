# IEP PDF to Neo4j Main Script (Modular Version)
import os, json, datetime, traceback
from .pdf_parser import process_iep_pdf
from .neo4j_importer import Neo4jImporter

# Default Configuration
DEFAULT_NEO4J_CONFIG = {
    'uri': 'bolt://10.242.30.37:7687',
    'user': 'neo4j',
    'password': 'password'
}
DEFAULT_ARCHIVE_DIR = r"./json_archives"

class IEPPipeline:
    def __init__(self, neo4j_config=None, archive_dir=None):
        """
        Initialize the pipeline with optional configuration overrides.
        """
        self.neo4j_config = neo4j_config or DEFAULT_NEO4J_CONFIG
        self.archive_dir = archive_dir or DEFAULT_ARCHIVE_DIR

    def run(self, pdf_path: str, child_id: str):
        """
        Executes the full pipeline: Parse -> Archive -> Ingest.
        """
        if not os.path.exists(pdf_path):
            print(f"✕ Error: PDF file not found at {pdf_path}")
            return False

        # Ensure archive directory exists
        if not os.path.exists(self.archive_dir):
            os.makedirs(self.archive_dir, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_filename = f"{child_id}_{timestamp}.json"
        archive_path = os.path.join(self.archive_dir, archive_filename)

        print(f"--- Starting IEP Pipeline for {child_id} ---")
        
        try:
            # Step 1: Parse PDF
            print(f"Step 1: Parsing PDF: {os.path.basename(pdf_path)}")
            data = process_iep_pdf(pdf_path, doc_id=child_id)
            
            # Step 2: Archive JSON
            print(f"Step 2: Archiving JSON to {archive_path}")
            with open(archive_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
            # Step 3: Ingest to Neo4j
            print(f"Step 3: Ingesting into Neo4j at {self.neo4j_config['uri']}")
            importer = Neo4jImporter(
                self.neo4j_config['uri'], 
                self.neo4j_config['user'], 
                self.neo4j_config['password']
            )
            try:
                importer.import_iep(data, report_id=child_id)
                print("✓ Neo4j Ingestion Successful")
            finally:
                importer.close()

            print(f"\n" + "="*50)
            print(f"✓ FULL FLOW COMPLETED SUCCESSFULLY")
            print(f"  Child ID: {child_id}")
            print(f"  Archive: {archive_filename}")
            print("="*50)
            return True, data

        except Exception as e:
            error_msg = str(e)
            print(f"✕ Error in IEP Pipeline: {error_msg}")
            traceback.print_exc()
            return False, error_msg

def integrate_iep_flow(pdf_path: str, child_id: str):
    """
    Simpler functional interface using default configurations.
    """
    pipeline = IEPPipeline()
    return pipeline.run(pdf_path, child_id)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process IEP PDF and Import to Neo4j")
    parser.add_argument("--pdf", default="IEP_ex.pdf", help="Path to the PDF file")
    parser.add_argument("--id", default="周子元_2024", help="ID for the child/report")
    args = parser.parse_args()

    integrate_iep_flow(args.pdf, args.id)
