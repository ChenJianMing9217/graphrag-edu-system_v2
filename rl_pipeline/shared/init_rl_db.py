import sys
import os
from sqlalchemy import text

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app, db

def init_db():
    with app.app_context():
        print("[DB-Init] Creating/Updating tables...")
        # create_all will create the table if it doesn't exist
        db.create_all()
        
        # Manually add retrieved_content column if it's missing (Alembic-like simple check)
        try:
            with db.engine.connect() as conn:
                # Check if column exists
                result = conn.execute(text("SHOW COLUMNS FROM search_feedback LIKE 'retrieved_content'"))
                if not result.fetchone():
                    print("[DB-Init] Adding 'retrieved_content' column to search_feedback table...")
                    conn.execute(text("ALTER TABLE search_feedback ADD COLUMN retrieved_content TEXT"))
                    conn.commit()
                else:
                    print("[DB-Init] 'retrieved_content' column already exists.")
        except Exception as e:
            print(f"[DB-Init] Warning during column update: {e}")
            
        print("[DB-Init] SearchFeedback table is ready.")

if __name__ == "__main__":
    init_db()
