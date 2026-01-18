#!/usr/bin/env python3
"""
Simple script to run the NotebookLM Streamlit app
"""

import subprocess
import sys
import os

def main():
    """Run the Streamlit app"""
    try:
        # Change to the directory containing the app
        app_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(app_dir)
        
        # Run streamlit
        cmd = [sys.executable, "-m", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
        
        print("🚀 Starting NotebookLM with Qdrant and Mistral...")
        print("📱 App will be available at: http://localhost:8501")
        print("🔑 Make sure to add your API keys in the sidebar:")
        print("   - Mistral API Key")
        print("   - AssemblyAI API Key") 
        print("   - Firecrawl API Key")
        print("   - Zep API Key")
        print("\n" + "="*50)
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down NotebookLM...")
    except Exception as e:
        print(f"❌ Error running app: {e}")

if __name__ == "__main__":
    main()