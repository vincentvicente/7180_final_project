@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
echo.
echo ========================================
echo Starting Streamlit App...
echo ========================================
echo.
echo The app will open in your browser shortly.
echo If it doesn't open automatically, go to:
echo http://localhost:8501
echo.
echo Press Ctrl+C to stop the server.
echo ========================================
echo.
streamlit run app/app.py --server.headless true
pause

