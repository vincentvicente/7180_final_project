#!/bin/bash

# Startup Success Prediction App Launcher for macOS/Linux

# Get the directory where the script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Navigate to the project directory
cd "$DIR"

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "Virtual environment not found. Running with system Python..."
fi

# Run the Streamlit app
echo "Starting Streamlit app..."
streamlit run app/app.py

