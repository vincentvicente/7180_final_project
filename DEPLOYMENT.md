# Deployment & Execution Guide

This guide provides multiple ways to run the Startup Success Prediction application, including local execution and Docker containerization.

## Option 1: Local Execution (Recommended for Development)

Run the application directly on your machine using Python and Streamlit.

### Prerequisites
- Python 3.10+ installed
- Virtual environment (optional but recommended)

### Steps

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Application**:
    ```bash
    streamlit run app/app.py
    ```

    *On macOS/Linux, you can also use the provided script:*
    ```bash
    chmod +x run_app.sh
    ./run_app.sh
    ```

---

## Option 2: Docker Container (Recommended for Deployment/Sharing)

Run the application in an isolated container. This ensures consistent behavior across different machines and is ideal for showing to instructors or deploying to a server.

### Prerequisites
- Docker installed on your machine.

### Steps

1.  **Build the Docker Image**:
    ```bash
    docker build -t startup-prediction-app .
    ```

2.  **Run the Container**:
    ```bash
    docker run -p 8501:8501 startup-prediction-app
    ```

3.  **Access the App**:
    Open your browser and go to: `http://localhost:8501`

---

## Option 3: Docker Compose (Easiest for Container Management)

If you have Docker Compose installed, this is the simplest command to build and run.

### Steps

1.  **Start the Service**:
    ```bash
    docker-compose up --build
    ```

2.  **Access the App**:
    Open your browser and go to: `http://localhost:8501`

3.  **Stop the Service**:
    Press `Ctrl+C` or run:
    ```bash
    docker-compose down
    ```

---

## Cloud Deployment (Optional)

To deploy this application to the web (so others can access it via a URL):

1.  **Streamlit Cloud (Easiest)**:
    - Push your code to GitHub.
    - Sign up for [Streamlit Community Cloud](https://streamlit.io/cloud).
    - Connect your GitHub repository and select `app/app.py` as the main file.

2.  **Heroku / Render / Railway**:
    - These platforms support Docker deployment.
    - You can deploy using the provided `Dockerfile`.

