# 🧠 Dementia Prediction

A machine learning–powered web application that predicts dementia risk using patient data.  
This project combines **Flask**, **machine learning**, and a simple **database-driven UI** for managing and analyzing dementia-related datasets.

---

## 🚀 Features
- Train and save ML models for dementia prediction (`train_model.py`).
- Web interface built with Flask (`app.py`).
- User-friendly frontend with login/registration, patient forms, and result visualization.
- Database integration (`mydata.db`, `dbmanagement.sql`) for storing patient records.
- Graphical analysis with plots (`Algo_bars.py`).
- Pre-trained model (`model.pkl`) included.

---

## 📂 Project Structure
```
DementiaPrediction/
│── app.py                # Flask application entry point
│── train_model.py         # Model training script
│── Algo_bars.py           # Graphs/algorithm visualization
│── model.pkl              # Pre-trained model
│── sql/dbmanagement.sql   # Database schema
│── instance/mydata.db     # SQLite database
│── templates/             # HTML templates (frontend)
│── static/css/            # CSS and images
│── DementiaPrediction/    # Datasets (CSV files)
```

---

## ⚙️ Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/DementiaPrediction.git
   cd DementiaPrediction
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Linux/Mac
   venv\Scripts\activate      # On Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Usage
1. Train the model (optional, already trained model is provided):
   ```bash
   python train_model.py
   ```

2. Run the Flask app:
   ```bash
   python app.py
   ```

3. Open the application in your browser:
   ```
   http://127.0.0.1:5000
   ```

---

## 📊 Dataset
- `dementia_dataset.csv`, `oasis_cross-sectional.csv`, `oasis_varied.csv`  
These datasets are used to train and validate the dementia prediction model.

---

## 🛠️ Tech Stack
- Python (Flask, scikit-learn, pandas, matplotlib)
- SQLite (for patient records)
- HTML, CSS (frontend templates)

---

## 🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
