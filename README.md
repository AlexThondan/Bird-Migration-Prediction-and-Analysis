# Bird Migration Prediction using Big Data Analytics & Machine Learning


## Project Overview
This postgraduate project (MSc Data Science – Kristu Jayanti Autonomous College, 2025–26) focuses on predicting bird migration patterns using big data analytics and supervised machine learning models.

The system processes a large dataset (~20,000 records) containing bird species, migration routes, reasons, and wingspan details. Through preprocessing, feature engineering, and model training, the model achieved **96% accuracy** with strong precision–recall performance.  
MLflow is used for experiment tracking, and a simple HTML/CSS dashboard visualizes the migration patterns and predictions.

---

## Dataset
- **Source:** Kaggle Bird Migration Dataset  
- **Format:** CSV  
- **Records:** ~20,000  
- **Key Features:**
  - species  
  - origin_location  
  - destination_location  
  - migration_reason  
  - wingspan (cm)

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/bird-migration-prediction.git
cd bird-migration-prediction
```

### 2. Create Virtual Environment
```bash
python -m venv venv
```

Activate:

**Windows**
```bash
venv\Scripts\activate
```

**Mac/Linux**
```bash
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

Sample `requirements.txt`:
```
pandas
numpy
scikit-learn
matplotlib
seaborn
mlflow
```

---

## Project Structure
```text
bird-migration-prediction/
├── data/
│   └── bird_migration.csv
├── notebooks/
│   └── EDA_and_Modeling.ipynb
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   └── predict.py
├── mlflow_runs/
├── visualization/
│   ├── index.html
│   └── style.css
├── requirements.txt
├── mlflow_tracking.py
└── README.md
```

---

## Usage

### Run Training Pipeline
```bash
python src/train.py
```

### Launch MLflow UI
```bash
mlflow ui
```
Open in browser:  
**http://localhost:5000**

### Open Frontend Visualization
Open:
```
visualization/index.html
```

---

## Results

- **Accuracy:** 96%  
- **F1-Score:** High across all classes  
- **Important factors:** Migration reason & wingspan  
- **Outputs Generated:**
  - Confusion Matrix  
  - Feature Importance  
  - Migration Path Charts  

---

## Screenshots (Aligned)

<p align="center">
  <img src="https://github.com/user-attachments/assets/8a2935aa-02c4-4392-a108-71f32c9dfa07" width="32%" />
  <img src="https://github.com/user-attachments/assets/7d214de4-e69a-4d42-bba7-6b0b71f283ae" width="32%" />
  <img src="https://github.com/user-attachments/assets/394f015c-cdc3-48a1-8491-6352b39d69c0" width="32%" />
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/612a4dac-045a-4700-b64c-4686b2cc9b17" width="32%" />
  <img src="https://github.com/user-attachments/assets/09c77d87-8ea6-4a33-b53c-a1a10b2cadbe" width="32%" />
  <img src="https://github.com/user-attachments/assets/16a68bc2-598c-4101-8007-740124d41d03" width="32%" />
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/aea22d4e-f3ac-4e37-8720-9b6132c32825" width="32%" />
  <img src="https://github.com/user-attachments/assets/e69391d6-8ce5-44e1-b876-8e94d882ee72" width="32%" />
  <img src="https://github.com/user-attachments/assets/fbd4b0d8-c9ea-4605-8909-237ac2748f42" width="32%" />
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/30d02c9f-4bcc-4fd2-bf87-1e94562e5cb5" width="32%" />
  <img src="https://github.com/user-attachments/assets/fff61c12-9c87-467d-a7a8-7a6d9245c547" width="32%" />
  <img src="https://github.com/user-attachments/assets/c2f95b07-2c8e-4088-b21d-9e788d946d50" width="32%" />
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/49324f3e-1fe2-4d15-b62e-91dee3830a9f" width="32%" />
  <img src="https://github.com/user-attachments/assets/9218ce48-7bba-4862-87f4-9d7aff125be0" width="32%" />
</p>

---

## Future Scope

- Integrate real-time GPS bird tracking data  
- Use climate and environmental models for dynamic predictions  
- Extend dataset to global migration patterns  
- Deploy using Flask/Dash  
- Mobile app for birdwatchers & researchers  

---

## Author
**Alex T Sabu**  
MSc Data Science, Kristu Jayanti Autonomous College  
Academic Year: *2025–26*

Feel free to reach out for collaboration or academic discussions.

---

## License
This project is licensed under the **MIT License**.

