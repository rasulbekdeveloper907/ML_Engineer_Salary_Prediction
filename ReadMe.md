# 💼 ML Engineer Salary Prediction  

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)  
[![Scikit-learn](https://img.shields.io/badge/ML-ScikitLearn-orange.svg)](https://scikit-learn.org/)  
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)  
[![Status](https://img.shields.io/badge/Project-Active-brightgreen.svg)]()  

---

## 📖 Loyiha haqida  

Ushbu loyiha **Machine Learning** yordamida **AI va ML mutaxassislari maoshini bashorat qilish** uchun ishlab chiqilgan.  
Model turli omillarni (tajriba darajasi, lavozim, kompaniya joylashuvi, hajmi va h.k.) hisobga olib, yillik maoshni prognoz qiladi.  

---

## 📂 Loyihaning Tuzilishi  

SML_R_Project/
│── Data/ # Xom va tozalangan datasetlar
│ ├── Raw_Data/ # Asl dataset
│ └── Processed_Data/ # Tozalangan dataset
│
│── Notebooks/ # Tahlil va model qurish uchun notebooklar
│── Scripts/ # ML pipeline kodlari
│── Source/ # Yordamchi funksiyalar
│── Errors/ # Xatolik loglari
│── Log/ # Model o‘qitish loglari
│── requirements.txt # Kutubxonalar ro‘yxati
│── README.md # Loyihaning tavsifi


---

## 📊 Dataset haqida  

- **Fayl nomi:** `global_ai_ml_data_salaries.csv`  
- **Manba:** Ochiq ma’lumotlar to‘plami (AI/ML bozoridagi ish o‘rinlari va maoshlar)  
- **Asosiy ustunlar:**
  - `work_year` – ishlagan yil  
  - `experience_level` – tajriba darajasi  
  - `employment_type` – ish turi  
  - `job_title` – lavozim  
  - `salary` – maosh  
  - `salary_currency` – valyuta  
  - `employee_residence` – yashash joyi  
  - `company_location` – kompaniya joylashuvi  
  - `company_size` – kompaniya hajmi  

---

## 🛠️ Texnologiyalar  

- **Python 3.x**  
- **Pandas, NumPy** – ma’lumotlarni qayta ishlash  
- **Matplotlib, Seaborn** – vizualizatsiya  
- **Scikit-learn** – ML algoritmlari  
- **Jupyter Notebook** – tahlil va test  

---

## 🚀 Model Qurish Jarayoni  

✅ **1. Data Cleaning** – null qiymatlar, noto‘g‘ri formatlar tozalandi  
✅ **2. EDA (Exploratory Data Analysis)** – grafiklar orqali maosh taqsimoti va tendensiyalar tahlil qilindi  
✅ **3. Feature Engineering** – kategorik ustunlar kodlandi  
✅ **4. Model qurish** – Linear Regression, Random Forest, XGBoost sinovdan o‘tkazildi  
✅ **5. Baholash** – MSE, RMSE va R² score orqali tekshirildi  
✅ **6. Prediction** – yangi ma’lumotlarga asoslangan maosh prognozi  

---

## 📈 Natijalar  

📊 **Salary Distribution**  
*(Bu yerga vizualizatsiya joylashtiriladi – histogram yoki barplot)*  

📊 **Model Performance (R² Score Comparison)**  
- Linear Regression → `0.65`  
- Random Forest → `0.82`  
- XGBoost → `0.85`  

---

## 🔮 Keyingi Rejalar  

- [ ] Modelni **Streamlit** orqali deploy qilish  
- [ ] **Flask/Django API** sifatida chiqarish  
- [ ] Docker bilan **containerization**  
- [ ] Real-time ma’lumotlar integratsiyasi  

---

## 📌 O‘rnatish  

1. Repo’ni clone qiling:  
   ```bash
   git clone https://github.com/rasulbekdeveloper907/ML_Engineer_Salary_Prediction.git
   cd ML_Engineer_Salary_Prediction

python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

pip install -r requirements.txt

jupyter notebook

👨‍💻 Muallif

👤 Rasulbek Developer
📧 Email: rassiazzi9218@gmail.com
🔗 GitHub: rasulbekdeveloper907
