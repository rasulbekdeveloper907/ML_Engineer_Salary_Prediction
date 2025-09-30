# 📊 SML Salary Prediction — ML Engineer

## 📄 Loyiha haqida
Ushbu loyiha **Machine Learning (ML)** yordamida **ML Engineer** lavozimidagi ish haqlarini taxmin qilish (salary prediction) modelini yaratish maqsadida ishlab chiqilgan. Loyiha turli regression algoritmlari, hyperparameter tuning va model baholash usullarini o‘z ichiga oladi.

---

## 🗂 Dataset haqida
- **Dataset nomi:** Salary Prediction for ML Engineers  
- **O‘lchami:** 4144 ta satr × 10 ta ustun  
- **Format:** CSV  
- **Asosiy ustunlar:**
  1. `work_year`
  2. `experience_level`
  3. `employment_type`
  4. `job_title`
  5. `salary`
  6. `salary_currency`
  7. `salary_in_usd`
  8. `employee_residence`
  9. `remote_ratio`
  10. `company_location`
  11. `company_size`

---

## 🔍 Loyihaning asosiy bosqichlari
1. **Data preprocessing**  
   - NaN qiymatlarni to‘ldirish yoki olib tashlash  
   - Kategorik ustunlarni kodlash  
   - Feature selection va correlation tekshirish  

2. **Model tanlash**  
   - Chiziqli modellar (Linear Regression, Ridge, Lasso)  
   - Daraxt asosidagi modellar (DecisionTreeRegressor, RandomForestRegressor)  
   - Support Vector Models (SVR)  
   - Ensemble modellar (XGBRegressor, CatBoostRegressor)  
   - KNN (KNeighborsRegressor)  

3. **Hyperparameter tuning**  
   - Grid Search CV  
   - Randomized Search CV  

4. **Model baholash**  
   - MSE (Mean Squared Error)  
   - MAE (Mean Absolute Error)  
   - R² (R-squared)  
   - Cross-validation  

5. **Modelni saqlash**  
   - Eng yaxshi model `.pkl` formatida saqlanadi  

---

## 📊 Loyihaning natijalari
- **Eng yaxshi model:** Randomized Search XGBRegressor  
- **Test R²:** ~0.6634  
- **Eng yaxshi hyperparameters:**  
  ```python
  {
      'subsample': 1.0,
      'n_estimators': 50,
      'max_depth': 5,
      'learning_rate': 0.1388888888888889,
      'colsample_bytree': 0.9
  }

