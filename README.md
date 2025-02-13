# 🧠 Machine Learning Projects 🚀  

Welcome to the **Machine Learning Projects Repository**! This repository contains multiple ML models covering various domains, from fraud detection to recommendation systems.  

[![Docker Hub](https://img.shields.io/badge/Docker-Hub-blue?logo=docker)](https://hub.docker.com/repository/docker/pranjalkumar09/machine_learning_projects/)  
[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://www.python.org/)  
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Sklearn%2C%20Pandas%2C%20TensorFlow-orange)](https://scikit-learn.org/)  
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)  

---

## 📁 **Project Structure**  

```
machine-learning-projects/
│── csv/                           # Dataset files (if any)
│── src/                           # ML models and scripts
│   ├── Big_market_sales.py
│   ├── Breast_cancer.py
│   ├── Calories_burnt.py
│   ├── Car_price_prediction.py
│   ├── Credit_card_fraud_detection.py
│   ├── Customer_Segmentation.py
│   ├── Diabetes_Prediction.py
│   ├── Fake_News_Prediction.py
│   ├── Heart_disease.py
│   ├── HousePricePrediction_Advanced_Regression.ipynb
│   ├── Loan_Prediction_Status.py
│   ├── Medical_Insurance.py
│   ├── Movie_Recommendation_System.py
│   ├── Sonar_Rock_Vs_Mine_Prediction.py
│   ├── Spam_mail_prediction_system.py
│   ├── titanic_disaster.ipynb
│   ├── Titanic_Survival_Prediction.py
│   ├── titanic_transported.ipynb
│   ├── Wine_quality_prediction.py
│── deployment.yaml                 # Kubernetes deployment config
│── service.yaml                     # Kubernetes service config
│── Dockerfile                        # Docker container setup
│── requirements.txt                   # Required Python libraries
│── .gitignore                         # Files to ignore in Git
```

---

## 📌 **Projects Included**  

| Project Name | Description |
|-------------|------------|
| **Big Market Sales** | Predicts sales in large retail stores |
| **Breast Cancer Prediction** | Classifies breast tumors as malignant or benign |
| **Calories Burnt Prediction** | Estimates calories burned based on activity data |
| **Car Price Prediction** | Predicts car prices using regression |
| **Credit Card Fraud Detection** | Identifies fraudulent transactions |
| **Customer Segmentation** | Clusters customers based on behavior |
| **Diabetes Prediction** | Predicts diabetes risk based on health parameters |
| **Fake News Prediction** | Detects fake news using NLP |
| **Heart Disease Prediction** | Predicts heart disease risk |
| **House Price Prediction** | Advanced regression model for housing prices |
| **Loan Prediction Status** | Predicts loan approval probability |
| **Medical Insurance Cost Prediction** | Estimates insurance costs |
| **Movie Recommendation System** | Suggests movies using collaborative filtering |
| **Sonar Rock vs Mine Prediction** | Classifies objects as rock or mine using sonar data |
| **Spam Mail Prediction** | Detects spam emails |
| **Titanic Survival Prediction** | Predicts passenger survival on the Titanic |
| **Wine Quality Prediction** | Classifies wine quality based on physicochemical tests |

---

## 🐳 **Run with Docker**  

You can run all models using **Docker**:  

```sh
docker pull pranjalkumar09/machine_learning_projects:latest
docker run -it pranjalkumar09/machine_learning_projects
```

🔗 [Docker Hub Repository](https://hub.docker.com/repository/docker/pranjalkumar09/machine_learning_projects/)  

---

## 🚀 **Setup & Installation**  

### 🔹 1. Clone the Repository  
```sh
git clone https://github.com/PranjalKumar09/machine-learning-projects.git
cd machine-learning-projects
```

### 🔹 2. Install Dependencies  
```sh
pip install -r requirements.txt
```

### 🔹 3. Run a Specific Model  
```sh
python src/Diabetes_Prediction.py
```

---

## ☁ **Deploy with Kubernetes**  

1️⃣ Apply the **deployment**:  
```sh
kubectl apply -f deployment.yaml
```

2️⃣ Expose the service:  
```sh
kubectl apply -f service.yaml
```

3️⃣ Check the running pods:  
```sh
kubectl get pods
```

---

## 📜 **Contributing**  

Want to improve the models or add a new project? 🎯 Feel free to:  
✅ Fork the repository  
✅ Make your changes  
✅ Submit a **pull request** 🚀  

---

## 🛠 **Technologies Used**  

- **Programming Language:** Python 🐍  
- **Libraries:** NumPy, Pandas, Scikit-learn, TensorFlow, Matplotlib  
- **Tools:** Jupyter Notebook, Docker, Kubernetes  

---

## 📩 **Contact & Support**  
📧 Email: [pranjal@example.com](mailto:coderkumarshukla@gmail.com)  
🔗 GitHub: [PranjalKumar09](https://github.com/PranjalKumar09)  

🙌 **Star ⭐ the repo if you find it useful!** 🚀
