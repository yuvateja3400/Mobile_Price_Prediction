
# 📱 Mobile Price Prediction

This project predicts mobile price ranges using machine learning. It includes model training, experiment logging, best model selection, FastAPI deployment, and Dockerization.

---

## 🧠 Features

- Data preprocessing with feature engineering (`screen_area`)
- Linear, Ridge, and Lasso regression
- Model tuning using `GridSearchCV`
- Metrics logged to `experiment_log.csv`
- Automatic best model selection (based on R²)
- FastAPI endpoint for predictions (`/predict`)
- Dockerized application for scalable deployment

---

## 🏗️ Architecture

```
     ┌───────────────┐
     │  CSV Dataset  │
     └──────┬────────┘
            ↓
 ┌────────────────────┐
 │   Data Preprocess  │ ◄── screen_area = sc_h × sc_w
 └──────┬─────────────┘
        ↓
┌────────────────────────────┐
│  Train Ridge, Lasso, etc.  │
└──────┬────────────┬────────┘
       ↓            ↓
   Log Metrics    Auto-select best (R²)
       ↓              ↓
    Save best_model.pkl
            ↓
 ┌───────────────────────┐
 │     FastAPI Server    │
 │  /predict endpoint     │
 └─────────┬─────────────┘
           ↓
 ┌───────────────────────┐
 │   Docker Container     │
 │  (API + Model inside)  │
 └───────────────────────┘
```

---

## 📂 Project Structure

```
mobile_price_prediction/
├── api/
│   └── predict.py           # FastAPI logic
├── config.yaml              # Dataset and target config
├── Dockerfile               # Docker image setup
├── main.py                  # Orchestrator script
├── models/
│   └── best_model.pkl       # Saved best model
├── results/
│   └── experiment_log.csv   # Metrics log
├── src/
│   ├── data_loader.py
│   ├── evaluator.py
│   ├── experiment_logger.py
│   ├── preprocess.py
│   └── models/
│       └── logistic_model.py
├── requirements.txt
└── README.md
```

---

## ⚙️ How to Run This Project

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Run Model Training and Logging
```bash
python main.py
```

This will:
- Train models (Ridge, Lasso, etc.)
- Log results to `results/experiment_log.csv`
- Auto-select best model
- Save it as `models/best_model.pkl`

### 3️⃣ Start the FastAPI Server
```bash
uvicorn api.predict:app --reload
```
Then go to:
[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

You can test your `/predict` endpoint with example input.

---

## 🔮 Example JSON Input

Paste this into `/docs` to test:

```json
{
  "battery_power": 1000,
  "blue": 1,
  "clock_speed": 2.2,
  "dual_sim": 1,
  "fc": 5,
  "four_g": 1,
  "int_memory": 32,
  "m_dep": 0.6,
  "mobile_wt": 180,
  "n_cores": 4,
  "pc": 10,
  "ram": 4096,
  "screen_area": 28,
  "talk_time": 10,
  "three_g": 1,
  "touch_screen": 1,
  "wifi": 1
}

```

---

## 🐳 Docker Instructions

### 1️⃣ Build Docker Image
```bash
docker build -t mobile-predictor .
```

### 2️⃣ Run Docker Container
```bash
docker run -d -p 8000:8000 mobile-predictor
```

Then visit: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🧾 .gitignore File (Optional)

To prevent large or unnecessary files from being uploaded to GitHub, add this `.gitignore` file:

```
models/
__pycache__/
*.pyc
.env
results/
*.pkl
```

---

## 📌 Future Enhancements

- Deploy to AWS SageMaker or EC2
- Add support for more model types
- Add authentication to the API

---

## 🙌 Author

**Yuva Teja Chunduru**  
🔗 [LinkedIn](https://www.linkedin.com/in/your-profile)  
📧 yuvatejachunduru3400@gmail.com 

---

## 🏁 License

This project is licensed for learning and personal use only.
