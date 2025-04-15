
# 🛠️ Turbofan Engine Remaining Useful Life (RUL) Prediction using LSTM

This project predicts the **Remaining Useful Life (RUL)** of aircraft engines using a deep learning model based on **stacked LSTM (Long Short-Term Memory)** layers. The model is trained and evaluated on the **NASA C-MAPSS FD001 dataset** and deployed via a **Streamlit web application** for real-time RUL inference from user-uploaded test data.

---

## 🚀 Project Overview

- **Goal**: Predict how many cycles (RUL) an aircraft engine has before failure using multivariate time-series sensor data.
- **Approach**: Preprocess the data, generate sequences, build a stacked LSTM model, and deploy a web app for inference.
- **Tools**: Python, TensorFlow, NumPy, Pandas, Scikit-learn, Matplotlib, Streamlit.

---

## 🧠 Model Architecture

```python
model = Sequential([
    LSTM(128, return_sequences=True, kernel_regularizer=l2(0.0005), input_shape=(sequence_length, num_features)),
    BatchNormalization(),
    Dropout(0.3),
    LSTM(64, return_sequences=True, kernel_regularizer=l2(0.0005)),
    BatchNormalization(),
    Dropout(0.3),
    LSTM(32, kernel_regularizer=l2(0.0005)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(1)
])
```

- **Sequence Input**: Engine sensor data over `sequence_length` time steps.
- **Regularization**: L2 kernel regularization + Dropout to reduce overfitting.
- **Output**: Single predicted RUL value per sequence.

---

## 📦 Dataset

- Source: [NASA CMAPSS](https://data.nasa.gov/dataset/C-MAPSS-Aircraft-Engine-Simulator-Data/xaut-bemq)
- File used: `train_FD001.txt`, `test_FD001.txt`, and `RUL_FD001.txt`
- Each file contains:
  - `engine_id`, `cycle`, and multiple sensor readings
  - Data is processed to compute true RUL values per time step

---

## ⚙️ Data Preprocessing

- Drop irrelevant or constant columns.
- Apply **RobustScaler** to sensor data to reduce the effect of outliers:
  ```python
  scaler = RobustScaler()
  scaled_data = scaler.fit_transform(sensor_values)
  ```
- Create **sliding windows** (sequences) of length `sequence_length` using:
  ```python
  def create_sequences(data, target, seq_length):
      X, y = [], []
      for i in range(len(data) - seq_length):
          X.append(data[i:i+seq_length])
          y.append(target[i+seq_length])
      return np.array(X), np.array(y)
  ```

---

## 📊 Metrics

- **Loss Function**: Mean Squared Error (MSE)
- **Evaluation**: 
  - R² Score (Coefficient of Determination)
  - Root Mean Squared Error (RMSE)

---

## 🖥️ Streamlit App

### 🔧 Features:
- Upload test data (`.txt` file)
- Run RUL prediction on uploaded engines
- View tabular results of predicted RULs
- Export results as CSV

### 🚀 How to Run:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the Streamlit app
streamlit run app.py
```

---

## 📁 Folder Structure

```
📦 turbofan-rul-lstm
├── app.py                # Streamlit frontend
├── model.py              # Model training script
├── preprocess.py         # Data preprocessing and sequence generation
├── utils.py              # Utility functions
├── trained_model.h5      # Saved LSTM model
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
├── /data
│   ├── train_FD001.txt
│   ├── test_FD001.txt
│   └── RUL_FD001.txt
```

---

## ✅ Results Snapshot

Sample output table from the app:

| engine_id | Predicted_RUL |
|-----------|----------------|
| 1         | 54.90          |
| 2         | 98.92          |
| 3         | 89.16          |
| ...       | ...            |

---

## 📌 Future Improvements

- Add attention mechanism to improve performance
- Deploy model as a cloud API
- Expand to other C-MAPSS datasets (FD002, FD003, FD004)

---

## 👨‍💻 Author

- **Your Name**
- GitHub: [your-username](https://github.com/your-username)
- Email: your.email@example.com

---

## 📜 License

This project is licensed under the MIT License.
