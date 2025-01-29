# Time Series Stock Price Predictor

## Overview

This project implements a time series stock price prediction model using various deep learning architectures. The models were trained to predict stock prices based on historical data, with the **CNN-LSTM hybrid model** tuned using **Keras Tuner** ultimately selected as the best-performing model.

Detailed analysis and model evaluations can be found in the **`StockPricePredictor.ipynb`** Jupyter Notebook.

---

## Dataset

The dataset consists of historical stock prices over a period of five years. The data was preprocessed and split into training, validation, and test sets to evaluate the model performance accurately.

---

## Requirements

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

**Dependencies:**

- Python == 3.11.3
- `numpy`
- `pandas`
- `matplotlib`
- `tensorflow`
- `keras`
- `scikit-learn`
- `keras-tuner`

---

## Cloning the Repository

To clone this repository onto your local machine:

1. Use the following command:

```bash
git clone https://github.com/FarisAnsara/TimeSeriesPredictor.git
```

2. Navigate to the project directory:

```bash
cd TimeSeriesPredictor
```

---

## How to Run

### 1. **Setting Up the Environment**

- Ensure you have Python 3.11.3 installed.
- Install the dependencies using:
  ```bash
  pip install -r requirements.txt
  ```

### 2. **Running the Model and Viewing Results**

- The **entire training, testing, and evaluation process** is documented in **`StockPricePredictor.ipynb`**.
- The saved best model (CNN-LSTM Hybrid) can be directly used for predictions within the notebook and is saved as `best_time_series_prediction.h5` in the repository.

---

## Models Tested

The following models were tested and evaluated:

### **1. Double LSTM** – A deeper version of LSTM with two stacked layers.

#### **Architecture:**
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

model_1 = Sequential([
    LSTM(64, activation='relu', return_sequences=True),
    Dropout(0.3),
    LSTM(32, activation='relu'),
    Dense(1)
])

model_1.compile(optimizer='adam', loss='mean_squared_error')

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)
```

---

### **2. CNN-LSTM Hybrid** – Combines **Conv1D** for feature extraction with LSTM for time series learning.

#### **Architecture:**
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

model_2 = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(sequence_length, 1)),
    MaxPooling1D(pool_size=2),
    LSTM(64, activation='relu'),
    Dense(1)
])

model_2.compile(optimizer='adam', loss='mean_squared_error')
```

---

### **3. Tuned Double LSTM** - An LSTM with two stacked layers tuned using the **Keras Tuner** for hyperparameter selection.

#### **Architecture:**
```python
import keras_tuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense

def build_model(hp):
    model = Sequential()
    model.add(LSTM(units=hp.Int('units_1', 32, 128, step=32), activation='relu', return_sequences=True, input_shape=(sequence_length, 1)))
    model.add(Dropout(hp.Float('dropout', 0.1, 0.5, step=0.1)))
    model.add(LSTM(units=hp.Int('units_2', 32, 128, step=32), activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model
```

#### **Best Hyperparameters:**
```
units_1: 128
dropout: 0.5
units_2: 128
```

---

### **4. Tuned CNN-LSTM Hybrid** – The final best model, optimized using **Keras Tuner** for hyperparameter selection.

#### **Architecture:**
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
import keras_tuner as kt

def build_model(hp):
    model = Sequential()
    
    kernel_size = hp.Int('kernel_size', min_value=1, max_value=min(sequence_length - 1, 5), step=1)
    model.add(Conv1D(
        filters=hp.Int('filters', min_value=32, max_value=128, step=32),
        kernel_size=kernel_size,
        activation='relu',
        input_shape=(sequence_length, 1)
    ))
    pool_size = hp.Int('pool_size', min_value=2, max_value=min(sequence_length - kernel_size, 3), step=1)
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(LSTM(units=hp.Int('lstm_units', min_value=32, max_value=128, step=32), activation='relu'))
    model.add(Dense(1))
    model.compile(
        optimizer='adam',
        loss='mean_squared_error'
    )
    
    return model
```

#### **Best Hyperparameters:**
```
kernel_size: 2
filters: 128
pool_size: 2
lstm_units: 128
```

---

## Results

- **Final Model:** Tuned CNN-LSTM Hybrid
- **Best Performance Metrics:**
  - **Normalized RMSE (Range):** 9.53%
  - **Normalized RMSE (Mean):** 7.14%
  - **Normalized MAE (Range):** 6.54%
  - **Normalized MAE (Mean):** 4.91%
- **Comparison:** The tuned CNN-LSTM outperformed the Double LSTM and standard LSTM models in both **long-term trend prediction and short-term price fluctuation capture**.

---

## Notes

- The project is designed to work seamlessly on both Windows and Linux platforms.
- Ensure all file paths are correctly set before running the notebook.
- For any issues or improvements, feel free to contribute to the repository.

**Best Model: Tuned CNN-LSTM Hybrid** – Optimized using Keras Tuner for the most reliable stock price forecasting.

