import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class HousePricePredictor:
    def __init__(self):
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.features = None
        
    def load_data(self, file_path):
        """載入並預處理資料"""
        try:
            data = pd.read_csv(file_path)
            print("資料載入成功！")
            return data
        except Exception as e:
            print(f"資料載入失敗：{str(e)}")
            return None
    
    def preprocess_data(self, data, features):
        """資料預處理"""
        self.features = features
        X = data[features]
        y = data['price']
        
        # 處理缺失值
        X = X.fillna(X.mean())
        
        # 特徵縮放
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y
    
    def train_model(self, X, y):
        """訓練模型"""
        # 分割訓練集和測試集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 訓練模型
        self.model.fit(X_train, y_train)
        
        # 預測並評估
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f'平均絕對誤差 (MAE): {mae:.2f}')
        print(f'均方誤差 (MSE): {mse:.2f}')
        print(f'R2 分數: {r2:.2f}')
    
    def predict_price(self, features_dict):
        """預測房價"""
        # 將輸入特徵轉換為 DataFrame
        X = pd.DataFrame([features_dict])
        
        # 特徵縮放
        X_scaled = self.scaler.transform(X)
        
        # 預測
        prediction = self.model.predict(X_scaled)[0]
        
        return prediction

class HousePriceApp:
    def __init__(self, root):
        self.root = root
        self.root.title('台北市房價預測系統')
        self.root.geometry('600x500')
        
        # 初始化預測器
        self.predictor = HousePricePredictor()
        
        # 載入資料
        self.data = self.predictor.load_data('C:/Users/user/Downloads/Taipei_house.csv')
        if self.data is None:
            messagebox.showerror('錯誤', '無法載入資料集')
            return
        
        # 定義特徵
        self.features = ['建物面積', '屋齡', '房間數', '廳數', '衛浴數', '樓層']
        
        # 訓練模型
        X, y = self.predictor.preprocess_data(self.data, self.features)
        self.predictor.train_model(X, y)
        
        self.create_widgets()
    
    def create_widgets(self):
        # 建立主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 標題
        title_label = ttk.Label(main_frame, text='房價預測系統', font=('Helvetica', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=10)
        
        # 輸入欄位
        self.entries = {}
        for i, feature in enumerate(self.features, 1):
            ttk.Label(main_frame, text=feature).grid(row=i, column=0, padx=5, pady=5)
            self.entries[feature] = ttk.Entry(main_frame)
            self.entries[feature].grid(row=i, column=1, padx=5, pady=5)
        
        # 預測按鈕
        predict_btn = ttk.Button(main_frame, text='預測房價', command=self.predict)
        predict_btn.grid(row=len(self.features)+1, column=0, columnspan=2, pady=20)
        
        # 結果顯示
        self.result_label = ttk.Label(main_frame, text='', font=('Helvetica', 12))
        self.result_label.grid(row=len(self.features)+2, column=0, columnspan=2)
    
    def predict(self):
        try:
            # 獲取輸入值
            features_dict = {}
            for feature in self.features:
                value = self.entries[feature].get()
                if not value:
                    messagebox.showwarning('警告', f'請輸入{feature}')
                    return
                features_dict[feature] = float(value)
            
            # 進行預測
            predicted_price = self.predictor.predict_price(features_dict)
            
            # 顯示結果
            self.result_label.config(
                text=f'預測房價: {predicted_price:.2f} 萬元',
                foreground='green'
            )
            
        except ValueError:
            messagebox.showerror('錯誤', '請確保所有輸入都是有效的數字')
        except Exception as e:
            messagebox.showerror('錯誤', f'預測過程發生錯誤：{str(e)}')

def main():
    root = tk.Tk()
    app = HousePriceApp(root)
    root.mainloop()

if __name__ == '__main__':
    main()