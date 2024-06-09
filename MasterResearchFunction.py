import os
import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import japanize_matplotlib

#Apple Watchからのモーションデータを読み込む関数
def process_apple_watch_csv(file_path):
    # CSVファイルの読み込み
    motion_data = pd.read_csv(file_path, header=0, names=['UnixTime', 'AccelerationX', 'AccelerationY', 'AccelerationZ', 'GyroX', 'GyroY', 'GyroZ'])
    # Unixタイムスタンプを日本時間に変換し、タイムゾーン情報を削除して表示する
    motion_data['Timestamp'] = pd.to_datetime(motion_data['UnixTime'], unit='s').dt.tz_localize('UTC').dt.tz_convert('Asia/Tokyo').dt.tz_localize(None)
    # 加速度のユークリッドノルムを計算してデータフレームに追加
    motion_data['EuclideanNorm'] = np.sqrt(motion_data['AccelerationX']**2 + motion_data['AccelerationY']**2 + motion_data['AccelerationZ']**2)
    #SG法を適用
    # SGフィルタのパラメータ設定
    window_length = 11  # 窓の長さ（奇数、10サンプル以上）
    polyorder = 2      # 多項式の次数（2次）
    # SGフィルタを適用
    motion_data['SG_AccelerationX'] = savgol_filter(motion_data['AccelerationX'], window_length, polyorder)
    motion_data['SG_AccelerationY'] = savgol_filter(motion_data['AccelerationY'], window_length, polyorder)
    motion_data['SG_AccelerationZ'] = savgol_filter(motion_data['AccelerationZ'], window_length, polyorder)
    motion_data['SG_EuclideanNorm'] = savgol_filter(motion_data['EuclideanNorm'], window_length, polyorder)
    # FFTのサンプル数を取得
    N = len(motion_data)
    # サンプリングレートを定義 (50Hzとする)
    Fs = 50
    # FFT解析を行い、3軸加速度のパワースペクトルを計算してデータフレームに追加
    for axis in ['AccelerationX', 'AccelerationY', 'AccelerationZ', 'EuclideanNorm']:
        accel_fft = np.fft.fft(motion_data[axis])
        power_spectrum = np.abs(accel_fft)**2
        motion_data[f'PowerSpectrum_{axis}'] = power_spectrum

    return motion_data

def process_all_apple_watch_csv_in_directory(directory):
    dataframes = []
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            df = process_apple_watch_csv(file_path)
            dataframes.append(df)
    return dataframes

#Tobii Pro Glasses 2からのアイトラッキングデータを読み込む関数
def process_tobii_csv(file_path):
    # CSVファイルの読み込み
    eye_data = pd.read_csv(file_path, header=0)
    #タイムスタンプをYYYY-MM-DD HH:MM:SS.fff形式に変換
    eye_data['Recording start time'] = pd.to_datetime(eye_data['Recording start time'], format='%H:%M:%S.%f')
    eye_data['Recording date'] = pd.to_datetime(eye_data['Recording date'], format='%Y/%m/%d')
    eye_data['Recording timestamp'] = pd.to_datetime(eye_data['Recording timestamp'], unit='us')
    eye_data['Timestamp'] = eye_data['Recording date'] + pd.to_timedelta(eye_data['Recording start time'].dt.strftime('%H:%M:%S.%f')) + pd.to_timedelta(eye_data['Recording timestamp'].dt.strftime('%H:%M:%S.%f'))

    return eye_data[(eye_data['Sensor'] == 'Eye Tracker')].drop(columns=['Recording timestamp', 'Computer timestamp', 'Recording start time UTC', 'Recording duration', 'Recording Fixation filter name', 'Project name', 'Export date', 'Recording name','Recording date', 'Recording date UTC', 'Recording start time', 'Recording media name', 'Recording media width', 'Recording media height'])

def dist(x, y):
    return (x - y) ** 2

def get_min(m0, m1, m2, i, j):
    if m0 < m1:
        if m0 < m2:
            return i - 1, j, m0
        else:
            return i - 1, j - 1, m2
    else:
        if m1 < m2:
            return i, j - 1, m1
        else:
            return i - 1, j - 1, m2

def spring_kawano(x, y, epsilon):
    Tx = len(x)
    Ty = len(y)

    C = np.zeros((Tx, Ty))
    B = np.zeros((Tx, Ty, 2), int)
    S = np.zeros((Tx, Ty), int)

    C[0, 0] = dist(x[0], y[0])

    for j in range(1, Ty):
        C[0, j] = C[0, j - 1] + dist(x[0], y[j])
        B[0, j] = [0, j - 1]
        S[0, j] = S[0, j - 1]

    for i in range(1, Tx):
        C[i, 0] = dist(x[i], y[0])
        B[i, 0] = [0, 0]
        S[i, 0] = i

        for j in range(1, Ty):
            pi, pj, m = get_min(C[i - 1, j],
                                C[i, j - 1],
                                C[i - 1, j - 1],
                                i, j)
            C[i, j] = dist(x[i], y[j]) + m
            B[i, j] = [pi, pj]
            S[i, j] = S[pi, pj]

        imin = np.argmin(C[:(i+1), -1])
        dmin = C[imin, -1]

        if dmin > epsilon:
            continue

        for j in range(1, Ty):
            if (C[i,j] < dmin) and (S[i, j] < imin):
                break
        else:
            path = [[imin, Ty - 1]]
            temp_i = imin
            temp_j = Ty - 1

            while (B[temp_i, temp_j][0] != 0 or B[temp_i, temp_j][1] != 0):
                path.append(B[temp_i, temp_j])
                temp_i, temp_j = B[temp_i, temp_j].astype(int)

            C[S <= imin] = 100000000
            yield np.array(path), dmin

def plot_spring_kawano(data_x, data_y, timestamps, epsilon):
    pathes = []
    times = []

    for path, cost in spring_kawano(data_x, data_y, epsilon):
        plt.figure(figsize=(72, 6))  # グラフを横長にする

        # マッチングパスをプロット
        for line in path:
            plt.plot([timestamps[line[0]], timestamps[line[1]]], [data_x[line[0]], data_y[line[1]]], linewidth=0.8, c="gray")

        # 長いストリームデータと短いパターンデータをプロット
        plt.plot(timestamps, data_x, label='long data')
        plt.plot(timestamps[:len(data_y)], data_y, label='gesture data')

        # マッチングパスの部分をプロット
        plt.plot(timestamps[path[:,0]], data_x[path[:,0]], c="C2", label='similar')

        plt.grid(True)  # グリッド表示
        plt.legend()  # 凡例表示
        plt.xticks(rotation=45)  # x軸のラベルを45度回転
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%M:%S:%f'))  # Timestampフォーマット設定
        plt.show()

        times.append(timestamps[path[:,0]])
        pathes.append(path)

    return pathes, times

def spring_ogawa(G, QG, Th):
    d = np.zeros((len(G)+1, len(QG)+1)) # distance
    s = np.zeros((len(G)+1, len(QG)+1), dtype=np.int32) # starting point of current distance calc.
    for j in range(1, len(QG)+1):
        d[0, j] = np.inf
    for i in range(1, len(G)+1):
        s[i, 0] = i-1

    seg = []
    d_min = np.inf
    t_s = t_e = 0

    for i in range(len(G)):
        for j in range(len(QG)):
            if d[i+1, j] <= d[i, j] and d[i+1, j] <= d[i, j+1]:
                d_best = d[i+1, j]
                s[i+1, j+1] = s[i+1, j]
            elif d[i, j+1] <= d[i, j]:
                d_best = d[i, j+1]
                s[i+1, j+1] = s[i, j+1]
            else:
                d_best = d[i, j]
                s[i+1, j+1] = s[i, j]

            d[i+1, j+1] = np.abs(G[i] - QG[j]) + d_best

        if d_min <= Th:
            flag = 1
            for j in range(len(QG)):
                if d[i+1, j+1] >= d_min or s[i+1, j+1] > t_e:
                    flag *= 1
                else:
                    flag *= 0
            if flag == 1:
                print("****")
                seg.append([i, d_min, t_s, t_e])
                d_min = np.inf
                for j in range(len(QG)):
                    if s[i+1, j+1] <= t_e:
                        d[i+1, j+1] = np.inf

        if d[i+1, -1] <= Th and d[i+1, -1] < d_min:
            d_min = d[i+1, -1]
            t_s = s[i+1, -1]
            t_e = i

    return seg
