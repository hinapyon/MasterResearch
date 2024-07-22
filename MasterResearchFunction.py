import os
import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import japanize_matplotlib

def process_apple_watch_csv(file_path):
    """
    Apple Watchからのモーションデータを読み込み、処理する関数

    Parameters:
    file_path (str): CSVファイルのパス

    Returns:
    pd.DataFrame: 処理されたモーションデータを含むデータフレーム
    """

    # CSVファイルの読み込み
    motion_data = pd.read_csv(file_path, header=0, names=[
        'UnixTime', 'AccelerationX', 'AccelerationY', 'AccelerationZ',
        'GyroX', 'GyroY', 'GyroZ'
    ])

    # Unixタイムスタンプを日本時間に変換し、タイムゾーン情報を削除して表示する
    motion_data['Timestamp'] = (
        pd.to_datetime(motion_data['UnixTime'], unit='s')
        .dt.tz_localize('UTC')
        .dt.tz_convert('Asia/Tokyo')
        .dt.tz_localize(None)
    )

    # 加速度のユークリッドノルムを計算してデータフレームに追加
    motion_data['EuclideanNorm'] = np.sqrt(
        motion_data['AccelerationX']**2 +
        motion_data['AccelerationY']**2 +
        motion_data['AccelerationZ']**2
    )

    # 角速度のユークリッドノルムを計算してデータフレームに追加
    motion_data['EuclideanNormGyro'] = np.sqrt(
        motion_data['GyroX']**2 +
        motion_data['GyroY']**2 +
        motion_data['GyroZ']**2
    )

    # Savitzky-Golayフィルタのパラメータ設定
    window_length = 11  # 窓の長さ（奇数、10サンプル以上）
    polyorder = 2       # 多項式の次数（2次）

    # Savitzky-Golayフィルタを適用
    for axis in ['AccelerationX', 'AccelerationY', 'AccelerationZ', 'EuclideanNorm']:
        motion_data[f'SG_{axis}'] = savgol_filter(motion_data[axis], window_length, polyorder)

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
    """
    指定されたディレクトリ内のすべてのApple WatchモーションデータCSVファイルを処理し、
    データフレームのリストとして返す関数。

    Parameters:
    directory (str): CSVファイルが格納されているディレクトリのパス

    Returns:
    list of pd.DataFrame: 処理されたモーションデータを含むデータフレームのリスト
    """

    dataframes = []

    # 指定されたディレクトリ内のすべてのファイルをチェック
    for filename in os.listdir(directory):
        # ファイルがCSVである場合のみ処理
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            # CSVファイルを処理
            df = process_apple_watch_csv(file_path)
            # 処理されたデータフレームをリストに追加
            dataframes.append(df)

    return dataframes

def process_tobii_csv(file_path):
    """
    Tobii Pro Glasses 2からのアイトラッキングデータを読み込み、処理する関数

    Parameters:
    file_path (str): CSVファイルのパス

    Returns:
    pd.DataFrame: 処理されたアイトラッキングデータを含むデータフレーム
    """

    # CSVファイルの読み込み
    eye_data = pd.read_csv(file_path, header=0)

    # タイムスタンプをYYYY-MM-DD HH:MM:SS.fff形式に変換
    eye_data['Recording start time'] = pd.to_datetime(eye_data['Recording start time'], format='%H:%M:%S.%f')
    eye_data['Recording date'] = pd.to_datetime(eye_data['Recording date'], format='%Y/%m/%d')
    eye_data['Recording timestamp'] = pd.to_datetime(eye_data['Recording timestamp'], unit='us')

    # 統合されたタイムスタンプを計算
    eye_data['Timestamp'] = (
        eye_data['Recording date'] +
        pd.to_timedelta(eye_data['Recording start time'].dt.strftime('%H:%M:%S.%f')) +
        pd.to_timedelta(eye_data['Recording timestamp'].dt.strftime('%H:%M:%S.%f'))
    )

    # 不要なカラムを削除し、'Eye Tracker'センサーのデータのみを抽出
    processed_eye_data = eye_data[
        eye_data['Sensor'] == 'Eye Tracker'
    ].drop(columns=[
        'Recording timestamp', 'Computer timestamp', 'Recording start time UTC',
        'Recording duration', 'Recording Fixation filter name', 'Project name',
        'Export date', 'Recording name', 'Recording date', 'Recording date UTC',
        'Recording start time', 'Recording media name', 'Recording media width',
        'Recording media height'
    ])

    return processed_eye_data

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

def three_axis_spring_ogawa(motion_data, train_data, Th1, Th2, Th3, Type):
    segx, segy, segz = [], [], []

    if Type == 'acc':
        for i in range(len(train_data)):
            segx.append(spring_ogawa(motion_data['AccelerationX'], train_data[i]['AccelerationX'], Th1))
            segy.append(spring_ogawa(motion_data['AccelerationY'], train_data[i]['AccelerationY'], Th2))
            segz.append(spring_ogawa(motion_data['AccelerationZ'], train_data[i]['AccelerationZ'], Th3))

    elif Type == 'sgacc':
        for i in range(len(train_data)):
            segx.append(spring_ogawa(motion_data['SG_AccelerationX'], train_data[i]['SG_AccelerationX'], Th1))
            segy.append(spring_ogawa(motion_data['SG_AccelerationY'], train_data[i]['SG_AccelerationY'], Th2))
            segz.append(spring_ogawa(motion_data['SG_AccelerationZ'], train_data[i]['SG_AccelerationZ'], Th3))

    elif Type == 'gyro':
        for i in range(len(train_data)):
            segx.append(spring_ogawa(motion_data['GyroX'], train_data[i]['GyroX'], Th1))
            segy.append(spring_ogawa(motion_data['GyroY'], train_data[i]['GyroY'], Th2))
            segz.append(spring_ogawa(motion_data['GyroZ'], train_data[i]['GyroZ'], Th3))

    return segx, segy, segz

#各セグメントをフィルタリングする関数
def filter_seg_by_elapsed_time(seg, motion_data, min_time, max_time):
    filtered_seg = []
    for segment in seg:  # 直接セグメントをイテレート
        temp = []
        for (l, d_min, t_s, t_e) in segment:
            start_time = motion_data['Timestamp'][t_s]
            end_time = motion_data['Timestamp'][t_e]
            elapsed_time = (end_time - start_time).total_seconds()
            if min_time < elapsed_time < max_time:
                temp.append((t_s, t_e))
        filtered_seg.append(temp)  # フィルタリング後にデータが存在しなくても空リストを追加
    return filtered_seg

#セグメントを統合
def combine_and_find_overlapping_segments(segx, segy, segz):
    # Combine all segments
    all_segments = []
    for seg in [segx, segy, segz]:
        for (i, d_min, t_s, t_e) in seg:
            all_segments.append((t_s, t_e))

    # Sort segments
    all_segments.sort()

    # Find overlapping segments
    overlap_ranges = []
    current_overlap = None
    current_count = 0

    for start, end in all_segments:
        if current_overlap is None:
            current_overlap = (start, end)
            current_count = 1
        else:
            current_start, current_end = current_overlap
            if start <= current_end:
                current_overlap = (current_start, max(current_end, end))
                current_count += 1
            else:
                if current_count >= 2:
                    overlap_ranges.append(current_overlap)
                current_overlap = (start, end)
                current_count = 1

    if current_count >= 2:
        overlap_ranges.append(current_overlap)

    return overlap_ranges

#フィルタリングされたセグメントを統合
def combine_and_find_overlapping_filtered_segments(segx, segy, segz):
    # Combine all segments
    all_segments = []
    for seg in [segx, segy, segz]:
        for (t_s, t_e) in seg:
            all_segments.append((t_s, t_e))

    # Sort segments
    all_segments.sort()

    # Find overlapping segments
    overlap_ranges = []
    current_overlap = None
    current_count = 0

    for start, end in all_segments:
        if current_overlap is None:
            current_overlap = (start, end)
            current_count = 1
        else:
            current_start, current_end = current_overlap
            if start <= current_end:
                current_overlap = (current_start, max(current_end, end))
                current_count += 1
            else:
                if current_count >= 2:
                    overlap_ranges.append(current_overlap)
                current_overlap = (start, end)
                current_count = 1

    if current_count >= 2:
        overlap_ranges.append(current_overlap)

    return overlap_ranges

# 各教師データの結果ごとにオーバーラップを検出
def overlap_from_three_segments(segx, segy, segz):
    overlap = []
    for i in range(len(segx)):
        overlap.append(combine_and_find_overlapping_segments(segx[i], segy[i], segz[i]))
    return overlap

# 各教師データの結果ごとに出したオーバーラップから、条件を満たさないセグメントを削除する関数
def filter_overlap_by_elapsed_time(overlap, motion_data, min_time, max_time):
    filtered_overlap = []
    for segments in overlap:
        temp = []
        for start, end in segments:
            start_time = motion_data['Timestamp'][start]
            end_time = motion_data['Timestamp'][end]
            elapsed_time = (end_time - start_time).total_seconds()
            if min_time < elapsed_time < max_time:
                temp.append((start, end))
        if temp:  # フィルタリング後にデータが存在する場合のみ追加
            filtered_overlap.append(temp)
    return filtered_overlap

#overlapの中からさらにコンバインするやつ
def combine_all_overlaps(overlap):
    # Combine all segments from multiple lists into one list
    combined_segments = []
    for segments in overlap:
        combined_segments.extend(segments)

    # Sort combined segments
    combined_segments.sort()

    # Find and merge overlapping segments
    final_segments = []
    current_overlap = None

    for start, end in combined_segments:
        if current_overlap is None:
            current_overlap = (start, end)
        else:
            current_start, current_end = current_overlap
            if start <= current_end:
                current_overlap = (current_start, max(current_end, end))
            else:
                final_segments.append(current_overlap)
                current_overlap = (start, end)

    if current_overlap is not None:
        final_segments.append(current_overlap)

    return final_segments

def new_combine_all_overlaps(overlap):
    # Combine all segments from multiple lists into one list
    combined_segments = []
    for segments in overlap:
        combined_segments.extend(segments)

    # Sort combined segments
    combined_segments.sort()

    # Find and merge overlapping segments
    final_segments = []
    current_overlap = None

    for start, end in combined_segments:
        if current_overlap is None:
            current_overlap = (start, end)
        else:
            current_start, current_end = current_overlap
            if start <= current_end:
                # Check if the current segment is completely within the previous segment
                if end <= current_end:
                    continue
                else:
                    # Extend the current overlap segment
                    current_overlap = (current_start, end)
            else:
                # If there is no overlap, add the current overlap to final segments
                final_segments.append(current_overlap)
                current_overlap = (start, end)

    if current_overlap is not None:
        final_segments.append(current_overlap)

    return final_segments

#検出された区間のタイムスタンプを出力するやつ
def extract_timestamp_from_overlap(motion_data, combine_overlap):
    results = []
    for start, end in combine_overlap:
        results.append([motion_data['Timestamp'][start], motion_data['Timestamp'][end]])
    return results

# 経過時間を算出して条件に合わないセグメントを削除する関数
def filter_segments_by_time_length(results, min_time, max_time):
    filtered_results = []
    for start_time, end_time in results:
        elapsed_time = (end_time - start_time).total_seconds()
        if min_time < elapsed_time < max_time:
            filtered_results.append([start_time, end_time])
    return filtered_results

# filtered_results の範囲内の eye_data['Timestamp'] を抽出する関数
def extract_eye_data_within_intervals(filtered_results, eye_data):
    extracted_eye_data = []
    for start_time, end_time in filtered_results:
        # タイムスタンプが範囲内のデータを抽出
        extracted_eye_data.append(eye_data[(eye_data['Timestamp'] >= start_time) & (eye_data['Timestamp'] <= end_time)])

    return extracted_eye_data

#視線データの移動距離の標準偏差でフィルタリング
def filter_by_std_gaze(extracted_eye_data, filtered_results, threshold):
    filtered_extracted_eye_data = []
    filtered_filtered_results = []

    for i in range(len(extracted_eye_data)):
        diff_x = extracted_eye_data[i]['Gaze point X'].interpolate(method='linear').diff()
        diff_y = extracted_eye_data[i]['Gaze point Y'].interpolate(method='linear').diff()
        distance = np.sqrt(diff_x**2 + diff_y**2)  # ユークリッド距離
        std = distance.std()  # ユークリッド距離の分散を計算
        diff_r = extracted_eye_data[i]['Pupil diameter right'].interpolate(method='linear').diff()
        diff_l = extracted_eye_data[i]['Pupil diameter left'].interpolate(method='linear').diff()
        std_r = diff_r.std()  # ユークリッド距離の分
        std_l = diff_l.std()  # ユークリッド距離の分
        print(std, std_r, std_l)

        if std < threshold:
            filtered_extracted_eye_data.append(extracted_eye_data[i])
            filtered_filtered_results.append(filtered_results[i])

    return filtered_extracted_eye_data, filtered_filtered_results

# 保存する変数を辞書にまとめる関数
def create_data_dict(segment_type):
    return {
        f'a_yuuma_{segment_type}_segx': globals()[f'a_yuuma_{segment_type}_segx'],
        f'a_yuuma_{segment_type}_segy': globals()[f'a_yuuma_{segment_type}_segy'],
        f'a_yuuma_{segment_type}_segz': globals()[f'a_yuuma_{segment_type}_segz'],
        f'b_sakamoto_{segment_type}_segx': globals()[f'b_sakamoto_{segment_type}_segx'],
        f'b_sakamoto_{segment_type}_segy': globals()[f'b_sakamoto_{segment_type}_segy'],
        f'b_sakamoto_{segment_type}_segz': globals()[f'b_sakamoto_{segment_type}_segz'],
        f'c_watabe_{segment_type}_segx': globals()[f'c_watabe_{segment_type}_segx'],
        f'c_watabe_{segment_type}_segy': globals()[f'c_watabe_{segment_type}_segy'],
        f'c_watabe_{segment_type}_segz': globals()[f'c_watabe_{segment_type}_segz'],
        f'd_nakazawa_{segment_type}_segx': globals()[f'd_nakazawa_{segment_type}_segx'],
        f'd_nakazawa_{segment_type}_segy': globals()[f'd_nakazawa_{segment_type}_segy'],
        f'd_nakazawa_{segment_type}_segz': globals()[f'd_nakazawa_{segment_type}_segz']
    }
