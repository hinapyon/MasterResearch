# 基本ライブラリ
import csv
import math
import os
import pickle
import random
import re
import statistics
from datetime import datetime, timedelta
from decimal import Decimal

# 数値計算とデータ処理
#import bottleneck as bn
import numpy as np
import pandas as pd

# 機械学習ライブラリ
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ディープラーニングライブラリ
import tensorflow as tf
from tensorflow.keras.layers import (
    Activation,
    Add,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling1D,
    Input,
    LayerNormalization,
    LSTM,
    Masking,
    MaxPooling1D,
    MultiHeadAttention,
)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


# プロットと可視化
import japanize_matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

# その他のライブラリ
from fastdtw import fastdtw
from scipy import signal, stats
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.signal import savgol_filter
from scipy.spatial.distance import euclidean
from tslearn.metrics import dtw_path

def process_apple_watch_csv(file_path: str) -> pd.DataFrame:
    """
    Apple Watchからのモーションデータを読み込み、処理する関数

    Parameters:
    file_path (str): CSVファイルのパス

    Returns:
    pd.DataFrame: 処理されたモーションデータを含むデータフレーム
    """

    # CSVファイルの読み込みとカラム名の設定
    motion_data = pd.read_csv(file_path, header=0, names=[
        'UnixTime', 'AccelerationX', 'AccelerationY', 'AccelerationZ',
        'GyroX', 'GyroY', 'GyroZ', 'Marking'
    ])

    # タイムスタンプの変換
    motion_data['Timestamp'] = pd.to_datetime(motion_data['UnixTime'], unit='s').dt.tz_localize('UTC').dt.tz_convert('Asia/Tokyo').dt.tz_localize(None)

    # ユークリッドノルムの計算
    motion_data['EuclideanNorm'] = np.linalg.norm(motion_data[['AccelerationX', 'AccelerationY', 'AccelerationZ']], axis=1)
    motion_data['EuclideanNormGyro'] = np.linalg.norm(motion_data[['GyroX', 'GyroY', 'GyroZ']], axis=1)

    # Savitzky-Golayフィルタの適用
    window_length = 11
    polyorder = 2
    for axis in ['AccelerationX', 'AccelerationY', 'AccelerationZ', 'EuclideanNorm']:
        motion_data[f'SG_{axis}'] = savgol_filter(motion_data[axis], window_length, polyorder)

    # FFTとパワースペクトルの計算
    N = len(motion_data)
    Fs = 50
    for axis in ['AccelerationX', 'AccelerationY', 'AccelerationZ', 'EuclideanNorm']:
        fft_values = np.fft.fft(motion_data[axis])
        motion_data[f'PowerSpectrum_{axis}'] = np.abs(fft_values)**2 / N

    return motion_data

def process_all_apple_watch_csv_in_directory(directory: str) -> list[pd.DataFrame]:
    """
    指定されたディレクトリ内のすべてのApple WatchモーションデータCSVファイルを処理し、
    データフレームのリストとして返す関数。

    Parameters:
    directory (str): CSVファイルが格納されているディレクトリのパス

    Returns:
    list of pd.DataFrame: 処理されたモーションデータを含むデータフレームのリスト
    """
    csv_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]
    return [process_apple_watch_csv(file_path) for file_path in csv_files]

def process_tobii_csv(file_path: str) -> pd.DataFrame:
    """
    Tobii Pro Glasses 2からのアイトラッキングデータを読み込み、処理し、
    元のデータと各種特徴量の標準偏差と平均を1つのデータフレームにまとめて出力する関数。

    Parameters:
    file_path (str): CSVファイルのパス

    Returns:
    pd.DataFrame: 元のデータと各特徴量の標準偏差および平均を含むデータフレーム
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

    # 'Eye Tracker'センサーのデータのみを抽出
    processed_eye_data = eye_data[
        eye_data['Sensor'] == 'Eye Tracker'
    ].drop(columns=[
        'Recording timestamp', 'Computer timestamp', 'Recording start time UTC',
        'Recording duration', 'Recording Fixation filter name', 'Project name',
        'Export date', 'Recording name', 'Recording date', 'Recording date UTC',
        'Recording start time', 'Recording media name', 'Recording media width',
        'Recording media height'
    ])

    # 特徴量計算
    def calculate_distance_2d(data, x_col, y_col):
        diff_x = data[x_col].interpolate(method='linear').diff()
        diff_y = data[y_col].interpolate(method='linear').diff()
        return np.sqrt(diff_x**2 + diff_y**2)

    def calculate_distance_3d(data, x_col, y_col, z_col):
        diff_x = data[x_col].interpolate(method='linear').diff()
        diff_y = data[y_col].interpolate(method='linear').diff()
        diff_z = data[z_col].interpolate(method='linear').diff()
        return np.sqrt(diff_x**2 + diff_y**2 + diff_z**2)

    # 各特徴量を計算
    features = {
        'Gaze2D_Distance': calculate_distance_2d(processed_eye_data, 'Gaze point X', 'Gaze point Y'),
        'Fixation_Distance': calculate_distance_2d(processed_eye_data, 'Fixation point X', 'Fixation point Y'),
        'Gaze3D_Distance': calculate_distance_3d(processed_eye_data, 'Gaze point 3D X', 'Gaze point 3D Y', 'Gaze point 3D Z'),
        'Pupil_Diameter_Change': (
            processed_eye_data['Pupil diameter right'].interpolate(method='linear').diff() +
            processed_eye_data['Pupil diameter left'].interpolate(method='linear').diff()
        ) / 2,
        'GazeDirection_Distance': (
            calculate_distance_3d(processed_eye_data, 'Gaze direction right X', 'Gaze direction right Y', 'Gaze direction right Z') +
            calculate_distance_3d(processed_eye_data, 'Gaze direction left X', 'Gaze direction left Y', 'Gaze direction left Z')
        ) / 2,
        'PupilPosition_Distance': (
            calculate_distance_3d(processed_eye_data, 'Pupil position right X', 'Pupil position right Y', 'Pupil position right Z') +
            calculate_distance_3d(processed_eye_data, 'Pupil position left X', 'Pupil position left Y', 'Pupil position left Z')
        ) / 2
    }

    # 特徴量をデータフレームに追加
    for feature_name, feature_data in features.items():
        processed_eye_data[feature_name] = feature_data

    return processed_eye_data

def load_motion_and_eye_data(name: str) -> tuple[list[pd.DataFrame], list[str]]:
    """
    指定された人物名とジェスチャーラベルに対応するデータを読み込み、
    不要なカラムを削除して前処理した目のデータとジェスチャーラベルのリストを返す関数。

    Parameters:
    name (str): データを読み込む人物の名前
    labels (List[str]): 読み込むジェスチャーラベルのリスト
    columns_to_drop (List[str]): 削除するカラム名のリスト

    Returns:
    Tuple[List[pd.DataFrame], List[str]]:
        - 前処理された目のデータ（データフレームのリスト）
        - 対応するジェスチャーラベルのリスト
    """
    motion_data_filename = f'datasets/new/{name}/motion/{name}_new_routine.csv'
    eye_data_filename = f'datasets/new/{name}/eye/{name}_eye_new_routine.csv'
    motion_data = process_apple_watch_csv(motion_data_filename)
    eye_data = process_tobii_csv(eye_data_filename)
    return motion_data, eye_data

def load_gesture_eye_data_pickle(name: str, labels: list[str], columns_to_drop: list[str]) -> tuple[list[pd.DataFrame], list[str]]:
    """
    指定された人物名とジェスチャーラベルに対応するデータを読み込み、
    不要なカラムを削除して前処理した目のデータとジェスチャーラベルのリストを返す関数。

    Parameters:
    name (str): データを読み込む人物の名前
    labels (List[str]): 読み込むジェスチャーラベルのリスト
    columns_to_drop (List[str]): 削除するカラム名のリスト

    Returns:
    Tuple[List[pd.DataFrame], List[str]]:
        - 前処理された目のデータ（データフレームのリスト）
        - 対応するジェスチャーラベルのリスト
    """
    gesture_eye_data = []
    gesture_labels = []
    for label in labels:
        filename = f'datasets/new/{name}/eye/train_pickle/{name}_{label}_eye.pkl'
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            # 指定されたカラムを削除
            processed_data = [df.drop(columns=columns_to_drop) for df in data]
            gesture_eye_data.extend(processed_data)
        gesture_labels.extend([label] * len(processed_data))
    return gesture_eye_data, gesture_labels

def load_trimmed_gesture_data(names, base_dir):
    """
    トリミング済みジェスチャーデータを読み込む関数。

    Parameters:
    - names (list[str]): 被験者名のリスト
    - base_dir (str): トリミング済みデータが保存されているディレクトリのベースパス

    Returns:
    - dict: 読み込んだトリミング済みデータ（辞書形式）
    """
    loaded_data = {}

    for name in names:
        file_path = os.path.join(base_dir, name, "motion", f"{name}_trimmed_gesture_data.pkl")

        if not os.path.exists(file_path):
            print(f"ファイルが見つかりません: {file_path}")
            continue

        with open(file_path, "rb") as f:
            trimmed_data = pickle.load(f)
            loaded_data[name] = trimmed_data
            print(f"{name} のトリミング済みデータを読み込みました。")

    return loaded_data

def load_trimmed_first_gesture_data(names, base_dir):
    """
    トリミング済みジェスチャーデータを読み込む関数。

    Parameters:
    - names (list[str]): 被験者名のリスト
    - base_dir (str): トリミング済みデータが保存されているディレクトリのベースパス

    Returns:
    - dict: 読み込んだトリミング済みデータ（辞書形式）
    """
    loaded_data = {}

    for name in names:
        file_path = os.path.join(base_dir, name, "motion", f"{name}_trimmed_first_gesture_data.pkl")

        if not os.path.exists(file_path):
            print(f"ファイルが見つかりません: {file_path}")
            continue

        with open(file_path, "rb") as f:
            trimmed_data = pickle.load(f)
            loaded_data[name] = trimmed_data
            print(f"{name} のファーストトリミング済みデータを読み込みました。")

    return loaded_data

def calculate_length_extremes(trimmed_data_dict):
    """
    トリミング済みデータの各ジェスチャラベルにおけるデータ長の最短と最長を計算。

    Parameters:
    - trimmed_data_dict (dict): トリミング済みジェスチャーデータの辞書

    Returns:
    - dict: 各nameとlabelに対応する最短・最長長さを格納した辞書
    """
    length_extremes = {}

    for name, labels_data in trimmed_data_dict.items():
        length_extremes[name] = {}
        for label, data_list in labels_data.items():
            if data_list:  # データリストが空でない場合
                lengths = [len(data) for data in data_list]
                min_length = min(lengths)
                max_length = max(lengths)
                length_extremes[name][label] = {"min_length": min_length, "max_length": max_length}
            else:
                length_extremes[name][label] = {"min_length": None, "max_length": None}

    return length_extremes

def spring(G: list[float], QG: list[float], Th: float) -> list[tuple[int, float, int, int]]:
    """
    SPRINGアルゴリズムを使用して、時間シリーズデータのセグメントを検索する関数。

    Parameters:
    G (list of float): 対象の時間シリーズデータ。
    QG (list of float): クエリ時間シリーズデータ。
    Th (float): マッチングのしきい値。

    Returns:
    list of list: 検出されたセグメントリスト。各セグメントは (i, d_min, t_s, t_e) のタプルで構成される。
    """
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

def three_axis_spring(motion_data: pd.DataFrame, train_data: list[pd.DataFrame], thresholds: tuple[float, float, float], data_type: str) -> tuple[list[int, float, int, int], list[int, float, int, int], list[int, float, int, int]]:
    """
    3軸のSPRINGアルゴリズムを実行し、結果を返す関数

    Parameters:
    motion_data (pd.DataFrame): 処理するモーションデータ
    train_data (list of pd.DataFrame): 各教師データ
    thresholds (tuple): 各軸に対するしきい値 (Th1, Th2, Th3)
    data_type (str): 'acc', 'sgacc', または 'gyro' のいずれか

    Returns:
    tuple: 各軸のSPRINGアルゴリズムの結果 (segx, segy, segz)
    """
    axis_map = {
        'acc': ['AccelerationX', 'AccelerationY', 'AccelerationZ'],
        'sgacc': ['SG_AccelerationX', 'SG_AccelerationY', 'SG_AccelerationZ'],
        'gyro': ['GyroX', 'GyroY', 'GyroZ']
    }

    axes = axis_map.get(data_type)
    if axes is None:
        raise ValueError("Invalid data type. Choose from 'acc', 'sgacc', or 'gyro'.")

    seg_results = [[], [], []]  # segx, segy, segz に相当

    for i in range(len(train_data)):
        for j, axis in enumerate(axes):
            seg_results[j].append(spring(motion_data[axis], train_data[i][axis], thresholds[j]))

    return tuple(seg_results)

def is_within_time_range(t_s: int, t_e: int, Hz: int, min_time: float, max_time: float) -> bool:
    """
    指定された開始時間と終了時間の間の経過時間が、指定された時間範囲内にあるかどうかを判定する関数。

    Parameters:
    t_s (int): 開始時間（サンプルインデックス）。
    t_e (int): 終了時間（サンプルインデックス）。
    Hz (int): サンプリング周波数（Hz）。1秒間に取得されるデータのサンプル数。
    min_time (float): 下限となる最小経過時間（秒）。
    max_time (float): 上限となる最大経過時間（秒）。

    Returns:
    bool: 経過時間が指定された範囲内であればTrue、そうでなければFalse。
    """
    elapsed_time = (t_e - t_s) / Hz
    return min_time < elapsed_time < max_time

def filter_segments_by_elapsed_time(segments: list[list[tuple[int, float, int, int]]], Hz: int, min_time: float, max_time: float) -> list[tuple[int, float, int, int]]:
    """
    セグメントの開始時間と終了時間の間の経過時間に基づいてセグメントをフィルタリングし、
    入力セグメントと同じ構造で返す関数。

    Parameters:
    segments (list of list): 各セグメントのリスト。セグメントは (l, d_min, t_s, t_e) のタプルで構成される。
    Hz (int): サンプリング周波数（Hz）。1秒間に取得されるデータのサンプル数。
    min_time (float): フィルタリングの下限となる最小経過時間（秒）。
    max_time (float): フィルタリングの上限となる最大経過時間（秒）。

    Returns:
    list of list: フィルタリングされたセグメントのリスト。セグメントは (l, d_min, t_s, t_e) のタプルで構成される。
    """

    filtered_segments = [
        [(l, d_min, t_s, t_e) for l, d_min, t_s, t_e in segment if is_within_time_range(t_s, t_e, Hz, min_time, max_time)]
        for segment in segments
    ]

    return filtered_segments

def three_axis_filter_segments_by_elapsed_time(segments_tuple: tuple[list[list[int, float, int, int]], list[list[int, float, int, int]], list[list[int, float, int, int]]], Hz: int, min_time: float, max_time: float) -> tuple[list[list[int, float, int, int]], list[list[int, float, int, int]], list[list[int, float, int, int]]]:
    """
    3軸分のセグメントをフィルタリングし、フィルタリング後の3軸分のセグメントタプルを返す関数。

    Parameters:
    segments_tuple (tuple of lists):
    3軸分のセグメントリストのタプル。
    各軸のセグメントリストは (l, d_min, t_s, t_e) のタプルで構成される。
    Hz (int): サンプリング周波数（Hz）。
    min_time (float): フィルタリングの下限となる最小経過時間（秒）。
    max_time (float): フィルタリングの上限となる最大経過時間（秒）。

    Returns:
    tuple of lists: フィルタリングされた3軸分のセグメントリストのタプル。
    """

    return tuple(
        filter_segments_by_elapsed_time(axis_segments, Hz, min_time, max_time)
        for axis_segments in segments_tuple
    )

def combine_and_find_overlapping_segments(
    segx: list[tuple[int, float, int, int]],
    segy: list[tuple[int, float, int, int]],
    segz: list[tuple[int, float, int, int]],
    overlap_count: int
) -> list[tuple[int, int]]:
    """
    3軸のセグメントを統合し、重なり合う時間範囲を見つける関数

    Parameters:
    - segx, segy, segz (list of tuples): 各軸のセグメントリスト (i, d_min, t_s, t_e)
    - overlap_count (int): どれだけ重なったかの下限

    Returns:
    overlap_ranges (list of tuples): 重なり合った時間範囲のリスト (start, end)
    """

    # セグメントの統合とソート
    all_segments = sorted((t_s, t_e) for seg in [segx, segy, segz] for _, _, t_s, t_e in seg)

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
                if current_count >= overlap_count:
                    overlap_ranges.append(current_overlap)
                current_overlap = (start, end)
                current_count = 1

    # 最後の重なり合い範囲を確認して追加
    if current_count >= overlap_count:
        overlap_ranges.append(current_overlap)

    return overlap_ranges

def combine_and_find_overlapping_all_segments(
    segments: list[list[list[tuple[int, float, int, int]]]],
    overlap_count: int
) -> list[list[tuple[int, int]]]:
    """
    各教師データの結果ごとにオーバーラップを検出する関数

    Parameters:
    segments (list of lists of tuples): 各軸のセグメントリスト。各セグメントリストは (l, d_min, t_s, t_e) のタプルを含む。
    overlap_count (int): どれだけ重なったかの下限

    Returns:
    list of lists of tuples: 重複している区間のリスト。各区間は (t_s, t_e) のタプルで構成される。
    """
    return [combine_and_find_overlapping_segments(segments[0][i], segments[1][i], segments[2][i], overlap_count) for i in range(len(segments[0]))]

def filter_overlaps_by_elapsed_time(
    overlap: list[list[tuple[int, int]]],
    Hz: int,
    min_time: float,
    max_time: float
) -> list[list[tuple[int, int]]]:
    """
    各教師データの結果ごとに出したオーバーラップから、経過時間に基づいて条件を満たさないセグメントを削除する関数

    Parameters:
    overlap (list of lists of tuples): 各セグメントリスト。各セグメントは (start, end) のタプルで構成される。
    Hz (int): サンプリング周波数（Hz）。1秒間に取得されるデータのサンプル数。
    min_time (float): フィルタリングの下限となる最小経過時間（秒）。
    max_time (float): フィルタリングの上限となる最大経過時間（秒）。

    Returns:
    list of lists of tuples: フィルタリングされたオーバーラップのリスト。各区間は (start, end) のタプルで構成される。
    """

    filtered_overlap = [
        [(start, end) for start, end in segments if is_within_time_range(start, end, Hz, min_time, max_time)]
        for segments in overlap if segments
    ]

    return filtered_overlap

def combine_overlapping_segments(overlap: list[list[tuple[int, int]]]) -> list[tuple[int, int]]:
    """
    複数のリストに分かれたセグメントを統合し、重なり合っている部分のみを統合する関数。

    Parameters:
    overlap (list of lists of tuples): 複数のセグメントリスト。各セグメントリストは (start, end) のタプルで構成される。

    Returns:
    list of tuples: 統合され、重なり合った部分がまとめられたセグメントリスト。
    """
    # 全てのセグメントを一つのリストに統合
    combined_segments = []
    for segments in overlap:
        combined_segments.extend(segments)

    if not combined_segments:
        return []

    # Sort segments by start time
    combined_segments.sort()
    final_segments = []
    current_start, current_end = combined_segments[0]

    for start, end in combined_segments[1:]:
        if start <= current_end:  # 重なり合っている場合
            current_end = max(current_end, end)  # 終了時間を延長
        else:  # 重ならない場合
            final_segments.append((current_start, current_end))  # 現在のセグメントを追加
            current_start, current_end = start, end  # 新しいセグメントを開始

    # 最後のセグメントを追加
    final_segments.append((current_start, current_end))

    return final_segments

def filter_and_combine_segments(
    segments: list[list[tuple[int, float, int, int]]],
    Hz: int,
    min_time: float,
    max_time: float,
    overlap_count: int
) -> list[tuple[int, int]]:
    """
    セグメントをフィルタリングし、重なり合っている部分を統合する関数。

    この関数は、3軸のセグメントデータに対して、指定された時間範囲に基づいてフィルタリングを行い、
    重なり合うセグメントを統合して、最終的なセグメントリストを返します。

    Parameters:
    segments (list of lists of tuples):
        3軸分のセグメントリスト。各軸のセグメントリストは (l, d_min, t_s, t_e) のタプルで構成されます。
    Hz (int):
        サンプリング周波数（Hz）。1秒間に取得されるデータのサンプル数。
    min_time (float):
        フィルタリングの下限となる最小経過時間（秒）。
    max_time (float):
        フィルタリングの上限となる最大経過時間（秒）。
    overlap_count (int): どれだけ重なったかの下限

    Returns:
    list of tuples:
        統合され、重なり合った部分がまとめられた最終的なセグメントリスト。各セグメントは (start, end) のタプルで構成されます。
    """

    # 3軸のセグメントを指定された時間範囲でフィルタリング
    filtered_segments = three_axis_filter_segments_by_elapsed_time(segments, Hz, min_time, max_time)

    # フィルタリングされたセグメントから、各教師データごとに重なり合いを検出
    overlapping_segments = combine_and_find_overlapping_all_segments(filtered_segments, overlap_count)

    # 重なり合っているセグメントを、経過時間に基づいてさらにフィルタリング
    filtered_overlaps = filter_overlaps_by_elapsed_time(overlapping_segments, Hz, min_time, max_time)

    # 重なり合っている部分を統合して最終的なセグメントリストを作成
    final_segments = combine_overlapping_segments(filtered_overlaps)

    return final_segments

# グラフプロット関数
def plot_detected_segments_with_highlights(data, marking_intervals, detected_intervals):
    """
    データの区間をハイライトしてプロットする関数。

    Parameters:
    - data (pd.DataFrame): プロットするデータ。
    - marking_intervals (list of tuple): ['Marking']がTrueであるインデックスのリスト。
    - detected_intervals (list of tuple): SPRINGで検出されたインデックスのリスト。
    """
    plt.figure(figsize=(12, 6))

    # 元データをプロット
    plt.plot(data.index, data['EuclideanNorm'], label='EuclideanNorm', color='blue')

    # ['Marking']がTrueの区間を黄色でハイライト
    for start, end in marking_intervals:
        plt.axvspan(start, end, color='yellow', alpha=0.3, label='Marking=True' if start == marking_intervals[0][0] else "")

    # SPRINGで検出された区間を青色でハイライト
    for start, end in detected_intervals:
        plt.axvspan(start, end, color='cyan', alpha=0.4, label='SPRING Detected' if start == detected_intervals[0][0] else "")

    # グラフ設定
    plt.title("Detected Segments with Highlights", fontsize=16)
    plt.xlabel("Time Step", fontsize=12)
    plt.ylabel("EuclideanNorm", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.show()

#検出された区間のタイムスタンプを出力するやつ
def extract_timestamp_from_overlap(motion_data, combine_overlap):
    results = []
    for start, end in combine_overlap:
        results.append([motion_data['Timestamp'][start], motion_data['Timestamp'][end]])
    return results

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

def find_true_intervals(df: pd.DataFrame, column_name: str = 'Marking') -> list:
    """
    データフレーム内で指定されたカラムがTrueとなっている区間の開始・終了インデックスを抽出する関数。

    Parameters:
    df (pd.DataFrame): 対象のデータフレーム
    column_name (str): 真偽値を持つカラム名（デフォルトは 'Marking'）

    Returns:
    list of tuples: 各区間の開始インデックスと終了インデックスのタプルのリスト
    """
    true_intervals = []
    start_index = None

    for index, row in df.iterrows():
        if row[column_name]:
            if start_index is None:
                start_index = index
        else:
            if start_index is not None:
                true_intervals.append((start_index, index - 1))
                start_index = None

    # 最後のTrueの区間がデータフレームの最後まで続く場合
    if start_index is not None:
        true_intervals.append((start_index, df.index[-1]))

    return true_intervals

def trim_and_save_gesture_data(gesture_data_dict, names, labels, output_base_path):
    """
    教師データのトリミングと保存を行う関数。

    Parameters:
    - gesture_data_dict (dict): 各被験者とジェスチャーのデータを格納した辞書。
    - names (list): 被験者名のリスト。
    - labels (list): ジェスチャーラベルのリスト。
    - output_base_path (str): トリミング後のデータを保存する基底パス。

    Returns:
    - None
    """
    # トリミング後のデータを保存する辞書
    trimmed_gesture_data = {}

    for name in names:
        print(f"Processing {name}...")
        trimmed_gesture_data[name] = {}

        for label in labels:
            print(f"Processing gesture: {label}...")
            data_list = gesture_data_dict[name, label]  # 該当被験者とジェスチャーのデータリスト
            trimmed_gesture_data[name][label] = []  # このジェスチャーのデータリストを初期化

            for i, data in enumerate(data_list):
                # データをプロットして確認
                plt.figure(figsize=(12, 6))
                plt.plot(data["EuclideanNorm"], label="EuclideanNorm", color="blue", linewidth=1.5)
                plt.plot(data["AccelerationX"], label="AccelerationX", color="red", alpha=0.6)
                plt.plot(data["AccelerationY"], label="AccelerationY", color="green", alpha=0.6)
                plt.plot(data["AccelerationZ"], label="AccelerationZ", color="orange", alpha=0.6)

                # 軸ラベルとグリッド
                plt.title(f"{name} - {label} - Sample {i}", fontsize=16)
                plt.xlabel("Time Step", fontsize=12)
                plt.ylabel("Value", fontsize=12)
                plt.legend(fontsize=12)
                plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

                # 横軸メモリを細かくする
                plt.xticks(
                    ticks=range(0, len(data), max(len(data) // 20, 1)),  # 最大20区切りになるよう調整
                    fontsize=10,
                    rotation=45
                )
                plt.tight_layout()
                plt.show()

                # トリミング範囲を入力
                print("このデータのトリミング範囲を指定してください：")
                start_idx = input(f"開始インデックス (0-{len(data)-1}, スキップする場合は空白): ")

                if not start_idx.strip():
                    print(f"データ {i} をスキップしました。")
                    continue

                start_idx = int(start_idx)
                end_idx = int(input(f"終了インデックス (0-{len(data)-1}): "))

                # トリミング
                trimmed_data = data.iloc[start_idx:end_idx].reset_index(drop=True)
                trimmed_gesture_data[name][label].append(trimmed_data)

                print(f"データ {i} をトリミングして保存しました： {start_idx} から {end_idx}")

        # トリミング後のデータを pkl ファイルに保存
        output_dir = os.path.join(output_base_path, name, "motion")
        os.makedirs(output_dir, exist_ok=True)  # ディレクトリが存在しない場合は作成
        output_file = os.path.join(output_dir, f"{name}_trimmed_gesture_data.pkl")

        with open(output_file, "wb") as f:
            pickle.dump(trimmed_gesture_data[name], f)

        print(f"{name} のトリミング後のデータを {output_file} に保存しました。")


#Transfformer機械学習モデル構築用の関数
def get_exclusion_intervals(motion_data: pd.DataFrame) -> pd.DataFrame:
    true_intervals = find_true_intervals(motion_data)
    intervals = [
        [motion_data.loc[start_idx, "Timestamp"], motion_data.loc[end_idx, "Timestamp"]]
        for start_idx, end_idx in true_intervals
    ]
    intervals_df = pd.DataFrame(intervals, columns=['start', 'end'])
    intervals_df = intervals_df.sort_values('start').reset_index(drop=True)

    # デバッグ: 各インターバルの型を確認
    for i, row in intervals_df.iterrows():
        print(f"Exclusion Interval {i}: start={row['start']}, end={row['end']}, types=({type(row['start'])}, {type(row['end'])})")

    return intervals_df

def get_available_intervals(
    exclusion_intervals: pd.DataFrame,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    continuity_threshold: float = 0.05  # 50ms
) -> list:
    """
    利用可能なインターバル（除外インターバル外の区間）を取得する関数。
    連続性を保つため、隣接するインターバル間のギャップが閾値以上の場合のみ利用可能とします。

    Parameters:
    exclusion_intervals (pd.DataFrame): 除外するインターバルのデータフレーム（'start' と 'end' カラム）
    start_time (pd.Timestamp): データ全体の開始時刻
    end_time (pd.Timestamp): データ全体の終了時刻
    continuity_threshold (float): インターバル間のギャップの最小許容時間（秒）

    Returns:
    list of tuples: 各利用可能なインターバルの開始・終了時刻のリスト
    """
    available_intervals = []

    if exclusion_intervals.empty:
        available_intervals.append((start_time, end_time))
        return available_intervals

    # 除外インターバルを開始時刻でソート
    exclusion_intervals = exclusion_intervals.sort_values('start').reset_index(drop=True)

    # 前の除外インターバルの終了時刻を初期化
    prev_end = start_time

    for idx, row in exclusion_intervals.iterrows():
        excl_start = row['start']
        excl_end = row['end']

        # 前の終了時刻と現在の開始時刻のギャップを計算
        gap_start = prev_end
        gap_end = excl_start

        gap_duration = (gap_end - gap_start).total_seconds()

        if gap_duration >= continuity_threshold:
            available_intervals.append((gap_start, gap_end))

        # 前の終了時刻を更新
        prev_end = excl_end

    # 最後の除外インターバル後のギャップを計算
    gap_start = prev_end
    gap_end = end_time

    gap_duration = (gap_end - gap_start).total_seconds()

    if gap_duration >= continuity_threshold:
        available_intervals.append((gap_start, gap_end))

    return available_intervals

def extract_random_intervals(
    eye_data: pd.DataFrame,
    available_intervals: list,
    min_interval: float,
    max_interval: float,
    num_intervals: int,
    continuity_threshold: float = 0.05  # 50ms
) -> list:
    """
    利用可能なインターバルから指定された長さのランダムなインターバルを抽出する関数。
    抽出するインターバル内のデータが連続していることを確認します。

    Parameters:
    eye_data (pd.DataFrame): 目のデータを含むデータフレーム（'Timestamp' カラムが必要）
    available_intervals (list): 利用可能なインターバルのリスト（開始時刻と終了時刻のタプル）
    min_interval (float): 最小インターバル長（秒）
    max_interval (float): 最大インターバル長（秒）
    num_intervals (int): 抽出するインターバルの数
    continuity_threshold (float): インターバル内のタイムスタンプの最大許容飛び（秒）

    Returns:
    list of pd.DataFrame: 抽出されたデータフレームのリスト
    """
    extracted_intervals = []
    attempts = 0
    max_attempts = num_intervals * 10  # 最大試行回数

    while len(extracted_intervals) < num_intervals and attempts < max_attempts:
        attempts += 1

        # 利用可能なインターバルをフィルタリング
        suitable_intervals = [
            interval for interval in available_intervals
            if (interval[1] - interval[0]).total_seconds() >= min_interval
        ]

        if not suitable_intervals:
            print("利用可能なインターバルが足りません。")
            break

        # ランダムに利用可能なインターバルを選択
        avail_start, avail_end = random.choice(suitable_intervals)
        avail_duration = (avail_end - avail_start).total_seconds()

        # 抽出するインターバルの長さを決定
        duration = random.uniform(min_interval, min(avail_duration, max_interval))

        # 抽出可能な開始時刻の範囲を計算
        max_start_time = avail_end - pd.Timedelta(seconds=duration)

        if max_start_time <= avail_start:
            continue

        # ランダムなオフセットを計算
        random_offset_seconds = random.uniform(0, (max_start_time - avail_start).total_seconds())
        start = avail_start + pd.Timedelta(seconds=random_offset_seconds)
        end = start + pd.Timedelta(seconds=duration)

        # インターバル内のデータを抽出
        data_in_interval = eye_data[
            (eye_data['Timestamp'] >= start) & (eye_data['Timestamp'] <= end)
        ]

        if data_in_interval.empty:
            continue

        # タイムスタンプの差分を計算し、連続性を確認
        time_diffs = data_in_interval['Timestamp'].diff().dt.total_seconds().dropna()
        if time_diffs.between(0.015, 0.025).all():  # 50Hzに近い（0.02秒）の許容範囲
            extracted_intervals.append(data_in_interval)
            print(f"Interval {len(extracted_intervals)} extracted successfully.")
        else:
            print(f"Interval {len(extracted_intervals)} has discontinuities. Skipping.")

    print(f"抽出されたインターバル数: {len(extracted_intervals)}")
    return extracted_intervals

def has_long_nan_block(series, threshold):
    """
    シリーズ内にthreshold以上の連続したNaNブロックが存在するかを判定する関数。

    Parameters:
    - series (pd.Series): チェック対象のシリーズ。
    - threshold (int): 許容する最大連続NaNの数。

    Returns:
    - bool: threshold以上の連続NaNブロックが存在する場合はTrue、そうでなければFalse。
    """
    is_nan = series.isna()
    # 連続するNaNのグループ番号を割り当て
    nan_groups = (is_nan != is_nan.shift()).cumsum()
    # 各NaNグループのサイズを取得
    nan_group_sizes = is_nan.groupby(nan_groups).sum()
    # threshold以上のNaNグループが存在するか判定
    return (nan_group_sizes >= threshold).any()

def preprocess_sequence(df, COLUMNS_TO_DROP, sampling_rate=50.0, max_gap_seconds=5.0):
    """
    単一のシーケンスデータに対する前処理を行う関数。

    Parameters:
    - df (pd.DataFrame): シーケンスデータのDataFrame。
    - COLUMNS_TO_DROP (list): 削除する不要なカラム名のリスト
    - sampling_rate (float): サンプリング周波数（Hz）。
    - max_gap_seconds (float): 補間可能な最大欠損期間（秒）。

    Returns:
    - pd.DataFrame: 前処理後のDataFrame。
    """
    max_gap_frames = int(max_gap_seconds * sampling_rate)  # 5秒 => 5フレーム

    # 0. 不要なカラムを削除
    df = df.drop(columns=[col for col in COLUMNS_TO_DROP if col in df.columns])

    # 1. 固視点座標の欠損値を -1 で埋める
    for coord in ['Fixation point X', 'Fixation point Y']:
        if coord in df.columns:
            # マスク特徴量の追加
            mask_col = f"{coord}_Mask"
            df[mask_col] = df[coord].notna().astype(int)
            # NaNを-1で埋める
            df[coord] = df[coord].fillna(-1)

    # 2. Gaze directionの線形補間と正規化
    gaze_directions = ['Gaze direction left X', 'Gaze direction left Y', 'Gaze direction left Z', 'Gaze direction right X', 'Gaze direction right Y', 'Gaze direction right Z']
    for gaze_col in gaze_directions:
        if gaze_col in df.columns:
            # 線形補間を適用
            df[gaze_col] = df[gaze_col].interpolate(method='linear', limit=max_gap_frames, limit_direction='both')
            # 欠損が残る場合は前後の値で補完
            df[gaze_col] = df[gaze_col].fillna(method='ffill').fillna(method='bfill')

    # Gaze directionベクトルの正規化
    for gaze_prefix in ['Gaze direction left', 'Gaze direction right']:
        gaze_cols = [f"{gaze_prefix} X", f"{gaze_prefix} Y", f"{gaze_prefix} Z"]
        if all(col in df.columns for col in gaze_cols):
            # ベクトルの大きさを計算
            magnitude = np.sqrt(df[gaze_cols].pow(2).sum(axis=1))
            # 大きさが0でない場合のみ正規化
            non_zero = magnitude != 0
            df.loc[non_zero, gaze_cols] = df.loc[non_zero, gaze_cols].div(magnitude[non_zero], axis=0)

    # 3. Gaze pointの線形補間
    gaze_points = ['Gaze point X', 'Gaze point Y']
    for gaze_col in gaze_points:
        if gaze_col in df.columns:
            # 線形補間を適用
            df[gaze_col] = df[gaze_col].interpolate(method='linear', limit=max_gap_frames, limit_direction='both')
            # 欠損が残る場合は前後の値で補完
            df[gaze_col] = df[gaze_col].fillna(method='ffill').fillna(method='bfill')

    # 4. 3D位置関連の3次スプライン補間
    spline_columns = ['Gaze point 3D X', 'Gaze point 3D Y', 'Gaze point 3D Z', 'Pupil position left X', 'Pupil position left Y', 'Pupil position left Z', 'Pupil position right X', 'Pupil position right Y', 'Pupil position right Z']
    for spline_col in spline_columns:
        if spline_col in df.columns:
            # スプライン補間を適用
            # スプライン補間には十分なデータ点が必要
            if df[spline_col].notna().sum() >= 4:
                try:
                    spline = UnivariateSpline(df.index[~df[spline_col].isna()], df[spline_col].dropna(), k=3, s=0)
                    df[spline_col] = spline(df.index)
                except Exception as e:
                    print(f"スプライン補間中にエラーが発生しました: {e}")
                    # 補間に失敗した場合は線形補間にフォールバック
                    df[spline_col] = df[spline_col].interpolate(method='linear', limit=max_gap_frames, limit_direction='both')
            else:
                # スプライン補間に必要なデータ点が不足している場合、線形補間にフォールバック
                df[spline_col] = df[spline_col].interpolate(method='linear', limit=max_gap_frames, limit_direction='both')
            # 欠損が残る場合は前後の値で補完
            df[spline_col] = df[spline_col].fillna(method='ffill').fillna(method='bfill')

    # 5. その他の数値特徴量の補間
    # 除外するカラム（マスク特徴量も含む）
    exclude_cols = ['Fixation point X', 'Fixation point Y'] + gaze_directions + gaze_points + spline_columns + [f"{coord}_Mask" for coord in ['Fixation point X', 'Fixation point Y']]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if col in exclude_cols:
            continue
        # 線形補間を適用
        df[col] = df[col].interpolate(method='linear', limit=max_gap_frames, limit_direction='both')
        # 欠損が残る場合は前後の値で補完
        df[col] = df[col].fillna(method='ffill').fillna(method='bfill')

    # 6. 残る NaN を 0 で埋める
    if df.isnull().values.any():
        print("残る NaN を 0 で埋めます。")
        df = df.fillna(0)

    return df

def preprocess_and_filter_sequences(
    X: list,
    y_labels: list,
    true_labels: list,
    COLUMNS_TO_DROP: list,
    sampling_rate=50.0,
    max_gap_seconds=0.1,
    max_nan_seconds=5.0,  # 1秒間のNaNを許容
) -> (list, list, list):
    """
    シーケンスデータの前処理とフィルタリングを行う関数。
    1秒間（50サンプル）以上連続するNaNが存在するシーケンスは削除します。

    Parameters:
    - X (list): シーケンスデータのリスト（各シーケンスは DataFrame または NumPy 配列）
    - y_labels (list): 各シーケンスのラベル
    - true_labels (list): 各シーケンスの真偽ラベル
    - sampling_rate (float): サンプリング周波数（Hz）
    - max_gap_seconds (float): 補間可能な最大欠損期間（秒）
    - max_nan_seconds (float): 削除対象とする最大欠損期間（秒）
    - COLUMNS_TO_DROP (list): 削除する不要なカラム名のリスト

    Returns:
    - filtered_X_scaled (list): 前処理とスケーリングを施したシーケンスデータのリスト
    - filtered_y_labels (list): フィルタリング後のラベルリスト
    - filtered_true_labels (list): フィルタリング後の真偽ラベルリスト
    - scaler (StandardScaler): フィット済みスケーラー
    """
    filtered_X = []
    filtered_y = []
    filtered_true = []
    max_nan_frames = int(max_nan_seconds * sampling_rate)  # 1秒 => 50フレーム

    for idx, sequence in enumerate(X):
        # シーケンスがDataFrameでない場合、DataFrameに変換
        if isinstance(sequence, np.ndarray):
            df = pd.DataFrame(sequence)
        elif isinstance(sequence, pd.DataFrame):
            df = sequence.copy()
        else:
            print(f"シーケンス {idx} がDataFrameでもNumPy配列でもありません。スキップします。")
            continue

        # 固視点座標の欠損チェック
        has_long_nan = False
        for coord in ['Fixation point X', 'Fixation point Y']:
            if coord in df.columns:
                if has_long_nan_block(df[coord], max_nan_frames):
                    has_long_nan = True
                    break

        if has_long_nan:
            print(f"シーケンス {idx} は1秒以上連続するNaNを含むため、削除されました。")
            continue

        # 前処理を適用
        df_processed = preprocess_sequence(df, COLUMNS_TO_DROP, sampling_rate, max_gap_seconds)

        # 数値型カラムのみを選択（再確認）
        df_processed = df_processed.drop(columns='Timestamp')

        filtered_X.append(df_processed)
        filtered_y.append(y_labels[idx])
        filtered_true.append(true_labels[idx])

    print(f"フィルタリング後のデータ数: {len(filtered_X)}")

    if not filtered_X:
        print("フィルタリング後にデータが存在しません。")
        return [], [], [], None

    return filtered_X, filtered_y, filtered_true

def preprocess_gesture_data(NAMES, LABELS, COLUMNS_TO_DROP, NON_GESTURE):
    """
    データを読み込んで前処理を行い、ジェスチャーデータと非ジェスチャーデータを統合します。

    Parameters:
        NAMES (list): 処理対象のデータ名のリスト。
        LABELS (list): ラベルのリスト。
        COLUMNS_TO_DROP (list): 不要なカラムのリスト。
        NON_GESTURE (int): 抽出する非ジェスチャーデータの個数。

    Returns:
        tuple: (X_filled, y_labels, true_labels, data_dict, label_dict, true_label_dict)
    """
    # データとラベルを格納するリストと辞書の初期化
    X_filled = []
    y_labels = []
    true_labels = []
    data_dict = {}
    label_dict = {}
    true_label_dict = {}

    for name in NAMES:
        print(f"Processing {name}...")

        # データの読み込みと前処理
        eye_data, gesture_labels = load_gesture_eye_data_pickle(name, LABELS, COLUMNS_TO_DROP)
        data_dict[name] = {'gesture': eye_data, 'non_gesture': []}
        label_dict[name] = {'gesture': gesture_labels, 'non_gesture': []}
        true_label_dict[name] = {'gesture': ['gesture'] * len(eye_data), 'non_gesture': []}

        # モーションデータとアイデータの読み込み
        motion_routine_data, eye_routine_data = load_motion_and_eye_data(name)
        eye_non_gesture_data = process_tobii_csv(f'datasets/new/{name}/eye/{name}_non_gesture.csv')

        # gestureデータの長さを算出
        if isinstance(eye_data, list):
            lengths = [df.shape[0] for df in eye_data]  # 各データフレームの行数を取得
            longest_length = max(lengths)
            shortest_length = min(lengths)
        else:
            longest_length = eye_data.shape[0]
            shortest_length = eye_data.shape[0]

        print(f"{name} の gesture データの長さ: 最長 {longest_length}, 最短 {shortest_length}")

        # ランダムな長さでデータを抽出
        random_non_gesture_data = []
        available_indices = list(range(eye_non_gesture_data.shape[0]))

        for _ in range(NON_GESTURE):
            # ランダムな長さを決定
            random_length = random.randint(shortest_length, longest_length)

            # 利用可能なインデックスからランダムに開始位置を選択
            possible_start_indices = [
                idx for idx in available_indices if idx + random_length <= len(available_indices)
            ]
            if not possible_start_indices:
                print(f"{name} の non-gesture データに十分なインデックスがありません。")
                break

            start_idx = random.choice(possible_start_indices)
            end_idx = start_idx + random_length

            # 抽出したデータを追加
            random_non_gesture_data.append(eye_non_gesture_data.iloc[start_idx:end_idx])

            # 使用済みインデックスを削除
            available_indices = [idx for idx in available_indices if idx < start_idx or idx >= end_idx]

        # 抽出したデータを辞書に格納
        data_dict[name]['non_gesture'] = random_non_gesture_data
        label_dict[name]['non_gesture'] = ['non_gesture'] * len(random_non_gesture_data)
        true_label_dict[name]['non_gesture'] = ['non_gesture'] * len(random_non_gesture_data)

        print(f"{name} の non-gesture データから {len(random_non_gesture_data)} 個のランダムデータを抽出しました。")

        # ジェスチャーデータの統合
        for seq, lbl, true_lbl in zip(eye_data, gesture_labels, true_label_dict[name]['gesture']):
            if seq is not None and len(seq) > 0:
                X_filled.append(seq)
                y_labels.append(lbl)
                true_labels.append(true_lbl)
            else:
                print(f"{name} のジェスチャーシーケンスが空です。対応するラベルをスキップします。")

        # 非ジェスチャーデータの統合
        for seq, lbl, true_lbl in zip(data_dict[name]['non_gesture'], label_dict[name]['non_gesture'], true_label_dict[name]['non_gesture']):
            if isinstance(seq, pd.DataFrame):
                if not seq.empty:
                    X_filled.append(seq)
                    y_labels.append(lbl)
                    true_labels.append(true_lbl)
                else:
                    print(f"{name} の非ジェスチャーシーケンスが空です。対応するラベルをスキップします。")
            else:
                print(f"{name} の非ジェスチャーシーケンスが DataFrame ではありません。スキップします。")

        print(f"総データ数: {len(X_filled)}")
        print(f"総ラベル数（y_labels）: {len(y_labels)}")
        print(f"総真偽ラベル数（true_labels）: {len(true_labels)}")

    return X_filled, y_labels, true_labels, data_dict, label_dict, true_label_dict

def check_timestamp_gaps(NAMES, data_dict):
    """
    タイムスタンプのギャップを確認します。

    Parameters:
        NAMES (list): 処理対象のデータ名のリスト。
        data_dict (dict): 非ジェスチャーデータを含む辞書。

    Returns:
        None
    """
    for name in NAMES:
        non_gesture_intervals = data_dict[name]['non_gesture']
        if non_gesture_intervals:
            for idx, eye_routine_data in enumerate(non_gesture_intervals):
                if isinstance(eye_routine_data, pd.DataFrame):
                    if not eye_routine_data.empty:
                        time_diffs = eye_routine_data['Timestamp'].diff().dt.total_seconds()
                        large_gaps = time_diffs[time_diffs > 0.05]
                        # print(f"{name} の non_gesture_intervals[{idx}] の大きなタイムスタンプギャップ数: {len(large_gaps)}")
                    else:
                        print(f"{name} の non_gesture_intervals[{idx}] が空です。")
                else:
                    print(f"{name} の non_gesture_intervals[{idx}] が DataFrame ではありません。")

def get_positional_encoding(max_len, d_model):
    """
    固定のサイン・コサイン位置エンコーディングを生成する関数。

    Parameters:
    - max_len (int): シーケンスの最大長
    - d_model (int): 埋め込み次元数

    Returns:
    - pos_encoding (np.array): 位置エンコーディング行列
    """
    angle_rads = np.arange(max_len)[:, np.newaxis] / np.power(10000, (2 * (np.arange(d_model)[np.newaxis, :]//2)) / np.float32(d_model))

    # 偶数次元はsin、奇数次元はcos
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]  # (1, max_len, d_model)
    return tf.cast(pos_encoding, dtype=tf.float32)

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    """
    Transformerエンコーダーブロックを構築する関数。
    """
    # 自己注意機構（マスクは自動的に適用される）
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = Dropout(dropout)(x)
    x = LayerNormalization(epsilon=1e-6)(inputs + x)

    # フィードフォワードネットワーク
    x_ff = Dense(ff_dim, activation='relu')(x)
    x_ff = Dense(inputs.shape[-1])(x_ff)
    x_ff = Dropout(dropout)(x_ff)
    out = LayerNormalization(epsilon=1e-6)(x + x_ff)
    return out

def build_transformer_model(max_length, num_features, head_size, num_heads, ff_dim, num_transformer_blocks, dropout):
    """
    Transformerベースの2値分類モデルを構築する関数。
    """
    inputs = Input(shape=(max_length, num_features))

    # マスキングレイヤーの追加（パディング値が0の場合）
    x = Masking(mask_value=0.0)(inputs)

    # 位置エンコーディングの追加
    pos_encoding = get_positional_encoding(max_length, num_features)
    x = x + pos_encoding[:,:max_length, :]

    # Transformerエンコーダーブロックの追加
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    # グローバルプーリング
    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout)(x)

    # 出力層
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model
