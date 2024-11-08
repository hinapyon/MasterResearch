import os
import numpy as np
import pandas as pd
import pickle
import random
from scipy import signal, stats
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import japanize_matplotlib

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
    segz: list[tuple[int, float, int, int]]
) -> list[tuple[int, int]]:
    """
    3軸のセグメントを統合し、重なり合う時間範囲を見つける関数

    Parameters:
    segx, segy, segz (list of tuples): 各軸のセグメントリスト (i, d_min, t_s, t_e)

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
                if current_count >= 3:
                    overlap_ranges.append(current_overlap)
                current_overlap = (start, end)
                current_count = 1

    # 最後の重なり合い範囲を確認して追加
    if current_count >= 3:
        overlap_ranges.append(current_overlap)

    return overlap_ranges


def combine_and_find_overlapping_all_segments(
    segments: list[list[list[tuple[int, float, int, int]]]]
) -> list[list[tuple[int, int]]]:
    """
    各教師データの結果ごとにオーバーラップを検出する関数

    Parameters:
    segments (list of lists of tuples): 各軸のセグメントリスト。各セグメントリストは (l, d_min, t_s, t_e) のタプルを含む。

    Returns:
    list of lists of tuples: 重複している区間のリスト。各区間は (t_s, t_e) のタプルで構成される。
    """
    return [combine_and_find_overlapping_segments(segments[0][i], segments[1][i], segments[2][i]) for i in range(len(segments[0]))]

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
    max_time: float
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

    Returns:
    list of tuples:
        統合され、重なり合った部分がまとめられた最終的なセグメントリスト。各セグメントは (start, end) のタプルで構成されます。
    """

    # 3軸のセグメントを指定された時間範囲でフィルタリング
    filtered_segments = three_axis_filter_segments_by_elapsed_time(segments, Hz, min_time, max_time)

    # フィルタリングされたセグメントから、各教師データごとに重なり合いを検出
    overlapping_segments = combine_and_find_overlapping_all_segments(filtered_segments)

    # 重なり合っているセグメントを、経過時間に基づいてさらにフィルタリング
    filtered_overlaps = filter_overlaps_by_elapsed_time(overlapping_segments, Hz, min_time, max_time)

    # 重なり合っている部分を統合して最終的なセグメントリストを作成
    final_segments = combine_overlapping_segments(filtered_overlaps)

    return final_segments

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
    連続性を保つため、隣接するインターバル間のギャップが閾値以下の場合は統合します。

    Parameters:
    exclusion_intervals (pd.DataFrame): 除外するインターバルのデータフレーム
    start_time (pd.Timestamp): データ全体の開始時刻
    end_time (pd.Timestamp): データ全体の終了時刻
    continuity_threshold (float): インターバル間のギャップの最大許容時間（秒）

    Returns:
    list of tuples: 各利用可能なインターバルの開始・終了時刻のリスト
    """
    available_intervals = []

    if exclusion_intervals.empty:
        available_intervals.append((start_time, end_time))
        return available_intervals

    # Sort exclusion intervals just in case
    exclusion_intervals = exclusion_intervals.sort_values('start').reset_index(drop=True)

    # Handle the first available interval before the first exclusion interval
    first_excl_start = exclusion_intervals.iloc[0]['start']
    if first_excl_start > start_time:
        available_intervals.append((start_time, first_excl_start))

    # Handle gaps between exclusion intervals
    for i in range(len(exclusion_intervals) - 1):
        current_excl_end = exclusion_intervals.iloc[i]['end']
        next_excl_start = exclusion_intervals.iloc[i + 1]['start']
        gap_duration = (next_excl_start - current_excl_end).total_seconds()
        if gap_duration >= continuity_threshold:
            available_intervals.append((current_excl_end, next_excl_start))

    # Handle the last available interval after the last exclusion interval
    last_excl_end = exclusion_intervals.iloc[-1]['end']
    if last_excl_end < end_time:
        available_intervals.append((last_excl_end, end_time))

    return available_intervals

def extract_random_intervals(
    eye_data: pd.DataFrame,
    available_intervals: list,
    min_interval: float,
    max_interval: float,
    num_intervals: int,
    continuity_threshold: float = 0.05  # 50ms
) -> list:
    extracted_intervals = []
    attempts = 0
    max_attempts = num_intervals * 10

    while len(extracted_intervals) < num_intervals and attempts < max_attempts:
        attempts += 1

        suitable_intervals = [
            interval for interval in available_intervals
            if (interval[1] - interval[0]).total_seconds() >= min_interval
        ]

        if not suitable_intervals:
            print("利用可能なインターバルが足りません。")
            break

        avail_start, avail_end = random.choice(suitable_intervals)
        avail_duration = (avail_end - avail_start).total_seconds()

        print(f"Attempt {attempts}: Selected interval ({avail_start}, {avail_end}) with duration {avail_duration} seconds.")

        max_possible_duration = min(avail_duration, max_interval)
        duration = random.uniform(min_interval, max_possible_duration)

        latest_start_time = avail_end - pd.Timedelta(seconds=duration)
        if latest_start_time <= avail_start:
            continue

        random_offset_seconds = random.uniform(0, (latest_start_time - avail_start).total_seconds())
        start = avail_start + pd.Timedelta(seconds=random_offset_seconds)
        end = start + pd.Timedelta(seconds=duration)

        print(f"Extracting interval from {start} to {end} (duration: {duration} seconds).")

        data_in_interval = eye_data[
            (eye_data['Timestamp'] >= start) & (eye_data['Timestamp'] <= end)
        ]

        if not data_in_interval.empty:
            time_diffs = data_in_interval['Timestamp'].diff().dt.total_seconds().dropna()
            if all(time_diffs.between(0.015, 0.025)):  # 50Hzに近い（0.02秒）の許容範囲
                extracted_intervals.append(data_in_interval)
                print(f"Interval {len(extracted_intervals)} extracted successfully.")
            else:
                print(f"Interval {len(extracted_intervals)} has discontinuities. Skipping.")
        else:
            continue

    print(f"抽出されたインターバル数: {len(extracted_intervals)}")
    return extracted_intervals
