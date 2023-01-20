from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.data.labels import STR_LABEL2ID
from src.errors import errors


def get_train_test_img_path_and_label_arr(
    train_dir_root: str,
    test_dir_root: str,
    annt_df: pd.DataFrame,
    str_label2id: Dict[str, int] = STR_LABEL2ID,
) -> Tuple[List[str], List[int], List[str], List[int]]:
    # 学習用画像のパスリストとラベルリストを取得
    train_img_path_arr = []
    train_img_label_arr = []
    for str_label, idx in str_label2id.items():
        for path in (Path(train_dir_root) / str_label).glob("*.jpg"):
            train_img_path_arr.append(str(path))
            train_img_label_arr.append(idx)
    if len(train_img_label_arr) == 0:
        raise errors.DatasetNotFound("学習対象の画像が見つかりません. ディレクトリ配置を再確認してください.")
    train_img_path_arr = np.array(train_img_path_arr)
    train_img_label_arr = to_onehot(train_img_label_arr, len(str_label2id))
    # テスト用画像のパスリストとラベルリストを取得
    test_img_path_arr = []
    test_img_label_arr = []
    for _, row in annt_df.iterrows():
        str_label = row["annotation"]
        idx = str_label2id[str_label]
        path = str(Path(test_dir_root) / Path(row["file_name"]).name)
        test_img_path_arr.append(path)
        test_img_label_arr.append(idx)
    if len(test_img_path_arr) == 0:
        raise errors.DatasetNotFound("テスト対象の画像が見つかりません. ディレクトリ配置を再確認してください.")
    test_img_path_arr = np.array(test_img_path_arr)
    test_img_label_arr = to_onehot(test_img_label_arr, len(str_label2id))
    return (
        train_img_path_arr,
        train_img_label_arr,
        test_img_path_arr,
        test_img_label_arr,
    )


def to_onehot(label_arr: List[int], num_classes: int):
    return np.eye(num_classes)[label_arr]
