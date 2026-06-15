import pandas as pd
import numpy as np


def load_file(uploaded_file):

    if uploaded_file.name.endswith(".csv"):
        # 自动检测分隔符（兼容逗号、制表符等）
        return pd.read_csv(uploaded_file, sep=None, engine="python")

    elif uploaded_file.name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file)

    else:
        raise ValueError("不支持的文件格式")


def parse_long_format(df, value_col, batch_col=None):
    """单列堆叠格式：返回 (单列DataFrame, 子组大小)"""
    if batch_col is not None and batch_col in df.columns:
        subgroup_size = int(df.groupby(batch_col).size().median())
    else:
        subgroup_size = 1
    data = df[value_col].dropna().values
    return pd.DataFrame({value_col: data}), subgroup_size


def parse_wide_format(df, value_cols, subgroup_col=None):
    """多列展开格式：将多个测量列展平为 (单列DataFrame, 子组大小)"""
    subgroup_size = len(value_cols)
    data = df[value_cols].values.flatten()
    data = data[~np.isnan(data)]
    return pd.DataFrame({"Value": data}), subgroup_size


def generate_profile(df):

    profile = pd.DataFrame({
        "列名": df.columns,
        "数据类型": df.dtypes.astype(str),
        "缺失值": df.isna().sum().values,
        "唯一值数": df.nunique().values
    })

    return profile

