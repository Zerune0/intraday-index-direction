"""基于日内区间形态与突破信号的股指方向预测分析。

在每日13:30的信息边界下，以二区、五区和五区突破三个离散信号预测
最终收盘价是否高于当日开盘价，并补充检验13:30至收盘的方向。

Visual Studio Code运行方式：
    1. 在VS Code中打开本仓库文件夹；
    2. 选择已安装requirements.txt依赖的Python解释器；
    3. 打开本文件，点击右上角“运行Python文件”。

终端等价命令：
    python intraday_index_direction_refactored.py

输入：本文件同目录下的三个指数年度分钟 CSV 文件夹。
输出：本文件同目录下的 analysis_outputs_refactored 文件夹。
随机过程固定 random_state=42；软件版本见 requirements.txt。
"""

# ==================================================
# 论文：
# 《基于日内区间形态与突破信号的股指趋势预测研究
# ——基于Logistic回归模型的实证分析》
# ==================================================

from pathlib import Path
from statistics import NormalDist
from typing import Any, Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import rcParams
from scipy.signal import find_peaks
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ==================================================
# 输入数据只读加载，全部分析结果写入独立输出目录。
# ==================================================

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "analysis_outputs_refactored"

DEBUG = False
SHOW_PLOTS = False
START_YEAR = 2015
TRAIN_END_YEAR = 2024
HOLDOUT_YEAR = 2025
END_YEAR = 2025
PREDICTION_THRESHOLD = 0.5
DEFAULT_PROMINENCE_RATIO = 0.20
BOOTSTRAP_RANDOM_STATE = 42
MODEL_FEATURE_COLUMNS = [
    "BREAK_SIGNAL",
    "ZONE2_SIGNAL",
    "ZONE5_SIGNAL",
]
FORWARD_FEATURE_COLUMNS = MODEL_FEATURE_COLUMNS.copy()
TIME_WINDOWS = {
    "morning": ("09:31:00", "11:30:00"),
    "zone2": ("10:01:00", "10:30:00"),
    "zone5": ("13:01:00", "13:30:00"),
}

SHAPE_RULES = {
    ("↑",): ("斜线向上", "涨"),
    ("↓",): ("斜线向下", "跌"),
    ("↓", "↑"): ("V", "涨"),
    ("↑", "↓"): ("倒V", "跌"),
    ("↑", "↓", "↑"): ("N", "涨"),
    ("↓", "↑", "↓"): ("倒N", "跌"),
    ("↓", "↑", "↓", "↑"): ("W", "涨"),
    ("↑", "↓", "↑", "↓"): ("M", "跌"),
}


# ==================================================
# Reusable analysis helpers
# ==================================================


def output_path(filename: str) -> Path:
    """Return an explicit path inside the analysis output directory."""

    return OUTPUT_DIR / filename


def select_time_window(
    minute_df: pd.DataFrame,
    window_name: str,
) -> pd.DataFrame:
    """Return rows inside one named inclusive intraday window."""

    start_time, end_time = TIME_WINDOWS[window_name]
    return minute_df[minute_df["TIME_ONLY"].between(start_time, end_time)]


def identify_shape_and_direction(
    price_list: Sequence[float],
    prominence_ratio: float = DEFAULT_PROMINENCE_RATIO,
) -> tuple[str, str]:
    """
    功能：
    根据某一区间的价格序列，自动识别走势形态，并判断该区间方向。

    参数：
    price_list：某一区间的收盘价列表
    prominence_ratio：拐点显著性阈值，默认使用区间振幅的20%

    返回：
    shape：识别出的形态
    direction：区间方向，分为“涨”“跌”“震荡”
    """

    # 计算该区间的价格振幅
    price_range = max(price_list) - min(price_list)

    # 如果区间没有波动，记为震荡
    if price_range == 0:
        return "震荡", "震荡"

    # 根据区间振幅动态设置寻峰阈值
    prominence_threshold = price_range * prominence_ratio

    # 寻找显著高点
    peaks, _ = find_peaks(price_list, prominence=prominence_threshold)

    # 寻找显著低点
    valleys, _ = find_peaks([-x for x in price_list], prominence=prominence_threshold)

    # 合并高点和低点
    turning_points = []

    for p in peaks:
        turning_points.append({"位置": int(p), "类型": "高点", "价格": price_list[p]})

    for v in valleys:
        turning_points.append({"位置": int(v), "类型": "低点", "价格": price_list[v]})

    # 按时间顺序排序
    turning_points = sorted(turning_points, key=lambda x: x["位置"])

    # 加入区间起点和终点
    full_points = []

    full_points.append({"位置": 0, "类型": "起点", "价格": price_list[0]})

    full_points.extend(turning_points)

    full_points.append(
        {"位置": len(price_list) - 1, "类型": "终点", "价格": price_list[-1]}
    )

    # 根据相邻关键点价格变化，生成方向序列
    direction_sequence = []

    for i in range(len(full_points) - 1):

        p1 = full_points[i]["价格"]
        p2 = full_points[i + 1]["价格"]

        if p2 > p1:
            direction_sequence.append("↑")
        else:
            direction_sequence.append("↓")

    matched_shape = SHAPE_RULES.get(tuple(direction_sequence))
    if matched_shape is not None:
        return matched_shape

    # 如果形态过于复杂，则根据整体方向归类为复杂上涨或复杂下跌。
    if price_list[-1] > price_list[0]:
        return "复杂上涨", "涨"
    if price_list[-1] < price_list[0]:
        return "复杂下跌", "跌"
    return "震荡", "震荡"


def extract_daily_components(
    minute_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """一次逐日遍历生成主流程需要的三张日级表。"""

    feature_records = []
    break_records = []
    real_records = []

    for (index_name, trade_date), day_data in minute_df.groupby(
        ["INDEX_NAME", "TRADE_DATE"]
    ):
        zone2_data = select_time_window(day_data, "zone2")
        zone5_data = select_time_window(day_data, "zone5")

        # 保留原流程的样本纳入规则：只有二区和五区都完整时才生成形态记录。
        if len(zone2_data) == 30 and len(zone5_data) == 30:
            zone2_shape, zone2_direction = identify_shape_and_direction(
                zone2_data["CLOSE_INDEX"].tolist()
            )
            zone5_shape, zone5_direction = identify_shape_and_direction(
                zone5_data["CLOSE_INDEX"].tolist()
            )

            feature_records.append(
                {
                    "INDEX_NAME": index_name,
                    "TRADE_DATE": str(trade_date),
                    "ZONE2_SHAPE": zone2_shape,
                    "ZONE2_DIRECTION": zone2_direction,
                    "ZONE5_SHAPE": zone5_shape,
                    "ZONE5_DIRECTION": zone5_direction,
                }
            )

        # 保留原流程的突破样本规则：这里只要求五区完整。
        if len(zone5_data) == 30:
            zone1_to_4_data = select_time_window(day_data, "morning")
            zone1_to_4_high = zone1_to_4_data["HIGH_INDEX"].max()
            zone1_to_4_low = zone1_to_4_data["LOW_INDEX"].min()
            zone5_high = zone5_data["HIGH_INDEX"].max()
            zone5_low = zone5_data["LOW_INDEX"].min()

            break_records.append(
                {
                    "INDEX_NAME": index_name,
                    "TRADE_DATE": str(trade_date),
                    "ZONE5_BREAK_HIGH": zone5_high > zone1_to_4_high,
                    "ZONE5_BREAK_LOW": zone5_low < zone1_to_4_low,
                }
            )

        # 保留原流程的13:30样本规则与按时间排序后的开收盘口径。
        sorted_day_data = day_data.sort_values("TDATE_TIME")
        midday_rows = sorted_day_data[
            sorted_day_data["TIME_ONLY"] == TIME_WINDOWS["zone5"][1]
        ]

        if len(midday_rows) == 0:
            continue

        day_open = sorted_day_data.iloc[0]["OPEN_INDEX"]
        day_close = sorted_day_data.iloc[-1]["CLOSE_INDEX"]
        close_1330 = midday_rows.iloc[0]["CLOSE_INDEX"]

        if day_close > day_open:
            real_direction = "涨"
        elif day_close < day_open:
            real_direction = "跌"
        else:
            real_direction = "平"

        if close_1330 > day_open:
            midday_position = "涨"
        elif close_1330 < day_open:
            midday_position = "跌"
        else:
            midday_position = "平"

        real_records.append(
            {
                "INDEX_NAME": index_name,
                "TRADE_DATE": str(trade_date),
                "DAY_OPEN": day_open,
                "DAY_CLOSE": day_close,
                "CLOSE_1330": close_1330,
                "OPEN_TO_1330_RETURN": close_1330 / day_open - 1,
                "MIDDAY_POSITION": midday_position,
                "REAL_DIRECTION": real_direction,
            }
        )

    return (
        pd.DataFrame(feature_records),
        pd.DataFrame(break_records),
        pd.DataFrame(real_records),
    )


def build_sensitivity_daily_cache(
    minute_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """缓存prominence敏感性检验中不随参数变化的日级输入。"""

    cached_days = []

    for (index_name, trade_date), day_data in minute_df.groupby(
        ["INDEX_NAME", "TRADE_DATE"]
    ):
        zone2_data = select_time_window(day_data, "zone2")
        zone5_data = select_time_window(day_data, "zone5")

        if len(zone2_data) != 30 or len(zone5_data) != 30:
            continue

        zone1_to_4_data = select_time_window(day_data, "morning")

        day_open = day_data.iloc[0]["OPEN_INDEX"]
        day_close = day_data.iloc[-1]["CLOSE_INDEX"]

        if day_close > day_open:
            real_direction = "涨"
        elif day_close < day_open:
            real_direction = "跌"
        else:
            real_direction = "平"

        cached_days.append(
            {
                "INDEX_NAME": index_name,
                "TRADE_DATE": str(trade_date),
                "ZONE2_PRICES": zone2_data["CLOSE_INDEX"].tolist(),
                "ZONE5_PRICES": zone5_data["CLOSE_INDEX"].tolist(),
                "ZONE5_BREAK_HIGH": (
                    zone5_data["HIGH_INDEX"].max() > zone1_to_4_data["HIGH_INDEX"].max()
                ),
                "ZONE5_BREAK_LOW": (
                    zone5_data["LOW_INDEX"].min() < zone1_to_4_data["LOW_INDEX"].min()
                ),
                "REAL_DIRECTION": real_direction,
            }
        )

    return cached_days


def predict_direction_series(feature_df: pd.DataFrame) -> np.ndarray:
    """按原规则一次性生成方向；条件顺序保留突破创新高的优先级。"""

    zone2_up = feature_df["ZONE2_DIRECTION"].eq("涨")
    zone2_down = feature_df["ZONE2_DIRECTION"].eq("跌")
    zone5_up = feature_df["ZONE5_DIRECTION"].eq("涨")
    zone5_down = feature_df["ZONE5_DIRECTION"].eq("跌")

    return np.select(
        [
            zone2_up & zone5_up,
            zone2_down & zone5_down,
            zone2_down & zone5_up & feature_df["ZONE5_BREAK_HIGH"],
            zone2_down & zone5_up,
            zone2_up & zone5_down & feature_df["ZONE5_BREAK_LOW"],
            zone2_up & zone5_down,
        ],
        ["涨", "跌", "涨", "跌", "跌", "涨"],
        default="无法判断",
    )


def break_signal_series(feature_df: pd.DataFrame) -> np.ndarray:
    """按原规则生成突破信号；同时创新高和创新低时仍优先记为创新高。"""

    return np.select(
        [feature_df["ZONE5_BREAK_HIGH"], feature_df["ZONE5_BREAK_LOW"]],
        [1, -1],
        default=0,
    )


def break_type_series(feature_df: pd.DataFrame) -> np.ndarray:
    """按原规则生成突破类型。"""

    return np.select(
        [feature_df["ZONE5_BREAK_HIGH"], feature_df["ZONE5_BREAK_LOW"]],
        ["创新高", "创新低"],
        default="无突破",
    )


def build_logistic_model() -> LogisticRegression:
    """使用论文设定构造Logistic模型。"""

    return LogisticRegression(
        penalty="l2",
        C=1.0,
        fit_intercept=True,
        class_weight=None,
        solver="lbfgs",
        max_iter=100,
    )


def fit_predict_logistic(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    probability_column: str,
    prediction_column: str,
    cast_target_to_int: bool = False,
) -> LogisticRegression:
    """按统一配置训练Logistic模型并写回概率与二元预测。"""

    fitted_model = build_logistic_model()
    training_target = train_data[target_column]
    if cast_target_to_int:
        training_target = training_target.astype(int)

    fitted_model.fit(
        train_data[feature_columns],
        training_target,
    )
    test_data[probability_column] = fitted_model.predict_proba(
        test_data[feature_columns]
    )[:, 1]
    test_data[prediction_column] = (
        test_data[probability_column] >= PREDICTION_THRESHOLD
    ).astype(int)

    return fitted_model


def fit_statsmodels_logit(
    train_data: pd.DataFrame,
    target_data: pd.Series,
    feature_columns: Sequence[str],
    model_name: str,
) -> dict[str, Any]:

    X_temp = train_data[list(feature_columns)].astype(float)

    X_temp_const = sm.add_constant(X_temp, has_constant="add")

    fitted_model = sm.Logit(target_data, X_temp_const).fit(disp=False)

    return {
        "Model": model_name,
        "Predictors": ", ".join(feature_columns),
        "Log_Likelihood": fitted_model.llf,
        "Pseudo_R2_McFadden": fitted_model.prsquared,
        "AIC": fitted_model.aic,
        "BIC": fitted_model.bic,
        "N": int(fitted_model.nobs),
    }


def wilson_ci_from_correct(
    correct_flags: Sequence[bool],
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")

    correct_flags = np.asarray(correct_flags).astype(int)
    n = len(correct_flags)
    estimate = correct_flags.mean()
    z = 1.959963984540054 if alpha == 0.05 else NormalDist().inv_cdf(1 - alpha / 2)
    denominator = 1 + z**2 / n
    center = (estimate + z**2 / (2 * n)) / denominator
    margin = (
        z * np.sqrt(estimate * (1 - estimate) / n + z**2 / (4 * n**2)) / denominator
    )
    return estimate, center - margin, center + margin


def bootstrap_auc_ci(
    y_true: Any,
    y_score: Any,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    random_state: int = BOOTSTRAP_RANDOM_STATE,
) -> tuple[float, float, float]:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    rng = np.random.default_rng(random_state)
    auc_values = []
    n = len(y_true)

    for _ in range(n_bootstrap):
        sample_idx = rng.integers(0, n, n)
        sample_y = y_true[sample_idx]

        if len(np.unique(sample_y)) < 2:
            continue

        sample_score = y_score[sample_idx]
        auc_values.append(roc_auc_score(sample_y, sample_score))

    auc_values = np.asarray(auc_values)
    estimate = roc_auc_score(y_true, y_score)
    lower = np.percentile(auc_values, 100 * alpha / 2)
    upper = np.percentile(auc_values, 100 * (1 - alpha / 2))
    return estimate, lower, upper


def _build_forward_target(data, target_map):
    """在不改变原TARGET的前提下构造13:30至收盘方向。"""

    result = data.copy()

    result["FORWARD_RETURN"] = result["DAY_CLOSE"] / result["CLOSE_1330"] - 1

    result["FORWARD_DIRECTION"] = np.where(
        result["DAY_CLOSE"] > result["CLOSE_1330"],
        "涨",
        np.where(
            result["DAY_CLOSE"] < result["CLOSE_1330"],
            "跌",
            "平",
        ),
    )

    result["FORWARD_TARGET"] = result["FORWARD_DIRECTION"].map(target_map)

    result["FORWARD_RULE_TARGET"] = result["PRED_DIRECTION"].map(target_map)

    result["MOMENTUM_TARGET"] = result["MIDDAY_POSITION"].map(target_map)

    return result


def _fit_statsmodels_forward_model(
    train_df,
    target,
    feature_columns,
    model_name,
):
    """估计前瞻Logit并返回模型比较指标。"""

    X_temp = sm.add_constant(
        train_df[feature_columns].astype(float),
        has_constant="add",
    )

    fitted_model = sm.Logit(
        target,
        X_temp,
    ).fit(
        disp=False,
    )

    return fitted_model, {
        "Model": model_name,
        "Predictors": ", ".join(feature_columns),
        "Log_Likelihood": fitted_model.llf,
        "Pseudo_R2_McFadden": fitted_model.prsquared,
        "AIC": fitted_model.aic,
        "BIC": fitted_model.bic,
        "N": int(fitted_model.nobs),
    }


def _append_accuracy_ci(
    output_rows,
    model_name,
    actual,
    predicted,
    wilson_ci_from_correct,
):
    """按原代码口径计算Wilson准确率置信区间。"""

    correct_flags = np.asarray(actual).astype(int) == np.asarray(predicted).astype(int)

    estimate, lower, upper = wilson_ci_from_correct(correct_flags)

    output_rows.append(
        {
            "Model": model_name,
            "Metric": "Accuracy",
            "Estimate": estimate,
            "CI_Lower": lower,
            "CI_Upper": upper,
            "Method": "Wilson",
        }
    )


def _build_szse_forward_state(sz_raw_df):
    """从深证成指分钟数据独立构造13:30状态和前瞻标签原始量。"""

    state_rows = []

    for (
        index_name,
        trade_date,
    ), day_data in sz_raw_df.groupby(
        [
            "INDEX_NAME",
            "TRADE_DATE",
        ]
    ):

        day_data = day_data.sort_values("TDATE_TIME")

        midday_rows = day_data[day_data["TIME_ONLY"] == TIME_WINDOWS["zone5"][1]]

        if len(midday_rows) == 0:
            continue

        day_open = day_data.iloc[0]["OPEN_INDEX"]
        close_1330 = midday_rows.iloc[0]["CLOSE_INDEX"]

        if close_1330 > day_open:
            midday_position = "涨"
        elif close_1330 < day_open:
            midday_position = "跌"
        else:
            midday_position = "平"

        state_rows.append(
            {
                "INDEX_NAME": index_name,
                "TRADE_DATE": str(trade_date),
                "CLOSE_1330": close_1330,
                "OPEN_TO_1330_RETURN": (close_1330 / day_open - 1),
                "MIDDAY_POSITION": midday_position,
            }
        )

    return pd.DataFrame(state_rows)


def _save_forward_plots(
    forward_output_dir,
    y_test,
    forward_test_df,
    rolling_results_df,
    logit_auc,
    momentum_auc,
    show_plots,
):
    """保存前瞻目标的ROC与滚动验证图。"""

    logit_fpr, logit_tpr, _ = roc_curve(
        y_test,
        forward_test_df["FORWARD_UP_PROB"],
    )
    momentum_fpr, momentum_tpr, _ = roc_curve(
        y_test,
        forward_test_df["OPEN_TO_1330_RETURN"],
    )

    plt.figure(figsize=(8.4, 6.2))
    plt.plot(
        logit_fpr,
        logit_tpr,
        linewidth=2.2,
        label=f"Three-signal logit (AUC = {logit_auc:.3f})",
    )
    plt.plot(
        momentum_fpr,
        momentum_tpr,
        linewidth=2.2,
        label=f"Pre-13:30 momentum (AUC = {momentum_auc:.3f})",
    )
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        linewidth=1.3,
    )
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Post-13:30 Direction ROC Curves, 2025 Holdout")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        forward_output_dir / "Forward_Target_ROC_2025.png",
        dpi=450,
        bbox_inches="tight",
    )

    if show_plots:
        plt.show()
    else:
        plt.close()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.5, 5.2),
    )

    accuracy_series = [
        ("Rule", "Rule_Accuracy"),
        ("Logit", "Logit_Accuracy"),
        ("Pre-13:30 momentum", "Momentum_Accuracy"),
    ]
    for label, column in accuracy_series:
        axes[0].plot(
            rolling_results_df["Test_Year"],
            rolling_results_df[column],
            marker="o",
            label=label,
        )

    axes[0].set_title("Post-13:30 Accuracy")
    axes[0].set_xlabel("Test Year")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    auc_series = [
        ("Logit", "Logit_AUC"),
        ("Pre-13:30 momentum", "Momentum_AUC"),
    ]
    for label, column in auc_series:
        axes[1].plot(
            rolling_results_df["Test_Year"],
            rolling_results_df[column],
            marker="o",
            label=label,
        )

    axes[1].axhline(
        0.5,
        linestyle="--",
        color="gray",
        linewidth=1.2,
    )
    axes[1].set_title("Post-13:30 AUC")
    axes[1].set_xlabel("Test Year")
    axes[1].set_ylabel("AUC")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(
        forward_output_dir / "Forward_Target_Rolling_Performance.png",
        dpi=450,
        bbox_inches="tight",
    )

    if show_plots:
        plt.show()
    else:
        plt.close(fig)


def _write_forward_summary(
    forward_output_dir,
    forward_train_df,
    forward_test_df,
    sz_yearly_results_df,
    logit_accuracy,
    logit_auc,
    logit_auc_ci,
    rule_accuracy,
    rule_auc,
    momentum_accuracy,
    momentum_auc,
):
    """保存并返回前瞻检验的文本摘要。"""

    sz_2025_row = sz_yearly_results_df[
        sz_yearly_results_df["Year"] == HOLDOUT_YEAR
    ].iloc[0]

    summary_lines = [
        "Supplementary Forward-Target Summary",
        "",
        f"Training binary target N: {len(forward_train_df)}",
        f"2025 holdout binary target N: {len(forward_test_df)}",
        ("2025 forward up rate: " f"{forward_test_df['FORWARD_TARGET'].mean():.6f}"),
        f"2025 three-signal logit accuracy: {logit_accuracy:.6f}",
        f"2025 three-signal logit AUC: {logit_auc:.6f}",
        (
            "2025 three-signal logit AUC 95% CI: "
            f"[{logit_auc_ci[1]:.6f}, {logit_auc_ci[2]:.6f}]"
        ),
        f"2025 deterministic rule accuracy: {rule_accuracy:.6f}",
        f"2025 deterministic rule binary-score AUC: {rule_auc:.6f}",
        f"2025 pre-13:30 momentum accuracy: {momentum_accuracy:.6f}",
        f"2025 pre-13:30 momentum AUC: {momentum_auc:.6f}",
        ("SZSE 2025 forward logit accuracy: " f"{sz_2025_row['Logit_Accuracy']:.6f}"),
        ("SZSE 2025 forward logit AUC: " f"{sz_2025_row['Logit_AUC']:.6f}"),
        "",
        "All leakage-audit checks passed.",
    ]

    with open(
        forward_output_dir / "forward_summary.txt",
        "w",
        encoding="utf-8",
    ) as summary_file:
        summary_file.write("\n".join(summary_lines))

    return summary_lines


def _run_forward_rolling_validation(
    forward_df: pd.DataFrame,
    forward_output_dir: Path,
) -> pd.DataFrame:
    """Run expanding-window validation for the post-13:30 target."""

    rolling_rows = []

    for test_year in range(2020, END_YEAR + 1):
        rolling_train_df = (
            forward_df[forward_df["TRADE_DATE_DT"].dt.year < test_year]
            .dropna(subset=FORWARD_FEATURE_COLUMNS + ["FORWARD_TARGET"])
            .copy()
        )
        rolling_test_df = (
            forward_df[forward_df["TRADE_DATE_DT"].dt.year == test_year]
            .dropna(subset=FORWARD_FEATURE_COLUMNS + ["FORWARD_TARGET"])
            .copy()
        )

        if rolling_train_df.empty or rolling_test_df.empty:
            continue

        fit_predict_logistic(
            rolling_train_df,
            rolling_test_df,
            FORWARD_FEATURE_COLUMNS,
            "FORWARD_TARGET",
            "FORWARD_UP_PROB",
            "FORWARD_PRED_TARGET",
            cast_target_to_int=True,
        )

        rolling_rule_df = rolling_test_df.dropna(subset=["FORWARD_RULE_TARGET"])
        rolling_momentum_df = rolling_test_df.dropna(subset=["MOMENTUM_TARGET"])
        rolling_majority_target = int(
            rolling_train_df["FORWARD_TARGET"].mean() >= PREDICTION_THRESHOLD
        )

        rolling_rows.append(
            {
                "Test_Year": test_year,
                "Training_Period": f"{START_YEAR}-{test_year - 1}",
                "N": len(rolling_test_df),
                "Up_Rate": rolling_test_df["FORWARD_TARGET"].mean(),
                "Majority_Accuracy": (
                    rolling_test_df["FORWARD_TARGET"].astype(int)
                    == rolling_majority_target
                ).mean(),
                "Rule_Accuracy": accuracy_score(
                    rolling_rule_df["FORWARD_TARGET"].astype(int),
                    rolling_rule_df["FORWARD_RULE_TARGET"].astype(int),
                ),
                "Logit_Accuracy": accuracy_score(
                    rolling_test_df["FORWARD_TARGET"].astype(int),
                    rolling_test_df["FORWARD_PRED_TARGET"],
                ),
                "Logit_AUC": roc_auc_score(
                    rolling_test_df["FORWARD_TARGET"].astype(int),
                    rolling_test_df["FORWARD_UP_PROB"],
                ),
                "Momentum_Accuracy": accuracy_score(
                    rolling_momentum_df["FORWARD_TARGET"].astype(int),
                    rolling_momentum_df["MOMENTUM_TARGET"].astype(int),
                ),
                "Momentum_AUC": roc_auc_score(
                    rolling_test_df["FORWARD_TARGET"].astype(int),
                    rolling_test_df["OPEN_TO_1330_RETURN"],
                ),
            }
        )

    rolling_results_df = pd.DataFrame(rolling_rows)
    rolling_results_df.to_csv(
        forward_output_dir / "forward_rolling_out_of_sample_results.csv",
        index=False,
    )
    return rolling_results_df


def _evaluate_szse_forward(
    sz_raw_df: pd.DataFrame,
    sz_feature_df: pd.DataFrame,
    target_map: dict[str, int],
    forward_model: LogisticRegression,
    forward_output_dir: Path,
) -> pd.DataFrame:
    """Evaluate the forward target on the external SZSE sample."""

    sz_forward_state_df = _build_szse_forward_state(sz_raw_df)
    sz_forward_df = sz_feature_df.merge(
        sz_forward_state_df,
        on=["INDEX_NAME", "TRADE_DATE"],
        how="left",
    )
    sz_forward_df["FORWARD_DIRECTION"] = np.where(
        sz_forward_df["DAY_CLOSE"] > sz_forward_df["CLOSE_1330"],
        "涨",
        np.where(
            sz_forward_df["DAY_CLOSE"] < sz_forward_df["CLOSE_1330"],
            "跌",
            "平",
        ),
    )
    sz_forward_df["FORWARD_RETURN"] = (
        sz_forward_df["DAY_CLOSE"] / sz_forward_df["CLOSE_1330"] - 1
    )
    sz_forward_df["FORWARD_TARGET"] = sz_forward_df["FORWARD_DIRECTION"].map(target_map)
    sz_forward_df["MOMENTUM_TARGET"] = sz_forward_df["MIDDAY_POSITION"].map(target_map)
    sz_forward_df["PRED_DIRECTION"] = predict_direction_series(sz_forward_df)
    sz_forward_df["FORWARD_RULE_TARGET"] = sz_forward_df["PRED_DIRECTION"].map(
        target_map
    )

    sz_model_df = sz_forward_df.dropna(
        subset=FORWARD_FEATURE_COLUMNS + ["FORWARD_TARGET"]
    ).copy()
    sz_model_df["FORWARD_UP_PROB"] = forward_model.predict_proba(
        sz_model_df[FORWARD_FEATURE_COLUMNS]
    )[:, 1]
    sz_model_df["FORWARD_PRED_TARGET"] = (
        sz_model_df["FORWARD_UP_PROB"] >= PREDICTION_THRESHOLD
    ).astype(int)

    yearly_rows = []
    for year in range(START_YEAR, END_YEAR + 1):
        year_df = sz_model_df[sz_model_df["TRADE_DATE_DT"].dt.year == year].copy()
        year_rule_df = year_df.dropna(subset=["FORWARD_RULE_TARGET"])
        year_momentum_df = year_df.dropna(subset=["MOMENTUM_TARGET"])

        yearly_rows.append(
            {
                "Year": year,
                "N": len(year_df),
                "Up_Rate": year_df["FORWARD_TARGET"].mean(),
                "Rule_Accuracy": accuracy_score(
                    year_rule_df["FORWARD_TARGET"].astype(int),
                    year_rule_df["FORWARD_RULE_TARGET"].astype(int),
                ),
                "Logit_Accuracy": accuracy_score(
                    year_df["FORWARD_TARGET"].astype(int),
                    year_df["FORWARD_PRED_TARGET"],
                ),
                "Logit_AUC": roc_auc_score(
                    year_df["FORWARD_TARGET"].astype(int),
                    year_df["FORWARD_UP_PROB"],
                ),
                "Momentum_Accuracy": accuracy_score(
                    year_momentum_df["FORWARD_TARGET"].astype(int),
                    year_momentum_df["MOMENTUM_TARGET"].astype(int),
                ),
                "Momentum_AUC": roc_auc_score(
                    year_df["FORWARD_TARGET"].astype(int),
                    year_df["OPEN_TO_1330_RETURN"],
                ),
            }
        )

    yearly_results_df = pd.DataFrame(yearly_rows)
    yearly_results_df.to_csv(
        forward_output_dir / "forward_szse_yearly_results.csv",
        index=False,
    )
    return yearly_results_df


def _prepare_forward_samples(
    feature_df: pd.DataFrame,
    raw_primary_df: pd.DataFrame,
    target_map: dict[str, int],
    forward_output_dir: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Build forward samples and verify their time and label boundaries."""

    main_forward_df = _build_forward_target(feature_df, target_map)
    main_forward_df["SAMPLE_GROUP"] = np.where(
        main_forward_df["TRADE_DATE_DT"].dt.year <= TRAIN_END_YEAR,
        "Training 2015-2024",
        "Holdout 2025",
    )

    sample_rows = []
    for sample_group, sample_df in main_forward_df.groupby("SAMPLE_GROUP"):
        sample_rows.append(
            {
                "Sample": sample_group,
                "Total_Index_Days": len(sample_df),
                "Forward_Up": (sample_df["FORWARD_DIRECTION"] == "涨").sum(),
                "Forward_Down": (sample_df["FORWARD_DIRECTION"] == "跌").sum(),
                "Forward_Flat": (sample_df["FORWARD_DIRECTION"] == "平").sum(),
                "Binary_Target_N": sample_df["FORWARD_TARGET"].notna().sum(),
                "Up_Rate": sample_df["FORWARD_TARGET"].mean(),
            }
        )

    sample_audit_df = pd.DataFrame(sample_rows)
    sample_audit_df.to_csv(
        forward_output_dir / "forward_sample_audit.csv",
        index=False,
    )

    print("\n第43步完成：13:30至收盘前瞻目标构造完成。")
    print(sample_audit_df)

    raw_close_1330_df = raw_primary_df[
        raw_primary_df["TIME_ONLY"] == TIME_WINDOWS["zone5"][1]
    ][["INDEX_NAME", "TRADE_DATE", "CLOSE_INDEX"]].rename(
        columns={"CLOSE_INDEX": "RAW_CLOSE_1330"}
    )
    raw_close_1330_df["TRADE_DATE"] = raw_close_1330_df["TRADE_DATE"].astype(str)

    audit_merge_df = main_forward_df.merge(
        raw_close_1330_df,
        on=["INDEX_NAME", "TRADE_DATE"],
        how="left",
    )
    close_1330_match = np.allclose(
        audit_merge_df["CLOSE_1330"],
        audit_merge_df["RAW_CLOSE_1330"],
        equal_nan=False,
    )
    forward_return_match = np.allclose(
        audit_merge_df["FORWARD_RETURN"],
        audit_merge_df["DAY_CLOSE"] / audit_merge_df["CLOSE_1330"] - 1,
        equal_nan=False,
    )
    forward_label_match = (
        (
            (audit_merge_df["FORWARD_TARGET"] == 1)
            == (audit_merge_df["DAY_CLOSE"] > audit_merge_df["CLOSE_1330"])
        )
        | (
            audit_merge_df["FORWARD_TARGET"].isna()
            & (audit_merge_df["DAY_CLOSE"] == audit_merge_df["CLOSE_1330"])
        )
    ).all()

    train_raw_df = main_forward_df[
        main_forward_df["TRADE_DATE_DT"].dt.year <= TRAIN_END_YEAR
    ].copy()
    test_raw_df = main_forward_df[
        main_forward_df["TRADE_DATE_DT"].dt.year == HOLDOUT_YEAR
    ].copy()
    date_boundary_match = (
        train_raw_df["TRADE_DATE_DT"].dt.year.max() <= TRAIN_END_YEAR
        and test_raw_df["TRADE_DATE_DT"].dt.year.min() == HOLDOUT_YEAR
    )
    predictor_boundary_match = FORWARD_FEATURE_COLUMNS == [
        "BREAK_SIGNAL",
        "ZONE2_SIGNAL",
        "ZONE5_SIGNAL",
    ]

    leakage_audit_df = pd.DataFrame(
        [
            {
                "Check": "13:30 close matches raw 13:30 record",
                "Passed": close_1330_match,
            },
            {
                "Check": "Forward return identity",
                "Passed": forward_return_match,
            },
            {
                "Check": "Forward binary label identity",
                "Passed": forward_label_match,
            },
            {
                "Check": "Training and 2025 date boundary",
                "Passed": date_boundary_match,
            },
            {
                "Check": "Predictors restricted to three pre-cutoff signals",
                "Passed": predictor_boundary_match,
            },
        ]
    )
    leakage_audit_df.to_csv(
        forward_output_dir / "forward_leakage_audit.csv",
        index=False,
    )
    if not leakage_audit_df["Passed"].all():
        raise AssertionError("前瞻目标信息边界检查未全部通过。")

    print("\n第44步完成：前瞻目标信息边界检查全部通过。")
    print(leakage_audit_df)

    return (
        main_forward_df,
        train_raw_df,
        test_raw_df,
        sample_audit_df,
        leakage_audit_df,
    )


def run_forward_extension(
    feature_df: pd.DataFrame,
    raw_primary_df: pd.DataFrame,
    sz_raw_df: pd.DataFrame,
    sz_feature_df: pd.DataFrame,
    target_map: dict[str, int],
    wilson_ci_from_correct: Callable[
        [Sequence[bool]],
        tuple[float, float, float],
    ],
    bootstrap_auc_ci: Callable[
        [Any, Any],
        tuple[float, float, float],
    ],
    output_dir: str | Path,
    show_plots: bool = False,
) -> dict[str, pd.DataFrame]:
    """运行13:30至收盘方向的补充检验。"""

    forward_output_dir = Path(output_dir) / "suggestion4"

    forward_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        main_forward_df,
        train_raw_df,
        test_raw_df,
        sample_audit_df,
        leakage_audit_df,
    ) = _prepare_forward_samples(
        feature_df,
        raw_primary_df,
        target_map,
        forward_output_dir,
    )

    # ==================================================
    # 第45步：训练前瞻Logistic模型
    # ==================================================

    forward_train_df = train_raw_df.dropna(
        subset=(FORWARD_FEATURE_COLUMNS + ["FORWARD_TARGET"])
    ).copy()

    forward_test_df = test_raw_df.dropna(
        subset=(FORWARD_FEATURE_COLUMNS + ["FORWARD_TARGET"])
    ).copy()

    X_train = forward_train_df[FORWARD_FEATURE_COLUMNS]

    y_train = forward_train_df["FORWARD_TARGET"].astype(int)

    X_test = forward_test_df[FORWARD_FEATURE_COLUMNS]

    y_test = forward_test_df["FORWARD_TARGET"].astype(int)

    forward_model = build_logistic_model()

    forward_model.fit(
        X_train,
        y_train,
    )

    forward_test_df["FORWARD_UP_PROB"] = forward_model.predict_proba(X_test)[:, 1]

    forward_test_df["FORWARD_PRED_TARGET"] = (
        forward_test_df["FORWARD_UP_PROB"] >= PREDICTION_THRESHOLD
    ).astype(int)

    sklearn_coef_df = pd.DataFrame(
        {
            "Variable": FORWARD_FEATURE_COLUMNS,
            "Coefficient": forward_model.coef_[0],
            "Odds_Ratio": np.exp(forward_model.coef_[0]),
        }
    )

    sklearn_coef_df = pd.concat(
        [
            pd.DataFrame(
                [
                    {
                        "Variable": "Intercept",
                        "Coefficient": forward_model.intercept_[0],
                        "Odds_Ratio": np.exp(forward_model.intercept_[0]),
                    }
                ]
            ),
            sklearn_coef_df,
        ],
        ignore_index=True,
    )

    sklearn_coef_df.to_csv(
        forward_output_dir / "forward_sklearn_coefficients.csv",
        index=False,
    )

    print("\n第45步完成：前瞻Logistic模型训练完成。")
    print(sklearn_coef_df)

    # ==================================================
    # 第45.1步：前瞻模型统计推断与模型比较
    # ==================================================

    full_statsmodels, full_model_metrics = _fit_statsmodels_forward_model(
        forward_train_df,
        y_train,
        FORWARD_FEATURE_COLUMNS,
        "Forward full: BREAK + ZONE2 + ZONE5",
    )

    _, break_model_metrics = _fit_statsmodels_forward_model(
        forward_train_df,
        y_train,
        ["BREAK_SIGNAL"],
        "Forward baseline: BREAK only",
    )

    statsmodels_coef_df = pd.DataFrame(
        {
            "Variable": full_statsmodels.params.index,
            "Coefficient": full_statsmodels.params.values,
            "Standard_Error": full_statsmodels.bse.values,
            "z_statistic": full_statsmodels.tvalues.values,
            "p_value": full_statsmodels.pvalues.values,
        }
    )

    statsmodels_coef_df["Odds_Ratio"] = np.exp(statsmodels_coef_df["Coefficient"])

    statsmodels_coef_df.to_csv(
        forward_output_dir / "forward_statsmodels_coefficients.csv",
        index=False,
    )

    model_comparison_df = pd.DataFrame(
        [
            break_model_metrics,
            full_model_metrics,
        ]
    )

    model_comparison_df.to_csv(
        forward_output_dir / "forward_model_comparison.csv",
        index=False,
    )

    print("\n第45.1步完成：前瞻模型统计推断完成。")
    print(statsmodels_coef_df)
    print(model_comparison_df)

    # ==================================================
    # 第46步：2025前瞻留出检验与基准比较
    # ==================================================

    rule_eval_df = forward_test_df.dropna(subset=["FORWARD_RULE_TARGET"]).copy()

    momentum_eval_df = forward_test_df.dropna(subset=["MOMENTUM_TARGET"]).copy()

    majority_target = int(y_train.mean() >= PREDICTION_THRESHOLD)

    forward_test_df["MAJORITY_TARGET"] = majority_target

    logit_accuracy = accuracy_score(
        y_test,
        forward_test_df["FORWARD_PRED_TARGET"],
    )

    logit_auc = roc_auc_score(
        y_test,
        forward_test_df["FORWARD_UP_PROB"],
    )

    rule_accuracy = accuracy_score(
        rule_eval_df["FORWARD_TARGET"].astype(int),
        rule_eval_df["FORWARD_RULE_TARGET"].astype(int),
    )

    momentum_accuracy = accuracy_score(
        momentum_eval_df["FORWARD_TARGET"].astype(int),
        momentum_eval_df["MOMENTUM_TARGET"].astype(int),
    )

    momentum_auc = roc_auc_score(
        y_test,
        forward_test_df["OPEN_TO_1330_RETURN"],
    )

    majority_accuracy = accuracy_score(
        y_test,
        forward_test_df["MAJORITY_TARGET"],
    )

    # 二元类别输出仅作为粗粒度排序分数计算AUC。
    # 这两个AUC只用于与论文表格对齐，不应与连续概率分数等同解释。
    majority_auc = roc_auc_score(
        y_test,
        forward_test_df["MAJORITY_TARGET"],
    )

    rule_auc = roc_auc_score(
        rule_eval_df["FORWARD_TARGET"].astype(int),
        rule_eval_df["FORWARD_RULE_TARGET"].astype(int),
    )

    results_2025_df = pd.DataFrame(
        [
            {
                "Model": "Majority-class baseline",
                "N": len(forward_test_df),
                "Accuracy": majority_accuracy,
                "AUC": majority_auc,
            },
            {
                "Model": "Deterministic rule",
                "N": len(rule_eval_df),
                "Accuracy": rule_accuracy,
                "AUC": rule_auc,
            },
            {
                "Model": "Pre-13:30 momentum-continuation baseline",
                "N": len(momentum_eval_df),
                "Accuracy": momentum_accuracy,
                "AUC": momentum_auc,
            },
            {
                "Model": "Three-signal LogisticRegression",
                "N": len(forward_test_df),
                "Accuracy": logit_accuracy,
                "AUC": logit_auc,
            },
        ]
    )

    results_2025_df.to_csv(
        forward_output_dir / "forward_2025_results.csv",
        index=False,
    )

    detail_columns = [
        "INDEX_NAME",
        "TRADE_DATE",
        "DAY_OPEN",
        "CLOSE_1330",
        "DAY_CLOSE",
        "FORWARD_RETURN",
        "FORWARD_TARGET",
        "FORWARD_UP_PROB",
        "FORWARD_PRED_TARGET",
        "FORWARD_RULE_TARGET",
        "MIDDAY_POSITION",
        "OPEN_TO_1330_RETURN",
        "MOMENTUM_TARGET",
    ]

    forward_test_df[detail_columns].to_csv(
        forward_output_dir / "forward_2025_prediction_details.csv",
        index=False,
    )

    by_index_rows = []

    for index_name, index_df in forward_test_df.groupby("INDEX_NAME"):

        by_index_rows.append(
            {
                "Index": index_name,
                "N": len(index_df),
                "Up_Rate": index_df["FORWARD_TARGET"].mean(),
                "Logit_Accuracy": accuracy_score(
                    index_df["FORWARD_TARGET"].astype(int),
                    index_df["FORWARD_PRED_TARGET"],
                ),
                "Logit_AUC": roc_auc_score(
                    index_df["FORWARD_TARGET"].astype(int),
                    index_df["FORWARD_UP_PROB"],
                ),
                "Momentum_Accuracy": accuracy_score(
                    index_df["FORWARD_TARGET"].astype(int),
                    index_df["MOMENTUM_TARGET"].astype(int),
                ),
                "Momentum_AUC": roc_auc_score(
                    index_df["FORWARD_TARGET"].astype(int),
                    index_df["OPEN_TO_1330_RETURN"],
                ),
            }
        )

    by_index_results_df = pd.DataFrame(by_index_rows)

    by_index_results_df.to_csv(
        forward_output_dir / "forward_2025_by_index_results.csv",
        index=False,
    )

    print("\n第46步完成：2025前瞻留出检验完成。")
    print(results_2025_df)
    print(by_index_results_df)

    # ==================================================
    # 第46.1步：2025前瞻结果置信区间
    # ==================================================

    ci_rows = []

    _append_accuracy_ci(
        ci_rows,
        "Majority-class baseline",
        y_test,
        forward_test_df["MAJORITY_TARGET"],
        wilson_ci_from_correct,
    )

    _append_accuracy_ci(
        ci_rows,
        "Deterministic rule",
        rule_eval_df["FORWARD_TARGET"],
        rule_eval_df["FORWARD_RULE_TARGET"],
        wilson_ci_from_correct,
    )

    _append_accuracy_ci(
        ci_rows,
        "Pre-13:30 momentum-continuation baseline",
        momentum_eval_df["FORWARD_TARGET"],
        momentum_eval_df["MOMENTUM_TARGET"],
        wilson_ci_from_correct,
    )

    _append_accuracy_ci(
        ci_rows,
        "Three-signal LogisticRegression",
        y_test,
        forward_test_df["FORWARD_PRED_TARGET"],
        wilson_ci_from_correct,
    )

    logit_auc_ci = bootstrap_auc_ci(
        y_test,
        forward_test_df["FORWARD_UP_PROB"],
    )

    ci_rows.append(
        {
            "Model": "Three-signal LogisticRegression",
            "Metric": "AUC",
            "Estimate": logit_auc_ci[0],
            "CI_Lower": logit_auc_ci[1],
            "CI_Upper": logit_auc_ci[2],
            "Method": "Bootstrap",
        }
    )

    momentum_auc_ci = bootstrap_auc_ci(
        y_test,
        forward_test_df["OPEN_TO_1330_RETURN"],
    )

    ci_rows.append(
        {
            "Model": "Pre-13:30 momentum-continuation baseline",
            "Metric": "AUC",
            "Estimate": momentum_auc_ci[0],
            "CI_Lower": momentum_auc_ci[1],
            "CI_Upper": momentum_auc_ci[2],
            "Method": "Bootstrap",
        }
    )

    confidence_intervals_df = pd.DataFrame(ci_rows)

    confidence_intervals_df.to_csv(
        forward_output_dir / "forward_2025_confidence_intervals.csv",
        index=False,
    )

    print("\n第46.1步完成：2025前瞻置信区间检验完成。")
    print(confidence_intervals_df)

    # ==================================================
    # 第47步：前瞻目标滚动样本外检验
    # ==================================================

    rolling_results_df = _run_forward_rolling_validation(
        main_forward_df,
        forward_output_dir,
    )

    print("\n第47步完成：前瞻目标滚动样本外检验完成。")
    print(rolling_results_df)

    # ==================================================
    # 第48步：深证成指前瞻外部验证
    # ==================================================

    sz_yearly_results_df = _evaluate_szse_forward(
        sz_raw_df,
        sz_feature_df,
        target_map,
        forward_model,
        forward_output_dir,
    )

    print("\n第48步完成：深证成指前瞻外部验证完成。")
    print(sz_yearly_results_df)

    # ==================================================
    # 第49步：绘制前瞻目标结果图
    # ==================================================

    _save_forward_plots(
        forward_output_dir,
        y_test,
        forward_test_df,
        rolling_results_df,
        logit_auc,
        momentum_auc,
        show_plots,
    )

    print("\n第49步完成：前瞻目标结果图完成。")

    # ==================================================
    # 第50步：保存补充检验数值摘要
    # ==================================================

    summary_lines = _write_forward_summary(
        forward_output_dir,
        forward_train_df,
        forward_test_df,
        sz_yearly_results_df,
        logit_accuracy,
        logit_auc,
        logit_auc_ci,
        rule_accuracy,
        rule_auc,
        momentum_accuracy,
        momentum_auc,
    )

    print("\n第50步完成：补充检验数值摘要已保存。")
    print("\n".join(summary_lines))

    return {
        "sample_audit": sample_audit_df,
        "leakage_audit": leakage_audit_df,
        "results_2025": results_2025_df,
        "by_index_results": by_index_results_df,
        "confidence_intervals": confidence_intervals_df,
        "rolling_results": rolling_results_df,
        "szse_results": sz_yearly_results_df,
        "statsmodels_coefficients": statsmodels_coef_df,
        "model_comparison": model_comparison_df,
    }


def audit_reported_value(rows, metric, reproduced, reported, tolerance):
    """记录计算值与论文报告值的差异及核对状态。"""

    rows.append(
        {
            "Metric": metric,
            "Reproduced": float(reproduced),
            "Manuscript_Reported": float(reported),
            "Tolerance": tolerance,
            "Passed": np.isclose(
                reproduced,
                reported,
                atol=tolerance,
                rtol=0,
            ),
        }
    )


def main() -> None:
    """Run the complete validated analysis pipeline."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ==================================================
    # 第1步：批量读取CSV文件
    # ==================================================

    data_config = {
        "沪深300": {"folder": str(SCRIPT_DIR / "1-沪深300数据"), "prefix": "1"},
        "上证指数": {"folder": str(SCRIPT_DIR / "2-上证指数数据"), "prefix": "2"},
    }

    df_list = []

    for index_name, config in data_config.items():

        for year in range(START_YEAR, END_YEAR + 1):

            file_path = f'{config["folder"]}/{config["prefix"]}-{year}.csv'

            temp_df = pd.read_csv(file_path)

            temp_df["INDEX_NAME"] = index_name
            temp_df["YEAR"] = year

            df_list.append(temp_df)

    df = pd.concat(df_list, ignore_index=True)

    print("所有数据读取完成。")
    print("总数据量：", len(df))
    print("已读取的数据范围：")
    print(df[["INDEX_NAME", "YEAR"]].drop_duplicates())

    # ==================================================
    # 第2步：查看数据
    # ==================================================

    if DEBUG:
        print(df.head())
        print(df["TDATE_TIME"].head(20))

    # ==================================================
    # 第3步：时间格式转换
    # ==================================================

    df["TDATE_TIME"] = pd.to_datetime(df["TDATE_TIME"], errors="coerce")

    if df["TDATE_TIME"].isnull().sum() == 0:
        print("时间格式转换成功。")
    else:
        print("发现无法转换的时间数据，请检查TDATE_TIME列。")
        print(df[df["TDATE_TIME"].isnull()].head())

    # ==================================================
    # 第4步：检查缺失值
    # ==================================================

    missing_count = df.isnull().sum()

    print("每一列缺失值数量：")
    print(missing_count)

    if missing_count.sum() == 0:
        print("数据不存在缺失值。")
    else:
        print("发现缺失值，请进一步检查。")

    # ==================================================
    # 第5步：检查重复时间记录
    # ==================================================

    duplicate_count = df.duplicated(subset=["INDEX_NAME", "TDATE_TIME"]).sum()

    if duplicate_count == 0:
        print("不存在重复时间记录。")
    else:
        print(f"发现 {duplicate_count} 条重复时间记录。")

    # ==================================================
    # 第6步：提取交易日期
    # ==================================================

    df["TRADE_DATE"] = df["TDATE_TIME"].dt.date

    if DEBUG:
        print(df[["INDEX_NAME", "YEAR", "TDATE_TIME", "TRADE_DATE"]].head(10))

    # ==================================================
    # 第7步：检查交易日数据完整性
    # ==================================================

    daily_count = df.groupby(["INDEX_NAME", "TRADE_DATE"]).size()

    print("每日分钟数据条数统计：")
    print(daily_count.describe())

    abnormal_days = daily_count[daily_count != 240]

    if len(abnormal_days) == 0:
        print("所有指数、所有交易日分钟数据完整，均为240条。")
    else:
        print("发现分钟数据条数异常的交易日：")
        print(abnormal_days)

    # ==================================================
    # 第8步：检查二区和五区关键时间点是否存在
    # ==================================================

    df["TIME_ONLY"] = df["TDATE_TIME"].dt.strftime("%H:%M:%S")

    required_times = [
        TIME_WINDOWS["zone2"][0],
        TIME_WINDOWS["zone2"][1],
        TIME_WINDOWS["zone5"][0],
        TIME_WINDOWS["zone5"][1],
    ]

    required_time_rows = df[df["TIME_ONLY"].isin(required_times)]
    required_time_count = required_time_rows.groupby(["INDEX_NAME", "TRADE_DATE"])[
        "TIME_ONLY"
    ].nunique()
    required_time_count = required_time_count.reindex(
        daily_count.index,
        fill_value=0,
    )

    abnormal_window_days = []
    for index_name, trade_date in required_time_count[
        required_time_count != len(required_times)
    ].index:
        existing_times = set(
            required_time_rows.loc[
                (required_time_rows["INDEX_NAME"] == index_name)
                & (required_time_rows["TRADE_DATE"] == trade_date),
                "TIME_ONLY",
            ]
        )
        abnormal_window_days.extend(
            (index_name, trade_date, required_time)
            for required_time in required_times
            if required_time not in existing_times
        )

    if len(abnormal_window_days) == 0:
        print("所有指数、所有交易日均包含二区和五区关键时间点。")
    else:
        print("发现缺少二区或五区关键时间点的交易日：")
        print(abnormal_window_days)

    # ==================================================
    # 第9步：正式提取二区数据，并检查二区是否完整
    # ==================================================

    zone2_df = select_time_window(df, "zone2")

    zone2_count = zone2_df.groupby(["INDEX_NAME", "TRADE_DATE"]).size()

    abnormal_zone2_days = zone2_count[zone2_count != 30]

    if len(abnormal_zone2_days) == 0:
        print("所有指数、所有交易日二区数据完整，均为30条。")
    else:
        print("发现二区数据不完整的交易日：")
        print(abnormal_zone2_days)

    # ==================================================
    # 第10步：正式提取五区数据，并检查五区是否完整
    # ==================================================

    zone5_df = select_time_window(df, "zone5")

    zone5_count = zone5_df.groupby(["INDEX_NAME", "TRADE_DATE"]).size()

    abnormal_zone5_days = zone5_count[zone5_count != 30]

    if len(abnormal_zone5_days) == 0:
        print("所有指数、所有交易日五区数据完整，均为30条。")
    else:
        print(f"发现 {len(abnormal_zone5_days)} 个指数-交易日五区数据不完整：")
        print(abnormal_zone5_days)

    # ==================================================
    # 第11步：封装区间形态识别函数
    # ==================================================

    # ==================================================
    # 第12步：批量提取二区和五区特征
    # ==================================================

    feature_df, break_df, real_df = extract_daily_components(df)

    print("第12步完成：二区和五区特征提取完成。")
    if DEBUG:
        print(feature_df.head())

    # ==================================================
    # 第12.1步：二区、五区方向与形态分布检查
    # ==================================================

    print("\n第12.1步：二区、五区方向与形态分布检查")

    print("\n二区方向分布：")
    print(feature_df["ZONE2_DIRECTION"].value_counts())
    print(feature_df["ZONE2_DIRECTION"].value_counts(normalize=True))

    print("\n五区方向分布：")
    print(feature_df["ZONE5_DIRECTION"].value_counts())
    print(feature_df["ZONE5_DIRECTION"].value_counts(normalize=True))

    print("\n二区×五区方向组合分布：")
    print(
        pd.crosstab(
            feature_df["ZONE2_DIRECTION"],
            feature_df["ZONE5_DIRECTION"],
            normalize="all",
        )
    )

    print("\n二区形态分布：")
    print(feature_df["ZONE2_SHAPE"].value_counts())
    print(feature_df["ZONE2_SHAPE"].value_counts(normalize=True))

    print("\n五区形态分布：")
    print(feature_df["ZONE5_SHAPE"].value_counts())
    print(feature_df["ZONE5_SHAPE"].value_counts(normalize=True))

    zone2_shock_rate = (feature_df["ZONE2_DIRECTION"] == "震荡").mean()

    zone5_shock_rate = (feature_df["ZONE5_DIRECTION"] == "震荡").mean()

    print("\n震荡占比汇总：")
    print("二区震荡占比：", round(zone2_shock_rate, 4))
    print("五区震荡占比：", round(zone5_shock_rate, 4))

    # ==================================================
    # 第13步：判断五区是否突破一区至四区高点或低点
    # ==================================================

    feature_df = feature_df.merge(break_df, on=["INDEX_NAME", "TRADE_DATE"], how="left")

    print("第13步完成：五区突破信息提取完成。")
    if DEBUG:
        print(feature_df.head())

    # ==================================================
    # 第14步：根据二区、五区及突破情况生成预测结果
    # ==================================================

    feature_df["PRED_DIRECTION"] = predict_direction_series(feature_df)

    print("第14步完成：预测结果生成完成。")
    if DEBUG:
        print(feature_df.head(20))

    # ==================================================
    # 第15步：计算真实涨跌，并验证预测准确率
    # ==================================================

    feature_df = feature_df.merge(real_df, on=["INDEX_NAME", "TRADE_DATE"], how="left")

    feature_df["IS_CORRECT"] = (
        feature_df["PRED_DIRECTION"] == feature_df["REAL_DIRECTION"]
    )

    valid_df = feature_df[
        (feature_df["PRED_DIRECTION"] != "无法判断")
        & (feature_df["REAL_DIRECTION"] != "平")
    ]

    total_count = len(valid_df)
    correct_count = valid_df["IS_CORRECT"].sum()
    accuracy = correct_count / total_count

    print("第15步完成：预测准确率验证完成。")
    print("有效样本数量：", total_count)
    print("预测正确数量：", correct_count)
    print("预测错误数量：", total_count - correct_count)
    print("预测准确率：", round(accuracy, 4))

    if DEBUG:
        print(
            valid_df[
                [
                    "INDEX_NAME",
                    "TRADE_DATE",
                    "ZONE2_SHAPE",
                    "ZONE2_DIRECTION",
                    "ZONE5_SHAPE",
                    "ZONE5_DIRECTION",
                    "ZONE5_BREAK_HIGH",
                    "ZONE5_BREAK_LOW",
                    "PRED_DIRECTION",
                    "REAL_DIRECTION",
                    "IS_CORRECT",
                ]
            ].head(20)
        )

    # ==================================================
    # 第16步：统计各类条件组合的预测准确率
    # 说明：第17-21步属于2015-2025全样本描述性统计，
    # 用于解释规律来源，不参与Logistic模型训练。
    # ==================================================

    print("\n第16步：规则贡献度分析")

    rule_df = valid_df.copy()

    # 构造规则标签：
    # 二区方向_五区方向_五区是否突破前高_五区是否突破前低
    rule_df["RULE_GROUP"] = (
        rule_df["ZONE2_DIRECTION"]
        + "_"
        + rule_df["ZONE5_DIRECTION"]
        + "_"
        + rule_df["ZONE5_BREAK_HIGH"].astype(str)
        + "_"
        + rule_df["ZONE5_BREAK_LOW"].astype(str)
    )

    # 按规则组合分组统计
    rule_stats = (
        rule_df.groupby("RULE_GROUP")
        .agg(TOTAL=("IS_CORRECT", "count"), CORRECT=("IS_CORRECT", "sum"))
        .reset_index()
    )

    # 计算每一种规则组合的准确率
    rule_stats["ACCURACY"] = rule_stats["CORRECT"] / rule_stats["TOTAL"]

    # 按样本数量从多到少排序，方便看主要规则
    rule_stats = rule_stats.sort_values(by="TOTAL", ascending=False)

    print(rule_stats)

    # ==================================================
    # 第17步：突破信号独立检验
    # ==================================================

    print("\n第17步：突破信号独立检验")

    # =========================
    # 创新高统计
    # =========================

    break_high_df = valid_df[valid_df["ZONE5_BREAK_HIGH"] == True]

    high_total = len(break_high_df)

    high_up = len(break_high_df[break_high_df["REAL_DIRECTION"] == "涨"])

    if high_total > 0:
        high_accuracy = high_up / high_total
    else:
        high_accuracy = 0

    print("\n五区创新高统计：")
    print("样本数量：", high_total)
    print("真实上涨数量：", high_up)
    print("上涨概率：", round(high_accuracy, 4))

    # =========================
    # 创新低统计
    # =========================

    break_low_df = valid_df[valid_df["ZONE5_BREAK_LOW"] == True]

    low_total = len(break_low_df)

    low_down = len(break_low_df[break_low_df["REAL_DIRECTION"] == "跌"])

    if low_total > 0:
        low_accuracy = low_down / low_total
    else:
        low_accuracy = 0

    print("\n五区创新低统计：")
    print("样本数量：", low_total)
    print("真实下跌数量：", low_down)
    print("下跌概率：", round(low_accuracy, 4))

    # =========================
    # 无突破统计
    # =========================

    no_break_df = valid_df[
        (valid_df["ZONE5_BREAK_HIGH"] == False) & (valid_df["ZONE5_BREAK_LOW"] == False)
    ]

    no_break_total = len(no_break_df)

    no_break_up = len(no_break_df[no_break_df["REAL_DIRECTION"] == "涨"])

    if no_break_total > 0:
        no_break_up_rate = no_break_up / no_break_total
    else:
        no_break_up_rate = 0

    print("\n无突破统计：")
    print("样本数量：", no_break_total)
    print("上涨概率：", round(no_break_up_rate, 4))

    # ==================================================
    # 第18步：形态贡献度分析
    # ==================================================

    print("\n第18步：形态贡献度分析")

    # ==========================================
    # 二区形态统计
    # ==========================================

    print("\n二区形态统计：")

    zone2_shape_stats = (
        feature_df.groupby("ZONE2_SHAPE")
        .agg(
            TOTAL=("REAL_DIRECTION", "count"),
            UP_COUNT=("REAL_DIRECTION", lambda x: (x == "涨").sum()),
            DOWN_COUNT=("REAL_DIRECTION", lambda x: (x == "跌").sum()),
        )
        .reset_index()
    )

    zone2_shape_stats["UP_RATE"] = (
        zone2_shape_stats["UP_COUNT"] / zone2_shape_stats["TOTAL"]
    )

    zone2_shape_stats["DOWN_RATE"] = (
        zone2_shape_stats["DOWN_COUNT"] / zone2_shape_stats["TOTAL"]
    )

    zone2_shape_stats = zone2_shape_stats.sort_values(by="TOTAL", ascending=False)

    print(zone2_shape_stats)

    # ==========================================
    # 五区形态统计
    # ==========================================

    print("\n五区形态统计：")

    zone5_shape_stats = (
        feature_df.groupby("ZONE5_SHAPE")
        .agg(
            TOTAL=("REAL_DIRECTION", "count"),
            UP_COUNT=("REAL_DIRECTION", lambda x: (x == "涨").sum()),
            DOWN_COUNT=("REAL_DIRECTION", lambda x: (x == "跌").sum()),
        )
        .reset_index()
    )

    zone5_shape_stats["UP_RATE"] = (
        zone5_shape_stats["UP_COUNT"] / zone5_shape_stats["TOTAL"]
    )

    zone5_shape_stats["DOWN_RATE"] = (
        zone5_shape_stats["DOWN_COUNT"] / zone5_shape_stats["TOTAL"]
    )

    zone5_shape_stats = zone5_shape_stats.sort_values(by="TOTAL", ascending=False)

    print(zone5_shape_stats)

    # ==================================================
    # 第19步：二区×五区形态组合胜率矩阵
    # ==================================================

    print("\n第19步：二区×五区形态组合胜率矩阵")

    # 构造形态组合标签

    feature_df["SHAPE_COMBINATION"] = (
        feature_df["ZONE2_SHAPE"] + "_" + feature_df["ZONE5_SHAPE"]
    )

    # 统计组合表现

    shape_stats = (
        feature_df.groupby("SHAPE_COMBINATION")
        .agg(
            TOTAL=("REAL_DIRECTION", "count"),
            UP_COUNT=("REAL_DIRECTION", lambda x: (x == "涨").sum()),
            DOWN_COUNT=("REAL_DIRECTION", lambda x: (x == "跌").sum()),
        )
        .reset_index()
    )

    # 计算上涨率

    shape_stats["UP_RATE"] = shape_stats["UP_COUNT"] / shape_stats["TOTAL"]

    # 计算下跌率

    shape_stats["DOWN_RATE"] = shape_stats["DOWN_COUNT"] / shape_stats["TOTAL"]

    # 样本数太少的不看
    shape_stats = shape_stats[shape_stats["TOTAL"] >= 5]

    # 按上涨率排序

    shape_stats = shape_stats.sort_values(by="UP_RATE", ascending=False)

    print(shape_stats)

    # ==================================================
    # 第20步：突破 × 二区方向 × 五区方向
    # ==================================================

    print("\n第20步：突破 × 二区方向 × 五区方向")

    # 构造突破类型
    feature_df["BREAK_TYPE"] = break_type_series(feature_df)

    # 构造组合标签

    feature_df["DIRECTION_GROUP"] = (
        feature_df["BREAK_TYPE"]
        + "_"
        + feature_df["ZONE2_DIRECTION"]
        + "_"
        + feature_df["ZONE5_DIRECTION"]
    )

    # 分组统计

    direction_stats = (
        feature_df.groupby("DIRECTION_GROUP")
        .agg(
            TOTAL=("REAL_DIRECTION", "count"),
            UP_COUNT=("REAL_DIRECTION", lambda x: (x == "涨").sum()),
            DOWN_COUNT=("REAL_DIRECTION", lambda x: (x == "跌").sum()),
        )
        .reset_index()
    )

    # 样本数过滤

    direction_stats = direction_stats[direction_stats["TOTAL"] >= 5]

    # 上涨率

    direction_stats["UP_RATE"] = direction_stats["UP_COUNT"] / direction_stats["TOTAL"]

    # 下跌率

    direction_stats["DOWN_RATE"] = (
        direction_stats["DOWN_COUNT"] / direction_stats["TOTAL"]
    )

    # 排序

    direction_stats = direction_stats.sort_values(by="UP_RATE", ascending=False)

    print(direction_stats)

    # ==================================================
    # 第二阶段：
    # 机器学习权重学习与样本外验证
    # ==================================================

    # ==================================================
    # 第21步：划分训练集和测试集
    # ==================================================

    # 将交易日期转为datetime，方便按年份划分
    feature_df["TRADE_DATE_DT"] = pd.to_datetime(feature_df["TRADE_DATE"])

    # 训练集：2015-2024
    train_df = feature_df[feature_df["TRADE_DATE_DT"].dt.year <= TRAIN_END_YEAR].copy()

    # 测试集：2025
    test_df = feature_df[feature_df["TRADE_DATE_DT"].dt.year == HOLDOUT_YEAR].copy()

    print("\n第21步完成：训练集和测试集划分完成。")
    print("训练集样本数量：", len(train_df))
    print("测试集样本数量：", len(test_df))

    print("训练集年份范围：")
    print(train_df["TRADE_DATE_DT"].dt.year.value_counts().sort_index())

    print("测试集年份范围：")
    print(test_df["TRADE_DATE_DT"].dt.year.value_counts().sort_index())

    # ==================================================
    # 第22步：构建机器学习特征
    # ==================================================

    direction_map = {"涨": 1, "跌": -1, "震荡": 0}

    target_map = {"涨": 1, "跌": 0}

    for data in [feature_df, train_df, test_df]:

        data["BREAK_SIGNAL"] = break_signal_series(data)

        data["ZONE2_SIGNAL"] = data["ZONE2_DIRECTION"].map(direction_map)

        data["ZONE5_SIGNAL"] = data["ZONE5_DIRECTION"].map(direction_map)

        data["TARGET"] = data["REAL_DIRECTION"].map(target_map)

        data["MIDDAY_POSITION_SIGNAL"] = data["MIDDAY_POSITION"].map(direction_map)

        data["MIDDAY_BASELINE_TARGET"] = (data["MIDDAY_POSITION"] == "涨").astype(int)

    print("\n第22步完成：机器学习特征构建完成。")
    if DEBUG:
        print(
            train_df[
                [
                    "INDEX_NAME",
                    "TRADE_DATE",
                    "BREAK_SIGNAL",
                    "ZONE2_SIGNAL",
                    "ZONE5_SIGNAL",
                    "TARGET",
                ]
            ].head()
        )

    # ==================================================
    # 第23步：Logistic回归训练与权重输出
    # ==================================================

    # 去除无法构建标签或特征的样本
    train_model_df = train_df.dropna(
        subset=["BREAK_SIGNAL", "ZONE2_SIGNAL", "ZONE5_SIGNAL", "TARGET"]
    ).copy()

    X_train = train_model_df[["BREAK_SIGNAL", "ZONE2_SIGNAL", "ZONE5_SIGNAL"]]

    y_train = train_model_df["TARGET"]

    model = build_logistic_model()

    model.fit(X_train, y_train)

    coef_df = pd.DataFrame({"变量": X_train.columns, "权重系数": model.coef_[0]})

    coef_df["绝对权重"] = coef_df["权重系数"].abs()

    coef_df = coef_df.sort_values(by="绝对权重", ascending=False)

    print("\n第23步完成：Logistic回归训练完成。")
    print("模型权重：")
    print(coef_df)

    print("截距项：", model.intercept_[0])

    # ==================================================
    # 第23.1步：statsmodels显著性检验、Baseline比较与VIF诊断
    # 说明：
    # 本步骤复用第23步已经构建完成的训练集特征矩阵，
    # 不重新读取数据，也不重新构造任何特征。
    # ==================================================

    stat_output_dir = OUTPUT_DIR / "statistical_enhancement_outputs"
    stat_output_dir.mkdir(parents=True, exist_ok=True)

    # ==========================
    # statsmodels Logit显著性检验
    # ==========================

    X_train_sm = X_train.astype(float)
    y_train_sm = y_train.astype(int)

    X_train_sm_const = sm.add_constant(X_train_sm, has_constant="add")

    statsmodels_logit = sm.Logit(y_train_sm, X_train_sm_const)

    statsmodels_result = statsmodels_logit.fit(disp=False)

    statsmodels_coef_df = pd.DataFrame(
        {
            "Variable": statsmodels_result.params.index,
            "Coefficient": statsmodels_result.params.values,
            "Standard_Error": statsmodels_result.bse.values,
            "z_statistic": statsmodels_result.tvalues.values,
            "p_value": statsmodels_result.pvalues.values,
        }
    )

    statsmodels_coef_df["Odds_Ratio"] = np.exp(statsmodels_coef_df["Coefficient"])

    statsmodels_coef_df["Significance"] = statsmodels_coef_df["p_value"].apply(
        lambda p: (
            "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))
        )
    )

    statsmodels_coef_df.to_csv(
        stat_output_dir / "statsmodels_logit_coefficients.csv", index=False
    )

    print("\n第23.1步：statsmodels显著性检验完成")
    print(statsmodels_coef_df)

    # ==========================
    # 按交易日聚类的稳健标准误
    # 说明：两个主要指数在同一交易日可能受到共同市场冲击，
    # 因此按TRADE_DATE_DT聚类，允许同一日期内的误差相关。
    # ==========================

    date_cluster_groups = train_model_df["TRADE_DATE_DT"].dt.strftime("%Y-%m-%d")

    statsmodels_date_clustered_result = sm.Logit(y_train_sm, X_train_sm_const).fit(
        disp=False,
        cov_type="cluster",
        cov_kwds={"groups": date_cluster_groups, "use_correction": True},
    )

    statsmodels_date_clustered_df = pd.DataFrame(
        {
            "Variable": statsmodels_date_clustered_result.params.index,
            "Coefficient": statsmodels_date_clustered_result.params.values,
            "Clustered_Standard_Error": statsmodels_date_clustered_result.bse.values,
            "z_statistic": statsmodels_date_clustered_result.tvalues.values,
            "p_value": statsmodels_date_clustered_result.pvalues.values,
        }
    )

    statsmodels_date_clustered_df["Odds_Ratio"] = np.exp(
        statsmodels_date_clustered_df["Coefficient"]
    )

    statsmodels_date_clustered_df["Cluster_Variable"] = "TRADE_DATE_DT"
    statsmodels_date_clustered_df["Number_of_Date_Clusters"] = (
        date_cluster_groups.nunique()
    )

    statsmodels_date_clustered_df.to_csv(
        stat_output_dir / "statsmodels_logit_coefficients_date_clustered.csv",
        index=False,
    )

    print("\n按交易日聚类的稳健标准误检验完成")
    print("交易日聚类数量：", date_cluster_groups.nunique())
    print(statsmodels_date_clustered_df)

    # ==========================
    # Baseline模型比较
    # ==========================

    baseline_compare_df = pd.DataFrame(
        [
            fit_statsmodels_logit(
                train_model_df, y_train_sm, ["BREAK_SIGNAL"], "Baseline: BREAK only"
            ),
            fit_statsmodels_logit(
                train_model_df,
                y_train_sm,
                ["BREAK_SIGNAL", "ZONE2_SIGNAL", "ZONE5_SIGNAL"],
                "Full: BREAK + ZONE2 + ZONE5",
            ),
        ]
    )

    baseline_compare_df.to_csv(
        stat_output_dir / "logit_model_comparison.csv", index=False
    )

    print("\nBaseline模型比较：")
    print(baseline_compare_df)

    # ==========================
    # VIF多重共线性检验
    # ==========================

    vif_list = []

    for i, column in enumerate(X_train_sm.columns):

        vif_list.append(
            {"Variable": column, "VIF": variance_inflation_factor(X_train_sm.values, i)}
        )

    vif_df = pd.DataFrame(vif_list)

    vif_df.to_csv(stat_output_dir / "vif_diagnostics.csv", index=False)

    print("\nVIF多重共线性检验：")
    print(vif_df)

    # ==================================================
    # 第24步：2025样本外预测验证
    # ==================================================

    # 与训练集保持一致，去除无法构建标签或特征的测试样本
    test_df = test_df.dropna(
        subset=["BREAK_SIGNAL", "ZONE2_SIGNAL", "ZONE5_SIGNAL", "TARGET"]
    ).copy()

    print("2025测试集有效建模样本数量：", len(test_df))

    # 测试集特征
    X_test = test_df[["BREAK_SIGNAL", "ZONE2_SIGNAL", "ZONE5_SIGNAL"]]

    # 测试集真实结果
    y_test = test_df["TARGET"]

    # 预测上涨概率
    test_df["UP_PROB"] = model.predict_proba(X_test)[:, 1]

    # 概率转涨跌
    test_df["PRED_TARGET"] = (test_df["UP_PROB"] >= PREDICTION_THRESHOLD).astype(int)

    # 准确率
    test_accuracy = accuracy_score(y_test, test_df["PRED_TARGET"])

    print("\n第24步完成：2025样本外验证完成。")

    print("2025测试集准确率：")
    print(round(test_accuracy, 4))

    # 查看前20条预测结果

    if DEBUG:
        print(
            test_df[
                ["INDEX_NAME", "TRADE_DATE", "UP_PROB", "TARGET", "PRED_TARGET"]
            ].head(20)
        )

    # ==================================================
    # 第25步：规则模型 VS Logistic模型
    # ==================================================

    # ==========================
    # 规则模型准确率（2025全样本口径）
    # ==========================

    rule_test_df = test_df.copy()

    rule_test_df["RULE_TARGET"] = (rule_test_df["REAL_DIRECTION"] == "涨").astype(int)

    rule_accuracy = (
        rule_test_df["PRED_DIRECTION"] == rule_test_df["REAL_DIRECTION"]
    ).mean()

    # ==========================
    # Logistic准确率（2025）
    # ==========================

    logistic_accuracy = accuracy_score(test_df["TARGET"], test_df["PRED_TARGET"])

    # ==========================
    # 输出结果
    # ==========================

    print("\n第25步完成：规则模型与Logistic模型比较")

    print("规则模型准确率（2025全样本口径）：")
    print(round(rule_accuracy, 4))

    print("Logistic模型准确率（2025全样本口径）：")
    print(round(logistic_accuracy, 4))

    print("提升幅度：")
    print(round(logistic_accuracy - rule_accuracy, 4))

    # ==================================================
    # 第25.2步：13:30 price-position baseline
    # ==================================================

    midday_eval_df = test_df[test_df["MIDDAY_POSITION"] != "平"].copy()

    midday_accuracy = accuracy_score(
        midday_eval_df["TARGET"], midday_eval_df["MIDDAY_BASELINE_TARGET"]
    )

    print("\n第25.2步完成：13:30 price-position baseline 验证完成")
    print("13:30 price-position baseline 准确率：")
    print(round(midday_accuracy, 4))
    print("有效样本数量：", len(midday_eval_df))

    # ==================================================
    # 第25.1步：2025测试集震荡与规则模型无法判断检查
    # ==================================================

    print("\n第25.1步：2025测试集震荡与规则模型无法判断检查")

    print("\n2025测试集二区方向分布：")
    print(test_df["ZONE2_DIRECTION"].value_counts())
    print(test_df["ZONE2_DIRECTION"].value_counts(normalize=True))

    print("\n2025测试集五区方向分布：")
    print(test_df["ZONE5_DIRECTION"].value_counts())
    print(test_df["ZONE5_DIRECTION"].value_counts(normalize=True))

    print("\n2025测试集二区×五区方向组合分布：")
    print(
        pd.crosstab(
            test_df["ZONE2_DIRECTION"], test_df["ZONE5_DIRECTION"], normalize="all"
        )
    )

    print("\n2025规则模型预测结果分布：")
    print(test_df["PRED_DIRECTION"].value_counts())
    print(test_df["PRED_DIRECTION"].value_counts(normalize=True))

    unable_count = (test_df["PRED_DIRECTION"] == "无法判断").sum()

    total_count = len(test_df)

    unable_rate = unable_count / total_count

    print("\n2025规则模型无法判断样本统计：")
    print("无法判断数量：", unable_count)
    print("总样本数量：", total_count)
    print("无法判断占比：", round(unable_rate, 4))

    # ==================================================
    # 第26步：分指数样本外验证
    # ==================================================

    for index_name in test_df["INDEX_NAME"].unique():

        temp_df = test_df[test_df["INDEX_NAME"] == index_name]

        acc = accuracy_score(temp_df["TARGET"], temp_df["PRED_TARGET"])

        print(f"\n{index_name} 2025测试集准确率：")
        print(round(acc, 4))

        print("样本数量：", len(temp_df))

    # ==================================================
    # 第27步：AUC检验
    # ==================================================

    auc = roc_auc_score(test_df["TARGET"], test_df["UP_PROB"])

    midday_auc = roc_auc_score(test_df["TARGET"], test_df["OPEN_TO_1330_RETURN"])

    print("\n第27步完成：AUC检验完成。")
    print("AUC值：")
    print(round(auc, 4))
    print("13:30 open-to-13:30 return baseline AUC：")
    print(round(midday_auc, 4))

    midday_baseline_results_df = pd.DataFrame(
        [
            {
                "Evaluation": "2025 13:30 price-position baseline",
                "N": len(midday_eval_df),
                "Accuracy": midday_accuracy,
                "AUC": midday_auc,
                "Up_Rate": test_df["TARGET"].mean(),
            }
        ]
    )

    midday_baseline_results_df.to_csv(
        output_path("midday_baseline_results.csv"), index=False
    )

    # ==================================================
    # 第27.0步：2025 holdout置信区间检验
    # 说明：
    # Accuracy使用Wilson二项比例置信区间；
    # AUC使用bootstrap置信区间。
    # 本步骤只评估2025样本外主结果，不改变模型训练或预测流程。
    # ==================================================

    holdout_ci_results = []

    rule_acc_ci = wilson_ci_from_correct(
        rule_test_df["PRED_DIRECTION"] == rule_test_df["REAL_DIRECTION"]
    )

    holdout_ci_results.append(
        {
            "Model": "Rule benchmark",
            "Metric": "Accuracy",
            "Estimate": rule_acc_ci[0],
            "CI_Lower": rule_acc_ci[1],
            "CI_Upper": rule_acc_ci[2],
            "Method": "Wilson",
        }
    )

    logit_acc_ci = wilson_ci_from_correct(test_df["TARGET"] == test_df["PRED_TARGET"])

    holdout_ci_results.append(
        {
            "Model": "Logistic",
            "Metric": "Accuracy",
            "Estimate": logit_acc_ci[0],
            "CI_Lower": logit_acc_ci[1],
            "CI_Upper": logit_acc_ci[2],
            "Method": "Wilson",
        }
    )

    midday_acc_ci = wilson_ci_from_correct(
        midday_eval_df["TARGET"] == midday_eval_df["MIDDAY_BASELINE_TARGET"]
    )

    holdout_ci_results.append(
        {
            "Model": "13:30 price-position baseline",
            "Metric": "Accuracy",
            "Estimate": midday_acc_ci[0],
            "CI_Lower": midday_acc_ci[1],
            "CI_Upper": midday_acc_ci[2],
            "Method": "Wilson",
        }
    )

    logit_auc_ci = bootstrap_auc_ci(test_df["TARGET"], test_df["UP_PROB"])

    holdout_ci_results.append(
        {
            "Model": "Logistic",
            "Metric": "AUC",
            "Estimate": logit_auc_ci[0],
            "CI_Lower": logit_auc_ci[1],
            "CI_Upper": logit_auc_ci[2],
            "Method": "Bootstrap",
        }
    )

    midday_auc_ci = bootstrap_auc_ci(test_df["TARGET"], test_df["OPEN_TO_1330_RETURN"])

    holdout_ci_results.append(
        {
            "Model": "13:30 price-position baseline",
            "Metric": "AUC",
            "Estimate": midday_auc_ci[0],
            "CI_Lower": midday_auc_ci[1],
            "CI_Upper": midday_auc_ci[2],
            "Method": "Bootstrap",
        }
    )

    holdout_ci_df = pd.DataFrame(holdout_ci_results)

    holdout_ci_df.to_csv(output_path("holdout_confidence_intervals.csv"), index=False)

    print("\n第27.0步完成：2025 holdout置信区间检验完成。")
    print(holdout_ci_df)
    print("置信区间结果已保存为：holdout_confidence_intervals.csv")

    # ==================================================
    # 第27.1步：statsmodels与scikit-learn预测结果一致性检查
    # 说明：
    # statsmodels用于显著性检验，scikit-learn用于原论文预测流程。
    # 本步骤验证两套实现对2025样本外预测结论是否一致。
    # ==================================================

    X_test_sm = X_test.astype(float)

    X_test_sm_const = sm.add_constant(X_test_sm, has_constant="add")

    test_df["UP_PROB_STATSMODELS"] = statsmodels_result.predict(X_test_sm_const)

    test_df["PRED_TARGET_STATSMODELS"] = (
        test_df["UP_PROB_STATSMODELS"] >= PREDICTION_THRESHOLD
    ).astype(int)

    statsmodels_accuracy = accuracy_score(
        test_df["TARGET"], test_df["PRED_TARGET_STATSMODELS"]
    )

    statsmodels_auc = roc_auc_score(test_df["TARGET"], test_df["UP_PROB_STATSMODELS"])

    prediction_compare_df = pd.DataFrame(
        [
            {"Model": "Rule benchmark", "Accuracy": rule_accuracy, "AUC": np.nan},
            {
                "Model": "13:30 price-position baseline",
                "Accuracy": midday_accuracy,
                "AUC": midday_auc,
            },
            {
                "Model": "scikit-learn LogisticRegression",
                "Accuracy": logistic_accuracy,
                "AUC": auc,
            },
            {
                "Model": "statsmodels Logit",
                "Accuracy": statsmodels_accuracy,
                "AUC": statsmodels_auc,
            },
        ]
    )

    prediction_compare_df.to_csv(
        output_path("prediction_framework_comparison.csv"), index=False
    )

    stat_summary_lines = [
        "Statistical enhancement completed inside test.py.",
        f"Training observations: {len(train_model_df)}",
        f"Testing observations: {len(test_df)}",
        "",
        "Statsmodels coefficient table:",
        statsmodels_coef_df.to_string(index=False),
        "",
        "Model comparison:",
        baseline_compare_df.to_string(index=False),
        "",
        "VIF diagnostics:",
        vif_df.to_string(index=False),
        "",
        "Prediction framework comparison:",
        prediction_compare_df.to_string(index=False),
    ]

    with open(
        stat_output_dir / "statistical_enhancement_summary.txt", "w", encoding="utf-8"
    ) as f:
        f.write("\n".join(stat_summary_lines))

    print("\n第27.1步完成：statsmodels与scikit-learn预测结果一致性检查完成")
    print(prediction_compare_df)

    # ==================================================
    # 第28步：概率分层检验（Calibration Analysis）
    # ==================================================

    calibration_df = test_df.copy()

    # 概率分箱
    calibration_df["PROB_GROUP"] = pd.cut(
        calibration_df["UP_PROB"],
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"],
    )

    result = (
        calibration_df.groupby("PROB_GROUP", observed=False)
        .agg(TOTAL=("TARGET", "count"), REAL_UP_RATE=("TARGET", "mean"))
        .reset_index()
    )

    print("\n第28步完成：概率分层检验完成。")
    print(result)

    # ==================================================
    # 第29步：高置信度交易信号检验
    # ==================================================

    high_up = test_df[test_df["UP_PROB"] >= 0.8].copy()

    high_down = test_df[test_df["UP_PROB"] <= 0.2].copy()

    # 看涨信号胜率
    up_win_rate = (high_up["TARGET"] == 1).mean()

    # 看跌信号胜率
    down_win_rate = (high_down["TARGET"] == 0).mean()

    print("\n第29步完成：高置信度信号检验完成。")

    print("\n高置信度看涨信号：")
    print("样本数量：", len(high_up))
    print("真实上涨率：", round(up_win_rate, 4))

    print("\n高置信度看跌信号：")
    print("样本数量：", len(high_down))
    print("真实下跌率：", round(down_win_rate, 4))

    # ==================================================
    # 第三阶段：
    # 论文图表
    # ==================================================
    # Use a publication-friendly font for English figure labels.
    rcParams["font.family"] = "Arial"

    # 避免负号显示异常
    rcParams["axes.unicode_minus"] = False

    # ==================================================
    # 第29.1步：滚动样本外验证（Expanding-window）
    # ==================================================

    rolling_results = []

    rolling_feature_columns = ["BREAK_SIGNAL", "ZONE2_SIGNAL", "ZONE5_SIGNAL"]

    for test_year in range(2020, 2026):

        rolling_train_df = (
            feature_df[feature_df["TRADE_DATE_DT"].dt.year < test_year]
            .dropna(subset=rolling_feature_columns + ["TARGET"])
            .copy()
        )

        rolling_test_df = (
            feature_df[feature_df["TRADE_DATE_DT"].dt.year == test_year]
            .dropna(subset=rolling_feature_columns + ["TARGET"])
            .copy()
        )

        if len(rolling_train_df) == 0 or len(rolling_test_df) == 0:
            continue

        fit_predict_logistic(
            rolling_train_df,
            rolling_test_df,
            rolling_feature_columns,
            "TARGET",
            "ROLLING_UP_PROB",
            "ROLLING_PRED_TARGET",
        )

        rolling_midday_eval_df = rolling_test_df[
            rolling_test_df["MIDDAY_POSITION"] != "平"
        ].copy()

        rolling_results.append(
            {
                "Test_Year": test_year,
                "Training_Period": f"2015-{test_year - 1}",
                "N": len(rolling_test_df),
                "Rule_Accuracy": (
                    rolling_test_df["PRED_DIRECTION"]
                    == rolling_test_df["REAL_DIRECTION"]
                ).mean(),
                "Logit_Accuracy": accuracy_score(
                    rolling_test_df["TARGET"], rolling_test_df["ROLLING_PRED_TARGET"]
                ),
                "Logit_AUC": roc_auc_score(
                    rolling_test_df["TARGET"], rolling_test_df["ROLLING_UP_PROB"]
                ),
                "Midday_Accuracy": accuracy_score(
                    rolling_midday_eval_df["TARGET"],
                    rolling_midday_eval_df["MIDDAY_BASELINE_TARGET"],
                ),
                "Midday_AUC": roc_auc_score(
                    rolling_test_df["TARGET"], rolling_test_df["OPEN_TO_1330_RETURN"]
                ),
            }
        )

    rolling_results_df = pd.DataFrame(rolling_results)

    rolling_results_df.to_csv(
        output_path("rolling_out_of_sample_results.csv"), index=False
    )

    print("\n第29.1步完成：滚动样本外验证完成。")
    print(rolling_results_df)

    accuracy_series = [
        ("Rule", "Rule_Accuracy"),
        ("Logit", "Logit_Accuracy"),
        ("13:30 baseline", "Midday_Accuracy"),
    ]

    auc_series = [
        ("Logit", "Logit_AUC"),
        ("13:30 baseline", "Midday_AUC"),
    ]

    plt.figure(figsize=(9.2, 5.6))

    for label, column in accuracy_series:
        plt.plot(
            rolling_results_df["Test_Year"],
            rolling_results_df[column],
            marker="o",
            linewidth=2.0,
            label=f"{label} accuracy",
        )

    plt.xlabel("Test year")
    plt.ylabel("Accuracy")
    plt.title("Rolling Out-of-Sample Accuracy")
    plt.ylim(0.50, 0.90)
    plt.xticks(rolling_results_df["Test_Year"])
    plt.grid(alpha=0.3)
    plt.legend()

    for _, row in rolling_results_df.iterrows():
        plt.text(
            row["Test_Year"],
            row["Logit_Accuracy"] + 0.012,
            f"{row['Logit_Accuracy']:.2%}",
            ha="center",
            fontsize=8,
        )

    plt.savefig(
        output_path("Rolling_Out_of_Sample_Accuracy.png"), dpi=450, bbox_inches="tight"
    )

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

    plt.figure(figsize=(9.2, 5.6))

    for label, column in auc_series:
        plt.plot(
            rolling_results_df["Test_Year"],
            rolling_results_df[column],
            marker="o",
            linewidth=2.0,
            label=f"{label} AUC",
        )

    plt.xlabel("Test year")
    plt.ylabel("AUC")
    plt.title("Rolling Out-of-Sample AUC")
    plt.ylim(0.55, 0.95)
    plt.xticks(rolling_results_df["Test_Year"])
    plt.grid(alpha=0.3)
    plt.legend()

    for _, row in rolling_results_df.iterrows():
        plt.text(
            row["Test_Year"],
            row["Logit_AUC"] + 0.012,
            f"{row['Logit_AUC']:.3f}",
            ha="center",
            fontsize=8,
        )

    plt.savefig(
        output_path("Rolling_Out_of_Sample_AUC.png"), dpi=450, bbox_inches="tight"
    )

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), dpi=450)

    for label, column in accuracy_series:
        axes[0].plot(
            rolling_results_df["Test_Year"],
            rolling_results_df[column],
            marker="o",
            linewidth=2.0,
            label=label,
        )

    axes[0].set_xlabel("Test year")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy")
    axes[0].set_ylim(0.50, 0.90)
    axes[0].set_xticks(rolling_results_df["Test_Year"])
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    for label, column in auc_series:
        axes[1].plot(
            rolling_results_df["Test_Year"],
            rolling_results_df[column],
            marker="o",
            linewidth=2.0,
            label=label,
        )

    axes[1].set_xlabel("Test year")
    axes[1].set_ylabel("AUC")
    axes[1].set_title("AUC")
    axes[1].set_ylim(0.55, 0.95)
    axes[1].set_xticks(rolling_results_df["Test_Year"])
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.suptitle("Rolling Out-of-Sample Validation", fontsize=13)
    fig.tight_layout()

    fig.savefig(
        output_path("Rolling_Out_of_Sample_Performance.png"),
        dpi=450,
        bbox_inches="tight",
    )

    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), dpi=450)

    for label, column in accuracy_series:
        axes[0].plot(
            rolling_results_df["Test_Year"],
            rolling_results_df[column],
            marker="o",
            linewidth=1.8,
            markersize=4.2,
            label=label,
        )

    axes[0].set_xlabel("Test year")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("(a) Accuracy")
    axes[0].set_ylim(0.50, 0.90)
    axes[0].set_xticks(rolling_results_df["Test_Year"])
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=7.5, frameon=False, loc="lower left")

    for label, column in auc_series:
        axes[1].plot(
            rolling_results_df["Test_Year"],
            rolling_results_df[column],
            marker="o",
            linewidth=1.8,
            markersize=4.2,
            label=label,
        )

    axes[1].set_xlabel("Test year")
    axes[1].set_ylabel("AUC")
    axes[1].set_title("(b) AUC")
    axes[1].set_ylim(0.55, 0.95)
    axes[1].set_xticks(rolling_results_df["Test_Year"])
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=7.5, frameon=False, loc="lower left")

    fig.tight_layout(w_pad=2.0)
    fig.savefig(
        output_path("Rolling_Out_of_Sample_Validation_Panels.png"),
        dpi=450,
        bbox_inches="tight",
    )

    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.35, 2.55), dpi=450)

    for label, column in accuracy_series:
        ax.plot(
            rolling_results_df["Test_Year"],
            rolling_results_df[column],
            marker="o",
            linewidth=1.6,
            markersize=3.8,
            label=label,
        )

    ax.set_xlabel("Test year", fontsize=8)
    ax.set_ylabel("Accuracy", fontsize=8)
    ax.set_ylim(0.50, 0.90)
    ax.set_xticks(rolling_results_df["Test_Year"])
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6.8, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(
        output_path("Rolling_Out_of_Sample_Accuracy_Publication.png"),
        dpi=450,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.35, 2.55), dpi=450)

    for label, column in auc_series:
        ax.plot(
            rolling_results_df["Test_Year"],
            rolling_results_df[column],
            marker="o",
            linewidth=1.6,
            markersize=3.8,
            label=label,
        )

    ax.set_xlabel("Test year", fontsize=8)
    ax.set_ylabel("AUC", fontsize=8)
    ax.set_ylim(0.55, 0.95)
    ax.set_xticks(rolling_results_df["Test_Year"])
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6.8, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(
        output_path("Rolling_Out_of_Sample_AUC_Publication.png"),
        dpi=450,
        bbox_inches="tight",
    )
    plt.close(fig)

    print("滚动样本外验证表已保存为：rolling_out_of_sample_results.csv")
    print("滚动准确率图已保存为：Rolling_Out_of_Sample_Accuracy.png")
    print("滚动AUC图已保存为：Rolling_Out_of_Sample_AUC.png")
    print("滚动验证合并图已保存为：Rolling_Out_of_Sample_Performance.png")
    print("滚动验证双面板图已保存为：Rolling_Out_of_Sample_Validation_Panels.png")
    print(
        "滚动验证正文Accuracy图已保存为：Rolling_Out_of_Sample_Accuracy_Publication.png"
    )
    print("滚动验证正文AUC图已保存为：Rolling_Out_of_Sample_AUC_Publication.png")
    # ==================================================
    # 第30步：绘制ROC曲线图
    # ==================================================

    # 计算ROC曲线所需数据
    fpr, tpr, thresholds = roc_curve(test_df["TARGET"], test_df["UP_PROB"])

    # 创建画布
    plt.figure(figsize=(9.5, 7.2))

    # 绘制ROC曲线
    plt.plot(fpr, tpr, linewidth=2.8, label=f"Logistic Regression (AUC = {auc:.3f})")

    # 绘制随机分类参考线
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.8, label="Random Classifier")

    plt.xlabel("False Positive Rate", fontsize=13)
    plt.ylabel("True Positive Rate", fontsize=13)
    plt.title("ROC Curve of the Logistic Regression Model", fontsize=15)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)

    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)

    # 保存图片
    plt.savefig(output_path("ROC_Curve.png"), dpi=450, bbox_inches="tight")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

    print("\n第30步完成：ROC曲线图绘制完成。")
    print("图片已保存为：ROC_Curve.png")

    # ==================================================
    # 第31步：绘制概率分层柱状图
    # ==================================================

    plt.figure(figsize=(8, 6))

    plt.bar(result["PROB_GROUP"].astype(str), result["REAL_UP_RATE"])

    plt.xlabel("Predicted Upward Probability Group")
    plt.ylabel("Observed Upward Rate")
    plt.title("Observed Upward Rate by Predicted Probability Group")

    plt.ylim(0, 1)

    # 在柱子上标注数值
    for i, value in enumerate(result["REAL_UP_RATE"]):
        plt.text(i, value + 0.02, f"{value:.2%}", ha="center")

    plt.grid(axis="y", alpha=0.3)

    plt.savefig(output_path("Probability_Group_Bar.png"), dpi=300, bbox_inches="tight")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

    print("\n第31步完成：概率分层柱状图绘制完成。")
    print("图片已保存为：Probability_Group_Bar.png")

    # ==================================================
    # 第32步：规则模型 VS Logistic模型准确率对比图
    # ==================================================

    plt.figure(figsize=(8, 6))

    model_names = ["Rule-Based Model", "Logistic Regression"]

    accuracy_values = [rule_accuracy, logistic_accuracy]

    bars = plt.bar(model_names, accuracy_values)

    plt.ylabel("Accuracy")
    plt.title("Accuracy Comparison: Rule-Based vs. Logistic Model")

    plt.ylim(0, 1)

    # 显示具体数值
    for bar, value in zip(bars, accuracy_values):

        plt.text(
            bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2%}", ha="center"
        )

    plt.grid(axis="y", alpha=0.3)

    plt.savefig(
        output_path("Model_Accuracy_Comparison.png"), dpi=300, bbox_inches="tight"
    )

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

    print("\n第32步完成：规则模型 VS Logistic模型对比图完成。")
    print("图片已保存为：Model_Accuracy_Comparison.png")

    # ==================================================
    # 第33步：分指数样本外准确率对比图
    # ==================================================

    index_acc = []

    for index_name in test_df["INDEX_NAME"].unique():

        temp_df = test_df[test_df["INDEX_NAME"] == index_name]

        acc = accuracy_score(temp_df["TARGET"], temp_df["PRED_TARGET"])

        index_acc.append({"INDEX_NAME": index_name, "ACCURACY": acc})

    index_acc_df = pd.DataFrame(index_acc)
    index_display_names = {
        "上证指数": "SSE Composite Index",
        "沪深300": "CSI 300 Index",
    }

    plt.figure(figsize=(8, 6))

    bars = plt.bar(
        index_acc_df["INDEX_NAME"].map(index_display_names), index_acc_df["ACCURACY"]
    )

    plt.ylabel("Accuracy")
    plt.title("Out-of-Sample Accuracy by Index, 2025")

    plt.ylim(0, 1)

    for bar, value in zip(bars, index_acc_df["ACCURACY"]):

        plt.text(
            bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2%}", ha="center"
        )

    plt.grid(axis="y", alpha=0.3)

    plt.savefig(
        output_path("Index_Accuracy_Comparison.png"), dpi=300, bbox_inches="tight"
    )

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

    print("\n第33步完成：分指数准确率对比图完成。")
    print("图片已保存为：Index_Accuracy_Comparison.png")

    # ==================================================
    # 第34步：二区×五区形态热力图
    # ==================================================

    print("\n第34步：绘制二区×五区形态热力图")

    # 构造矩阵数据

    heatmap_df = (
        feature_df.groupby(["ZONE2_SHAPE", "ZONE5_SHAPE"])
        .agg(UP_RATE=("REAL_DIRECTION", lambda x: (x == "涨").mean()))
        .reset_index()
    )

    # 转成透视表

    pivot_df = heatmap_df.pivot(
        index="ZONE2_SHAPE", columns="ZONE5_SHAPE", values="UP_RATE"
    )
    shape_display_names = {
        "斜线向上": "Monotonic Up",
        "斜线向下": "Monotonic Down",
        "V": "V",
        "倒V": "Inverted V",
        "N": "N",
        "倒N": "Inverted N",
        "W": "W",
        "M": "M",
        "复杂上涨": "Complex Up",
        "复杂下跌": "Complex Down",
        "震荡": "Sideways",
    }

    # 画图

    plt.figure(figsize=(12, 9.5))

    im = plt.imshow(pivot_df, cmap="RdYlBu_r", aspect="auto")

    # 坐标轴

    plt.xticks(
        range(len(pivot_df.columns)),
        [shape_display_names.get(x, x) for x in pivot_df.columns],
        rotation=45,
        ha="right",
        fontsize=11,
    )

    plt.yticks(
        range(len(pivot_df.index)),
        [shape_display_names.get(x, x) for x in pivot_df.index],
        fontsize=11,
    )

    # 颜色条

    cbar = plt.colorbar(im)
    cbar.set_label("Observed Upward Rate", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    # 写数值

    for i in range(len(pivot_df.index)):
        for j in range(len(pivot_df.columns)):

            value = pivot_df.iloc[i, j]

            if not np.isnan(value):

                plt.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=9)

    plt.title("Observed Upward Rate by Zone 2 and Zone 5 Pattern", fontsize=15)
    plt.xlabel("Zone 5 Pattern", fontsize=13)
    plt.ylabel("Zone 2 Pattern", fontsize=13)

    plt.tight_layout()

    plt.savefig(output_path("Shape_Heatmap.png"), dpi=450, bbox_inches="tight")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

    print("图片已保存为：Shape_Heatmap.png")

    # 生成适合论文单栏排版的高可读性版本。
    # 原始12英寸画布缩放至单栏后会同步缩小字号；这里直接按单栏物理尺寸绘制，
    # 并使用较短的英文标签，使坐标和单元格数值在约3.2英寸宽度下仍可辨认。
    shape_short_names = {
        "斜线向上": "Mono. Up",
        "斜线向下": "Mono. Down",
        "V": "V",
        "倒V": "Inv. V",
        "N": "N",
        "倒N": "Inv. N",
        "W": "W",
        "M": "M",
        "复杂上涨": "C. Up",
        "复杂下跌": "C. Down",
        "震荡": "Sideways",
    }

    fig, ax = plt.subplots(figsize=(3.35, 3.05), dpi=600)
    im = ax.imshow(pivot_df, cmap="RdYlBu_r", aspect="auto")
    ax.set_xticks(range(len(pivot_df.columns)))
    ax.set_xticklabels(
        [shape_short_names.get(x, x) for x in pivot_df.columns],
        rotation=55,
        ha="right",
        fontsize=5.6,
    )
    ax.set_yticks(range(len(pivot_df.index)))
    ax.set_yticklabels(
        [shape_short_names.get(x, x) for x in pivot_df.index], fontsize=5.6
    )

    for i in range(len(pivot_df.index)):
        for j in range(len(pivot_df.columns)):
            value = pivot_df.iloc[i, j]
            if not np.isnan(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=4.6)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.025)
    cbar.set_label("Observed upward rate", fontsize=5.8)
    cbar.ax.tick_params(labelsize=5.2)
    ax.set_title("Upward Rate by Zone 2 and Zone 5 Pattern", fontsize=7.0, pad=4)
    ax.set_xlabel("Zone 5 pattern", fontsize=6.2)
    ax.set_ylabel("Zone 2 pattern", fontsize=6.2)
    fig.tight_layout(pad=0.35)
    fig.savefig(output_path("Shape_Heatmap_Column.png"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    print("单栏优化热力图已保存为：Shape_Heatmap_Column.png")

    # ==================================================
    # 第35步：Logistic特征重要性图
    # ==================================================

    print("\n第35步：绘制Logistic特征重要性图")

    # 按绝对值排序
    plot_df = coef_df.sort_values(by="绝对权重", ascending=False)

    plt.figure(figsize=(8, 5))

    bars = plt.bar(plot_df["变量"], plot_df["绝对权重"])

    # 写数值

    for bar in bars:

        height = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
        )

    plt.ylabel("Absolute Coefficient")

    plt.title("Feature Importance in the Logistic Regression Model")

    plt.tight_layout()

    plt.savefig(output_path("Feature_Importance.png"), dpi=300, bbox_inches="tight")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

    print("图片已保存为：Feature_Importance.png")

    # ==================================================
    # 第37步：Prominence参数敏感性检验
    # ==================================================

    print("\n第37步：Prominence参数敏感性检验")

    prominence_list = [0.10, 0.15, 0.20, 0.25, 0.30]
    sensitivity_feature_columns = [
        "BREAK_SIGNAL",
        "ZONE2_SIGNAL",
        "ZONE5_SIGNAL",
    ]

    sensitivity_result = []
    sensitivity_daily_cache = build_sensitivity_daily_cache(df)

    for ratio in prominence_list:

        print(f"\n正在测试 prominence_ratio = {ratio}")

        temp_feature_list = []

        for cached_day in sensitivity_daily_cache:
            zone2_shape, zone2_direction = identify_shape_and_direction(
                cached_day["ZONE2_PRICES"], prominence_ratio=ratio
            )
            zone5_shape, zone5_direction = identify_shape_and_direction(
                cached_day["ZONE5_PRICES"], prominence_ratio=ratio
            )

            temp_feature_list.append(
                {
                    "INDEX_NAME": cached_day["INDEX_NAME"],
                    "TRADE_DATE": cached_day["TRADE_DATE"],
                    "ZONE2_DIRECTION": zone2_direction,
                    "ZONE5_DIRECTION": zone5_direction,
                    "ZONE5_BREAK_HIGH": cached_day["ZONE5_BREAK_HIGH"],
                    "ZONE5_BREAK_LOW": cached_day["ZONE5_BREAK_LOW"],
                    "REAL_DIRECTION": cached_day["REAL_DIRECTION"],
                }
            )

        temp_feature_df = pd.DataFrame(temp_feature_list)

        # 日期转换
        temp_feature_df["TRADE_DATE_DT"] = pd.to_datetime(temp_feature_df["TRADE_DATE"])

        # 构造机器学习特征
        temp_feature_df["BREAK_SIGNAL"] = break_signal_series(temp_feature_df)

        temp_feature_df["ZONE2_SIGNAL"] = temp_feature_df["ZONE2_DIRECTION"].map(
            direction_map
        )

        temp_feature_df["ZONE5_SIGNAL"] = temp_feature_df["ZONE5_DIRECTION"].map(
            direction_map
        )

        temp_feature_df["TARGET"] = temp_feature_df["REAL_DIRECTION"].map(target_map)

        # 划分训练集和测试集
        temp_train_df = temp_feature_df[
            temp_feature_df["TRADE_DATE_DT"].dt.year <= TRAIN_END_YEAR
        ].copy()

        temp_test_df = temp_feature_df[
            temp_feature_df["TRADE_DATE_DT"].dt.year == HOLDOUT_YEAR
        ].copy()

        # 去除无法建模样本
        temp_train_df = temp_train_df.dropna(
            subset=["BREAK_SIGNAL", "ZONE2_SIGNAL", "ZONE5_SIGNAL", "TARGET"]
        )

        temp_test_df = temp_test_df.dropna(
            subset=["BREAK_SIGNAL", "ZONE2_SIGNAL", "ZONE5_SIGNAL", "TARGET"]
        )

        y_test = temp_test_df["TARGET"]

        fit_predict_logistic(
            temp_train_df,
            temp_test_df,
            sensitivity_feature_columns,
            "TARGET",
            "SENSITIVITY_UP_PROB",
            "SENSITIVITY_PRED_TARGET",
        )

        up_prob = temp_test_df["SENSITIVITY_UP_PROB"]
        pred_target = temp_test_df["SENSITIVITY_PRED_TARGET"]

        # 计算准确率和AUC
        temp_accuracy = accuracy_score(y_test, pred_target)

        temp_auc = roc_auc_score(y_test, up_prob)

        sensitivity_result.append(
            {
                "PROMINENCE_RATIO": ratio,
                "TEST_SAMPLE": len(temp_test_df),
                "ACCURACY": temp_accuracy,
                "AUC": temp_auc,
            }
        )

    sensitivity_df = pd.DataFrame(sensitivity_result)

    print("\n第37步完成：Prominence参数敏感性检验完成。")
    print(sensitivity_df)

    sensitivity_output_df = sensitivity_df.copy()
    sensitivity_output_df["ACCURACY_PERCENT"] = (
        sensitivity_output_df["ACCURACY"] * 100
    ).round(2)
    sensitivity_output_df["AUC_ROUND"] = (sensitivity_output_df["AUC"]).round(3)
    sensitivity_output_df = sensitivity_output_df[
        ["PROMINENCE_RATIO", "TEST_SAMPLE", "ACCURACY_PERCENT", "AUC_ROUND"]
    ]
    sensitivity_output_df.columns = ["Ratio", "Test Sample", "Accuracy (%)", "AUC"]
    sensitivity_output_df.to_csv(
        output_path("prominence_sensitivity_results.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    print("Prominence敏感性结果表已保存为：prominence_sensitivity_results.csv")

    # ==================================================
    # 第38步：Prominence参数敏感性结果图
    # ==================================================

    plt.figure(figsize=(11.5, 7))

    plt.plot(
        sensitivity_df["PROMINENCE_RATIO"],
        sensitivity_df["ACCURACY"],
        marker="o",
        linewidth=2,
        label="Accuracy",
    )

    plt.plot(
        sensitivity_df["PROMINENCE_RATIO"],
        sensitivity_df["AUC"],
        marker="s",
        linewidth=2,
        label="AUC",
    )

    plt.xlabel("Prominence Ratio")
    plt.ylabel("Metric Value")
    plt.title("Sensitivity Analysis of the Prominence Parameter")

    plt.ylim(0.65, 0.80)

    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()

    plt.savefig(output_path("Parameter_Sensitivity.png"), dpi=300, bbox_inches="tight")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

    print("\n第38步完成：参数敏感性结果图完成。")
    print("图片已保存为：Parameter_Sensitivity.png")
    # ==================================================
    # 第四阶段：深证成指数据作为外部验证集
    # ==================================================

    # ==================================================
    # 第39步：读取深证成指数据作为外部验证集
    # ==================================================

    print("\n第39步：读取深证成指外部验证数据")

    sz_list = []

    for year in range(START_YEAR, END_YEAR + 1):

        file_path = SCRIPT_DIR / "3-深证成指" / f"3-{year}.csv"

        temp_df = pd.read_csv(file_path)

        temp_df["INDEX_NAME"] = "深证成指"
        temp_df["YEAR"] = year

        sz_list.append(temp_df)

    sz_df = pd.concat(sz_list, ignore_index=True)

    sz_df["TDATE_TIME"] = pd.to_datetime(sz_df["TDATE_TIME"], errors="coerce")

    sz_df["TRADE_DATE"] = sz_df["TDATE_TIME"].dt.date
    sz_df["TIME_ONLY"] = sz_df["TDATE_TIME"].dt.strftime("%H:%M:%S")

    print("深证成指数据读取完成。")
    print("总数据量：", len(sz_df))
    print(sz_df[["INDEX_NAME", "YEAR"]].drop_duplicates())

    # ==================================================
    # 第40步：深证成指二区、五区与突破特征提取
    # ==================================================

    print("\n第40步：深证成指特征提取")

    sz_feature_df, sz_break_df, sz_real_df = extract_daily_components(sz_df)

    sz_feature_df = sz_feature_df.merge(
        sz_break_df, on=["INDEX_NAME", "TRADE_DATE"], how="left"
    ).merge(
        sz_real_df[
            [
                "INDEX_NAME",
                "TRADE_DATE",
                "DAY_OPEN",
                "DAY_CLOSE",
                "REAL_DIRECTION",
            ]
        ],
        on=["INDEX_NAME", "TRADE_DATE"],
        how="left",
    )

    sz_feature_df["TRADE_DATE_DT"] = pd.to_datetime(sz_feature_df["TRADE_DATE"])

    print("深证成指特征提取完成。")
    if DEBUG:
        print(sz_feature_df.head())

    # ==================================================
    # 第41步：深证成指2015-2025跨指数长期验证
    # ==================================================

    print("\n第41步：深证成指2015-2025跨指数长期验证")

    # 构造深证成指机器学习特征
    sz_feature_df["BREAK_SIGNAL"] = break_signal_series(sz_feature_df)

    sz_feature_df["ZONE2_SIGNAL"] = sz_feature_df["ZONE2_DIRECTION"].map(direction_map)

    sz_feature_df["ZONE5_SIGNAL"] = sz_feature_df["ZONE5_DIRECTION"].map(direction_map)

    sz_feature_df["TARGET"] = sz_feature_df["REAL_DIRECTION"].map(target_map)

    # 去除无法建模的样本
    sz_model_df = sz_feature_df.dropna(
        subset=["BREAK_SIGNAL", "ZONE2_SIGNAL", "ZONE5_SIGNAL", "TARGET"]
    ).copy()

    sz_feature_columns = [
        "BREAK_SIGNAL",
        "ZONE2_SIGNAL",
        "ZONE5_SIGNAL",
    ]

    # 使用原Logistic模型一次性预测全部外部验证样本，不重新训练。
    sz_model_df["UP_PROB"] = model.predict_proba(sz_model_df[sz_feature_columns])[:, 1]
    sz_model_df["PRED_TARGET"] = (
        sz_model_df["UP_PROB"] >= PREDICTION_THRESHOLD
    ).astype(int)

    # 用来保存逐年验证结果
    sz_yearly_result = []

    for year in range(START_YEAR, END_YEAR + 1):

        temp_df = sz_model_df[sz_model_df["TRADE_DATE_DT"].dt.year == year].copy()

        y_sz_year = temp_df["TARGET"]

        year_accuracy = accuracy_score(y_sz_year, temp_df["PRED_TARGET"])

        year_auc = roc_auc_score(y_sz_year, temp_df["UP_PROB"])

        sz_yearly_result.append(
            {
                "YEAR": year,
                "SAMPLE": len(temp_df),
                "ACCURACY": year_accuracy,
                "AUC": year_auc,
            }
        )

    sz_yearly_df = pd.DataFrame(sz_yearly_result)

    # 生成论文用统计表
    sz_yearly_table = sz_yearly_df.copy()

    sz_yearly_table["ACCURACY_PERCENT"] = (sz_yearly_table["ACCURACY"] * 100).round(2)

    sz_yearly_table["AUC_ROUND"] = (sz_yearly_table["AUC"]).round(4)

    sz_yearly_table = sz_yearly_table[
        ["YEAR", "SAMPLE", "ACCURACY_PERCENT", "AUC_ROUND"]
    ]

    sz_yearly_table.columns = ["年份", "样本数量", "准确率(%)", "AUC"]

    print("深证成指2015-2025逐年验证完成。")
    print(sz_yearly_df)

    print("\n深证成指Accuracy和AUC统计表：")
    print(sz_yearly_table)

    # 保存统计表
    sz_yearly_table.to_csv(
        output_path("SZSE_Component_Index_Yearly_Accuracy_AUC_Table.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    print("统计表已保存为：SZSE_Component_Index_Yearly_Accuracy_AUC_Table.csv")

    print("\n深证成指长期验证汇总：")
    print("平均准确率：", round(sz_yearly_df["ACCURACY"].mean(), 4))
    print("最高准确率：", round(sz_yearly_df["ACCURACY"].max(), 4))
    print("最低准确率：", round(sz_yearly_df["ACCURACY"].min(), 4))
    print("平均AUC：", round(sz_yearly_df["AUC"].mean(), 4))
    print("最高AUC：", round(sz_yearly_df["AUC"].max(), 4))
    print("最低AUC：", round(sz_yearly_df["AUC"].min(), 4))

    # 单独提取深证成指2025结果，供第42步三指数对比使用
    sz_2025_accuracy = sz_yearly_df[sz_yearly_df["YEAR"] == HOLDOUT_YEAR][
        "ACCURACY"
    ].values[0]

    sz_2025_auc = sz_yearly_df[sz_yearly_df["YEAR"] == HOLDOUT_YEAR]["AUC"].values[0]

    print("\n深证成指2025外部验证结果：")
    print("预测准确率：", round(sz_2025_accuracy, 4))
    print("AUC值：", round(sz_2025_auc, 4))

    # ==================================================
    # 第42步：深证成指逐年Accuracy、AUC图与三指数2025准确率对比图
    # ==================================================

    print("\n第42步：绘制深证成指逐年Accuracy、AUC图与三指数2025对比图")

    # ==========================
    # 图1：深证成指2015-2025逐年准确率图
    # ==========================

    plt.figure(figsize=(10, 6))

    plt.plot(
        sz_yearly_df["YEAR"],
        sz_yearly_df["ACCURACY"],
        marker="o",
        linewidth=2.6,
        markersize=7,
    )

    plt.xlabel("Year", fontsize=13)
    plt.ylabel("Accuracy", fontsize=13)
    plt.title("Annual External-Validation Accuracy: SZSE Component Index", fontsize=15)
    plt.xticks(sz_yearly_df["YEAR"], rotation=30, fontsize=10)
    plt.yticks(fontsize=11)

    plt.ylim(0, 1)

    for x, y in zip(sz_yearly_df["YEAR"], sz_yearly_df["ACCURACY"]):

        plt.text(x, y + 0.02, f"{y:.2%}", ha="center", fontsize=9)

    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        output_path("SZSE_Component_Index_Yearly_Accuracy.png"),
        dpi=450,
        bbox_inches="tight",
    )

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

    print("图片已保存为：SZSE_Component_Index_Yearly_Accuracy.png")

    # ==========================
    # 图2：深证成指2015-2025逐年AUC图
    # ==========================

    plt.figure(figsize=(11.5, 7))

    plt.plot(
        sz_yearly_df["YEAR"],
        sz_yearly_df["AUC"],
        marker="s",
        linewidth=2.6,
        markersize=7,
    )

    plt.xlabel("Year", fontsize=13)
    plt.ylabel("AUC", fontsize=13)
    plt.title("Annual External-Validation AUC: SZSE Component Index", fontsize=15)
    plt.xticks(sz_yearly_df["YEAR"], rotation=30, fontsize=10)
    plt.yticks(fontsize=11)

    plt.ylim(0, 1)

    for x, y in zip(sz_yearly_df["YEAR"], sz_yearly_df["AUC"]):

        plt.text(x, y + 0.02, f"{y:.3f}", ha="center", fontsize=9)

    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        output_path("SZSE_Component_Index_Yearly_AUC.png"), dpi=450, bbox_inches="tight"
    )

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

    print("图片已保存为：SZSE_Component_Index_Yearly_AUC.png")

    # ==========================
    # 图3：2025三指数准确率对比图
    # ==========================

    three_index_acc_df = pd.DataFrame(
        [
            {
                "INDEX_NAME": "上证指数",
                "ACCURACY": index_acc_df[index_acc_df["INDEX_NAME"] == "上证指数"][
                    "ACCURACY"
                ].values[0],
            },
            {
                "INDEX_NAME": "沪深300",
                "ACCURACY": index_acc_df[index_acc_df["INDEX_NAME"] == "沪深300"][
                    "ACCURACY"
                ].values[0],
            },
            {"INDEX_NAME": "深证成指", "ACCURACY": sz_2025_accuracy},
        ]
    )
    three_index_display_names = {
        "上证指数": "SSE Composite",
        "沪深300": "CSI 300",
        "深证成指": "SZSE Component",
    }

    plt.figure(figsize=(8, 6))

    bars = plt.bar(
        three_index_acc_df["INDEX_NAME"].map(three_index_display_names),
        three_index_acc_df["ACCURACY"],
    )

    plt.ylabel("Accuracy")
    plt.title("Prediction Accuracy across Three Indices, 2025")

    plt.ylim(0, 1)

    for bar, value in zip(bars, three_index_acc_df["ACCURACY"]):

        plt.text(
            bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2%}", ha="center"
        )

    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        output_path("Three_Index_Accuracy_Comparison.png"), dpi=300, bbox_inches="tight"
    )

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

    print("图片已保存为：Three_Index_Accuracy_Comparison.png")

    print("\n第42步完成：深证成指逐年Accuracy图、AUC图、统计表与三指数2025对比图完成。")

    # ==================================================
    # 第五阶段：
    # 13:30至收盘前瞻方向检验
    # ==================================================

    """
    本节复用三个既定信号，不改变特征定义、样本划分、模型参数或
    13:30信息边界。
    """

    forward_results = run_forward_extension(
        feature_df=feature_df,
        raw_primary_df=df,
        sz_raw_df=sz_df,
        sz_feature_df=sz_feature_df,
        target_map=target_map,
        wilson_ci_from_correct=wilson_ci_from_correct,
        bootstrap_auc_ci=bootstrap_auc_ci,
        output_dir=OUTPUT_DIR,
        show_plots=SHOW_PLOTS,
    )

    # ==================================================
    # 第51步：核对论文报告的关键样本量与结果
    # 本步骤不参与训练、调参或预测。
    # ==================================================

    manuscript_audit_rows = []

    # 表1：样本划分与日期聚类数
    audit_reported_value(
        manuscript_audit_rows,
        "Training observations",
        len(train_model_df),
        4861,
        0,
    )
    audit_reported_value(
        manuscript_audit_rows,
        "2025 holdout observations",
        len(test_df),
        486,
        0,
    )
    audit_reported_value(
        manuscript_audit_rows,
        "SZSE observations",
        len(sz_model_df),
        2674,
        0,
    )
    audit_reported_value(
        manuscript_audit_rows,
        "Trading-date clusters",
        date_cluster_groups.nunique(),
        2431,
        0,
    )
    audit_reported_value(
        manuscript_audit_rows,
        "Simultaneous high-and-low breakouts",
        (feature_df["ZONE5_BREAK_HIGH"] & feature_df["ZONE5_BREAK_LOW"]).sum(),
        9,
        0,
    )

    # 表2：按交易日聚类的未正则化Logit系数
    reported_coefficients = {
        "BREAK_SIGNAL": 1.7181,
        "ZONE2_SIGNAL": 0.2055,
        "ZONE5_SIGNAL": 0.1548,
        "const": 0.1797,
    }

    for variable, reported_value in reported_coefficients.items():
        reproduced_value = statsmodels_date_clustered_df.loc[
            statsmodels_date_clustered_df["Variable"] == variable,
            "Coefficient",
        ].iloc[0]

        audit_reported_value(
            manuscript_audit_rows,
            f"Clustered Logit coefficient: {variable}",
            reproduced_value,
            reported_value,
            0.00005,
        )

    # 表3：主要预测结果
    for metric, reproduced, reported, tolerance in [
        ("2025 rule accuracy", rule_accuracy, 0.6543, 0.00005),
        ("2025 price-position accuracy", midday_accuracy, 0.8354, 0.00005),
        ("2025 price-position AUC", midday_auc, 0.893, 0.0005),
        ("2025 three-signal Logit accuracy", logistic_accuracy, 0.6955, 0.00005),
        ("2025 three-signal Logit AUC", auc, 0.768, 0.0005),
        ("SZSE 2025 Logit accuracy", sz_2025_accuracy, 0.7490, 0.00005),
        ("SZSE 2025 Logit AUC", sz_2025_auc, 0.8205, 0.00005),
    ]:
        audit_reported_value(
            manuscript_audit_rows,
            metric,
            reproduced,
            reported,
            tolerance,
        )

    # 表4：13:30后补充边界检验
    forward_table = forward_results["results_2025"].set_index("Model")

    forward_reported_values = [
        ("Majority-class baseline", "Accuracy", 0.5576, 0.00005),
        ("Majority-class baseline", "AUC", 0.500, 0.0005),
        ("Deterministic rule", "Accuracy", 0.4691, 0.00005),
        ("Deterministic rule", "AUC", 0.461, 0.0005),
        ("Pre-13:30 momentum-continuation baseline", "Accuracy", 0.5432, 0.00005),
        ("Pre-13:30 momentum-continuation baseline", "AUC", 0.532, 0.0005),
        ("Three-signal LogisticRegression", "Accuracy", 0.5412, 0.00005),
        ("Three-signal LogisticRegression", "AUC", 0.481, 0.0005),
    ]

    for model_name, metric_name, reported_value, tolerance in forward_reported_values:
        audit_reported_value(
            manuscript_audit_rows,
            f"Forward {model_name} {metric_name}",
            forward_table.loc[model_name, metric_name],
            reported_value,
            tolerance,
        )

    manuscript_audit_df = pd.DataFrame(manuscript_audit_rows)
    manuscript_audit_df.to_csv(
        OUTPUT_DIR / "manuscript_result_audit.csv",
        index=False,
    )

    if not manuscript_audit_df["Passed"].all():
        failed_checks = manuscript_audit_df.loc[~manuscript_audit_df["Passed"]]
        raise AssertionError(
            "计算结果与论文报告值不一致：\n" + failed_checks.to_string(index=False)
        )

    print("\n第51步完成：关键样本量与论文报告结果一致。")
    print(manuscript_audit_df)


if __name__ == "__main__":
    main()
