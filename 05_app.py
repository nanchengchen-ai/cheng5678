"""銅箔製程異常分析與 Random Forest 預測 Streamlit 應用程式。"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
DATA_FILE = APP_DIR / "copper_foil_cleaned_labeled.csv"
MODEL_FILE = APP_DIR / "random_forest_anomaly.joblib"

st.set_page_config(
    page_title="銅箔製程異常分析",
    page_icon=":material/analytics:",
    layout="wide",
)


@st.cache_data(ttl="15m", max_entries=3)
def load_data(path: str, modified_time: float) -> pd.DataFrame:
    """載入清理後的標註資料；以檔案修改時間更新快取。"""
    del modified_time
    return pd.read_csv(path, encoding="utf-8-sig")


@st.cache_resource
def load_model(path: str, modified_time: float) -> dict:
    """載入訓練 Notebook 匯出的模型與欄位中繼資料。"""
    del modified_time
    return joblib.load(path)


def render_overview() -> None:
    """顯示主要製程與厚度資料圖表，可依缺陷類別篩選。"""
    st.title("製程資料視覺化")
    st.caption("探索銅箔製程參數、厚度分布與異常標記。")

    if not DATA_FILE.exists():
        st.error(
            "找不到清理後資料檔。請先執行 `02_data_preparation.ipynb`，"
            "並確認 `copper_foil_cleaned_labeled.csv` 位於應用程式同一資料夾。",
            title="資料檔尚未準備",
        )
        return

    try:
        data = load_data(str(DATA_FILE), DATA_FILE.stat().st_mtime)
    except Exception as exc:
        st.error(f"讀取資料失敗：{exc}", title="無法載入資料")
        return

    if data.empty:
        st.warning("資料檔沒有資料列。", title="目前沒有可顯示的資料")
        return

    if "Defect_Type" in data.columns:
        defect_types = sorted(data["Defect_Type"].dropna().unique().tolist())
        selected_types = st.sidebar.multiselect(
            "缺陷類別篩選",
            options=defect_types,
            default=defect_types,
            key="overview_defect_types",
        )
        view = data[data["Defect_Type"].isin(selected_types)]
    else:
        view = data

    if view.empty:
        st.info("目前篩選條件沒有符合的資料。", title="沒有資料")
        return

    total = len(view)
    anomaly_count = int(pd.to_numeric(view.get("Is_Anomaly", 0), errors="coerce").fillna(0).sum())
    anomaly_rate = anomaly_count / total if total else 0.0
    mean_thickness = pd.to_numeric(view.get("Thick_Mean_um"), errors="coerce").mean()
    metric_cols = st.columns(3)
    metric_cols[0].metric("樣本數", f"{total:,}", border=True)
    metric_cols[1].metric("異常筆數", f"{anomaly_count:,}", border=True)
    metric_cols[2].metric(
        "平均厚度",
        f"{mean_thickness:.3f} µm" if pd.notna(mean_thickness) else "無資料",
        f"異常率 {anomaly_rate:.1%}",
        border=True,
    )

    st.subheader("製程參數趨勢")
    numeric_options = [
        col for col in [
            "Line_Speed_mpm", "Current_ASD", "Bath_Temp_C", "Cu_Conc_gL",
            "Additive_ppm", "Jet_Pressure_bar", "Edge_Current_Bias",
        ] if col in view.columns
    ]
    if numeric_options:
        selected_feature = st.selectbox(
            "選擇製程特徵",
            numeric_options,
            format_func=lambda name: {
                "Line_Speed_mpm": "線速 (m/min)",
                "Current_ASD": "電流密度 (A/dm²)",
                "Bath_Temp_C": "槽液溫度 (°C)",
                "Cu_Conc_gL": "銅離子濃度 (g/L)",
                "Additive_ppm": "添加劑 (ppm)",
                "Jet_Pressure_bar": "噴流壓力 (bar)",
                "Edge_Current_Bias": "邊緣電流偏差",
            }.get(name, name),
        )
        if "Timestamp" in view.columns:
            time_data = view[["Timestamp", selected_feature]].copy()
            time_data["Timestamp"] = pd.to_datetime(time_data["Timestamp"], errors="coerce")
            time_data[selected_feature] = pd.to_numeric(time_data[selected_feature], errors="coerce")
            time_data = time_data.dropna().sort_values("Timestamp").set_index("Timestamp")
            st.line_chart(
                time_data,
                y=selected_feature,
                alt=f"銅箔製程特徵 {selected_feature} 隨時間變化",
            )
        else:
            st.line_chart(view[selected_feature], y=selected_feature, alt=f"{selected_feature} 樣本趨勢")

    thickness_cols = [
        col for col in ["Thick_Left_um", "Thick_Center_um", "Thick_Right_um"]
        if col in view.columns
    ]
    left_col, right_col = st.columns(2)
    with left_col:
        st.subheader("厚度量測分布")
        if thickness_cols:
            thickness_long = view[thickness_cols].melt(var_name="量測位置", value_name="厚度 (µm)")
            st.bar_chart(
                thickness_long.groupby("量測位置", as_index=True)["厚度 (µm)"].mean(),
                y="厚度 (µm)",
                alt="左側、中心及右側厚度的平均值比較",
            )
            st.dataframe(
                view[thickness_cols].describe().T.round(3),
                alt="各量測位置厚度的描述統計",
            )
        else:
            st.info("資料中沒有三點厚度欄位。")
    with right_col:
        st.subheader("缺陷類別筆數")
        if "Defect_Type" in view.columns:
            counts = view["Defect_Type"].fillna("未標註").value_counts().rename_axis("缺陷類別")
            st.bar_chart(counts, y="count", alt="各缺陷類別的樣本筆數")
        else:
            st.info("資料中沒有 Defect_Type 欄位。")

    st.subheader("製程參數與平均厚度")
    if "Thick_Mean_um" in view.columns and numeric_options:
        scatter_feature = st.selectbox(
            "散佈圖 X 軸特徵",
            numeric_options,
            key="scatter_feature",
            format_func=lambda name: name.replace("_", " "),
        )
        scatter = view[[scatter_feature, "Thick_Mean_um"]].copy()
        if "Defect_Type" in view.columns:
            scatter["Defect_Type"] = view["Defect_Type"].astype("string").fillna("未標註")
        st.scatter_chart(
            scatter,
            x=scatter_feature,
            y="Thick_Mean_um",
            color="Defect_Type" if "Defect_Type" in scatter.columns else None,
            alt=f"{scatter_feature} 與平均厚度的關係",
        )

    with st.expander("查看目前篩選資料"):
        st.dataframe(view, alt="目前篩選後的銅箔製程資料")


def render_prediction() -> None:
    """依訓練時欄位中繼資料建立輸入表單並執行單筆預測。"""
    st.title("單筆異常預測")
    st.caption("輸入一筆製程量測，使用已訓練的 Random Forest 模型預測異常。")

    if not MODEL_FILE.exists():
        st.error(
            "找不到 `random_forest_anomaly.joblib`。請先執行 `03_random_forest_training.ipynb`，"
            "將模型檔放在 `05_app.py` 同一資料夾。",
            title="模型檔尚未準備",
        )
        return

    try:
        bundle = load_model(str(MODEL_FILE), MODEL_FILE.stat().st_mtime)
        model = bundle["model"]
        feature_columns = bundle["feature_columns"]
        classes = bundle["classes"]
    except Exception as exc:
        st.error(f"載入模型失敗：{exc}", title="無法載入模型")
        return

    # 使用已準備資料的中位數作為表單預設值，讓輸入起點貼近訓練分布。
    reference = None
    if DATA_FILE.exists():
        try:
            labeled = load_data(str(DATA_FILE), DATA_FILE.stat().st_mtime)
            available = [col for col in feature_columns if col in labeled.columns]
            reference = labeled[available]
        except Exception:
            reference = None

    numeric_defaults = {}
    categorical_choices = {}
    for feature in feature_columns:
        if reference is not None and feature in reference.columns:
            series = reference[feature]
        else:
            series = pd.Series(dtype="float64")
        if pd.api.types.is_numeric_dtype(series):
            values = pd.to_numeric(series, errors="coerce").dropna()
            numeric_defaults[feature] = float(values.median()) if not values.empty else 0.0
        else:
            choices = series.dropna().astype(str).value_counts().index.tolist()
            categorical_choices[feature] = choices or ["未知"]

    st.info(
        "輸入欄位會依模型特徵清單自動產生。缺失旗標及時間衍生欄位也需輸入；"
        "如有原始時間，可依訓練資料規則換算為小時、星期及週末旗標。",
        title="輸入提示",
    )

    input_row = {}
    with st.form("prediction_form", border=True):
        st.subheader("製程與厚度特徵")
        columns = st.columns(2)
        for index, feature in enumerate(feature_columns):
            parent = columns[index % 2]
            label = feature.replace("_", " ")
            with parent:
                if feature in categorical_choices:
                    choices = categorical_choices[feature]
                    input_row[feature] = st.selectbox(
                        label,
                        options=choices,
                        key=f"input_{feature}",
                        help=f"模型訓練時記錄的類別值：{', '.join(choices[:8])}",
                    )
                else:
                    default = numeric_defaults[feature]
                    # 缺失標記和旗標欄位限制為 0/1，其餘連續量可自由輸入。
                    is_flag = feature.endswith("_Was_Missing") or feature in {"Is_Weekend"}
                    if is_flag:
                        input_row[feature] = st.selectbox(
                            label,
                            options=[0, 1],
                            index=int(default >= 0.5),
                            key=f"input_{feature}",
                        )
                    elif feature in {"Timestamp_Hour"}:
                        input_row[feature] = st.number_input(
                            label, min_value=0, max_value=23,
                            value=int(np.clip(round(default), 0, 23)), step=1,
                            key=f"input_{feature}",
                        )
                    elif feature == "Day_Of_Week":
                        input_row[feature] = st.number_input(
                            label, min_value=0, max_value=6,
                            value=int(np.clip(round(default), 0, 6)), step=1,
                            key=f"input_{feature}",
                            help="星期一為 0，星期日為 6。",
                        )
                    else:
                        input_row[feature] = st.number_input(
                            label,
                            value=default,
                            format="%.4f",
                            key=f"input_{feature}",
                        )
        submitted = st.form_submit_button(
            "執行異常預測",
            type="primary",
            icon=":material/search:",
            width="stretch",
        )

    if submitted:
        input_frame = pd.DataFrame([input_row], columns=feature_columns)
        try:
            predicted = int(model.predict(input_frame)[0])
            class_values = np.asarray(classes)
            anomaly_idx = np.flatnonzero(class_values == 1)
            anomaly_probability = None
            if len(anomaly_idx) and hasattr(model, "predict_proba"):
                anomaly_probability = float(model.predict_proba(input_frame)[0, anomaly_idx[0]])
        except Exception as exc:
            st.error(f"預測失敗：{exc}", title="模型推論錯誤")
            return

        if predicted == 1:
            st.error("模型判定：異常", title="預測結果")
        else:
            st.success("模型判定：正常", title="預測結果")
        if anomaly_probability is not None:
            st.metric("異常機率", f"{anomaly_probability:.1%}")
            st.progress(anomaly_probability, text="Random Forest 輸出的異常類別機率")
        st.caption("此結果為模型預測，請依現場品質規範與量測資訊判讀。")
        with st.expander("檢視送入模型的資料"):
            st.dataframe(input_frame, alt="送入模型的單筆製程特徵")


pages = [
    st.Page(render_overview, title="資料視覺化", icon=":material/monitoring:", default=True),
    st.Page(render_prediction, title="異常預測", icon=":material/psychology:")
]
selected_page = st.navigation(pages, position="sidebar")
with st.sidebar:
    st.caption("銅箔製程品質分析")
selected_page.run()
