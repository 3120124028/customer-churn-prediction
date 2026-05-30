from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "Data" / "churn_data_customer.csv"
MODEL_PATH = ROOT / "models" / "best_model.joblib"
TARGET_COL = "churn_target"

SUBSCRIPTION_MAP = {"Basic": 0, "Premium": 1, "Standard": 2}
PROMO_MAP = {"No": 0, "Yes": 1}

FEATURE_GROUPS = {
    "Thông tin cơ bản": [
        "age",
        "account_tenure_months",
        "subscription_type",
        "has_active_promo",
        "monthly_charges",
    ],
    "Tần suất và mức độ sử dụng": [
        "monthly_active_days",
        "avg_daily_usage_minutes",
        "num_sessions_per_week",
        "app_rating",
        "num_notification_opt_out",
        "customer_support_satisfaction",
    ],
    "Lịch sử và xu hướng": [
        "support_tickets_30d",
        "payment_delay_days",
        "avg_active_days_3m",
        "usage_minutes_3m",
        "support_tickets_3m",
        "avg_active_days_6m",
        "usage_minutes_6m",
        "support_tickets_6m",
        "avg_active_days_12m",
        "usage_minutes_12m",
        "support_tickets_12m",
    ],
}


st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def load_model_artifact() -> tuple[object, float, list[str]]:
    artifact = joblib.load(MODEL_PATH)
    return artifact["model"], float(artifact["threshold"]), list(artifact["feature_columns"])


@st.cache_data(show_spinner=False)
def load_reference_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def inject_css() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(59, 130, 246, 0.08), transparent 28%),
                    radial-gradient(circle at top right, rgba(16, 185, 129, 0.08), transparent 25%),
                    linear-gradient(180deg, #ffffff 0%, #f7fbff 100%);
            }
            .hero {
                background: linear-gradient(135deg, #ffffff 0%, #eff6ff 55%, #ecfeff 100%);
                color: #0f172a;
                border-radius: 24px;
                padding: 28px 30px;
                border: 1px solid rgba(148, 163, 184, 0.20);
                box-shadow: 0 18px 42px rgba(15, 23, 42, 0.08);
            }
            .hero h1 {
                margin: 0;
                font-size: 2.1rem;
                line-height: 1.1;
            }
            .hero p {
                margin: 0.6rem 0 0;
                opacity: 0.92;
                font-size: 1rem;
            }
            .prediction-box {
                border-radius: 20px;
                padding: 20px;
                color: #0f172a;
                box-shadow: 0 16px 36px rgba(15, 23, 42, 0.08);
                border: 1px solid rgba(148, 163, 184, 0.18);
            }
            .positive {
                background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
            }
            .negative {
                background: linear-gradient(135deg, #fff1f2 0%, #ffe4e6 100%);
            }
            .prediction-box h2, .prediction-box h3, .prediction-box p {
                margin: 0;
            }
            .small-note {
                font-size: 0.92rem;
                color: #475569;
            }
            .soft-card {
                background: rgba(255, 255, 255, 0.95);
                border: 1px solid rgba(148, 163, 184, 0.16);
                border-radius: 18px;
                padding: 18px;
                box-shadow: 0 12px 28px rgba(15, 23, 42, 0.05);
            }
            section[data-testid="stSidebar"] {
                background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
                border-right: 1px solid rgba(148, 163, 184, 0.18);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def pretty_label(column_name: str) -> str:
    return column_name.replace("_", " ").title()


def summarize_numeric(series: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(series, errors="coerce")
    return {"min": float(values.min()), "max": float(values.max()), "median": float(values.median())}


def preprocess_raw_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    if "customer_id" in df.columns:
        df = df.drop(columns=["customer_id"])
    if TARGET_COL in df.columns:
        df = df.drop(columns=[TARGET_COL])

    df["engagement_score"] = (
        (df["monthly_active_days"] / 30) * 0.4
        + np.clip(df["avg_daily_usage_minutes"], 0, 120) / 120 * 0.4
        + np.clip(df["num_sessions_per_week"], 0, 14) / 14 * 0.2
    ).round(4)
    df["satisfaction_score"] = (
        df["app_rating"] / 5 * 0.5 + df["customer_support_satisfaction"] / 5 * 0.5
    ).round(4)
    df["usage_trend_3m_6m"] = (
        (df["usage_minutes_3m"] - df["usage_minutes_6m"]) / (df["usage_minutes_6m"] + 1)
    ).round(4)

    df["subscription_type_enc"] = df["subscription_type"].map(SUBSCRIPTION_MAP)
    df["has_active_promo_enc"] = df["has_active_promo"].map(PROMO_MAP)
    df = df.drop(columns=["subscription_type", "has_active_promo"])
    return df


def align_features(raw_df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    prepared = preprocess_raw_frame(raw_df)
    for column in feature_columns:
        if column not in prepared.columns:
            prepared[column] = 0
    return prepared[feature_columns].copy()


def predict_churn(model: object, threshold: float, feature_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    proba = model.predict_proba(feature_df)[:, 1]
    pred = (proba >= threshold).astype(int)
    return pred, proba


def build_customer_input(reference_df: pd.DataFrame) -> pd.DataFrame | None:
    raw = reference_df.drop(columns=["customer_id", TARGET_COL]).copy()
    defaults: dict[str, object] = {}
    for column in raw.columns:
        if column in {"subscription_type", "has_active_promo"}:
            defaults[column] = raw[column].mode(dropna=True).iloc[0]
            continue
        stats = summarize_numeric(raw[column])
        if pd.api.types.is_integer_dtype(raw[column]):
            defaults[column] = int(round(stats["median"]))
        else:
            defaults[column] = float(stats["median"])

    with st.form("customer_input_form"):
        values: dict[str, object] = {}
        for group_name, columns in FEATURE_GROUPS.items():
            st.subheader(group_name)
            for row_start in range(0, len(columns), 3):
                row_cols = columns[row_start : row_start + 3]
                widgets = st.columns(len(row_cols))
                for widget, column in zip(widgets, row_cols):
                    with widget:
                        if column == "subscription_type":
                            options = ["Basic", "Premium", "Standard"]
                            values[column] = st.selectbox(
                                pretty_label(column),
                                options,
                                index=options.index(defaults[column]),
                            )
                        elif column == "has_active_promo":
                            options = ["No", "Yes"]
                            values[column] = st.selectbox(
                                pretty_label(column),
                                options,
                                index=options.index(defaults[column]),
                            )
                        else:
                            stats = summarize_numeric(raw[column])
                            min_value = float(stats["min"])
                            max_value = float(stats["max"])
                            if pd.api.types.is_integer_dtype(raw[column]):
                                values[column] = st.number_input(
                                    pretty_label(column),
                                    min_value=int(np.floor(min_value)),
                                    max_value=int(np.ceil(max_value)),
                                    value=int(defaults[column]),
                                    step=1,
                                )
                            else:
                                step = max(round((max_value - min_value) / 200, 2), 0.1)
                                values[column] = st.number_input(
                                    pretty_label(column),
                                    min_value=float(min_value),
                                    max_value=float(max_value),
                                    value=float(defaults[column]),
                                    step=step,
                                )

        submitted = st.form_submit_button("Dự đoán churn", use_container_width=True)

    if not submitted:
        return None

    return pd.DataFrame([values])


def make_indicator(probability: float, threshold: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=float(probability * 100),
            number={"suffix": "%"},
            delta={"reference": threshold * 100, "increasing": {"color": "#ef4444"}},
            title={"text": "Xác suất churn"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#2563eb"},
                "steps": [
                    {"range": [0, 35], "color": "#eff6ff"},
                    {"range": [35, 60], "color": "#dbeafe"},
                    {"range": [60, 100], "color": "#bfdbfe"},
                ],
                "threshold": {"line": {"color": "#2563eb", "width": 4}, "value": threshold * 100},
            },
        )
    )
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=55, b=20))
    return fig


def make_feature_importance_plot(model: object, feature_columns: list[str]) -> go.Figure:
    importances = pd.Series(getattr(model, "feature_importances_", np.zeros(len(feature_columns))), index=feature_columns)
    importance_df = importances.sort_values(ascending=True).reset_index()
    importance_df.columns = ["Feature", "Importance"]
    fig = px.bar(
        importance_df.tail(10),
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale=["#eff6ff", "#bfdbfe", "#60a5fa"],
        title="Top feature importance",
    )
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10), coloraxis_showscale=False)
    return fig


def compare_customer_to_reference(
    customer_row: pd.DataFrame,
    reference_features: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    ref_median = reference_features[feature_columns].median()
    customer = customer_row.iloc[0]
    rows = []
    for feature in feature_columns:
        rows.append(
            {
                "Feature": feature,
                "Customer": float(customer[feature]),
                "Median": float(ref_median[feature]),
                "Gap": float(customer[feature] - ref_median[feature]),
            }
        )
    return pd.DataFrame(rows)


def risk_label(probability: float, threshold: float) -> tuple[str, str]:
    if probability >= threshold:
        return "Nguy cơ cao", "negative"
    if probability >= threshold * 0.75:
        return "Can theo doi", "negative"
    return "Nguy cơ thấp", "positive"


inject_css()
model, threshold, feature_columns = load_model_artifact()
reference_df = load_reference_data()
reference_features = align_features(reference_df, feature_columns)

st.markdown(
    """
    <div class="hero">
        <h1>Churn Prediction Dashboard</h1>
        <p>Ứng dụng Streamlit tích hợp mô hình Machine Learning đã huấn luyện, hỗ trợ nhập dữ liệu, dự đoán và giải thích kết quả bằng biểu đồ trực quan.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")
col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Mô hình", type(model).__name__)
col_b.metric("Ngưỡng tối ưu", f"{threshold:.2f}")
col_c.metric("Số feature đầu vào", len(feature_columns))
col_d.metric("Mẫu tham chiếu", f"{len(reference_df):,}")

with st.sidebar:
    st.markdown("### Công Ty TAD")
    st.info(
        "Dự đoán khách hàng rời bỏ ứng dụng mua bán hàng"
    )
    st.markdown("---")
    st.metric("Threshold", f"{threshold:.2f}")
    st.metric("Feature model", len(feature_columns))
    st.metric("Loại model", type(model).__name__)

tab_predict, tab_batch, tab_explain = st.tabs(["Dự đoán 1 khách hàng", "Dự đoán hàng loạt", "Trực quan hoá và giải thích"])

with tab_predict:
    st.markdown("### Nhập thông tin khách hàng")
    customer_input = build_customer_input(reference_df)

    if customer_input is None:
        st.info("Hãy nhập các trường bên trên và bấm nút Dự đoán churn để xem kết quả.")
    else:
        customer_features = align_features(customer_input, feature_columns)
        prediction, probability = predict_churn(model, threshold, customer_features)
        churn_probability = float(probability[0])
        churn_pred = int(prediction[0])
        label, tone = risk_label(churn_probability, threshold)

        st.session_state["latest_probability"] = churn_probability
        st.session_state["latest_prediction"] = churn_pred
        st.session_state["latest_label"] = label

        top_left, top_right = st.columns([1.05, 1])
        with top_left:
            st.plotly_chart(make_indicator(churn_probability, threshold), use_container_width=True)
            box_class = "negative" if churn_pred == 1 else "positive"
            box_title = (
                "Dự đoán : KHÁCH HÀNG CÓ NGUY CƠ RỜI BỎ APP"
                if churn_pred == 1
                else "Dự đoán: KHÁCH HÀNG ỔN ĐỊNH"
            )
            action_text = (
                "Khuyến nghị CSKH chủ động: ưu tiên chăm sóc, nhận voucher và điều chỉnh nội dung giữ chân khách hàng."
                if churn_pred == 1
                else "Khách hàng đang ổn định, có thể tiếp tục theo dõi hành vi sử dụng và mời ưu đãi nếu có cơ hội phù hợp."
            )
            st.markdown(
                f"""
                <div class="prediction-box {box_class}">
                    <h2>{box_title}</h2>
                    <p style="margin-top: 10px; font-size: 1.08rem;">Xác suất churn: <strong>{churn_probability:.2%}</strong></p>
                    <p style="margin-top: 8px; font-size: 1.0rem;">{action_text}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with top_right:
            st.markdown("### So sánh với mẫu tham chiếu")
            compare_df = compare_customer_to_reference(customer_features, reference_features, feature_columns)
            key_features = compare_df.sort_values("Gap", key=lambda s: s.abs(), ascending=False).head(6)
            chart_df = pd.concat(
                [
                    key_features.assign(Series="Khach hang", Value=key_features["Customer"])[["Feature", "Series", "Value"]],
                    key_features.assign(Series="Trung vi dataset", Value=key_features["Median"])[["Feature", "Series", "Value"]],
                ],
                ignore_index=True,
            )
            feature_chart = px.bar(
                chart_df,
                x="Feature",
                y="Value",
                color="Series",
                barmode="group",
                color_discrete_map={"Khach hang": "#2563eb", "Trung vi dataset": "#94a3b8"},
                title="Top feature so voi dataset median",
            )
            feature_chart.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(feature_chart, use_container_width=True)

        st.markdown("### Giải thích nguyên nhân")
        importance_series = pd.Series(getattr(model, "feature_importances_", np.zeros(len(feature_columns))), index=feature_columns)
        explanation_df = compare_df.assign(Importance=importance_series.reindex(compare_df["Feature"]).values)
        explanation_df = explanation_df.sort_values("Importance", ascending=False).head(8)
        explanation_df["Xu huong"] = np.where(
            explanation_df["Gap"] > 0,
            "cao hơn median",
            np.where(explanation_df["Gap"] < 0, "thấp hơn median", "bằng median"),
        )
        st.dataframe(
            explanation_df[["Feature", "Customer", "Median", "Gap", "Importance", "Xu huong"]].round(4),
            use_container_width=True,
            hide_index=True,
        )

with tab_batch:
    st.markdown("### Dự đoán hàng loạt từ file CSV")
    uploaded_file = st.file_uploader("Tải lên file CSV", type=["csv"])
    if uploaded_file is not None:
        batch_raw = pd.read_csv(uploaded_file)
        batch_features = align_features(batch_raw, feature_columns)
        batch_pred, batch_proba = predict_churn(model, threshold, batch_features)

        batch_output = batch_raw.copy()
        batch_output["churn_probability"] = batch_proba
        batch_output["churn_prediction"] = batch_pred
        batch_output["risk_label"] = np.where(batch_pred == 1, "High risk", "Low risk")

        st.success(f"Đã tạo dự đoán cho {len(batch_output):,} dòng dữ liệu.")
        st.dataframe(batch_output.head(20), use_container_width=True)
        st.download_button(
            "Tải kết quả CSV",
            batch_output.to_csv(index=False).encode("utf-8"),
            file_name="churn_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.info("“Nếu bạn muốn phân tích nhiều khách hàng cùng lúc, hãy tải lên file CSV có cùng schema với dataset gốc.")

with tab_explain:
    st.markdown("### Mô hình và các biểu đồ giải thích")
    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(make_feature_importance_plot(model, feature_columns), use_container_width=True)
    with right:
        st.markdown("#### Khuôn mẫu đánh giá nhanh")
        latest_probability = st.session_state.get("latest_probability")
        latest_prediction = st.session_state.get("latest_prediction")
        latest_label = st.session_state.get("latest_label")
        quick_rows = pd.DataFrame(
            {
                "Chỉ số": ["Xác suất churn", "Threshold", "Nhóm rủi ro"],
                "Giá trị": [
                    f"{latest_probability:.2%}" if latest_probability is not None else "-",
                    f"{threshold:.2f}",
                    latest_label if latest_label is not None else "-",
                ],
            }
        )
        st.dataframe(quick_rows, use_container_width=True, hide_index=True)
        st.markdown(
            """
            <p class="small-note">
                Biểu đồ Feature Importance cho thấy những đặc trưng ảnh hưởng mạnh nhất đến mô hình.
                Bảng so sánh giúp hiểu vị trí của khách hàng hiện tại so với giá trị trung vị của dữ liệu gốc.
            </p>
            """,
            unsafe_allow_html=True,
        )
        if latest_prediction is not None:
            st.caption("Kết quả gần nhất đã được lưu trong session để bạn đối chiếu nhanh.")
