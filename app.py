from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Rappi Competitive Intelligence Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
)


ROOT = Path(__file__).resolve().parent
RESOURCES = ROOT / "Resources"
DATA_PATH = next(RESOURCES.glob("*.xlsx"))

ROLLING_WEEKS = [f"L{i}W_ROLL" for i in range(8, -1, -1)]
ORDERS_WEEKS = [f"L{i}W" for i in range(8, -1, -1)]
LATEST_ROLL = "L0W_ROLL"
LATEST_ORDERS = "L0W"


def normalize_zone_type(value: str) -> str:
    if pd.isna(value):
        return value
    mapping = {
        "Non Wealthy": "Non-Wealthy",
        "Non-Wealthy": "Non-Wealthy",
        "Mixed": "Mixed",
        "Wealthy": "Wealthy",
    }
    return mapping.get(str(value), str(value))


def weighted_average(frame: pd.DataFrame, value_col: str, weight_col: str = "ORDERS") -> float:
    valid = frame[[value_col, weight_col]].dropna()
    if valid.empty or valid[weight_col].sum() == 0:
        return np.nan
    return float(np.average(valid[value_col], weights=valid[weight_col]))


def zscore(series: pd.Series, invert: bool = False) -> pd.Series:
    std = series.std(ddof=0)
    if pd.isna(std) or std == 0:
        scored = pd.Series(0.0, index=series.index)
    else:
        scored = (series - series.mean()) / std
    return -scored if invert else scored


@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_excel(DATA_PATH, sheet_name="Input_metrics")
    orders = pd.read_excel(DATA_PATH, sheet_name="Orders")
    metrics["ZONE_TYPE_STD"] = metrics["ZONE_TYPE"].map(normalize_zone_type)
    return metrics, orders


@st.cache_data(show_spinner=False)
def build_views() -> dict[str, pd.DataFrame]:
    metrics, orders = load_data()

    fee_metrics = ["Delivery Fee (MXN)", "Service Fee (%)"]
    rappi_price_metrics = ["Restaurants Markdowns / GMV", "Gross Profit UE"]
    ops_metrics = [
        "Perfect Orders",
        "Restaurants SST > SS CVR",
        "Retail SST > SS CVR",
        "Turbo Adoption",
        "Lead Penetration",
        "Non-Pro PTC > OP",
    ]
    geo_metrics = rappi_price_metrics + ops_metrics

    fee_snapshot = (
        metrics.loc[
            metrics["COMPETITOR"].isin(["Uber Eats", "DiDi Food"])
            & metrics["METRIC"].isin(fee_metrics),
            ["COMPETITOR", "ZONE", "ZONE_TYPE_STD", "METRIC", LATEST_ROLL],
        ]
        .pivot_table(
            index=["COMPETITOR", "ZONE", "ZONE_TYPE_STD"],
            columns="METRIC",
            values=LATEST_ROLL,
        )
        .reset_index()
    )
    fee_snapshot = fee_snapshot.merge(
        orders.loc[
            orders["COMPETITOR"].isin(["Uber Eats", "DiDi Food"]),
            ["COMPETITOR", "ZONE", LATEST_ORDERS],
        ].rename(columns={LATEST_ORDERS: "ORDERS"}),
        on=["COMPETITOR", "ZONE"],
        how="left",
    )
    for basket in [150, 200]:
        fee_snapshot[f"Effective Fee {basket}"] = (
            fee_snapshot["Delivery Fee (MXN)"] + basket * fee_snapshot["Service Fee (%)"]
        )

    time_snapshot = metrics.loc[
        metrics["COMPETITOR"].isin(["Uber Eats", "DiDi Food"])
        & (metrics["METRIC"] == "Avg Delivery Time (mins)"),
        ["COMPETITOR", "ZONE", "ZONE_TYPE_STD", LATEST_ROLL],
    ].rename(columns={LATEST_ROLL: "Avg Delivery Time (mins)"})
    time_snapshot = time_snapshot.merge(
        orders.loc[
            orders["COMPETITOR"].isin(["Uber Eats", "DiDi Food"]),
            ["COMPETITOR", "ZONE", LATEST_ORDERS],
        ].rename(columns={LATEST_ORDERS: "ORDERS"}),
        on=["COMPETITOR", "ZONE"],
        how="left",
    )

    rappi_price_snapshot = (
        metrics.loc[
            (metrics["COMPETITOR"] == "Rappi") & metrics["METRIC"].isin(rappi_price_metrics),
            ["ZONE", "ZONE_TYPE_STD", "METRIC", LATEST_ROLL],
        ]
        .pivot_table(index=["ZONE", "ZONE_TYPE_STD"], columns="METRIC", values=LATEST_ROLL)
        .reset_index()
    )
    rappi_price_snapshot["COMPETITOR"] = "Rappi"
    rappi_price_snapshot = rappi_price_snapshot.merge(
        orders.loc[orders["COMPETITOR"] == "Rappi", ["ZONE", LATEST_ORDERS]].rename(
            columns={LATEST_ORDERS: "ORDERS"}
        ),
        on="ZONE",
        how="left",
    )

    rappi_ops_snapshot = (
        metrics.loc[
            (metrics["COMPETITOR"] == "Rappi") & metrics["METRIC"].isin(ops_metrics),
            ["ZONE", "ZONE_TYPE_STD", "METRIC", LATEST_ROLL],
        ]
        .pivot_table(index=["ZONE", "ZONE_TYPE_STD"], columns="METRIC", values=LATEST_ROLL)
        .reset_index()
    )
    rappi_ops_snapshot["COMPETITOR"] = "Rappi"
    rappi_ops_snapshot = rappi_ops_snapshot.merge(
        orders.loc[orders["COMPETITOR"] == "Rappi", ["ZONE", LATEST_ORDERS]].rename(
            columns={LATEST_ORDERS: "ORDERS"}
        ),
        on="ZONE",
        how="left",
    )
    rappi_ops_snapshot["operational_health_score"] = (
        zscore(rappi_ops_snapshot["Perfect Orders"])
        + zscore(rappi_ops_snapshot["Restaurants SST > SS CVR"])
        + zscore(rappi_ops_snapshot["Retail SST > SS CVR"])
        + zscore(rappi_ops_snapshot["Turbo Adoption"])
        + zscore(rappi_ops_snapshot["Lead Penetration"])
        + zscore(rappi_ops_snapshot["Non-Pro PTC > OP"])
    ) / 6

    rappi_geo = (
        metrics.loc[
            (metrics["COMPETITOR"] == "Rappi") & metrics["METRIC"].isin(geo_metrics),
            ["ZONE", "ZONE_TYPE_STD", "METRIC", LATEST_ROLL],
        ]
        .pivot_table(index=["ZONE", "ZONE_TYPE_STD"], columns="METRIC", values=LATEST_ROLL)
        .reset_index()
    )
    rappi_geo["COMPETITOR"] = "Rappi"
    rappi_geo = rappi_geo.merge(
        orders.loc[orders["COMPETITOR"] == "Rappi", ["ZONE", LATEST_ORDERS]].rename(
            columns={LATEST_ORDERS: "ORDERS"}
        ),
        on="ZONE",
        how="left",
    )
    rappi_geo["pricing_competitiveness_score"] = (
        zscore(rappi_geo["Restaurants Markdowns / GMV"], invert=True)
        + zscore(rappi_geo["Gross Profit UE"])
    ) / 2
    rappi_geo["operational_competitiveness_score"] = (
        zscore(rappi_geo["Perfect Orders"])
        + zscore(rappi_geo["Restaurants SST > SS CVR"])
        + zscore(rappi_geo["Retail SST > SS CVR"])
        + zscore(rappi_geo["Turbo Adoption"])
        + zscore(rappi_geo["Lead Penetration"])
        + zscore(rappi_geo["Non-Pro PTC > OP"])
    ) / 6
    rappi_geo["overall_competitiveness_score"] = (
        rappi_geo["pricing_competitiveness_score"] + rappi_geo["operational_competitiveness_score"]
    ) / 2
    rappi_geo["competitive_quadrant"] = [
        f"{'High' if pricing >= 0 else 'Low'} price proxy / "
        f"{'High' if ops >= 0 else 'Low'} ops"
        for pricing, ops in zip(
            rappi_geo["pricing_competitiveness_score"],
            rappi_geo["operational_competitiveness_score"],
        )
    ]

    fee_trend_rows = []
    for roll_week, orders_week in zip(ROLLING_WEEKS, ORDERS_WEEKS):
        week_frame = (
            metrics.loc[
                metrics["COMPETITOR"].isin(["Uber Eats", "DiDi Food"])
                & metrics["METRIC"].isin(fee_metrics),
                ["COMPETITOR", "ZONE", "METRIC", roll_week],
            ]
            .pivot_table(index=["COMPETITOR", "ZONE"], columns="METRIC", values=roll_week)
            .reset_index()
        )
        week_frame = week_frame.merge(
            orders.loc[
                orders["COMPETITOR"].isin(["Uber Eats", "DiDi Food"]),
                ["COMPETITOR", "ZONE", orders_week],
            ].rename(columns={orders_week: "ORDERS"}),
            on=["COMPETITOR", "ZONE"],
            how="left",
        )
        week_frame["Effective Fee 150"] = (
            week_frame["Delivery Fee (MXN)"] + 150 * week_frame["Service Fee (%)"]
        )
        for competitor, group in week_frame.groupby("COMPETITOR"):
            fee_trend_rows.append(
                {
                    "week": roll_week,
                    "COMPETITOR": competitor,
                    "Delivery Fee (MXN)": weighted_average(group, "Delivery Fee (MXN)"),
                    "Service Fee (%)": weighted_average(group, "Service Fee (%)"),
                    "Effective Fee 150": weighted_average(group, "Effective Fee 150"),
                }
            )
    fee_trend = pd.DataFrame(fee_trend_rows)

    rappi_promo_rows = []
    for roll_week, orders_week in zip(ROLLING_WEEKS, ORDERS_WEEKS):
        week_frame = (
            metrics.loc[
                (metrics["COMPETITOR"] == "Rappi")
                & metrics["METRIC"].isin(rappi_price_metrics),
                ["ZONE", "METRIC", roll_week],
            ]
            .pivot_table(index="ZONE", columns="METRIC", values=roll_week)
            .reset_index()
        )
        week_frame = week_frame.merge(
            orders.loc[orders["COMPETITOR"] == "Rappi", ["ZONE", orders_week]].rename(
                columns={orders_week: "ORDERS"}
            ),
            on="ZONE",
            how="left",
        )
        rappi_promo_rows.append(
            {
                "week": roll_week,
                "COMPETITOR": "Rappi",
                "Restaurants Markdowns / GMV": weighted_average(
                    week_frame, "Restaurants Markdowns / GMV"
                ),
                "Gross Profit UE": weighted_average(week_frame, "Gross Profit UE"),
            }
        )
    rappi_promo_trend = pd.DataFrame(rappi_promo_rows)

    return {
        "fee_snapshot": fee_snapshot,
        "time_snapshot": time_snapshot,
        "rappi_price_snapshot": rappi_price_snapshot,
        "rappi_ops_snapshot": rappi_ops_snapshot,
        "rappi_geo": rappi_geo,
        "fee_trend": fee_trend,
        "rappi_promo_trend": rappi_promo_trend,
    }


def available_zones(views: dict[str, pd.DataFrame], selected_competitors: list[str], zone_types: list[str]) -> list[str]:
    frames = [
        views["fee_snapshot"][["COMPETITOR", "ZONE", "ZONE_TYPE_STD"]],
        views["time_snapshot"][["COMPETITOR", "ZONE", "ZONE_TYPE_STD"]],
        views["rappi_price_snapshot"][["COMPETITOR", "ZONE", "ZONE_TYPE_STD"]],
        views["rappi_ops_snapshot"][["COMPETITOR", "ZONE", "ZONE_TYPE_STD"]],
        views["rappi_geo"][["COMPETITOR", "ZONE", "ZONE_TYPE_STD"]],
    ]
    combined = pd.concat(frames, ignore_index=True).drop_duplicates()
    filtered = combined[combined["COMPETITOR"].isin(selected_competitors)]
    filtered = filtered[filtered["ZONE_TYPE_STD"].isin(zone_types)]
    return sorted(filtered["ZONE"].dropna().unique().tolist())


def apply_common_filters(
    frame: pd.DataFrame, selected_competitors: list[str], selected_zone_types: list[str], selected_zones: list[str]
) -> pd.DataFrame:
    filtered = frame.copy()
    if "COMPETITOR" in filtered.columns:
        filtered = filtered[filtered["COMPETITOR"].isin(selected_competitors)]
    if "ZONE_TYPE_STD" in filtered.columns:
        filtered = filtered[filtered["ZONE_TYPE_STD"].isin(selected_zone_types)]
    if "ZONE" in filtered.columns:
        filtered = filtered[filtered["ZONE"].isin(selected_zones)]
    return filtered


def metric_card_row(metrics_map: list[tuple[str, str]]) -> None:
    cols = st.columns(len(metrics_map))
    for col, (label, value) in zip(cols, metrics_map):
        col.metric(label, value)


def format_currency(value: float) -> str:
    return "N/A" if pd.isna(value) else f"${value:,.2f}"


def format_percent(value: float) -> str:
    return "N/A" if pd.isna(value) else f"{value:.1%}"


views = build_views()

st.title("Rappi Competitive Intelligence Dashboard")
st.caption(f"Fuente: `{DATA_PATH.name}`")
st.info(
    "Metodología: Uber Eats y DiDi Food tienen métricas directas de fees y tiempos. "
    "Rappi entra parcialmente con proxies en precio, promoción, operación y geografía."
)


all_competitors = ["Rappi", "Uber Eats", "DiDi Food"]
all_zone_types = ["Wealthy", "Mixed", "Non-Wealthy"]

st.sidebar.header("Filtros")
selected_competitors = st.sidebar.multiselect(
    "Competidor",
    options=all_competitors,
    default=all_competitors,
)
if not selected_competitors:
    selected_competitors = all_competitors

selected_zone_types = st.sidebar.multiselect(
    "Tipo de zona",
    options=all_zone_types,
    default=all_zone_types,
)
if not selected_zone_types:
    selected_zone_types = all_zone_types

zone_options = available_zones(views, selected_competitors, selected_zone_types)
selected_zones = st.sidebar.multiselect(
    "Zona",
    options=zone_options,
    default=zone_options,
)
if not selected_zones:
    selected_zones = zone_options


tab_prices, tab_ops, tab_promo, tab_geo = st.tabs(
    ["1. Precios", "2. Operación", "3. Promoción", "4. Geografía"]
)


with tab_prices:
    st.subheader("Posicionamiento de precios")
    fee_view = apply_common_filters(
        views["fee_snapshot"], selected_competitors, selected_zone_types, selected_zones
    )
    rappi_price_view = apply_common_filters(
        views["rappi_price_snapshot"], selected_competitors, selected_zone_types, selected_zones
    )

    direct_competitors = [c for c in ["Uber Eats", "DiDi Food"] if c in selected_competitors]
    if fee_view.empty:
        st.info("No hay datos directos de fees para la selección actual.")
    else:
        summary_rows = []
        for competitor, group in fee_view.groupby("COMPETITOR"):
            summary_rows.append(
                {
                    "COMPETITOR": competitor,
                    "Delivery Fee": weighted_average(group, "Delivery Fee (MXN)"),
                    "Service Fee": weighted_average(group, "Service Fee (%)"),
                    "Effective Fee 150": weighted_average(group, "Effective Fee 150"),
                    "Effective Fee 200": weighted_average(group, "Effective Fee 200"),
                }
            )
        fee_summary = pd.DataFrame(summary_rows).sort_values("COMPETITOR")
        if not fee_summary.empty:
            metric_card_row(
                [
                    (
                        f"{row['COMPETITOR']} Delivery",
                        format_currency(row["Delivery Fee"]),
                    )
                    for _, row in fee_summary.iterrows()
                ]
            )
            metric_card_row(
                [
                    (
                        f"{row['COMPETITOR']} Effective @150",
                        format_currency(row["Effective Fee 150"]),
                    )
                    for _, row in fee_summary.iterrows()
                ]
            )

            plot_summary = fee_summary.melt(
                id_vars="COMPETITOR",
                value_vars=["Delivery Fee", "Service Fee", "Effective Fee 150", "Effective Fee 200"],
                var_name="metric",
                value_name="value",
            )
            fig = px.bar(
                plot_summary,
                x="metric",
                y="value",
                color="COMPETITOR",
                barmode="group",
                title="Estructura de fees ponderada por órdenes",
                text_auto=".2f",
            )
            fig.update_layout(template="plotly_white", yaxis_title="Valor")
            st.plotly_chart(fig, use_container_width=True)

        if set(direct_competitors) == {"Uber Eats", "DiDi Food"}:
            zone_gap = (
                fee_view.pivot_table(
                    index=["ZONE", "ZONE_TYPE_STD"],
                    columns="COMPETITOR",
                    values="Effective Fee 150",
                )
                .dropna()
                .reset_index()
            )
            if not zone_gap.empty:
                zone_gap["gap_uber_minus_didi"] = zone_gap["Uber Eats"] - zone_gap["DiDi Food"]
                zone_gap["cheapest"] = np.where(
                    zone_gap["gap_uber_minus_didi"] < 0, "Uber Eats", "DiDi Food"
                )
                fig = px.bar(
                    zone_gap.sort_values("gap_uber_minus_didi"),
                    x="gap_uber_minus_didi",
                    y="ZONE",
                    color="cheapest",
                    orientation="h",
                    title="Gap de fee efectivo por zona (Uber Eats - DiDi Food)",
                    labels={"gap_uber_minus_didi": "MXN", "ZONE": "Zona"},
                )
                fig.add_vline(x=0, line_dash="dash", line_color="black")
                fig.update_layout(template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
    if "Rappi" in selected_competitors:
        if rappi_price_view.empty:
            st.info("Rappi no tiene datos visibles de precio/proxy para esta selección.")
        else:
            st.warning(
                "Rappi no tiene Delivery Fee ni Service Fee directos en el dataset. "
                "Se muestra competitividad de precio vía subsidio y rentabilidad."
            )
            fig = px.scatter(
                rappi_price_view,
                x="Restaurants Markdowns / GMV",
                y="Gross Profit UE",
                size="ORDERS",
                color="ZONE_TYPE_STD",
                hover_name="ZONE",
                title="Rappi: intensidad promocional vs rentabilidad unitaria",
            )
            fig.update_layout(template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(
                rappi_price_view[
                    [
                        "ZONE",
                        "ZONE_TYPE_STD",
                        "Restaurants Markdowns / GMV",
                        "Gross Profit UE",
                        "ORDERS",
                    ]
                ].sort_values("Restaurants Markdowns / GMV", ascending=False),
                use_container_width=True,
            )


with tab_ops:
    st.subheader("Ventaja / desventaja operacional")
    time_view = apply_common_filters(
        views["time_snapshot"], selected_competitors, selected_zone_types, selected_zones
    )
    rappi_ops_view = apply_common_filters(
        views["rappi_ops_snapshot"], selected_competitors, selected_zone_types, selected_zones
    )

    if time_view.empty:
        st.info("No hay datos directos de tiempos de entrega para la selección actual.")
    else:
        time_summary = pd.DataFrame(
            [
                {
                    "COMPETITOR": competitor,
                    "Weighted Time": weighted_average(group, "Avg Delivery Time (mins)"),
                }
                for competitor, group in time_view.groupby("COMPETITOR")
            ]
        ).sort_values("COMPETITOR")
        metric_card_row(
            [
                (f"{row['COMPETITOR']} Delivery Time", f"{row['Weighted Time']:.1f} min")
                for _, row in time_summary.iterrows()
            ]
        )
        fig = px.bar(
            time_summary,
            x="COMPETITOR",
            y="Weighted Time",
            color="COMPETITOR",
            title="Tiempo de entrega ponderado por órdenes",
            text_auto=".1f",
        )
        fig.update_layout(template="plotly_white", yaxis_title="Minutos")
        st.plotly_chart(fig, use_container_width=True)

        if set(direct_competitors) == {"Uber Eats", "DiDi Food"}:
            zone_time = (
                time_view.pivot_table(
                    index=["ZONE", "ZONE_TYPE_STD"],
                    columns="COMPETITOR",
                    values="Avg Delivery Time (mins)",
                )
                .dropna()
                .reset_index()
            )
            zone_time["gap_uber_minus_didi"] = zone_time["Uber Eats"] - zone_time["DiDi Food"]
            zone_time["faster"] = np.where(
                zone_time["gap_uber_minus_didi"] < 0, "Uber Eats", "DiDi Food"
            )
            fig = px.bar(
                zone_time.sort_values("gap_uber_minus_didi"),
                x="gap_uber_minus_didi",
                y="ZONE",
                color="faster",
                orientation="h",
                title="Gap de tiempo por zona (Uber Eats - DiDi Food)",
                labels={"gap_uber_minus_didi": "Minutos", "ZONE": "Zona"},
            )
            fig.add_vline(x=0, line_dash="dash", line_color="black")
            fig.update_layout(template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

    if "Rappi" in selected_competitors:
        if rappi_ops_view.empty:
            st.info("Rappi no tiene proxies operativos visibles para esta selección.")
        else:
            st.warning(
                "Rappi no tiene `Avg Delivery Time (mins)` en el dataset. "
                "Se muestra salud operacional proxy."
            )
            fig = px.scatter(
                rappi_ops_view,
                x="Turbo Adoption",
                y="Perfect Orders",
                size="ORDERS",
                color="ZONE_TYPE_STD",
                hover_name="ZONE",
                title="Rappi: mapa indirecto de salud operacional",
            )
            fig.update_layout(template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(
                rappi_ops_view[
                    [
                        "ZONE",
                        "ZONE_TYPE_STD",
                        "operational_health_score",
                        "Perfect Orders",
                        "Turbo Adoption",
                        "Lead Penetration",
                        "ORDERS",
                    ]
                ].sort_values("operational_health_score"),
                use_container_width=True,
            )


with tab_promo:
    st.subheader("Estrategia promocional")
    fee_trend_view = views["fee_trend"].copy()
    if direct_competitors:
        fee_trend_view = fee_trend_view[fee_trend_view["COMPETITOR"].isin(direct_competitors)]
    if not fee_trend_view.empty:
        fig = px.line(
            fee_trend_view,
            x="week",
            y="Effective Fee 150",
            color="COMPETITOR",
            markers=True,
            title="Tendencia del fee efectivo ponderado por órdenes",
        )
        fig.update_layout(template="plotly_white", yaxis_title="MXN")
        st.plotly_chart(fig, use_container_width=True)
        fee_vol = (
            fee_trend_view.groupby("COMPETITOR")["Effective Fee 150"]
            .agg(["mean", "std", "min", "max"])
            .reset_index()
        )
        st.dataframe(fee_vol, use_container_width=True)
    else:
        st.info("No hay competidores con fee observable en la selección actual.")

    if "Rappi" in selected_competitors:
        rappi_promo_trend = views["rappi_promo_trend"].copy()
        rappi_promo_snapshot = apply_common_filters(
            views["rappi_price_snapshot"], ["Rappi"], selected_zone_types, selected_zones
        )
        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=rappi_promo_trend["week"],
                    y=rappi_promo_trend["Restaurants Markdowns / GMV"],
                    mode="lines+markers",
                    name="Markdowns / GMV",
                    yaxis="y1",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=rappi_promo_trend["week"],
                    y=rappi_promo_trend["Gross Profit UE"],
                    mode="lines+markers",
                    name="Gross Profit UE",
                    yaxis="y2",
                )
            )
            fig.update_layout(
                title="Rappi: subsidio vs rentabilidad",
                template="plotly_white",
                yaxis=dict(title="Markdowns / GMV"),
                yaxis2=dict(title="Gross Profit UE", overlaying="y", side="right"),
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            if rappi_promo_snapshot.empty:
                st.info("Rappi no tiene datos promocionales visibles para esta selección.")
            else:
                fig = px.scatter(
                    rappi_promo_snapshot,
                    x="Restaurants Markdowns / GMV",
                    y="Gross Profit UE",
                    size="ORDERS",
                    color="ZONE_TYPE_STD",
                    hover_name="ZONE",
                    title="Rappi: zonas con mayor presión promocional",
                )
                fig.update_layout(template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)


with tab_geo:
    st.subheader("Variabilidad geográfica")
    if "Rappi" not in selected_competitors:
        st.info("Esta vista es Rappi-céntrica. Selecciona Rappi para ver competitividad geográfica.")
    else:
        geo_view = apply_common_filters(
            views["rappi_geo"], ["Rappi"], selected_zone_types, selected_zones
        )
        if geo_view.empty:
            st.info("No hay datos geográficos de Rappi para la selección actual.")
        else:
            zone_type_summary = (
                geo_view.groupby("ZONE_TYPE_STD")
                .apply(
                    lambda g: pd.Series(
                        {
                            "pricing_score": g["pricing_competitiveness_score"].mean(),
                            "ops_score": g["operational_competitiveness_score"].mean(),
                            "overall_score": weighted_average(
                                g.rename(columns={"overall_competitiveness_score": "value"}),
                                "value",
                            ),
                        }
                    )
                )
                .reset_index()
            )
            metric_card_row(
                [
                    (
                        "Best Zone",
                        geo_view.sort_values("overall_competitiveness_score", ascending=False)
                        .iloc[0]["ZONE"],
                    ),
                    (
                        "Weakest Zone",
                        geo_view.sort_values("overall_competitiveness_score")
                        .iloc[0]["ZONE"],
                    ),
                    (
                        "Weighted Score",
                        f"{weighted_average(geo_view.rename(columns={'overall_competitiveness_score': 'value'}), 'value'):.2f}",
                    ),
                ]
            )

            col1, col2 = st.columns(2)
            with col1:
                fig = px.scatter(
                    geo_view,
                    x="pricing_competitiveness_score",
                    y="operational_competitiveness_score",
                    size="ORDERS",
                    color="ZONE_TYPE_STD",
                    hover_name="ZONE",
                    title="Mapa de competitividad por zona",
                )
                fig.add_vline(x=0, line_dash="dash", line_color="black")
                fig.add_hline(y=0, line_dash="dash", line_color="black")
                fig.update_layout(template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.bar(
                    geo_view.sort_values("overall_competitiveness_score"),
                    x="overall_competitiveness_score",
                    y="ZONE",
                    color="ZONE_TYPE_STD",
                    orientation="h",
                    title="Ranking de competitividad total por zona",
                )
                fig.add_vline(x=0, line_dash="dash", line_color="black")
                fig.update_layout(template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                zone_type_summary.sort_values("overall_score", ascending=False),
                use_container_width=True,
            )
            st.dataframe(
                geo_view[
                    [
                        "ZONE",
                        "ZONE_TYPE_STD",
                        "ORDERS",
                        "pricing_competitiveness_score",
                        "operational_competitiveness_score",
                        "overall_competitiveness_score",
                        "competitive_quadrant",
                    ]
                ].sort_values("overall_competitiveness_score", ascending=False),
                use_container_width=True,
            )
