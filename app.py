import streamlit as st
import pandas as pd
import plotly.express as px
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
import re
import numpy as np
import plotly.graph_objects as go
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from geopy.distance import geodesic
import asyncio
from playwright.sync_api import sync_playwright
import os
import math

st.set_page_config(page_title="Terminal Report", layout="wide")

def dms_to_dd(dms_str):
    """Convert DMS (e.g. 22°41'44.84"N) or decimal string to float."""
    if pd.isna(dms_str):
        return None
    dms_str = str(dms_str).strip()
    try:
        return float(dms_str)  
    except ValueError:
        pass
    regex = r"(\d+)°(\d+)'([\d.]+)\"?([NSEW])"
    match = re.match(regex, dms_str)
    if not match:
        return None
    degrees, minutes, seconds, direction = match.groups()
    dd = float(degrees) + float(minutes) / 60 + float(seconds) / 3600
    if direction in ['S', 'W']:
        dd *= -1
    return dd

def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance (km) between two points, vectorized."""
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))

@st.cache_data
def load_excel_data():
    """Load base Excel data (analytical part)."""
    data = pd.read_excel("port-data.xlsx")
    data['Departure Date'] = pd.to_datetime(data['Departure Date'], errors='coerce')
    data['Arrival Date'] = pd.to_datetime(data['Arrival Date'], errors='coerce')
    return data

@st.cache_data
def load_terminal_data():
    """Load terminal coordinates."""
    df = pd.read_excel("terminal-data.xlsx")
    df['Latitude'] = df['Latitude'].apply(dms_to_dd)
    df['Longitude'] = df['Longitude'].apply(dms_to_dd)
    df = df.dropna(subset=['Latitude', 'Longitude'])
    return df

@st.cache_data
def load_project_data():
    """Load project coordinates."""
    df_project = pd.read_excel("project-data.xlsx")
    df_project['Latitude'] = df_project['Latitude'].apply(dms_to_dd)
    df_project['Longitude'] = df_project['Longitude'].apply(dms_to_dd)
    df_project = df_project.dropna(subset=['Latitude', 'Longitude'])
    return df_project

@st.cache_data
def load_lad_data():
    """Load project coordinates."""
    df_lad = pd.read_excel("lad-data.xlsx")
    df_lad['Latitude'] = df_lad['Latitude'].apply(dms_to_dd)
    df_lad['Longitude'] = df_lad['Longitude'].apply(dms_to_dd)
    df_lad = df_lad.dropna(subset=['Latitude', 'Longitude'])
    return df_lad

@st.cache_data
def attach_cargo_weights(data, df):
    """Attach cargo weight to terminals for geospatial heatmaps."""
    cargo_weights = (
        pd.concat([
            data[["Origin Terminal", "Quantity (MT)"]].rename(columns={"Origin Terminal": "Terminal"}),
            data[["Destination Terminal", "Quantity (MT)"]].rename(columns={"Destination Terminal": "Terminal"})
        ])
        .groupby("Terminal")["Quantity (MT)"].sum()
    )
    df["CargoWeight"] = df["Terminal Name"].map(cargo_weights).fillna(1)
    return df

@st.cache_data
def generate_pdf(tab_name, kpis=None, figures=None):
    """Generate PDF report (cached)."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph(f"IWAI Report - {tab_name}", styles["Title"]))
    elements.append(Spacer(1, 12))

    if kpis:
        if isinstance(kpis, dict):
            for k, v in kpis.items():
                elements.append(Paragraph(f"<b>{k}:</b> {v}", styles["Normal"]))
        elif isinstance(kpis, list):
            for item in kpis:
                elements.append(Paragraph(str(item), styles["Normal"]))
        elements.append(Spacer(1, 12))

    if figures:
        for fig in figures:
            img_bytes = fig.to_image(format="png")
            img_buffer = BytesIO(img_bytes)
            elements.append(Image(img_buffer, width=450, height=250))
            elements.append(Spacer(1, 12))

    doc.build(elements)
    buffer.seek(0)
    return buffer

@st.cache_data
def get_heatmap_data(click_lat, click_lon, df, selected_range):
    """Subset terminals within range of click."""
    distances = haversine(click_lat, click_lon, df["Latitude"].values, df["Longitude"].values)
    mask = distances <= selected_range
    return df.loc[mask, ["Latitude", "Longitude", "CargoWeight"]].values.tolist()

report_type = "Analytical Reports"
col1, col2 = st.columns([2,5])
with col1:
    report_type = st.selectbox("Choose Report Type", ["Analytical Reports", "Geospatial Reports"])
with col2:
    st.title("Dashboard")

# ----------------- ANALYTICAL REPORTS -----------------
if report_type == "Analytical Reports":
    data = load_excel_data()
    header_col1, header_col2 = st.columns([8,2])
    with header_col2:
        year = st.selectbox(
        "Select Year",
        sorted(data["Year"].unique()),
        key="year_selectbox"
    )
    with header_col1:
        st.header("Analytical Reports")

    tab1, tab2, tab3, tab4 = st.tabs(["Commodity Analysis", 
                                    "Origin-Destination Report", 
                                    "Yearly Trend", 
                                    "Operational Efficiency Metrics",])

    filtered_df = data[data['Year'] == year]

    # Tab 1: Commodity-wise Analysis
    with tab1:
        #st.subheader("Commodity-wise Distribution")
            
        commodity_summary = (
            filtered_df.groupby('Commodity')['Quantity (MT)']
            .sum()
            .reset_index()
            .sort_values(by='Quantity (MT)', ascending=False)
        )

        if not commodity_summary.empty:
            top10_commodities = commodity_summary.head(10)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                #bar chart
                title1 = f"Commodity-wise Cargo Volume ({year})"
                fig1 = px.bar(
                        top10_commodities,
                        x="Commodity",
                        y="Quantity (MT)",
                        color="Commodity",
                        title=title1,
                        
                        color_discrete_sequence=["#0077b6"]
                )
                fig1.update_layout(showlegend=False)
                fig1.update_layout(hoverlabel=dict(font_color="black"))
                fig1.update_layout(
                    yaxis=dict(
                        title_font=dict(color="black"),
                        tickfont=dict(color="black")
                    ),
                    xaxis=dict(
                        title_font=dict(color="black"),
                        tickfont=dict(color="black")
                    )
                )

                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                title2 = f"Commodity Share ({year})"
                fig2 = px.pie(
                            top10_commodities,
                            names="Commodity",
                            values="Quantity (MT)",
                            title=title2,
                            color_discrete_sequence=px.colors.sequential.Blues_r
                    )
                fig2.update_traces(textinfo='percent', textposition='outside') 
                fig2.update_layout(showlegend=False)
                fig2.update_layout(hoverlabel=dict(font_color="black"))
                fig2.update_traces(
                    textinfo='percent',
                    textfont=dict(
                        color="black"
                    )
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            with col3:
                passenger_data = data[(data['Commodity'] == "Copper") & (data['Year'] == year)].copy()
                passenger_data['Month'] = passenger_data['Arrival Date'].dt.strftime('%b')
                passenger_data['Copper'] = (passenger_data['Quantity (MT)'] / 60).round().astype('Int64')
                monthly_passengers = (
                    passenger_data.groupby('Month')['Copper']
                    .sum()
                    .reindex(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
                    .reset_index()
                )
                title3 = f"Month-wise Number of Copper({year})"
                fig3 = px.bar(
                    monthly_passengers,
                    x='Month',
                    y='Copper',
                    color='Month',
                    color_discrete_sequence=["#0077b6"],
                    text='Copper',
                    title=title3
                )

                fig3.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
                fig3.update_traces(textfont=dict(color="black"))
                fig3.update_layout(
                    xaxis_title="Month",
                    yaxis_title="Number of Copper",
                    showlegend=False,
                    hoverlabel=dict(font_color="black"),
                    yaxis=dict(title_font=dict(color="black"), tickfont=dict(color="black")),
                    xaxis=dict(title_font=dict(color="black"), tickfont=dict(color="black"))
                )

                st.plotly_chart(fig3, use_container_width=True)

            #Report Download
            figures = [fig1, fig2, fig3] if not commodity_summary.empty else []
            pdf_buffer = generate_pdf("Commodity Analysis", kpis=None, figures=figures)
            st.download_button(
                "Download Report",
                data=pdf_buffer,
                file_name=f"iwai_report_{year}_commodity.pdf",
                mime="application/pdf"
            )

        else:
            st.warning("No data available for selected filters.")
                
            
    # Tab 2: Origin-Destination
    with tab2:
        #st.subheader(f"Origin–Destination Flows Report ({year})")
        if filtered_df.empty:
            st.warning(f"No data available for {year}")
        else:
            # KPIs 
            total_qty = filtered_df['Quantity (MT)'].sum()
            unique_routes = filtered_df.groupby(['Origin Terminal', 'Destination Terminal']).ngroups
            top_origin = filtered_df.groupby('Origin Terminal')['Quantity (MT)'].sum().idxmax()
            top_dest = filtered_df.groupby('Destination Terminal')['Quantity (MT)'].sum().idxmax()

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Quantity (MT)", f"{total_qty:,.0f}")
            col2.metric("Unique Routes", unique_routes)
            col3.metric("Top Origin", top_origin)
            col4.metric("Top Destination", top_dest)

            # Top 20 Routes 
            st.subheader("Top 20 Origin-Destination Routes by Quantity")
            route_summary = (
                filtered_df.groupby(['Origin Terminal', 'Destination Terminal'])['Quantity (MT)']
                .sum()
                .reset_index()
                .sort_values(by='Quantity (MT)', ascending=False)
                .head(20)
            )
            title1="Top 20 Routes"
            fig1 = px.bar(
                route_summary,
                x='Quantity (MT)',
                y='Origin Terminal',
                color='Destination Terminal',
                orientation='h',
                title=title1,
                color_discrete_sequence=px.colors.sequential.Blues_r,
                text='Quantity (MT)'
            )
            fig1.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
            fig1.update_layout(hoverlabel=dict(font_color="black"))
            fig1.update_layout(
                yaxis=dict(
                    title_font=dict(color="black"),
                    tickfont=dict(color="black")
                ),
                xaxis=dict(
                    title_font=dict(color="black"),
                    tickfont=dict(color="black")
                )
            )
            fig1.update_traces(textfont=dict(color="black"))
            st.plotly_chart(fig1, use_container_width=True, key="od_top_routes")

            # Sankey Diagram 
            # data['Origin_clean'] = data['Origin Terminal'].astype(str).str.strip()
            # data['Destination_clean'] = data['Destination Terminal'].astype(str).str.strip()

            # route_summary = (
            #     data.groupby(['Origin_clean', 'Destination_clean'], as_index=False)['Quantity (MT)']
            #         .sum()
            # )
     
            # col1, col2 = st.columns([5, 2])
            # with col2:
            #     origin = st.selectbox(
            #         "Select Origin Terminal",
            #         sorted(data['Origin_clean'].unique()),
            #         key="origin_selectbox_terminal"
            #     )

            # with col1:
            #     title2=st.markdown(f"Cargo Flows from **{origin}** to all Destinations")

            # filtered_routes = route_summary[route_summary['Origin_clean'] == origin]

            # if not filtered_routes.empty:
            #     labels = list(
            #         pd.concat([filtered_routes["Origin_clean"], filtered_routes["Destination_clean"]])
            #         .dropna()
            #         .unique()
            #     )
            #     source_indices = [labels.index(o) for o in filtered_routes["Origin_clean"]]
            #     target_indices = [labels.index(d) for d in filtered_routes["Destination_clean"]]
            #     values = filtered_routes["Quantity (MT)"].tolist()

            #     fig2 = go.Figure(data=[go.Sankey(
            #         node=dict(pad=15, 
            #                   thickness=20, 
            #                   line=dict(color="blue", width=1), 
            #                   label=labels,
            #                   color="blue"),

            #         link=dict(source=source_indices, 
            #                   target=target_indices, 
            #                   value=values)
            #     )])
            #     num_nodes = len(labels)
            #     dynamic_height = min(max(100 + num_nodes * 40, 400), 1000)
            #     fig2.update_layout(
            #         width=1200,   
            #         height=dynamic_height,   
            #         font=dict(size= 25, color="black"),
            #         hoverlabel=dict(font_color="black"),
            #         plot_bgcolor='blue',
            #         paper_bgcolor='white',
            #         font_shadow=""
            #     )  
            #     st.plotly_chart(fig2, use_container_width=True, key=f"sankey_{origin}")
            # else:
            #     st.warning(f"No data available for Origin Terminal: {origin}")

            #-----------------------------------------------------
            
            data['Origin_clean'] = data['Origin NW'].astype(str).str.strip()
            data['Destination_clean'] = data['Destination NW'].astype(str).str.strip()

            route_summary = (
                data.groupby(['Origin_clean', 'Destination_clean'], as_index=False)['Quantity (MT)']
                    .sum()
            )

            col1, col2 = st.columns([5, 2])
            with col2:
                origin = st.selectbox(
                    "Select Origin NW",
                    sorted(data['Origin_clean'].unique()),
                    key="origin_selectbox_nw"
                )

            with col1:
                title3=st.markdown(f"Cargo Flows from **{origin}** to all Destinations")

            filtered_routes = route_summary[route_summary['Origin_clean'] == origin]
            if not filtered_routes.empty:
                labels = list(
                    pd.concat([filtered_routes["Origin_clean"], filtered_routes["Destination_clean"]])
                    .dropna()
                    .unique()
                )

                source_indices = [labels.index(o) for o in filtered_routes["Origin_clean"]]
                target_indices = [labels.index(d) for d in filtered_routes["Destination_clean"]]
                values = filtered_routes["Quantity (MT)"].tolist()

                fig3 = go.Figure(data=[go.Sankey(
                    node=dict(pad=15, thickness=20, line=dict(color="blue", width=1), label=labels, color="blue"),
                    link=dict(source=source_indices, target=target_indices, value=values)
                )])
                fig3.update_layout(
                    width=1200,   
                    height=600,   
                    font=dict(size= 25, color="black"),
                    hoverlabel=dict(font_color="black"),
                    plot_bgcolor='blue',
                    paper_bgcolor='white',
                    font_shadow=""
                ) 
                st.plotly_chart(fig3, use_container_width=True, key=f"sankey_{origin}")
            else:
                st.warning(f"No data available for Origin Terminal: {origin}")

            # Report Download
            kpis = {
                "Total Quantity (MT)": f"{total_qty:,.0f}",
                "Unique Routes": unique_routes,
                "Top Origin": top_origin,
                "Top Destination": top_dest
            }
            figures = [fig1, fig2, fig3]
            pdf_buffer = generate_pdf("Origin–Destination Report", kpis=kpis, figures=figures)
            st.download_button(
                "Download Report",
                data=pdf_buffer,
                file_name=f"iwai_report_{year}_origin_destination.pdf",
                mime="application/pdf"
            )


    #Tab 3: Yearly & Monthly Trends
    with tab3:
        #st.subheader("Yearly Cargo Trend")
        col1,col2=st.columns(2)
        with col1:
            yearly_trend = (
                data.groupby('Year')['Quantity (MT)']
                .sum()
                .reset_index()
                .sort_values(by='Year')
            )
            title1 = "Yearly Cargo Volume Trend"
            fig1 = px.line(
                yearly_trend,
                x="Year",
                y="Quantity (MT)",
                markers=True,
                title=title1,
                color_discrete_sequence=["#0077b6"]
            )
            fig1.update_layout(hoverlabel=dict(font_color="black"))
            fig1.update_layout(
                yaxis=dict(
                    title_font=dict(color="black"),
                    tickfont=dict(color="black")
                ),
                xaxis=dict(
                    title_font=dict(color="black"),
                    tickfont=dict(color="black")
                )
            )
            st.plotly_chart(fig1, use_container_width=True)
  
        #--------------------------------------------------
        with col2:
            data['Month'] = data['Departure Date'].dt.strftime('%b')
            monthly_qty = data.groupby('Month')['Quantity (MT)'].sum().reindex(
                ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
            ).reset_index()
            title2=f"Month-wise Quantity for {'Commodity'}"
            fig2 = px.bar(
                monthly_qty,
                x='Month',
                y='Quantity (MT)',
                color='Month',
                color_discrete_sequence=["#0077b6"],
                text='Quantity (MT)',
                title=title2
            )
            fig2.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
            fig2.update_layout(xaxis_title="Month", yaxis_title="Quantity (MT)", showlegend=False)
            fig2.update_layout(hoverlabel=dict(font_color="black"))
            fig2.update_layout(
                yaxis=dict(
                    title_font=dict(color="black"),
                    tickfont=dict(color="black")
                ),
                xaxis=dict(
                    title_font=dict(color="black"),
                    tickfont=dict(color="black")
                )
            )
            fig2.update_traces(textfont=dict(color="black"))
            st.plotly_chart(fig2, use_container_width=True)

        # Report Download
        figures = [fig1, fig2]
        pdf_buffer = generate_pdf("Yearly & Monthly Trends", kpis=None, figures=figures)
        st.download_button(
            "Download Report",
            data=pdf_buffer,
            file_name=f"iwai_report_{year}_trends.pdf",
            mime="application/pdf"
        )


    # Tab 4: Operational Efficiency Metrics    
    with tab4:
        #st.subheader("Operational Efficiency Metrics")
        # year = 2025
        data['Arrival Date'] = pd.to_datetime(data['Arrival Date'])
        data['Departure Date'] = pd.to_datetime(data['Departure Date'])

        yearly_data = data[data['Year'] == year].copy() 
        yearly_data.loc[:, 'Turnaround Time (Days)'] = ( (yearly_data['Arrival Date'] - yearly_data['Departure Date']).dt.total_seconds() / (60 * 60 * 24) ).round().abs().astype('Int64')

        yearly_data = yearly_data[yearly_data['Turnaround Time (Days)'] > 0]

        turnaround_avg = (
            yearly_data.groupby('Origin NW')['Turnaround Time (Days)']
            .mean()
            .round()
            .astype('Int64')
            .reset_index()
        )

        if not turnaround_avg.empty:
            overall_turnaround = int(turnaround_avg['Turnaround Time (Days)'].mean())
            fastest_tat = int(yearly_data['Turnaround Time (Days)'].min())
            slowest_tat = int(yearly_data['Turnaround Time (Days)'].max())
            fastest_port = yearly_data.loc[yearly_data['Turnaround Time (Days)'] == fastest_tat, 'Origin NW'].values[0]
            slowest_port = yearly_data.loc[yearly_data['Turnaround Time (Days)'] == slowest_tat, 'Origin NW'].values[0]
            num_voyages = len(yearly_data)
        else:
            overall_turnaround = fastest_tat = slowest_tat = num_voyages = 0
            fastest_port = slowest_port = "-"

        
        st.metric(f"Average Turnaround Time ({year})", f"{overall_turnaround} Days")

        col1,col2 = st.columns(2)
        with col1:
            if not turnaround_avg.empty:
                title1 = f'Average Turnaround Time per Port - {year}'
                fig1 = px.bar(
                    turnaround_avg,
                    x='Origin NW',
                    y='Turnaround Time (Days)',
                    color='Origin NW',
                    color_discrete_sequence=["#0077b6"],
                    title=title1,
                    text='Turnaround Time (Days)',
                    labels={'Turnaround Time (Days)': 'Avg Turnaround Time (Days)'}
                )
                fig1.update_traces(textposition='outside', texttemplate='%{text} days')
                fig1.update_traces(textfont=dict(color="black"))
                fig1.update_layout(hoverlabel=dict(font_color="black"))
                fig1.update_layout(showlegend = False)
                fig1.update_layout(
                    yaxis=dict(
                        title_font=dict(color="black"),
                        tickfont=dict(color="black")
                    ),
                    xaxis=dict(
                        title_font=dict(color="black"),
                        tickfont=dict(color="black")
                    )
                )
                st.plotly_chart(fig1)
        
        with col2:
            monthly_tat = (
                yearly_data.groupby(yearly_data['Arrival Date'].dt.to_period('M'))['Turnaround Time (Days)']
                .mean()
                .reset_index()
            )
            monthly_tat['Arrival Date'] = monthly_tat['Arrival Date'].dt.to_timestamp()
            title2 = f'Monthly Average Turnaround Time Trend - {year}'
            if not monthly_tat.empty:
                fig2 = px.line(
                    monthly_tat,
                    x='Arrival Date',
                    y='Turnaround Time (Days)',
                    color_discrete_sequence=["#0077b6"],
                    title=title2,
                    markers=True
                )
                fig2.update_layout(hoverlabel=dict(font_color="black"))
                fig2.update_layout(
                    yaxis=dict(
                        title_font=dict(color="black"),
                        tickfont=dict(color="black")
                    ),
                    xaxis=dict(
                        title_font=dict(color="black"),
                        tickfont=dict(color="black")
                    )
                )
                st.plotly_chart(fig2)
            
        kpis = {
            "Average Turnaround Time": f"{overall_turnaround} Days",
            "Fastest Port": f"{fastest_port} ({fastest_tat} days)",
            "Slowest Port": f"{slowest_port} ({slowest_tat} days)",
            "Total Voyages": num_voyages
        }
        figures = []
        if not turnaround_avg.empty:
            figures.append(fig1)
        if not monthly_tat.empty:
            figures.append(fig2)

        pdf_buffer = generate_pdf("Operational Efficiency Metrics", kpis=kpis, figures=figures)
        st.download_button(
            "Download Efficiency Report (PDF)",
            data=pdf_buffer,
            file_name=f"iwai_report_{year}_efficiency.pdf",
            mime="application/pdf"
        )


# ----------------- GEOSPATIAL REPORTS -----------------
elif report_type == "Geospatial Reports":
    #st.title("Geospatial Analysis")

    data = load_excel_data()
    df = load_terminal_data()
    df = attach_cargo_weights(data, df)
    df_project = load_project_data()
    df_lad = load_lad_data()

    tab1, tab2 = st.tabs(["Terminal Map", "Project Map"])

    with tab1:
        def get_nearby_terminals(selected_terminal, df, selected_range):
            terminal_row = df[df["Terminal Name"] == selected_terminal].iloc[0]
            center_point = (terminal_row["Latitude"], terminal_row["Longitude"])
            nearby = []
            for _, row in df.iterrows():
                point = (row["Latitude"], row["Longitude"])
                distance = geodesic(center_point, point).km
                if distance <= selected_range:
                    nearby.append([row["Latitude"], row["Longitude"], row["CargoWeight"]])
            return nearby, center_point

        col1, col2 = st.columns([5, 2])
        with col1:
            terminal_options = df['Terminal Name'].dropna().unique()
            selected_terminal = st.selectbox("Select Terminal", terminal_options)
        with col2:
            selected_range = st.slider("Select Range (km)", min_value=10, max_value=1000, step=10, value=100)

        heat_data, (center_lat, center_lon) = get_nearby_terminals(selected_terminal, df, selected_range)
        m = folium.Map(location=[center_lat, center_lon], zoom_start=8)

        if heat_data:
            HeatMap(heat_data, radius=12, blur=8, max_zoom=4).add_to(m)
        map_state = st_folium(m, width='90%', height=350, key="main_map")

        if map_state and map_state["last_clicked"] is not None:
            click_lat = map_state["last_clicked"]["lat"]
            click_lon = map_state["last_clicked"]["lng"]

            df["distance"] = df.apply(
                lambda r: geodesic((click_lat, click_lon), (r["Latitude"], r["Longitude"])).km, axis=1
            )
            nearest_terminal = df.loc[df["distance"].idxmin()]
            st.subheader("Terminal Details")
            st.table(pd.DataFrame({
                "Terminal Name": [nearest_terminal["Terminal Name"]],
                "NW Name": [nearest_terminal["NW Name"]],
                "Quantity (MT)": [nearest_terminal["CargoWeight"]]
            }))
        
        if st.button("Download Map PDF Snapshot"):
            png_file = capture_map(m)
            pdf_file = png_to_pdf(png_file)
            st.download_button(
                "Download Terminal Map PDF",
                data=pdf_file,
                file_name="terminal_map_snapshot.pdf",
                mime="application/pdf"
            )
    
    with tab2:

        col1, col2 = st.columns(2)

        with col1:
            nw_options = df_project["NW"].dropna().unique().tolist()
            selected_nw = st.selectbox("Select NW", ["All"] + nw_options)

        with col2:
            status_options = df_project["Status"].dropna().unique().tolist()
            selected_status = st.selectbox("Select Project Status", ["All"] + status_options)

        filtered_df = df_project.copy()
        if selected_nw != "All":
            filtered_df = filtered_df[filtered_df["NW"] == selected_nw]

        if selected_status != "All":
            filtered_df = filtered_df[filtered_df["Status"] == selected_status]

        if not filtered_df.empty:
            center_lat = filtered_df["Latitude"].mean()
            center_lon = filtered_df["Longitude"].mean()
        else:
            center_lat, center_lon = 22.5, 79.0  

        m = folium.Map(location=[center_lat, center_lon], zoom_start=6)

        for _, row in filtered_df.iterrows():
            popup_text = f"""
            <b>Project:</b> {row['Project_Name']}<br>
            <b>NW:</b> {row['NW']}<br>
            <b>Status:</b> {row['Status']}
            """
            folium.Marker(
                location=[row["Latitude"], row["Longitude"]],
                popup=popup_text,
                tooltip=row["Project_Name"],
                icon=folium.Icon(color="blue" if row["Status"] == "Ongoing" else "green")
            ).add_to(m)

        st_folium(m, width="90%", height=350)
    
    