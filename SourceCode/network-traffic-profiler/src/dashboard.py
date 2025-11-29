# step 5: visualise statistics on dashboard

import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Network Traffic Profiler", layout="wide")
st.title("Network Traffic Dashboard")

# Loads validated CSV 
df = pd.read_csv("csv_output/test_flows_validated.csv")

# Removes invalid rows
df = df[df["is_valid"] == True]

# Filters by protocol
protocols = st.multiselect("Filter by protocol", df["protocol_name"].unique())
filtered = df[df["protocol_name"].isin(protocols)] if protocols else df


# Packet per protocol bar chart
st.subheader("Packet Count per Protocol")
fig1 = px.bar(filtered, x="protocol_name", y="packet_count")
st.plotly_chart(fig1)

# Traffic over time
st.subheader("Traffic Over Time (Bytes per Packet Index)")
time_df = filtered.groupby("first_packet_index")["byte_count"].sum().reset_index()

fig_time = px.bar(
    time_df,
    x="first_packet_index",
    y="byte_count",
    title="Traffic Volume Over Time (Byte Count)"
)

st.plotly_chart(fig_time, use_container_width=True)

# Table
st.subheader("Flow Table")

table_cols = [
    "src_ip", "dst_ip",
    "src_port", "dst_port",
    "protocol", "protocol_name",
    "packet_count", "byte_count",
    "avg_packet_size",
    "first_packet_index", "last_packet_index",
    "duration"
]

st.dataframe(filtered[table_cols], use_container_width=True)
