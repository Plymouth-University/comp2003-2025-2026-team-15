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

# Filtering section
st.sidebar.header("Filters")

# Filter by protocol
protocols = st.sidebar.multiselect(
    "Protocol",
    df["protocol_name"].unique()
)

# Filter by source IP
src_ips = st.sidebar.multiselect(
    "Source IP",
    df["src_ip"].unique()
)

# Filter by destination IP
dst_ips = st.sidebar.multiselect(
    "Destination IP",
    df["dst_ip"].unique()
)

# Filter by source port
src_ports = st.sidebar.multiselect(
    "Source Port",
    sorted(df["src_port"].dropna().unique())
)

# Filter by destination port
dst_ports = st.sidebar.multiselect(
    "Destination Port",
    sorted(df["dst_port"].dropna().unique())
)

filtered = df.copy()

# Apply filters
if protocols:
    filtered = filtered[filtered["protocol_name"].isin(protocols)]

if src_ips:
    filtered = filtered[filtered["src_ip"].isin(src_ips)]

if dst_ips:
    filtered = filtered[filtered["dst_ip"].isin(dst_ips)]

if src_ports:
    filtered = filtered[filtered["src_port"].isin(src_ports)]

if dst_ports:
    filtered = filtered[filtered["dst_port"].isin(dst_ports)]

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
