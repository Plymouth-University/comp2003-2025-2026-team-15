# step 5: visualise statistics on dashboard

import streamlit as st
import pandas as pd
import plotly.express as px
import main
import tempfile

st.set_page_config(page_title="Network Traffic Profiler", layout="wide")
st.title("Network Traffic Dashboard")

# ---------------------
# Run pipeline
# ---------------------

# Caches result to speed up filtering data
# Only bypasses the cache if func arguments change
# show_spinner=False prevents a loading spinner from being displayed,
# of which the content of is un-modifable. A custom spinner is used later.
@st.cache_data(show_spinner=False)
def load_dataset(path):
    return main.run_pipeline(path)

# Initialise session state keys
# These enable the code to check if a new PCAP file has been uploaded
# in order to bypass cached pipeline data
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None
if "df" not in st.session_state:
    st.session_state.df = None

# PCAP File Upload
st.sidebar.header("File Upload")
uploaded_file = st.sidebar.file_uploader("Upload a .pcap/.pcapng file", type=["pcap", "pcapng"], accept_multiple_files=False)
# If no file has been uploaded, stop the app
if uploaded_file is None:
    st.info("Upload a PCAP file to start!")
    st.stop()
else:
    # If a new file has been uploaded, run the pipeline
    if uploaded_file.name != st.session_state.uploaded_file_name:
        # Write to a temporary file as web applications cannot access
        # local files on a computer
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".pcap")
        temp.write(uploaded_file.read())
        # Run pipeline passing through the temporary file
        with st.spinner("Processing PCAP…"):
            flows_df, numeric_df, anomaly_info = load_dataset(temp.name)

        # Update session state to track uploaded file
        st.session_state.flows = flows_df
        st.session_state.uploaded_file_name = uploaded_file.name
        st.session_state.numeric_df = numeric_df
        st.session_state.anomaly_info = anomaly_info

# Copy the extracted data into a dataframe
df = st.session_state.flows.copy()

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

# Table View Filters
st.sidebar.header("Table Display Options")

show_top_endpoints = st.sidebar.checkbox("Top Endpoints", value=True)
show_flow_table = st.sidebar.checkbox("All Flows", value=True)
show_anomalous_table = st.sidebar.checkbox("Anomalous Flows", value=True)
show_invalid_flows = st.sidebar.checkbox("Invalid Flows", value=True)

# Dashboard visualisation section

# Anomaly summary
if st.session_state.anomaly_info is not None:
    info = st.session_state.anomaly_info

    st.subheader("Anomaly Detection Summary")

    col1, col2, col3 = st.columns(3)
    col1.metric("Anomalous Flows", info["anomaly_count"])
    col2.metric("Total Flows", info["total_flows"])
    col3.metric("Anomaly %", f"{info['anomaly_percentage']:.2f}%")

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
# Top Endpoints Table
if show_top_endpoints:
    st.subheader("Top Endpoints")
    endpoints_df = (
        filtered.groupby(["src_ip", "dst_ip"])["byte_count"]
        .sum()
        .reset_index()
        .sort_values(by="byte_count", ascending=False)
        .head(20)
    )
    st.dataframe(endpoints_df, use_container_width=True)

# All Flows Table
if show_flow_table:
    st.subheader("All Flows Table")

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

# Anomalous Flows Table
if show_anomalous_table:
    st.subheader("Anomalous Flows")

    numeric_df = st.session_state.numeric_df
    anomalous = numeric_df[numeric_df["anomaly"] == True]

    if anomalous.empty:
      st.info("No anomalous flows detected")
    else:
      st.dataframe(anomalous, use_container_width=True)
      
# Invalid Flows Table    
if show_invalid_flows:
    st.subheader("Invalid Flows")
    invalid_flows = st.session_state.flows[st.session_state.flows["is_valid"] == False]
    
    if invalid_flows.empty:
        st.info("No invalid flows detected")
    else:
        cols = ["error_reason"] + [c for c in invalid_flows.columns if c not in ("error_reason", "is_valid")] 
        st.dataframe(invalid_flows[cols], use_container_width=True)
