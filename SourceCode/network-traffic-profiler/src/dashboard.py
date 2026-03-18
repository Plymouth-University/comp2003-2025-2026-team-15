# step 5: visualise statistics on dashboard

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
# In dashboard.py
@st.cache_data(show_spinner=False)
def load_dataset(path):
    # Create a grouped progress area
    with st.status("Processing...", state="running", expanded=True) as status:
        # Pass the status object to the pipeline
        results = main.run_pipeline(path, status)
        status.update(label="Processing Complete! Loading results...", state="complete", expanded=False)
    return results



# Initialise session state keys
# These enable the code to check if a new PCAP file has been uploaded
# in order to bypass cached pipeline data
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None
if "df" not in st.session_state:
    st.session_state.df = None

# Initialise session state for file upload status
if "flows" not in st.session_state:
    st.session_state.flows = None
if "file_pending_upload" not in st.session_state:
    st.session_state.file_pending_upload = False
if "current_file" not in st.session_state:
    st.session_state.current_file = None

# PCAP File Upload
st.sidebar.header("File Upload")
uploaded_file = st.sidebar.file_uploader("Upload a .pcap/.pcapng file", type=["pcap", "pcapng"], accept_multiple_files=False, max_upload_size=5000)
# If no file has been uploaded, stop the app
if uploaded_file is None:
    st.info("Upload a PCAP file to start!")

    # Legal notice for user to upload data only under compliance with UK law
    st.markdown(
    """
    <div style="
        background-color:#A63037;
        padding:16px;
        border-radius:8px;
    ">
        <p style="color:#FFFFFF; font-size:16px; line-height:1.5;">
        <b>Legal Notice</b><br><br>
        When analysing network traffic, you must have permission to capture and inspect the data.<br><br>  
        Analysing traffic that you did not generate, or do not have consent to inspect, may breach 
        UK laws relating to the acquisition, interception, or handling of communication of data under 
        the <b>Investigatory Powers Act 2016 (IPA)</b>.<br><br>
        You should only analyse data captured from your <b>own personal network</b>, and data 
        that <b>does not include other people's information</b>, unless they have explicitly agreed.
        </p>
    </div>
    """,
    unsafe_allow_html=True
    )
    st.stop()

# If user selects new file, reset state to prompt for confirmation again
if uploaded_file != st.session_state.current_file:
    st.session_state.current_file = uploaded_file
    st.session_state.file_pending_upload = True
    st.session_state.flows = None # Clear old results

# Checkbox to confirm ownership/authorisation
if st.session_state.file_pending_upload:
    warning_placeholder = st.sidebar.empty()
    confirm = st.sidebar.checkbox("I confirm that I own or am authorised to analyse this PCAP file (required)")

    # Button to upload PCAP file
    upload_clicked = st.sidebar.button("Process PCAP")

    if upload_clicked and not confirm:
        st.error("You must confirm that you are authorised to analyse this PCAP file before proceeding.")
        st.stop()



    file_size_bytes = uploaded_file.size
    file_size_mb = file_size_bytes / (1024 * 1024)
    if file_size_mb > 500:
        warning_placeholder.warning('Please note that large files may induce long processing times and require large amounts of RAM to be available on the host machine.', icon="⚠️")

    # If the user checked the box and clicked the button, process the file
    if confirm and upload_clicked:
        # Write to a temporary file as web applications cannot access
        # local files on a computer
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".pcap")
        temp.write(uploaded_file.read())
        # Run pipeline passing through the temporary file
        
        flows_df, numeric_df, anomaly_info = load_dataset(temp.name)

        # Update session state to track uploaded file
        st.session_state.flows = flows_df
        st.session_state.uploaded_file_name = uploaded_file.name
        st.session_state.numeric_df = numeric_df
        st.session_state.anomaly_info = anomaly_info

        # Mark file as processed, hides checkbox & button
        st.session_state.file_pending_upload = False
        st.rerun() # Immediate refresh

# If still no processed data, stop here
if st.session_state.flows is None:
    st.info("Please confirm to process the uploaded PCAP file.")
    st.stop()

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
show_top_conversations = st.sidebar.checkbox("Top Conversations", value=True)
show_flow_table = st.sidebar.checkbox("All Flows", value=True)
show_anomalous_table = st.sidebar.checkbox("Anomalous Flows", value=True)
show_flagged_flows = st.sidebar.checkbox("Flagged Flows", value=True)

# CSV Download
st.sidebar.header("Download Data")
st.sidebar.download_button(
    label="Download Filtered CSV",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="filtered_flows.csv",
    mime="text/csv"
)

# Convert label names
LABEL_NAMES = {
    "src_ip": "Source Address",
    "dst_ip": "Destination Address",
    "src_port": "Source Port",
    "dst_port": "Destination Port",
    "protocol": "Protocol Number",
    "protocol_name": "Protocol Name",
    "packet_count": "Packet Count",
    "byte_count": "Byte Count",
    "avg_packet_size": "Average Packet Size",
    "first_packet_index": "First Packet Index",
    "last_packet_index": "Last Packet Index",
    "duration": "Flow Duration (s)",
    "error_reason": "Flagged Reason",
    "anomaly": "Anomaly",
    "conversation": "Conversation Pair",
    "total_bytes": "Total Bytes",
    "total_packets": "Total Packets",
    "flow_count": "Flow Count",
    "action_type": "User Action Type" 
}

def rename_cols(df):
    return df.rename(columns=LABEL_NAMES)

# Dashboard visualisation section

# Activity Analysis 
st.header("Activity Analysis")
col_a, col_b = st.columns(2) # two side by side columns 
# label colours
colors = {
    "Comment": "rgb(255, 107, 107)",
    "Like": "rgb(255, 169, 77)",
    "Play": "rgb(255, 212, 59)",
    "Search": "rgb(105, 219, 124)",
    "Subscribe": "rgb(177, 151, 252)",
    "Non-YouTube": "rgb(192, 192, 192)",
    "Background": "rgb(192, 192, 192)"
}

# pie chart
with col_a:
    st.write("Traffic Composition (Data Transferred per Action)")

    pie_df = filtered.groupby("action_type", dropna=False)["byte_count"].sum().reset_index()
    if not pie_df.empty:
        fig_pie = px.pie(
            pie_df,
            values="byte_count",
            names="action_type",
            hole=0.4,
            color="action_type",
            color_discrete_map=colors
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("No actions to display.")

# timeline
with col_b:
    st.write("Activity Timeline (When Action Occured)")
    
    # create coloured dots for user actions
    actions = filtered[
        filtered["action_type"].notna() &
        ~filtered["action_type"].isin(["Background Traffic", "Non-YouTube"])
    ].sort_values("first_packet_index")

    # display non yt data/background traffic as baseline
    non_youtube = filtered[
        filtered["action_type"].isin(["Non-YouTube", "Background Traffic"])
    ].sort_values("first_packet_index")

    if not actions.empty:
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=non_youtube["first_packet_index"], # when traffic happened (packet number)
            y=non_youtube["byte_count"], # how much data was transferred in the flow
            mode="lines", # connect the dots to a line
            name="Non-YouTube",
            line={"color": colors["Non-YouTube"], "width": 2},
            hovertemplate="<b>%{x}</b><br>Non-YouTube Traffic: %{y:.2s}B<extra></extra>" # include only 2 fp
        ))

        for action, df_action in actions.groupby("action_type"):
            fig.add_trace(go.Scatter(
                x=df_action["first_packet_index"],
                y=df_action["byte_count"],
                mode="markers", # display actions as dots
                name=action,
                hovertemplate=f"<b>%{{x}}</b><br>{action}: %{{y:.2s}}B<extra></extra>",
                marker={"size": 10, "color": colors.get(action, "white"), "line": {"width": 1, "color": "black"}}
            ))

        fig.update_layout(
            xaxis_title="Time (Packet Index)",
            yaxis_title="Byte Count",
            yaxis={"rangemode": "nonnegative"}, # prevent neg values
            legend_title="Action"
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No timeline data available.")
    
# Anomaly summary
if st.session_state.anomaly_info is not None:
    info = st.session_state.anomaly_info

    st.subheader("Anomaly Detection Summary")
    st.write("Overview of how many flows were detected as anomalous based on statistical ML rules.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Anomalous Flows", info["anomaly_count"])
    col2.metric("Total Flows", info["total_flows"])
    col3.metric("Anomaly %", f"{info['anomaly_percentage']:.2f}%")

# Traffic over time
st.subheader("Traffic Volume Over Time (Byte Count)")
st.write("Visualises how much traffic occurred over time, helping identify spikes, bursts, or unusual peaks in network activity.")
time_df = filtered.groupby("first_packet_index")["byte_count"].sum().reset_index()

fig_time = px.bar(
    time_df,
    x="first_packet_index",
    y="byte_count",
)

st.plotly_chart(fig_time, use_container_width=True)

# Table
# Top Endpoints Table
if show_top_endpoints:
    st.subheader("Top Endpoints")
    st.write("Shows which IP pairs exchanged the most data. Useful for identifying the busiest devices on the network.")
    endpoints_df = (
        filtered.groupby(["src_ip", "dst_ip"])["byte_count"]
        .sum()
        .reset_index()
        .sort_values(by="byte_count", ascending=False)
        .head(20)
    )
    st.dataframe(rename_cols(endpoints_df), use_container_width=True)


# Top Conversations Table 
# Add IP-pair conversation key
df["conversation"] = df.apply(
    lambda r: tuple(sorted([r["src_ip"], r["dst_ip"]])),
    axis=1
)

if show_top_conversations:
    st.subheader("Top Conversations")
    st.write("Top network conversations between two IP addresses ranked on total bytes transferred.")
    # Group by conversation pair
    top_conversations = (
        df.groupby("conversation")
        .agg(
            total_bytes=("byte_count", "sum"),
            total_packets=("packet_count", "sum"),
            avg_packet_size=("avg_packet_size", "mean"),
            flow_count=("conversation", "count"),
        )
        .sort_values(by="total_bytes", ascending=False)
        .reset_index()
    )
    # Split pair into individual columns for display
    top_conversations["Source IP"] = top_conversations["conversation"].apply(lambda x: x[0])
    top_conversations["Dest IP"] = top_conversations["conversation"].apply(lambda x: x[1])
    top_conversations = top_conversations[
        ["Source IP", "Dest IP", "total_bytes", "total_packets", "avg_packet_size", "flow_count"]
    ]
    st.dataframe(rename_cols(top_conversations.head(10)), use_container_width=True)

    
# All Flows Table
if show_flow_table:
        st.subheader("All Flows Table")
        st.write("A network flow is a sequence of packets sent from a source to a destination that all share a common set of characteristics: Source IP, Destination IP, Source Port, Destination Port & Protocol, and classified by action type (e.g. Play, Search, Like).")
        table_cols = [
            "src_ip", "dst_ip",
            "src_port", "dst_port",
            "start_time", "end_time",
            "protocol", "protocol_name",
            "packet_count", "byte_count",
            "avg_packet_size",
            "first_packet_index", "last_packet_index",
            "duration",
            "action_type" # Add action_type column
        ]
        # Only show action_type if present
        display_cols = [col for col in table_cols if col in filtered.columns]
        st.dataframe(rename_cols(filtered[display_cols]), use_container_width=True)

# Anomalous Flows Table
if show_anomalous_table:
    st.subheader("Anomalous Flows")
    st.write("Flows that likely contain statistical anomalies")

    numeric_df = st.session_state.numeric_df
    anomalous = numeric_df[numeric_df["anomaly"] == True]

    if anomalous.empty:
      st.info("No anomalous flows detected")
    else:
      st.dataframe(rename_cols(anomalous), use_container_width=True)
      
# Flagged Flows Table    
if show_flagged_flows:
    st.subheader("Flagged Flows")
    st.write("Flows that failed one or more validation rules, indicating potential errors or anomalies in the data.")
    invalid_flows = st.session_state.flows[st.session_state.flows["is_valid"] == False]
    
    if invalid_flows.empty:
        st.info("No flagged flows detected")
    else:
        cols = ["error_reason"] + [c for c in invalid_flows.columns if c not in ("error_reason", "is_valid")] 
        st.dataframe(rename_cols(invalid_flows[cols]), use_container_width=True)

# plots for visualisiung anomalities in data

# traffic activity comparison: packet vs byte count

st.subheader("Flow Activity Distribution")
st.write("Packet Count (number of packets in a flow) vs Byte Count (total amount of data transferred).")

x = "packet_count"
y = "byte_count"

fig = px.scatter(
    numeric_df,
    x=x,
    y=y,
    color="anomaly",
    color_discrete_map={True: "red", False: "blue"}
)

st.plotly_chart(fig, use_container_width=True)
