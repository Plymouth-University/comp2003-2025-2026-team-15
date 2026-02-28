import pytest
import pandas as pd
from scapy.all import IP, TCP
from action_classification.extract_ml_features import is_google_youtube_ip, extract_ml_features

def test_is_google_or_youtube_ip():
    assert is_google_youtube_ip("172.217.1.1") is True
    assert is_google_youtube_ip("1.1.1.1") is False

def test_extract_ml_features_empty_or_missing(tmp_path):
    # Test non existant file
    assert extract_ml_features("non_existent.pcap") is None
    
    # Test empty file
    with pytest.MonkeyPatch().context() as m:
        # Temporarily replace the Scapy rdpcap func with the below lambda
        m.setattr("action_classification.extract_ml_features.rdpcap", lambda x: [])
        
        pcap = tmp_path / "empty.pcap"
        pcap.write_text("")
        df = extract_ml_features(str(pcap))
        assert isinstance(df, pd.DataFrame)
        assert df.empty

def test_feature_calculation_logic(tmp_path):
    # Create some dummy packets 
    # both should be recognised as one flow
    pkt1 = IP(src="1.1.1.1", dst="172.217.0.1")/TCP(sport=12345, dport=443)
    pkt1.time = 100
    pkt2 = IP(src="1.1.1.1", dst="172.217.0.1")/TCP(sport=12345, dport=443)
    pkt2.time = 105 

    pcap = tmp_path / "test.pcap"
    pcap.write_text("dummy")
    
    with pytest.MonkeyPatch().context() as m:
        # Temporarily replace the Scapy rdpcap func with the below lambda to mock two packets
        m.setattr("action_classification.extract_ml_features.rdpcap", lambda x: [pkt1, pkt2])
        
        df = extract_ml_features(str(pcap))
        
        assert len(df) == 1
        assert df.iloc[0]['duration'] == 5.0
        assert df.iloc[0]['pk_count'] == 2