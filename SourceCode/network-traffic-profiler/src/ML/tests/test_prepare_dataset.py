import pytest
from action_classification.prepare_dataset import prepare_dataset

@pytest.fixture # Only run this function once for all tests
def setup_dataset(tmp_path, monkeypatch):
    # Create a mock file structure

    # Create the base directory
    base_dir = tmp_path / "datasets"
    like_dir = base_dir / "Like"
    sub_dir = like_dir / "subfolder"
    
    sub_dir.mkdir(parents=True)
    
    # Create dummy pcap files
    (sub_dir / "test1.pcap").write_text("test data")
    (like_dir / "test2.pcap").write_text("test data")
    (like_dir / "ignore_me.txt").write_text("not a pcap")
    
    # monkeypatch changes the working directory to the temp folder
    # so the script looks for 'datasets' in the right place
    monkeypatch.chdir(tmp_path)
    
    return base_dir

def test_organises_files(setup_dataset):
    # Run the function
    prepare_dataset()
    
    like_path = setup_dataset / "Like"
    
    # Check that the new files exist
    assert (like_path / "Like_0.pcap").exists()
    assert (like_path / "Like_1.pcap").exists()
    
    # Check that the original subfolder file is gone (moved)
    assert not (like_path / "subfolder" / "test1.pcap").exists()
    
    # Check that the non pcap file wasn't touched
    assert (like_path / "ignore_me.txt").exists()

def test_missing_dir(capsys, tmp_path, monkeypatch):
    # Create an empty datasets folder with no sub-folders
    empty_base = tmp_path / "datasets"
    empty_base.mkdir()
    monkeypatch.chdir(tmp_path)
    prepare_dataset()
    # Hijacks and captures the output
    captured = capsys.readouterr()
    assert "Directory not found: datasets/Like" in captured.out