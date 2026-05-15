import os
import subprocess
import zipfile
from pathlib import Path

def download_fer2013():
    print("Downloading FER2013 dataset using Kaggle API...")
    # Requires ~/.kaggle/kaggle.json to be set up!
    root_dir = Path(__file__).parent.parent
    raw_dir = root_dir / 'data' / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Download using kaggle CLI
    # We use deadskull7/fer2013 which provides fer2013.csv directly
    try:
        subprocess.run(
            ['kaggle', 'datasets', 'download', '-d', 'deadskull7/fer2013', '-p', str(raw_dir)],
            check=True
        )
        print("Download complete. Extracting...")
        
        zip_path = raw_dir / 'fer2013.zip'
        if zip_path.exists():
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(raw_dir)
            print("Extraction complete.")
            os.remove(zip_path) # Clean up the zip file
        else:
            print("Zip file not found. Please check download.")
            
    except subprocess.CalledProcessError as e:
        print(f"Failed to download dataset. Ensure you have kaggle credentials configured. Error: {e}")
    except FileNotFoundError:
        print("Kaggle CLI not found. Make sure kaggle is installed in your environment.")

if __name__ == '__main__':
    download_fer2013()
