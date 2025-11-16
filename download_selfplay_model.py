"""
Download self-play trained model from Modal volume
"""

import modal

app = modal.App("chess-selfplay-downloader")
volume = modal.Volume.from_name("chess-selfplay-data", create_if_missing=False)

@app.function(volumes={"/data": volume})
def download():
    """Download model from Modal volume"""
    import os

    # List files to help debug
    files = os.listdir('/data')
    print(f"Files in volume: {files}")

    # Find the latest model
    model_files = [f for f in files if f.endswith('.pth')]
    if not model_files:
        raise Exception("No .pth model files found in volume!")

    # Get the latest iteration model
    model_files.sort()
    latest_model = model_files[-1]
    print(f"Downloading: {latest_model}")

    with open(f'/data/{latest_model}', 'rb') as f:
        return f.read(), latest_model

@app.local_entrypoint()
def main():
    print("📥 Downloading self-play model from Modal...")

    try:
        model_bytes, filename = download.remote()

        with open('src/chess_model.pth', 'wb') as f:
            f.write(model_bytes)

        print(f"✓ Model '{filename}' downloaded to src/chess_model.pth")
        print("🎉 Done! Restart your dev server to use the new model.")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure you have:")
        print("  1. Trained a model using 'modal run train_self_play_modal.py'")
        print("  2. Set up Modal with 'modal setup'")
