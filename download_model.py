"""
Download trained model from Modal volume
"""

import modal

app = modal.App("chess-model-downloader")
volume = modal.Volume.from_name("chess-data", create_if_missing=False)

@app.function(volumes={"/data": volume})
def download():
    """Download model from Modal volume"""
    with open('/data/chess_model.pth', 'rb') as f:
        return f.read()

@app.local_entrypoint()
def main():
    print("📥 Downloading model from Modal...")

    try:
        model_bytes = download.remote()

        with open('src/chess_model.pth', 'wb') as f:
            f.write(model_bytes)

        print("✓ Model downloaded to src/chess_model.pth")
        print("🎉 Done! Restart your dev server to use the new model.")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure you have:")
        print("  1. Trained a model using 'modal run train_modal.py'")
        print("  2. Set up Modal with 'modal setup'")
