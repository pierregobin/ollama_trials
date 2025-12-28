
# Install

    sudo snap install ollama
    alloma pull phi3
    pip install -r requirements.txt
    # to install sentence transformers in local ( ~/.cache/huggingface)
    python - <<EOF
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print("Model downloaded and cached locally")
    EOF
    # to force using local cache
    export HF_HUB_OFFLINE=1

# Usage

    ./phi3.py


