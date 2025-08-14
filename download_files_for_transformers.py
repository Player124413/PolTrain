import os
import sys
import wget

assets_folder = sys.argv[1]
embedder_name = sys.argv[2]

os.makedirs(assets_folder, exist_ok=True)

repo_url = "https://huggingface.co/Politrees/RVC_resources/resolve/main"
file_urls = {
    "rmvpe/rmvpe.pt": f"{repo_url}/predictors/rmvpe.pt",
    "hubert/config.json": f"https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/{embedder_name}/config.json",
    "hubert/pytorch_model.bin": f"https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/{embedder_name}/pytorch_model.bin",
}

for file, url in file_urls.items():
    file_path = os.path.join(assets_folder, file)
    if not os.path.exists(file_path):
        wget.download(url, out=file_path)
