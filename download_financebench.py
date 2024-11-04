import datasets
import requests
from pathlib import Path
from tqdm import tqdm

if __name__ == "__main__":
    ds = datasets.load_dataset("PatronusAI/financebench", split="train")

    for x in tqdm(ds):
        outp = Path("data", "financebench", x["doc_name"] + ".pdf")
        if not outp.is_file():
            try:
                r = requests.get(x["doc_link"])
                with open(outp, "wb") as f:
                    f.write(r.content)
            except Exception as e:
                print(e)


