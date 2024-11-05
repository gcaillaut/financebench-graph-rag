from neo4j import GraphDatabase
import datasets
from pathlib import Path
from tqdm import tqdm


def query_database(driver, neo4j_query, params={}):
    with driver.session() as session:
        result = session.run(neo4j_query, params)
        output = [r.values() for r in result]
        output.insert(0, result.keys())
        return output


if __name__ == "__main__":
    import os, json
    from dotenv import load_dotenv

    load_dotenv()

    URL = os.environ["NEO4J_URL"]
    USER = os.environ["NEO4J_USER"]
    PASSWORD = os.environ["NEO4J_PASSWORD"]

    driver = GraphDatabase.driver(URL, auth=(USER, PASSWORD), telemetry_disabled=True)

    ds = datasets.load_dataset("PatronusAI/financebench", split="train")

    chunks_and_triples = []

    for docname in tqdm(ds["doc_name"]):
        q = f"""
MATCH (c:Chunk)-[:PART_OF]->(:Document {{fileName: '{docname}.pdf'}})
OPTIONAL MATCH (c:Chunk)-[:HAS_ENTITY]->(e)-[r]->(v:!Chunk)
RETURN c.text, collect(e.id), collect(type(r)), collect(v.id);"""
        res = query_database(driver, q)

        chunks_and_triples.extend(
            [
                {
                    "document": docname,
                    "text": x[0],
                    "triples": list(zip(x[1], x[2], x[3])),
                }
                for x in res[1:]
            ]
        )

    output_path = Path("data", "financebench_neo4j.json")
    output_path.parent.mkdir(exist_ok=True, parents=True)
    with open(output_path, "wt", encoding="utf-8") as f:
        json.dump(chunks_and_triples, f, indent=4, ensure_ascii=False)
