import argparse
import re
import ast
import random
import time
import requests
from bs4 import BeautifulSoup
from spacy_help_functions import get_entities, get_sentences,get_entity_span
from spanbert import SpanBERT
from gemini_helper_6111 import get_gemini_completion
from googleapiclient.discovery import build
import spacy
RELATION_MAP = {
    1: "Schools_Attended",
    2: "Work_For",
    3: "Live_In",
    4: "Top_Member_Employees"
}

INTERNAL_MAP = {
    "Schools_Attended": "per:schools_attended",
    "Work_For": "per:employee_of",
    "Live_In": "per:cities_of_residence",
    "Top_Member_Employees": "org:top_members/employees"
}

ENTITY_CONSTRAINTS = {
    "Schools_Attended": ("PERSON", "ORGANIZATION"),
    "Work_For": ("PERSON", "ORGANIZATION"),
    "Live_In": ("PERSON", {"LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"}),
    "Top_Member_Employees": ("ORGANIZATION", "PERSON")
}
def is_valid_pair(relation, e1, e2):
    subj_type, obj_type = ENTITY_CONSTRAINTS[relation]
    return (
        (e1[1] == subj_type and (e2[1] in obj_type if isinstance(obj_type, set) else e2[1] == obj_type)) or
        (e2[1] == subj_type and (e1[1] in obj_type if isinstance(obj_type, set) else e1[1] == obj_type))
    )

def search(query, api_key, engine_id):
    service = build("customsearch", "v1", developerKey=api_key)
    result = service.cse().list(q=query, cx=engine_id).execute()
    return [item["link"] for item in result.get("items", [])]

def run_spanbert(query, relation, k, threshold, client_key, engine_id):
    from spacy_help_functions import nlp  

    print("\n=========== Using SpanBERT for Relation Extraction ===========")
    print("Loading pre-trained spanBERT from ./pretrained_spanbert\n")

    seen_urls = set()
    extracted = dict()
    used_queries = set()
    spanbert = SpanBERT("./pretrained_spanbert")

    current_query = query
    print("\nLoading necessary libraries; This should take a minute or so ...)\n")
    print(f"=========== Iteration: 0 - Query: {current_query} ===========\n")

    while len(extracted) < k:
        urls = search(current_query, client_key, engine_id)
        for i, url in enumerate(urls):
            if url in seen_urls:
                continue
            seen_urls.add(url)

            print(f"URL ( {i+1} / {len(urls)}): {url}")
            print("\tFetching text from url ...")
            try:
                html = requests.get(url, timeout=10).text
                soup = BeautifulSoup(html, "html.parser")
                text = soup.get_text()
                text=text.replace('\xa0',' ')
                text=re.sub(r'\n+','\n',text)
                text=re.sub(r'\s{2,}',' ',text)
                text=text.strip()
                if len(text) > 10000:
                    print(f"\tTrimming webpage content from {len(text)} to 10000 characters")
                    text = text[:10000]
                print(f"\tWebpage length (num characters): {len(text)}")
                try:
                    sentences = get_sentences(text)
                    print(f"\t# Sentences extracted = {len(sentences)}")
                except Exception as e:
                    print(f"\tError in get_sentences: {e}")
                    continue

                print(f"\tExtracted {len(sentences)} sentences. Processing each sentence one by one to check for presence of right pair of named entity types; if so, will run the second pipeline ...")

                print("\tAnnotating the webpage using spacy...")
                print(f"\tExtracted {len(sentences)} sentences. Processing each sentence one by one to check for presence of right pair of named entity types; if so, will run the second pipeline ...\n")
            except Exception:
                print("\tSkipping due to error.")
                continue

            extracted_count = 0
            for idx, sentence in enumerate(sentences):
                if (idx + 1) % 5 == 0 or idx == len(sentences) - 1:
                    print(f"\tProcessed {idx + 1} / {len(sentences)} sentences")

                entities = get_entity_span(nlp(sentence), ENTITY_CONSTRAINTS[relation])
                if len(entities) < 2:
                    continue

                pairs = [(e1, e2) for e1 in entities for e2 in entities
                         if e1 != e2 and is_valid_pair(relation, e1, e2)]
                if not pairs:
                    continue

                tokens = [token.text for token in nlp(sentence)]
                examples = []
                for a, b in pairs:
                    examples.append({
                        "tokens": tokens,
                        "subj": (a[0], a[1], (a[2], a[3])),
                        "obj": (b[0], b[1], (b[2], b[3]))
                    })
                    examples.append({
                        "tokens": tokens,
                        "subj": (b[0], b[1], (b[2], b[3])),
                        "obj": (a[0], a[1], (a[2], a[3]))
                    })

                labels, probs = spanbert.predict(examples)

                for ex, label, prob in zip(examples, labels, probs):
                    if label!=INTERNAL_MAP[relation]:
                        continue
                    if label == INTERNAL_MAP[relation] and prob >= threshold:
                        subj = ex["subj"][0]
                        obj = ex["obj"][0]
                        print("\n\t\t=== Extracted Relation ===")
                        print(f"\t\tInput tokens: {ex['tokens']}")
                        print(f"\t\tOutput Confidence: {prob:.8f} ; Subject: {subj} ; Object: {obj}")
                        tup = (subj, obj)
                        if tup not in extracted or extracted[tup] < prob:
                            print("\t\tAdding to set of extracted relations")
                            extracted[tup] = prob
                        else:
                            print("\t\tDuplicate with lower confidence than existing record. Ignoring.")
                        print("\t\t==========")
                        extracted_count += 1

            print(f"\n\tExtracted annotations for {extracted_count} out of total {len(sentences)} sentences")
            print(f"\tRelations extracted from this website: {extracted_count} (Overall: {len(extracted)})\n")

            if len(extracted) >= k:
                break

        if len(extracted) >= k:
            break

        candidates = sorted([(t, p) for t, p in extracted.items() if " ".join(t) not in used_queries], key=lambda x: -x[1])
        if not candidates:
            print("No new queries available. Stopping.")
            break
        top_query = " ".join(candidates[0][0])
        used_queries.add(top_query)
        print(f"\n=========== Iteration: {len(used_queries)} - Query: {top_query} ===========\n")
        current_query = top_query

    print(f"================== ALL RELATIONS for {relation} =================")
    sorted_final = sorted(extracted.items(), key=lambda x: -x[1])[:k]
    for (s, o), p in sorted_final:
        print(f"Confidence: {p:.8f} \t\t| Subject: {s} \t\t| Object: {o}")
    print(f"Total # of iterations = {len(used_queries) + 1}")

import random

def run_gemini(query, relation, k, client_key, engine_id, gemini_key):
    # Mapping from internal to semantic keywords
    RELATION_KEYWORDS = {
        "Schools_Attended": ["school", "university", "college", "attended", "alma mater"],
        "Work_For": ["works for", "worked at", "employed", "employee", "co-founder", "chair", "founder"],
        "Live_In": ["lives in", "resides in", "resident of", "based in"],
        "Top_Member_Employees": ["chairman","president","ceo", "cto", "coo", "executive", "head", "leader", "chief"]
    }

    seen_urls = set()
    extracted = {}
    current_query = query
    iteration = 0

    print("=========== Using Gemini for Relation Extraction ===========")

    while len(extracted) < k:
        print(f"\n=========== Iteration: {iteration} - Query: {current_query} ===========\n")
        urls = search(current_query, client_key, engine_id)

        for idx, url in enumerate(urls):
            if url in seen_urls:
                continue
            seen_urls.add(url)

            print(f"\nURL ( {idx + 1} / {len(urls)}): {url}")
            print("\tFetching text from url ...")

            try:
                html = requests.get(url, timeout=10).text
                soup = BeautifulSoup(html, "html.parser")
                text = soup.get_text()
                if len(text) > 10000:
                    print(f"\tTrimming webpage content from {len(text)} to 10000 characters")
                    text = text[:10000]
                print(f"\tWebpage length (num characters): {len(text)}")
                print("\tAnnotating the webpage using spacy...")
                sentences = get_sentences(text)
                print(f"\tExtracted {len(sentences)} sentences. Processing each sentence one by one to check for presence of right pair of named entity types; if so, will run the second pipeline ...")
            except Exception as e:
                print(f"\tSkipping due to error: {e}")
                continue

            local_extracted = 0
            for i, sent in enumerate(sentences):
                if (i+1) % 5 == 0 or (i+1) == len(sentences):
                    print(f"\tProcessed {i+1} / {len(sentences)} sentences")

                entities = get_entities(sent, ENTITY_CONSTRAINTS[relation])
                if len(entities) < 2:
                    continue
                if not any(is_valid_pair(relation, e1, e2) for e1 in entities for e2 in entities if e1 != e2):
                    continue

                prompt =f"""Extract all {relation} relations from the sentence below.
Output format: [["Subject","Relation","Object"]]
Sentence: {sent}
"""

                time.sleep(random.randint(7, 12))  

                try:
                    response = get_gemini_completion(prompt, gemini_key=gemini_key)
                    lines = response.strip().split('\n')
                    for line in lines:
                        line=line.strip().strip("`").strip()
                        line=re.sub(r'```(json)?','',line).strip()
                        match=re.search(r'\[\[.*?\]\]',line)
                        if match:
                            line=match.group(0)
                        else:
                            continue
                        if '[' in line and ']' in line:
                            try:
                                parsed = ast.literal_eval(line)
                                if not parsed:
                                    continue
                                if isinstance(parsed[0], str): 
                                    parsed = [parsed]
                                for triple in parsed:
                                    normalized_relation=relation.lower().replace("_","")
                                    normalized_label=triple[1].lower().replace("_","").replace(" ","")
                                    if (
                                        isinstance(triple, list)
                                        and len(triple) == 3
                                        and isinstance(triple[0], str)
                                        and isinstance(triple[2], str)
                                        and (
                                            normalized_label==normalized_relation  or
                                            any(keyword in triple[1].lower() for keyword in RELATION_KEYWORDS[relation])
                                        )
                                    ):
                                        subj, obj = triple[0], triple[2]
                                        key = (subj, obj)
                                        print("\n\t\t=== Extracted Relation ===")
                                        cleaned_sent = sent.replace('\n', ' ')[:300]
                                        print(f"\t\tSentence: {cleaned_sent}")
                                        print(f"\t\tSubject: {subj} ; Object: {obj} ;")
                                        if key not in extracted:
                                            extracted[key] = 1.0
                                            print("\t\tAdding to set of extracted relations")
                                            local_extracted += 1
                                        else:
                                            print("\t\tDuplicate. Ignoring this.")
                                        print("\t\t==========\n")
                            except Exception as parse_error:
                                print(f"\t\tInvalid Gemini line: {line.strip()} — Error: {parse_error}")
                                continue
                except Exception as e:
                    print(f"\tError with Gemini: {e}")
                    if "429" in str(e):
                        wait = random.randint(60, 90)
                        print(f"\tRate limit hit. Sleeping for {wait} seconds...")
                        time.sleep(wait)
                    continue

            print(f"\n\tExtracted annotations for {local_extracted} out of total {len(sentences)} sentences")
            print(f"\tRelations extracted from this website: {local_extracted} (Overall: {len(extracted)})")

            if len(extracted) >= k:
                break

        if len(extracted) >= k:
            break

        remaining = [f"{s} {o}" for (s, o) in extracted.keys() if f"{s} {o}" != current_query]
        if not remaining:
            print("No more queries. Stopping.")
            break

        current_query = remaining[-1]
        iteration += 1

    print(f"\n================== ALL RELATIONS for {relation} ({len(extracted)}) ==================")
    for (s, o), conf in list(extracted.items())[:k]:
        print(f"Subject: {s} \t\t| Object: {o}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=["spanbert", "gemini"])
    parser.add_argument("client_key")
    parser.add_argument("engine_id")
    parser.add_argument("gemini_key")
    parser.add_argument("relation", type=int)
    parser.add_argument("threshold", type=float)
    parser.add_argument("query")
    parser.add_argument("k", type=int)
    args = parser.parse_args()

    relation_name = RELATION_MAP[args.relation]

    print("____")
    print("Parameters:")
    print(f"Client key      = {args.client_key}")
    print(f"Engine key      = {args.engine_id}")
    print(f"Gemini key      = {args.gemini_key}")
    print(f"Method          = {args.method}")
    print(f"Relation        = {relation_name}")
    print(f"Threshold       = {args.threshold}")
    print(f"Query           = {args.query}")
    print(f"# of Tuples     = {args.k}")

    if args.method == "spanbert":
        print("\n=========== Using SpanBERT for Relation Extraction ===========")
        run_spanbert(args.query, relation_name, args.k, args.threshold, args.client_key, args.engine_id)
    else:
        print("\n=========== Using Gemini for Relation Extraction ===========")
        run_gemini(args.query, relation_name, args.k, args.client_key, args.engine_id,args.gemini_key)
