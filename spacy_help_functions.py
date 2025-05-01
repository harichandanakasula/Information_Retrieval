import spacy
import re
from collections import defaultdict
from nltk.tokenize import sent_tokenize

spacy2bert = {
    "ORG": "ORGANIZATION",
    "PERSON": "PERSON",
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "DATE": "DATE"
}

bert2spacy = {
    "ORGANIZATION": "ORG",
    "PERSON": "PERSON",
    "LOCATION": "LOC",
    "CITY": "GPE",
    "COUNTRY": "GPE",
    "STATE_OR_PROVINCE": "GPE",
    "DATE": "DATE"
}

nlp = spacy.load("en_core_web_lg")

def clean_entity(text):
    return re.sub(r'\[\d+\]', '', text).strip()

def get_sentences(text):
    nlp=spacy.load("en_core_web_lg")
    doc=nlp(text)
    return [sent.text.strip() for sent  in doc.sents]

def get_entities(sentence, entities_of_interest):
    doc = nlp(sentence)
    return [(clean_entity(ent.text.strip()), spacy2bert[ent.label_])
            for ent in doc.ents if ent.label_ in spacy2bert and spacy2bert[ent.label_] in entities_of_interest]
def is_valid_pair(ent1, ent2, relation_type):
    if relation_type == "Work_For":
        return ent1[1] == "PERSON" and ent2[1] == "ORGANIZATION"
    elif relation_type == "Live_In":
        return ent1[1] == "PERSON" and ent2[1] == "LOCATION"
    elif relation_type == "Schools_Attended":
        return ent1[1] == "PERSON" and ent2[1] == "ORGANIZATION"
    elif relation_type == "Top_Member_Employees":
        return ent1[1] == "ORGANIZATION" and ent2[1] == "PERSON"
    return False
def create_entity_pairs(sents_doc, entities_of_interest, window_size=40,relation_type=None):
    if entities_of_interest is not None:
        entities_of_interest = {bert2spacy[b] for b in entities_of_interest}
    ents = sents_doc.ents
    length_doc = len(sents_doc)
    entity_pairs = []
    for i in range(len(ents)):
        e1 = ents[i]
        if entities_of_interest is not None and e1.label_ not in entities_of_interest:
            continue
        for j in range(1, len(ents) - i):
            e2 = ents[i + j]
            if entities_of_interest is not None and e2.label_ not in entities_of_interest:
                continue
            if e1.text.lower() == e2.text.lower():
                continue
            if (1 <= (e2.start - e1.end) <= window_size):
                punc_token = False
                start = e1.start - 1 - sents_doc.start
                if start > 0:
                    while not punc_token:
                        punc_token = sents_doc[start].is_punct
                        start -= 1
                        if start < 0:
                            break
                    left_r = start + 2 if start > 0 else 0
                else:
                    left_r = 0
                punc_token = False
                start = e2.end - sents_doc.start
                if start < length_doc:
                    while not punc_token:
                        punc_token = sents_doc[start].is_punct
                        start += 1
                        if start == length_doc:
                            break
                    right_r = start if start < length_doc else length_doc
                else:
                    right_r = length_doc
                if (right_r - left_r) > window_size:
                    continue
                x = [token.text for token in sents_doc[left_r:right_r]]
                gap = sents_doc.start + left_r
                e1_info = (clean_entity(e1.text), spacy2bert[e1.label_], (e1.start - gap, e1.end - gap - 1))
                e2_info = (clean_entity(e2.text), spacy2bert[e2.label_], (e2.start - gap, e2.end - gap - 1))
                if not is_valid_pair(e1_info, e2_info, relation_type):
                    continue
                entity_pairs.append((x, e1_info, e2_info))
    return entity_pairs

def extract_relations(doc, spanbert, entities_of_interest=None, conf=0.7,relation_type=None):
    num_sentences = len([s for s in doc.sents])
    print("Total # sentences = {}".format(num_sentences))
    res = defaultdict(int)
    relation_mapping = {
        "Work_For": "per:employee_of",
        "Live_In": "per:cities_of_residence",
        "Schools_Attended": "per:schools_attended",
        "Top_Member_Employees": "org:top_members/employees"
    }
    for sentence in doc.sents:
        print("\tprocessing sentence: {}".format(sentence))
        entity_pairs = create_entity_pairs(sentence, entities_of_interest,relation_type=relation_type)
        examples = []
        for ep in entity_pairs:
            examples.append({"tokens": ep[0], "subj": ep[1], "obj": ep[2]})
            examples.append({"tokens": ep[0], "subj": ep[2], "obj": ep[1]})
        preds = spanbert.predict(examples)
        for ex, pred in list(zip(examples, preds)):
            relation = pred[0]
            if relation!=relation_mapping[relation_type]:
                continue
            if relation == 'no_relation':
                continue
            print("\t\t=== Extracted Relation ===")
            print("\t\tTokens: {}".format(ex['tokens']))
            subj = ex["subj"][0]
            obj = ex["obj"][0]
            confidence = pred[1]
            print("\t\tRelation: {} (Confidence: {:.3f})\nSubject: {}\tObject: {}".format(
                relation, confidence, subj, obj))
            if confidence > conf:
                if res[(subj, relation, obj)] < confidence:
                    res[(subj, relation, obj)] = confidence
                    print("\t\tAdding to set of extracted relations")
                else:
                    print("\t\tDuplicate with lower confidence. Ignoring.")
            else:
                print("\t\tConfidence below threshold. Ignoring.")
            print("\t\t==========")
    return res

def get_entity_span(doc, entities_of_interest):
    spans = []
    for ent in doc.ents:
        if ent.label_ in spacy2bert and spacy2bert[ent.label_] in entities_of_interest:
            spans.append((clean_entity(ent.text), spacy2bert[ent.label_], ent.start, ent.end - 1))
    return spans
