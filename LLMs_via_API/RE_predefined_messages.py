import os
import json
from data_utlis import DocREDLoader

# Loads relations {"P6": "head of government", "P17": "country", ...
with open('../data/DocRED/rel_info.json', 'r', encoding='utf-8') as file:
    relations = json.load(file)

# Loads descriptions taken & corrected to match docred from https://github.com/THUDM/AutoRE/blob/main/AutoRE/data/relations_desc/relation_description_redocred.json
with open('RE_descriptions/relation_description_redocred.json', 'r', encoding='utf-8') as file:
    descriptions = json.load(file)


def get_example():
    dr_loader = DocREDLoader('..')

    document = dr_loader.load_docs(docred_type='redocred', split='train')[0]

    for i in range(len(document['vertexSet'])):
        entity = document['vertexSet'][i][0]
        document['sents'][entity['sent_id']][entity['pos'][0]] = '<<' + str(i) + '>>' + \
                                                                 document['sents'][entity['sent_id']][entity['pos'][0]]
        document['sents'][entity['sent_id']][entity['pos'][1] - 1] = document['sents'][entity['sent_id']][
                                                                         entity['pos'][1] - 1] + '<<' + str(i) + '>>'

    concatenated_sents = []
    for i in range(len(document['sents'])):
        concatenated_sents.append(' '.join(document['sents'][i]))
    concatenated_sents = ' '.join(concatenated_sents)
    return concatenated_sents, document['labels']


system_msg_gpt_v1 = f"""You are a highly skilled RELATION EXTRACTION model. Extract directed relations between entities marked with special token <<entity_id>> at its start and its end. The relation could be spanning multiple sentences.
I want you to return only a valid JSON surrounded by $$$ of all relations found in the text in the following format: $$$[{'{'}"h_idx":0,"t_idx":1,"r":"P175","evidence":[0,1]{'}'}, {'{'}"h_idx":4,"t_idx":1,"r":"P20","evidence":[3]{'}'}]$$$.
h_idx is the head <<entity_id>> index, t_idx is the tail <<entity_id>> index, r is the relation type connecting head and tail entity, evidence is a list of sentences where the relation was found.
List of allowed relation types to extract is the following subset of properties from wikidata: {json.dumps(relations, ensure_ascii=False)}.
If relation connecting marked entities is not present in the provided subset do not include it in the output."""

system_msg_gpt_v2 = f"""
You are a highly skilled RELATION EXTRACTION system. Analyze the given text and extract DIRECTED RELATIONS between entities explicitly marked with <<n>> index tags. Follow these guidelines:

1. Entity Identification:
   - Entities are wrapped in <<n>> tags where n is their pre-defined index
   - Example: <<0>>Paris<<0>> = index 0
   - Preserve EXACT index numbers from the markers

2. Relation Extraction:
   - Extract ONLY relations from this Wikidata subset:
     {json.dumps(relations, ensure_ascii=False)}
   - Relations must be DIRECTED (h_idx -> t_idx)
   - Relations can span MULTIPLE sentences
   - Return empty list if no valid relations exist
   - STRICTLY ignore relations not in allowed list

3. Evidence Handling:
   - List ALL supporting sentence indices (0-based)
   - Sort evidence indices ASCENDING
   - Include cross-sentence evidence

4. Output Requirements:
   - ONLY output valid JSON between $$$ delimiters
   - Use EXACT structure:
     [{'{'}
       "h_idx": (marker number), 
       "t_idx": (marker number),
       "r": (P-code), 
       "evidence": [sorted_indices]
     {'}'}]
   - Validate JSON syntax

Example Input:
{get_example()[0]}

Example Output:
$$$
{json.dumps(get_example()[1], ensure_ascii=False)}
$$$"""


# I don't know what this is. Skip
# system_msg_gpt_v3 = f"""You are a highly skilled RELATION EXTRACTION model. Extract directed relation between entities marked with special token <<entity_id>> at its start and its end. The relation could be spanning multiple sentences.
# I want you to return only a valid JSON surrounded by $$$ of entity pairs found in the text in the following format: $$$[{'{'}"h_idx":0,"t_idx":1,"r":"P175","evidence":[0,1]{'}'}, {'{'}"h_idx":4,"t_idx":1,"r":"P175","evidence":[3]{'}'}]$$$.
# h_idx is the head <<entity_id>> index, t_idx is the tail <<entity_id>> index, r is the relation type connecting head and tail entity, evidence is a list of sentences where the evidence for relation was found.
# If no pair of marked entities is connected by provided relation type return empty JSON $$$[]$$$.
# The relation to extract is:"""

# Skip this - it was about querying one entitiy at the time
# system_msg_gpt_v3 = f"""You are a highly skilled RELATION EXTRACTION model. Extract directed relations between entities marked with special token <<entity_id>> at its start and its end. The relation could be spanning multiple sentences.
# I want you to return only a valid JSON surrounded by $$$ of all relations found in the text in the following format: $$$[{'{'}"h_idx":0,"t_idx":1,"r":"P175","evidence":[0,1]{'}'}, {'{'}"h_idx":4,"t_idx":1,"r":"P20","evidence":[3]{'}'}]$$$.
# h_idx is the head <<entity_id>> index, t_idx is the tail <<entity_id>> index, r is the relation type connecting head and tail entity, evidence is a list of sentences where the relation was found..
# If relation connecting marked entities is not present in the provided subset do not include it in the output.
# List of allowed relation types to extract is the following subset of properties from wikidata:"""


def remap_example_from_r_ids_to_names(ex):
    example_remapped = []
    for rel in ex:
        if rel["r"] in relations:
            new_rel = rel.copy()
            new_rel["r"] = relations[rel["r"]]
            example_remapped.append(new_rel)
        else:
            raise Exception(f"Relation {rel['r']} not found in relation_dict")
    return example_remapped


system_msg_gpt_v4 = f"""
You are a highly skilled RELATION EXTRACTION system. Analyze the given text and extract DIRECTED RELATIONS between entities explicitly marked with <<n>> index tags. Follow these guidelines:

1. Entity Identification:
   - Entities are wrapped in <<n>> tags where n is their pre-defined index
   - Example: <<0>>Paris<<0>> = index 0
   - Preserve EXACT index numbers from the markers

2. Relation Extraction:
   - Extract ONLY relations from this Wikidata subset:
     {json.dumps(descriptions, ensure_ascii=False)}
   - Relations must be DIRECTED (h_idx -> t_idx)
   - Relations can span MULTIPLE sentences
   - Return empty list if no valid relations exist
   - STRICTLY ignore relations not in allowed list

3. Evidence Handling:
   - List ALL supporting sentence indices (0-based)
   - Sort evidence indices ASCENDING
   - Include cross-sentence evidence

4. Output Requirements:
   - ONLY output valid JSON between $$$ delimiters
   - Use EXACT structure:
     [{'{'}
       "h_idx": (marker number), 
       "t_idx": (marker number),
       "r": (relation key), 
       "evidence": [sorted_indices]
     {'}'}]
   - Validate JSON syntax

Example Input:
{get_example()[0]}

Example Output:
$$$
{json.dumps(remap_example_from_r_ids_to_names(get_example()[1]), ensure_ascii=False)}
$$$"""

system_msg_gpt_v3 = f"""
You are a highly skilled RELATION EXTRACTION system. Analyze the given text and extract DIRECTED RELATIONS between entities explicitly marked with <<n>> index tags. Follow these guidelines:

1. Entity Identification:
   - Entities are wrapped in <<n>> tags where n is their pre-defined index
   - Example: <<0>>Paris<<0>> = index 0
   - Preserve EXACT index numbers from the markers

2. Relation Extraction:
   - Extract ONLY relations from this Wikidata subset:
     {json.dumps(descriptions, ensure_ascii=False)}
   - Relations must be DIRECTED (h_idx -> t_idx)
   - Relations can span MULTIPLE sentences
   - Return empty list if no valid relations exist
   - STRICTLY ignore relations not in allowed list

3. Evidence Handling:
   - List ALL supporting sentence indices (0-based)
   - Sort evidence indices ASCENDING
   - Include cross-sentence evidence

4. Output Requirements:
   - ONLY output valid JSON between $$$ delimiters
   - Use EXACT structure:
     $$$[{'{'}
       "h_idx": (marker number), 
       "t_idx": (marker number),
       "r": (relation key), 
       "evidence": [sorted_indices]
     {'}'}]$$$
   - Validate JSON syntax
"""

with open('RE_descriptions/property_dict.json', 'r') as file:
    wikidata_descriptions = json.load(file)
system_msg_gpt_v5 = f"""
You are a highly skilled RELATION EXTRACTION system. Analyze the given text and extract DIRECTED RELATIONS between entities explicitly marked with <<n>> index tags. Follow these guidelines:

1. Entity Identification:
   - Entities are wrapped in <<n>> tags where n is their pre-defined index
   - Example: <<0>>Paris<<0>> = index 0
   - Preserve EXACT index numbers from the markers

2. Relation Extraction:
   - Extract ONLY relations from this Wikidata subset:
     {json.dumps(wikidata_descriptions, ensure_ascii=False)}
   - Relations must be DIRECTED (h_idx -> t_idx)
   - Relations can span MULTIPLE sentences
   - Return empty list if no valid relations exist
   - STRICTLY ignore relations not in allowed list

3. Evidence Handling:
   - List ALL supporting sentence indices (0-based)
   - Sort evidence indices ASCENDING
   - Include cross-sentence evidence

4. Output Requirements:
   - ONLY output valid JSON between $$$ delimiters
   - Use EXACT structure:
     $$$[{'{'}
       "h_idx": (marker number), 
       "t_idx": (marker number),
       "r": (relation key), 
       "evidence": [sorted_indices]
     {'}'}]$$$
   - Validate JSON syntax
"""


# CHECK OF NOTOVERLAPPING RELATIONS
# print(relations)
# print(descriptions)
# for g_t in relations.values():
#     found = False
#     for c_t in descriptions.keys():
#         if g_t == c_t:
#             found = True
#             break
#     if not found:
#         print(f'Not found docred relation {g_t} in descriptions')
# print(len(relations), len(descriptions))
# for g_t in descriptions.keys():
#     found = False
#     for c_t in relations.values():
#         if g_t == c_t:
#             found = True
#             break
#     if not found:
#         print(f'extra relation from redocred {g_t} in descriptions')

def get_specific_examples(doc_id_from_list):
    dr_loader = DocREDLoader('..')

    docs_coverage = [2720, 1651, 2975, 810, 3007, 1003, 2384, 655, 1656, 2301, 176, 328, 2380, 584, 88, 46, 887, 299]
    document = dr_loader.load_docs(docred_type='redocred', split='train')[docs_coverage[doc_id_from_list]]

    for i in range(len(document['vertexSet'])):
        entity = document['vertexSet'][i][0]
        document['sents'][entity['sent_id']][entity['pos'][0]] = '<<' + str(i) + '>>' + \
                                                                 document['sents'][entity['sent_id']][entity['pos'][0]]
        document['sents'][entity['sent_id']][entity['pos'][1] - 1] = document['sents'][entity['sent_id']][
                                                                         entity['pos'][1] - 1] + '<<' + str(i) + '>>'

    concatenated_sents = []
    for i in range(len(document['sents'])):
        concatenated_sents.append(' '.join(document['sents'][i]))
    concatenated_sents = ' '.join(concatenated_sents)
    return concatenated_sents, document['labels']


def func_system_msg_gpt_examples(how_many):
    system_msg_gpt = f"""
    You are a highly skilled RELATION EXTRACTION system. Analyze the given text and extract DIRECTED RELATIONS between entities explicitly marked with <<n>> index tags. Follow these guidelines:
    
    1. Entity Identification:
       - Entities are wrapped in <<n>> tags where n is their pre-defined index
       - Example: <<0>>Paris<<0>> = index 0
       - Preserve EXACT index numbers from the markers
    
    2. Relation Extraction:
       - Extract ONLY relations from this Wikidata subset:
         {json.dumps(descriptions, ensure_ascii=False)}
       - Relations must be DIRECTED (h_idx -> t_idx)
       - Relations can span MULTIPLE sentences
       - Return empty list if no valid relations exist
       - STRICTLY ignore relations not in allowed list
    
    3. Evidence Handling:
       - List ALL supporting sentence indices (0-based)
       - Sort evidence indices ASCENDING
       - Include cross-sentence evidence
    
    4. Output Requirements:
       - ONLY output valid JSON between $$$ delimiters
       - Use EXACT structure:
         [{'{'}
           "h_idx": (marker number), 
           "t_idx": (marker number),
           "r": (relation key), 
           "evidence": [sorted_indices]
         {'}'}]
       - Validate JSON syntax
    """
    for i in range(how_many):
        ex_input, ex_output = get_specific_examples(i)
        system_msg_gpt += f"""
        Example Input:
        {ex_input}
        Example Output:
        $$${json.dumps(remap_example_from_r_ids_to_names(ex_output), ensure_ascii=False)}$$$"""

    return system_msg_gpt


system_msg_gpt_v6 = func_system_msg_gpt_examples(5)
system_msg_gpt_v7 = func_system_msg_gpt_examples(10)

experiment_prompts = {
    'system_v1': system_msg_gpt_v1,
    'system_v2': system_msg_gpt_v2,
    'system_v3': system_msg_gpt_v3,
    'system_v4': system_msg_gpt_v4,
    'system_v5': system_msg_gpt_v5,
    'system_v6': system_msg_gpt_v6,
    'system_v7': system_msg_gpt_v7,
    # 'refine_v1': refine_instructions_v1,
    # 'refine_v2': refine_instructions_v2,

}
