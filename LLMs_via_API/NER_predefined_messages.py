import json
from data_utlis import DocREDLoader

system_msg_gpt_v1 = """You are a highly skilled NER extraction model. Extract entities from text and categorize them as:
                LOCATION (LOC), MISCELLANEOUS (MISC), PERSON (PER), NUMBER (NUM), TIME (TIME), ORGANIZATION (ORG).
                I will provide a document as a list of sentences. Each sentence is tokenized and represented as a list of words.
                Effectively you get document as a list of lists with sentences as a first nesting level and words as a second nesting level.
                Provide position of named entity, you need to return index of a sentence in which it was found counting from 0 as "sent_id".
                Provide position of named entity within sentence counting from 0 as "pos": [index of a word starting named entity, index of a word ending named entity + 1]
                Return only valid JSON format surrounded by $$$: $$$[{"name": "entity", "type": "TYPE", "sent_id": sentence_index, "pos": [start_word_index, end_word_index + 1]}]$$$.
                If named entity appears more than once, do not forget to extract all valid occurrences.
                If no entities found, return empty list."""

system_msg_gpt_v2 = """You are a clinical-grade NER extractor. Extract entities from tokenized text with:
- Types: 1. LOC (Geographically defined locations, including mountains, waters, etc. Politically defined locations, including countries, cities, states, streets, etc. Facilities, including buildings, museums, stadiums, hospitals, factories, airports, etc.)
         2. TIME (Absolute or relative dates or periods)
         3. NUM (Percents, money, quantities)
         4. PER (People, including fictional)
         5. ORG (Companies, universities, institutions, political or religious groups, etc.)
         6. MISC (Products, including vehicles, weapons, etc. Events, including elections, battles, sporting events, etc. Laws, cases, languages, etc.)
- Position format: closed left and open right half-open interval [start, end) indices (e.g., "New York" as [3,5] for tokens 3-4)
- Validation: make sure occurrences of given NER are extracted, additionally reject entities if: 1. pos exceeds sentence length 2. Text reconstruction mismatch

Process this structure:
Input: [["Paris","hosted", "is", "the", "capital", "and", "largest", "city", "of", "France".], ["Paris", "hosted", "2023","Climate","Summit"]]
Output: $$$[{"name":"Paris","type":"LOC","sent_id":0,"pos":[0,1]}, {"name":"France","type":"LOC","sent_id":0,"pos":[9,10]},{"name":"Paris","type":"LOC","sent_id":1,"pos":[0,1]},
{"name":"2023 Climate Summit","type":"MISC","sent_id":1,"pos":[2,5]}]$$$

Return ONLY valid JSON matching these rules surrounded by $$$. The list should be empty if no entities are found."""


def few_shot(docred_type, split):
    dr_loader = DocREDLoader('..')
    dev_true = dr_loader.load_docs(docred_type=docred_type, split=split)

    examples = ''
    for i in range(len(dev_true) - 1, len(dev_true) - 6, -1):
        examples += 'Input: ' + json.dumps(dev_true[i]['sents'], ensure_ascii=False) + '\n'
        output = []
        for entity_group in dev_true[i]['vertexSet']:
            output += entity_group
        examples += 'Output: $$$' + json.dumps(output, ensure_ascii=False) + '$$$\n'

    return examples


def system_msg_gpt_v3(docred_type, split):
    system = f"""You are a clinical-grade NER extractor. Extract entities from tokenized text with:
    - Types: 1. LOC (Geographically defined locations, including mountains, waters, etc. Politically defined locations, including countries, cities, states, streets, etc. Facilities, including buildings, museums, stadiums, hospitals, factories, airports, etc.)
             2. TIME (Absolute or relative dates or periods)
             3. NUM (Percents, money, quantities)
             4. PER (People, including fictional)
             5. ORG (Companies, universities, institutions, political or religious groups, etc.)
             6. MISC (Products, including vehicles, weapons, etc. Events, including elections, battles, sporting events, etc. Laws, cases, languages, etc.)
    - Position format: closed left and open right half-open interval [start, end) indices (e.g., "New York" as [3,5] for tokens 3-4)
    - Validation: make sure occurrences of given NER are extracted, additionally reject entities if: 1. pos exceeds sentence length 2. Text reconstruction mismatch
    
    Return ONLY valid JSON matching these rules surrounded by $$$. The list should be empty if no entities are found.
    
    Gold standard examples:
    {few_shot(docred_type, split)}"""

    return system


def few_shot2(docred_type, split):
    dr_loader = DocREDLoader('..')
    dev_true = dr_loader.load_docs(docred_type=docred_type, split=split)
    examples = ''
    for i in range(len(dev_true) - 1, len(dev_true) - 11, -1):
        examples += 'Input: ' + json.dumps(dev_true[i]['sents'], ensure_ascii=False) + '\n'
        output = []
        for entity_group in dev_true[i]['vertexSet']:
            output += entity_group
        examples += 'Output: $$$' + json.dumps(output, ensure_ascii=False) + '$$$\n'

    return examples


def system_msg_gpt_v4(docred_type, split):
    system = f"""You are a clinical-grade NER extractor. Extract entities from tokenized text with:
    - Types: 1. LOC (Geographically defined locations, including mountains, waters, etc. Politically defined locations, including countries, cities, states, streets, etc. Facilities, including buildings, museums, stadiums, hospitals, factories, airports, etc.)
             2. TIME (Absolute or relative dates or periods)
             3. NUM (Percents, money, quantities)
             4. PER (People, including fictional)
             5. ORG (Companies, universities, institutions, political or religious groups, etc.)
             6. MISC (Products, including vehicles, weapons, etc. Events, including elections, battles, sporting events, etc. Laws, cases, languages, etc.)
    - Position format: closed left and open right half-open interval [start, end) indices (e.g., "New York" as [3,5] for tokens 3-4)
    - Validation: make sure occurrences of given NER are extracted, additionally reject entities if: 1. pos exceeds sentence length 2. Text reconstruction mismatch
    
    Return ONLY valid JSON matching these rules surrounded by $$$. The list should be empty if no entities are found.
    
    Gold standard examples:
    {few_shot2(docred_type, split)}"""
    return system


def system_msg_gpt_v5(docred_type, split):
    system = f"{few_shot2(docred_type, split)}"

    return system


def refine_instructions_v1(other_prediction):
    refine_instructions = f'''
    Other NER extraction model response was @@@{other_prediction}@@@. Please reconsider your previous response by comparing it with response of another expert model. Be aware that both you and the other expert model are making mistakes. 
    Refinement steps:
    1. identify any discrepancies between responses. Type of entity tagged is not important, only whether tokens were tagged or not tagged at all. 
    2. analyze the discrepancies by paying equal attention to reducing False Positives and False Negatives: If the other expert model included an entity that you did not, think of reasons for and against including it. If you have included some entity and the second expert did not, think of reasons for and against excluding it. 
    3. Refine your previous response based on the analysis from step 2. 
    In your answer firstly include your analysis of the differences between responses, then provide $$$JSON$$$ with refined response. Analysis of the discrepancies should not be part of refined JSON'''
    return refine_instructions


def refine_instructions_v2(other_prediction):
    refine_instructions = f"""
    Other NER extraction model response was @@@{other_prediction}@@@.  
    Refine your previous prediction by resolving discrepancies as follows:  
    
    1. **Identify Discrepancies**:  
       - List tokens where your prediction and the other model’s prediction **disagree** (tagged vs. not tagged).  
    
    2. **Analyze Discrepancies**:  
       For **each conflicting token**, evaluate:  
       - **Contextual Plausibility**:  
         - Does the surrounding text (e.g., verbs, prepositions) strongly support the entity’s presence?  
         - Example: "Criticized Pfizer" → "criticized" implies an organization.  
       - **Semantic Similarity**:  
         - Compare the token’s context to typical entity contexts (e.g., "visited [LOC]" vs. "launched [ORG]").  
       - **Document-Level Consistency**:  
         - Does the entity appear elsewhere in the document? If yes, favor tagging.  
    
    3. **Refine Your Prediction**:  
       - If **context/semantics strongly support tagging** (even if only one model tagged it), **include the entity**.  
       - If **context/semantics contradict tagging** (e.g., non-entity usage), **exclude the entity**.  
       - If **document repeats the entity**, **include it** (reduce FN).  
    
    4. **Output**:  
       - First, write a concise **analysis** of discrepancies using the criteria above.  
       - Then, provide a $$$JSON$$$ with your refined entities (only entities, no analysis).  

    """

    return refine_instructions


def verifier():
    refine_instructions = f"""
    Please carefully review your previous Named Entity Recognition (NER) extraction and identify potential errors. Specifically:
    1. ASSESS COMPLETENESS: Could any entities be missing?
    2. HAVE YOU TAGGED ANYTHING INCORRECTLY: Are there any entities that should not have been tagged?
    3. EVALUATE CONSISTENCY: Are similar mentions treated identically throughout the text?

    Verification of entity types has secondary importance. The main focus is on the presence or absence of entities.

    For any identified issues:
    - Explain the potential error
    - Correct any issues while keeping the rest of the response unchanged
    - Provide response in the same format as previously
    - Do not include any comments within json structure
    """

    return refine_instructions


experiment_prompts = {
    'system_v1': system_msg_gpt_v1,
    'system_v2': system_msg_gpt_v2,
    'system_v3': system_msg_gpt_v3,
    'system_v4': system_msg_gpt_v4,
    'system_v5': system_msg_gpt_v5,
    'refine_v1': refine_instructions_v1,
    'refine_v2': refine_instructions_v2,
    'verifier': verifier
}
