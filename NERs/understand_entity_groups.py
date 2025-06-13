"""This checks that entities group are not established by name or type"""

from data_utlis import DocREDLoader

splits = ['train', 'dev', 'test']
docred_type = ['docred', 'redocred']

dr_loader = DocREDLoader('..')

for docred in docred_type:
    for split in splits:
        problem = False
        docs = dr_loader.load_docs(docred_type=docred, split=split)
        print(f"Loaded {len(docs)} documents from {docred} - {split}")
        c = 0
        for doc in docs:
            if problem:
                break
            # print(f"Title: {doc['title']}")
            for eg in doc['vertexSet']:
                if problem:
                    break
                for i in range(len(eg)):
                    if problem:
                        break
                    if i != 0:
                        if eg[i]['type'] != eg[i - 1]['type']:  # TODO you can try name/type.
                            # print(f"Entity {i} in {doc['title']} does not match previous entity name.")
                            # print(eg[i]['name'], '-----', eg[i - 1]['name'])
                            print(eg)
                            c += 1

                            problem = True
                            break
        print(c)
