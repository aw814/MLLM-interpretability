from wimbd.es import WimbdSearch

# Examples

# # Count documents containing a phrase (AND logic)
# count = ws.count_documents_containing_phrases(
#     index="mc4_de", 
#     phrases=["künstliche intelligenz"], 
#     all_phrases=True
# )
# print(f"Found {count} documents.")

# # Query multiple languages at once
# res = ws.count_documents_containing_phrases(
#     index="mc4_de,mc4_fr,mc4_es", 
#     phrases=["data science"], 
#     all_phrases=True
# )

ws = WimbdSearch(es_url="http://localhost:9200", api_key=None)

# Query Dataset A
res_a = ws.count_documents_containing_phrases("mc4_de", ["winter"], all_phrases=True)

# Query Dataset B (Hebrew)
res_b = ws.count_documents_containing_phrases("corpus_hebrew", ["שלום"], all_phrases=True)

# Query BOTH at once
res_both = ws.count_documents_containing_phrases("mc4_de,mc4_fr,mc4_es", ["data"], all_phrases=True)

