from libs.indexer import Indexer
from libs.local_llm import generate
from libs.token_lib import CustomTokenizer

new_indexer = Indexer()
new_indexer.load_from_json("libs/indexer_data.json")

prompt = "Donne moi un historique des évolution du conflit en Ukraine. Tu peux aborder ça sous l'angle tactique, stratégique, économique, diplomatique, politique, donner des noms, mais ça doit être une analyse chronologique."
query = "Donne moi un historique des évolution du conflit en Ukraine. Tu peux aborder ça sous l'angle tactique, stratégique, économique, diplomatique, politique, donner des noms, mais ça doit être une analyse chronologique."

relevant_chunks = new_indexer.retrieve_chunks(query)
enriched_context = new_indexer.enrich_context(relevant_chunks)
print(f"prompt : {prompt}")
for chunks, position in relevant_chunks:
    print("Chunk pertinent :", position)
for context in enriched_context:
    print("Chunk pertinent :", context)
print("Contexte enrichi :", enriched_context)


custom_tokenizer = CustomTokenizer()

context = " ".join(enriched_context)
system_prompt = prompt + f" Tu n'as le droit que d'utiliser les élements suivant et strictement rien d'autre : {context}. Ne répond qu'en Français."

context = custom_tokenizer.get_tokens_from_string(''.join(enriched_context))
# generate(query, context)
print("main : " + system_prompt)
generate(system_prompt, [])
