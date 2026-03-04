### Output Data Structure (long format)

| Column | Description | Where did the value come from|
|:-------|:-------------|:-------------|
| **q_id** | Unique question identifier | From Dataset |
| **source_lang** | Source language of the QA pair |  From Dataset |
| **target_lang** | Target language of the QA pair |  From Dataset |
| **q_src** | Question asked in the source language|  From Dataset |
| **q_tgt** | Question asked in the target language|  From Dataset |
| **a_src** | Answer when asking Q in source language|  From LLM |
| **a_tgt** | Answer when asking Q in target language|  From LLM |
| **correct_source** | whether the a_src is correct|  From LLM-as-a-judge |
| **correct_target** | Answer a_tgt is correct|  From LLM-as-a-judge |
| **original_content(c_src)** | Original passage/content text from Wikipedia |  From Dataset |
| **content (c_tgt)** | Target-language content text from Wikipedia |  From Dataset |
| **original_answer (gold_answer_src)** | Original answer text |  From Dataset |
| **answer (gold_answer_tgt)** | Target-language answer text |  From Dataset |
| **translated** | Flag (1 = translated, 0 = original) |  From Dataset |
| **title**, **url** | Metadata retained from the original dataset |  From Dataset |



