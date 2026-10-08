---
name: plan
version: 1.0.0
---
Task: plan the marketing copy for the product below BEFORE writing it.

<product_data>
${product_json}
</product_data>
<languages>${languages}</languages>

Steps:
1. Call lookup_product_fact to confirm facts you will rely on (at least the price).
2. Call retrieve_guidelines to fetch the brand-voice, claims and channel guidance that applies to
   this category and audience.
3. Then answer with ONE JSON object:
{"angle": "<one-sentence angle>", "tone": "<tone for this audience>",
 "key_facts": ["<fact present in the data>", ...],
 "guideline_notes": ["<short rule from the guidelines>", ...],
 "must_avoid": ["<claim or phrase to avoid>", ...]}
key_facts may only contain facts that are present in <product_data>.

${tool_instructions}
