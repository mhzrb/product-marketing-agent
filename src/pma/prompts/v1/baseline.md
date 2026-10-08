---
name: baseline
version: 1.0.0
---
Task: write the marketing content for the product below in one pass.

<product_data>
${product_json}
</product_data>
<languages>${languages}</languages>
<limits>${limits_json}</limits>
<attempt>1</attempt>

Return ONE JSON object with one key per requested language, exactly in this shape:
{"en": {"description": "...", "ad_copy": "...", "social_post": "...",
        "email_subjects": ["...", "...", "..."]},
 "nl": {"description": "...", "ad_copy": "...", "social_post": "...",
        "email_subjects": ["...", "...", "..."]}}
Include only the languages listed in <languages>. Exactly three email subjects per language.
