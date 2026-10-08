---
name: revise
version: 1.0.0
---
Task: revise your previous draft. An automated reviewer found problems. Fix every problem listed
in <feedback> and change nothing else that was fine.

<product_data>
${product_json}
</product_data>
<plan>
${plan_json}
</plan>
<guidelines>
${guidelines}
</guidelines>
<languages>${languages}</languages>
<limits>${limits_json}</limits>
<attempt>${attempt}</attempt>
<previous_draft>
${previous_draft}
</previous_draft>
<feedback>
${feedback}
</feedback>

Return ONE JSON object in the same shape as the previous draft (one key per requested language,
exactly three email_subjects each) and nothing else.
