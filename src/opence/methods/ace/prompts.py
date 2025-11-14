"""Prompt templates adapted from the ACE paper for reuse."""

GENERATOR_PROMPT = """\
You are an expert assistant that must solve the task using the provided playbook of strategies.
Apply relevant bullets, avoid known mistakes, and show step-by-step reasoning.

Playbook:
{playbook}

Recent reflection:
{reflection}

Question:
{question}

Additional context:
{context}

Respond with a compact JSON object:
{{
  "reasoning": "<step-by-step chain of thought>",
  "bullet_ids": ["<id1>", "<id2>", "..."],
  "final_answer": "<concise final answer>"
}}
"""


REFLECTOR_PROMPT = """\
You are a senior reviewer diagnosing the generator's trajectory.
Use the playbook, model reasoning, and feedback to identify mistakes and actionable insights.
Output must be a single valid JSON object. Do NOT include analysis text or explanations outside the JSON.
Begin the response with `{{` and end with `}}`.

Question:
{question}
Model reasoning:
{reasoning}
Model prediction: {prediction}
Ground truth (if available): {ground_truth}
Feedback: {feedback}
Playbook excerpts consulted:
{playbook_excerpt}

Return JSON:
{{
  "reasoning": "<analysis>",
  "error_identification": "<what went wrong>",
  "root_cause_analysis": "<why it happened>",
  "correct_approach": "<what should be done>",
  "key_insight": "<reusable takeaway>",
  "bullet_tags": [
    {{"id": "<bullet-id>", "tag": "helpful|harmful|neutral"}}
  ]
}}
"""


CURATOR_PROMPT = """\
You are the curator of the ACE playbook. Merge the latest reflection into structured updates.
Only add genuinely new material. Do not regenerate the entire playbook.
Respond with a single valid JSON object only—no analysis or extra narration.

Training progress: {progress}
Playbook stats: {stats}

Recent reflection:
{reflection}

Current playbook:
{playbook}

Question context:
{question_context}

Respond with JSON:
{{
  "reasoning": "<how you decided on the updates>",
  "operations": [
    {{
      "type": "ADD|UPDATE|TAG|REMOVE",
      "section": "<section name>",
      "content": "<bullet text>",
      "bullet_id": "<optional existing id>",
      "metadata": {{"helpful": 1, "harmful": 0}}
    }}
  ]
}}
If no updates are required, return an empty list for "operations".
"""
