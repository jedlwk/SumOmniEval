# EVALUATION TASK
Evaluate the quality of the generated summary provided below.

### DATA INPUTS
**Generated Summary:** {{ generated_summary }}

{% if source %}
**Source Text:** {{ source }}
{% endif %}

{% if reference_summary %}
**Reference Summary:** {{ reference_summary }}
{% endif %}