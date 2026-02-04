# Task: {{ task }}

## Status: {{ status | default("in_progress") }}

## Subtasks

{% for subtask in subtasks %}
- [{% if subtask.completed %}x{% else %} {% endif %}] {{ subtask.description }}
{% else %}
- [ ] Define subtasks for this task
{% endfor %}

## Notes

{{ notes | default("Add any relevant notes or context here.") }}

---
*Plan created by planning-with-files skill*
