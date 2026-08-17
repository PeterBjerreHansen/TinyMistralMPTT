# Stage 2 protocol development

The purpose of this directory is to learn how the selected starting models
should be trained before spending serious compute on the main comparison.

Current evidence is grouped by question:

- `learning_rate/`: joint adaptation versus frozen controls and backbone-LR dose
  response;
- `pass_depth/`: fixed-K development, including the recent K=3 continuation;
- `recurrence/`: exact-K versus collapsed recurrent inference health checks.

These records are evidence, not an implicit model lineage. A protocol is not
mainline until `../main/LOCKED_PROTOCOL.md` says so.
