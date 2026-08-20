# Development benchmarks

Use this directory for structured studies that inform the protocol without
being the final claim-establishing campaign. A development study should exist
only when it answers a protocol question worth retaining.

The active program is defined in `experimental_pipeline.md` and split into:

- `stage_0_implementation_gates/`: code, test, and wiring preflight gates;
- `stage_1_wiring/`: local Phase-A adaptation of added feedback pathways;
- `stage_2_local_smoke/`: 1M-token local Phase-B integration checks;
- `stage_3_cloud_pilot/`: resumable 5M/10M seed-1337 cloud pilots;
- `stage_4_confirmation/`: two additional seeds for promoted arms.

Pass-depth stability, memory interventions, and exact-vs-recurrent drift are
reusable checkpoint diagnostics rather than separate development studies. Run
them from `scripts/` whenever a validation round needs those measurements.
