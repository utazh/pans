# Cold execution isolation

The initial mixed warm/cold suite reused one process across eight ordinary-cache arms before its first cold arm. Strict mincore rejected that first cold attempt: 7,884 of 69,160 target pages remained resident, with 19,928 cache-file mappings still present. No cold timing was admitted.

All eight completed warm arms remain unchanged. Failed cold artifacts were renamed from v24_time_native_b0_identity to v24_rejected_warm_to_cold_b0_identity; the exact paths are recorded in audit/v24_cold_restart.json.

A fresh-process two-request native probe passed the unchanged application-empty and zero-page checks with no payload writes. Every formal cold arm now starts a separate process, performs the originally configured eight runtime warmups, then uses the existing v23 ColdController before each of its 32 measured requests. Model loading and cold setup remain outside TTFT. Both identity and shared, native and preserve use this same process-isolation policy.

No old cold function, cache controller, inference kernel, file-page certification, or tolerance was changed. No live tensor mapping was forcibly unmapped. Raw failed records remain available; only protocol-valid requests enter formal cold statistics.

scripts/run_v24_cold_jobs.py implements the isolated-arm executor. scripts/resume_v24_cold.py resumed this turn without re-running completed quality or warm results. The main v24 orchestration script now separates warm and isolated cold execution for future fresh-output runs. The initially executed mixed driver is preserved as audit/v24_mixed_phase_driver_rejected.py.
