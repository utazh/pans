# v2.4 server experiment

See EXPERIMENT_REPORT_V24_ZH.md and results/v24_final_analysis.json. All relative paths are under /home/panzihang/src/prism_layerkv. Original prism_pans and previous implementation files are unchanged.

Scripts/configs retain exact commands. Outputs refuse overwrite; reruns need fresh output paths. Use PYTHONPATH=src:runtime_deps_v24 and the existing vllm-stable Python. Raw profiler captures remain in profiles/v24. Delivery is an additive code/results overlay excluding weights, bulk KV and raw profiler traces.
