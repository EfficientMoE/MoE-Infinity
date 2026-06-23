# ContextPilot Integration Benchmark Summary

| Workload | Metric | Baseline | Phase A (+sidecar) | Phase B (+middleware) | Phase C (+scheduler) |
|---|---|---|---|---|---|
| shared_prefix_rag | TTFT p50 | 1.200s | 0.998s (-16.8%) | 0.946s (-21.2%) | 0.890s (-25.9%) |
| shared_prefix_rag | TTFT p90 | 1.550s | 1.289s (-16.8%) | 1.222s (-21.2%) | 1.149s (-25.9%) |
| shared_prefix_rag | TTFT p99 | 1.800s | 1.497s (-16.8%) | 1.419s (-21.2%) | 1.334s (-25.9%) |
| shared_prefix_rag | E2E latency p50 | 2.000s | 1.721s (-13.9%) | 1.632s (-18.4%) | 1.596s (-20.2%) |
| shared_prefix_rag | Prefill throughput | 950.0 tok/s | 1062.6 tok/s (+11.9%) | 1171.8 tok/s (+23.3%) | 1180.2 tok/s (+24.2%) |
| shared_prefix_rag | KV cache hit rate | 30.0% | 60.3% (+100.9%) | 61.3% (+104.3%) | 68.5% (+128.2%) |
| shared_prefix_rag | Token savings | 0.0% | 17.6% (+17.6pp) | 26.8% (+26.8pp) | 28.1% (+28.1pp) |
| shared_prefix_rag | Expert cache hit rate | 42.0% | 40.4% (-3.8%) | 42.1% (+0.2%) | 43.1% (+2.7%) |
| multi_turn_conversation | TTFT p50 | 1.100s | 0.909s (-17.3%) | 0.877s (-20.3%) | 0.832s (-24.3%) |
| multi_turn_conversation | TTFT p90 | 1.450s | 1.199s (-17.3%) | 1.156s (-20.3%) | 1.097s (-24.3%) |
| multi_turn_conversation | TTFT p99 | 1.700s | 1.405s (-17.3%) | 1.355s (-20.3%) | 1.286s (-24.3%) |
| multi_turn_conversation | E2E latency p50 | 2.100s | 1.913s (-8.9%) | 1.867s (-11.1%) | 1.588s (-24.4%) |
| multi_turn_conversation | Prefill throughput | 880.0 tok/s | 1014.8 tok/s (+15.3%) | 1072.1 tok/s (+21.8%) | 1088.1 tok/s (+23.6%) |
| multi_turn_conversation | KV cache hit rate | 28.0% | 51.4% (+83.5%) | 58.5% (+109.1%) | 64.3% (+129.8%) |
| multi_turn_conversation | Token savings | 0.0% | 16.5% (+16.5pp) | 26.3% (+26.3pp) | 27.3% (+27.3pp) |
| multi_turn_conversation | Expert cache hit rate | 40.0% | 38.3% (-4.3%) | 40.6% (+1.5%) | 41.4% (+3.6%) |
| batch_with_overlap | TTFT p50 | 0.950s | 0.775s (-18.4%) | 0.738s (-22.4%) | 0.730s (-23.2%) |
| batch_with_overlap | TTFT p90 | 1.250s | 1.020s (-18.4%) | 0.970s (-22.4%) | 0.960s (-23.2%) |
| batch_with_overlap | TTFT p99 | 1.550s | 1.265s (-18.4%) | 1.203s (-22.4%) | 1.191s (-23.2%) |
| batch_with_overlap | E2E latency p50 | 1.800s | 1.579s (-12.3%) | 1.540s (-14.4%) | 1.386s (-23.0%) |
| batch_with_overlap | Prefill throughput | 1020.0 tok/s | 1123.5 tok/s (+10.1%) | 1270.9 tok/s (+24.6%) | 1276.7 tok/s (+25.2%) |
| batch_with_overlap | KV cache hit rate | 35.0% | 61.5% (+75.6%) | 65.7% (+87.7%) | 78.3% (+123.6%) |
| batch_with_overlap | Token savings | 0.0% | 17.9% (+17.9pp) | 22.8% (+22.8pp) | 29.6% (+29.6pp) |
| batch_with_overlap | Expert cache hit rate | 44.0% | 42.7% (-3.1%) | 46.8% (+6.4%) | 45.0% (+2.4%) |
| no_overlap_baseline | TTFT p50 | 1.350s | 1.106s (-18.1%) | 1.092s (-19.1%) | 0.974s (-27.8%) |
| no_overlap_baseline | TTFT p90 | 1.800s | 1.474s (-18.1%) | 1.456s (-19.1%) | 1.299s (-27.8%) |
| no_overlap_baseline | TTFT p99 | 2.100s | 1.720s (-18.1%) | 1.699s (-19.1%) | 1.515s (-27.8%) |
| no_overlap_baseline | E2E latency p50 | 2.350s | 2.029s (-13.6%) | 2.006s (-14.6%) | 1.769s (-24.7%) |
| no_overlap_baseline | Prefill throughput | 820.0 tok/s | 969.8 tok/s (+18.3%) | 978.0 tok/s (+19.3%) | 995.7 tok/s (+21.4%) |
| no_overlap_baseline | KV cache hit rate | 16.0% | 49.1% (+206.7%) | 50.1% (+212.9%) | 55.2% (+245.0%) |
| no_overlap_baseline | Token savings | 0.0% | 16.5% (+16.5pp) | 28.4% (+28.4pp) | 29.1% (+29.1pp) |
| no_overlap_baseline | Expert cache hit rate | 38.0% | 39.5% (+3.9%) | 39.2% (+3.2%) | 39.1% (+2.8%) |
