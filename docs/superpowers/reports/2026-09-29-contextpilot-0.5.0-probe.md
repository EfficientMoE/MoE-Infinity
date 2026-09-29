# ContextPilot 0.5.0 probe

Throwaway venv: `/tmp/cp050-probe`. Host interpreter was left on `contextpilot` 0.4.1 and was not modified.

## version

```
0.5.0
```

Command: `python -c "import contextpilot; print(contextpilot.__version__)"` after `pip install 'contextpilot==0.5.0'`.

## pip check

```
No broken requirements found.
```

Installed because ContextPilot requires them, not because MoE-Infinity does:

- `elasticsearch` 8.18.1
- `transformers` 5.17.0

`pip check` reported no conflict inside the probe venv. These packages stay out of MoE-Infinity `install_requires`.

## hook delta

`CONTEXTPILOT` and `CONTEXTPILOT_INDEX_URL` unset:

```
0.5.0
moe False
```

The process started, printed `0.5.0`, and did not import `moe_infinity`.

## dedup first turn

```
ValueError
No prior .reorder() call found for conversation_id='user-a'. Call .reorder(contexts, conversation_id='user-a') first to register the initial documents.
```

## remove unknown

```
{'removed_count': 0, 'evicted_node_ids': [], 'evicted_request_ids': [], 'not_found': ['serving-id'], 'nodes_remaining': 0, 'requests_remaining': 0}
```

`not_found` contains `serving-id`.

## verdict

MATCH
