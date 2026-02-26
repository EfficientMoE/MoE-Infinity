#!/usr/bin/env python3
"""
build_ast_index.py — Build and persist a Tree-sitter AST function index.

Scans moe_infinity/ (Python) and core/ (C++/CUDA) and writes
.ast_index.json at the repo root containing:
  - definitions: list of {name, file, line, end_line, lang}
  - call_graph:  {caller -> [callee, ...]}

Usage:
    # From repo root
    python scripts/build_ast_index.py

    # Scan specific paths
    python scripts/build_ast_index.py moe_infinity/ core/extensions/

    # Print summary only (no file write)
    python scripts/build_ast_index.py --dry-run

    # Pretty-print the index
    python scripts/build_ast_index.py --summary

Requires (tree-sitter 0.23+):
    pip install tree-sitter tree-sitter-python tree-sitter-cpp
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Language config — tree-sitter 0.23 API (Query.matches / Query.captures)
# ---------------------------------------------------------------------------
LANG_CONFIG = {
    "python": {
        "extensions": {".py", ".pyi"},
        "pip_package": "tree-sitter-python",
        "module": "tree_sitter_python",
        "queries": {
            "definitions": """
                (function_definition
                  name: (identifier) @name) @def
            """,
            "calls": """
                (call function: (identifier) @name)
                (call function: (attribute attribute: (identifier) @name))
            """,
        },
    },
    "cpp": {
        "extensions": {".cpp", ".cc", ".cxx", ".cu", ".hpp", ".hxx", ".hh", ".h"},
        "pip_package": "tree-sitter-cpp",
        "module": "tree_sitter_cpp",
        "queries": {
            "definitions": """
                (function_definition
                  declarator: (function_declarator
                    declarator: (identifier) @name)) @def
                (function_definition
                  declarator: (function_declarator
                    declarator: (qualified_identifier
                      name: (identifier) @name))) @def
            """,
            "calls": """
                (call_expression function: (identifier) @name)
                (call_expression function: (qualified_identifier
                  name: (identifier) @name))
                (call_expression function: (field_expression
                  field: (field_identifier) @name))
            """,
        },
    },
}

EXT_TO_LANG = {}
for _lang, _cfg in LANG_CONFIG.items():
    for _ext in _cfg["extensions"]:
        EXT_TO_LANG[_ext] = _lang

# ---------------------------------------------------------------------------
# Parser cache
# ---------------------------------------------------------------------------
_PARSERS = {}  # lang -> (parser, language)


def _get_parser(lang_name):
    if lang_name in _PARSERS:
        return _PARSERS[lang_name]
    from tree_sitter import Language, Parser

    cfg = LANG_CONFIG[lang_name]
    mod = __import__(cfg["module"])
    lang_fn = getattr(mod, "language", None) or getattr(mod, "LANGUAGE", None)
    if lang_fn is None:
        raise RuntimeError(f"Cannot find language() in {cfg['module']}")
    language = Language(lang_fn())
    parser = Parser(language)
    _PARSERS[lang_name] = (parser, language)
    return parser, language


# ---------------------------------------------------------------------------
# Core parse logic (tree-sitter 0.23: Query.matches / Query.captures)
# ---------------------------------------------------------------------------
def _parse_file(filepath, lang_name):
    """Return (definitions, calls_per_func) for one file."""
    from tree_sitter import Query

    parser, language = _get_parser(lang_name)
    cfg = LANG_CONFIG[lang_name]
    source = Path(filepath).read_bytes()
    tree = parser.parse(source)
    root = tree.root_node

    # --- definitions ---
    def_q = Query(language, cfg["queries"]["definitions"])
    def_matches = def_q.matches(root)  # list of (pat_idx, {name: [nodes], def: [nodes]})

    definitions = []
    def_spans = []  # (name_str, start_byte, end_byte, line)

    for _pat, captures in def_matches:
        name_nodes = captures.get("name", [])
        def_nodes = captures.get("def", [])
        if not name_nodes:
            continue
        name_node = name_nodes[0]
        def_node = def_nodes[0] if def_nodes else name_node.parent
        fname = name_node.text.decode("utf8")
        definitions.append(
            {
                "name": fname,
                "file": str(filepath),
                "line": name_node.start_point[0] + 1,
                "end_line": def_node.end_point[0] + 1 if def_node else name_node.end_point[0] + 1,
                "lang": lang_name,
            }
        )
        def_spans.append(
            (
                fname,
                def_node.start_byte if def_node else name_node.start_byte,
                def_node.end_byte if def_node else name_node.end_byte,
            )
        )

    # --- calls ---
    call_q = Query(language, cfg["queries"]["calls"])
    call_captures = call_q.captures(root)  # {name: [nodes]}
    call_nodes = call_captures.get("name", [])

    calls_per_func = defaultdict(list)

    for call_node in call_nodes:
        call_name = call_node.text.decode("utf8")
        cb = call_node.start_byte
        enclosing = None
        enc_size = None
        for fname, sb, eb in def_spans:
            if sb <= cb <= eb:
                size = eb - sb
                if enc_size is None or size < enc_size:
                    enclosing = fname
                    enc_size = size
        if enclosing:
            calls_per_func[enclosing].append(call_name)

    return definitions, dict(calls_per_func)


# ---------------------------------------------------------------------------
# File collection
# ---------------------------------------------------------------------------
def _collect_files(paths):
    files = []
    for p in paths:
        path = Path(p)
        if path.is_file():
            lang = EXT_TO_LANG.get(path.suffix)
            if lang:
                files.append((path, lang))
        elif path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file():
                    lang = EXT_TO_LANG.get(child.suffix)
                    if lang:
                        files.append((child, lang))
    return files


# ---------------------------------------------------------------------------
# Build index
# ---------------------------------------------------------------------------
def build_index(paths):
    """Build the full AST index from the given paths."""
    files = _collect_files(paths)
    if not files:
        print("No supported source files found.", file=sys.stderr)
        sys.exit(1)

    all_defs = []
    all_calls = defaultdict(list)
    errors = []

    for filepath, lang_name in files:
        try:
            defs, calls = _parse_file(filepath, lang_name)
            all_defs.extend(defs)
            for fn, callees in calls.items():
                all_calls[fn].extend(callees)
        except Exception as exc:
            errors.append({"file": str(filepath), "error": str(exc)})

    # Deduplicate callees
    call_graph = {fn: sorted(set(callees)) for fn, callees in all_calls.items()}

    return {
        "definitions": all_defs,
        "call_graph": call_graph,
        "files_scanned": len(files),
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
DEFAULT_PATHS = ["moe_infinity", "core", "extensions"]
INDEX_FILE = ".ast_index.json"


def main():
    ap = argparse.ArgumentParser(description="Build Tree-sitter AST index for MoE-Infinity")
    ap.add_argument(
        "paths",
        nargs="*",
        default=DEFAULT_PATHS,
        help=f"Paths to scan (default: {', '.join(DEFAULT_PATHS)})",
    )
    ap.add_argument(
        "--output",
        default=INDEX_FILE,
        help=f"Output JSON file (default: {INDEX_FILE})",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summary but do not write output file",
    )
    ap.add_argument(
        "--summary",
        action="store_true",
        help="Print a human-readable summary after writing",
    )
    args = ap.parse_args()

    # Resolve paths relative to repo root (where this script lives in scripts/)
    repo_root = Path(__file__).parent.parent
    resolved = [repo_root / p for p in args.paths]

    print(f"Scanning: {', '.join(str(r) for r in resolved)}", file=sys.stderr)
    index = build_index(resolved)

    n_defs = len(index["definitions"])
    n_callers = len(index["call_graph"])
    n_files = index["files_scanned"]
    n_errors = len(index["errors"])

    for e in index["errors"]:
        print(f"  WARN parse error {e['file']}: {e['error']}", file=sys.stderr)

    print(
        f"Index: {n_files} files | {n_defs} definitions | {n_callers} callers "
        f"| {n_errors} errors",
        file=sys.stderr,
    )

    if not args.dry_run:
        out = repo_root / args.output
        out.write_text(json.dumps(index, indent=2))
        print(f"Written: {out}", file=sys.stderr)

    if args.summary:
        langs = defaultdict(int)
        for d in index["definitions"]:
            langs[d["lang"]] += 1
        print("\nDefinitions by language:")
        for lang, cnt in sorted(langs.items()):
            print(f"  {lang:10s}: {cnt}")
        print("\nTop 10 most-called functions:")
        callee_counts = defaultdict(int)
        for callees in index["call_graph"].values():
            for c in callees:
                callee_counts[c] += 1
        for fn, cnt in sorted(callee_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"  {cnt:4d}x  {fn}")


if __name__ == "__main__":
    main()
