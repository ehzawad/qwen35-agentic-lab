"""The three suite v1 task families.

  lookup_chain   pure multi-hop retrieval through kb_lookup (H 2/4/8/12)
  typed_relay    typed orchestration across kb_lookup, unit_convert,
                 calculator (H 2/4/8/12)
  fulfillment    stateful warehouse environment with two environment-scoped
                 tools, warehouse_query and warehouse_update (H 4/8/14/20)

Each module exposes the same generator interface:

  FAMILY                     family name
  HORIZONS                   supported horizons
  has_unit_convert(h)        whether the cell contains a unit_convert node
  generate_task(rng, h)      -> schema.TaskDraft
  render_prompt(template_id, fields) -> str   (12 paraphrase templates)

plus a thin TRL-facing Env class whose reset(**row) rebuilds the episode
runtime exactly from serialized spec/kb/oracle rows.
"""

from __future__ import annotations


def family_module(family: str):
    if family == "lookup_chain":
        from . import lookup_chain as m
    elif family == "typed_relay":
        from . import typed_relay as m
    elif family == "fulfillment":
        from . import fulfillment as m
    else:
        raise ValueError(f"unknown family {family!r}")
    return m
