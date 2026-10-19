"""Pure planning of contiguous MoE rows from already computed real routes.

Only frozen experts avoid volatile slot capacity. Current volatile hits still
occupy slots and must be included in each chunk's protected union. This module
does not import Torch, predict routes, reorder rows, or mutate residency.
"""
from numbers import Integral


def plan_route_chunks(routes, frozen, layer, stage_limit, *, max_rows=42, max_slots=256):
    """Return maximal feasible contiguous [start,end) spans, in original order.

    ``routes`` is a rectangular Python sequence from one actual indices.tolist
    transfer; ``frozen`` supports ordinary (layer,expert) key membership. It must
    stay unchanged until all spans finish. Single-row impossibility is rejected
    before any planned span is executed. Routed-slot count includes duplicates.
    """
    # A caller may explicitly select384/1536/3072 routed-slot tiers. Those count
    # repeated routes, not weight capacity: actual cold union stays separately
    # bounded by stage_limit. Defaults remain42/256 and never read env.
    expanded = max_slots == 384 and not isinstance(max_slots, bool)
    large = max_slots == 1536 and not isinstance(max_slots, bool)
    largest = max_slots == 3072 and not isinstance(max_slots, bool)
    joint1024 = max_slots == 6144 and not isinstance(max_slots, bool)
    for name,value,minimum,maximum in (("layer",layer,0,None),("stage_limit",stage_limit,0,None),
                                      ("max_rows",max_rows,1,1024 if joint1024 else 512 if largest else 256 if large else 64 if expanded else 42),
                                      ("max_slots",max_slots,1,6144 if joint1024 else 3072 if largest else 1536 if large else 384 if expanded else 256)):
        if (isinstance(value,bool) or not isinstance(value,Integral) or value<minimum
                or maximum is not None and value>maximum):
            raise ValueError(f"invalid {name}")
    if not routes:
        return []
    topk=len(routes[0])
    if topk<1 or topk>max_slots:
        raise ValueError("one route row exceeds the routed-slot limit or is empty")
    row_limit=min(int(max_rows),int(max_slots)//topk)
    cold_rows=[]
    for row in routes:
        if len(row)!=topk:
            raise ValueError("route rows must have the same topk")
        if any(isinstance(expert,bool) or not isinstance(expert,Integral) or not 0<=expert<384 for expert in row):
            raise ValueError("route expert IDs must be integers in [0,384)")
        cold={int(expert) for expert in row if (layer,int(expert)) not in frozen}
        if len(cold)>stage_limit:
            raise ValueError("one route row needs more non-frozen experts than volatile capacity")
        cold_rows.append(cold)
    spans=[]
    start=0
    union=set()
    for end,cold in enumerate(cold_rows):
        enlarged=union|cold
        if end-start>=row_limit or len(enlarged)>stage_limit:
            spans.append((start,end))
            start=end
            union=set(cold)
        else:
            union=enlarged
    spans.append((start,len(cold_rows)))
    return spans
