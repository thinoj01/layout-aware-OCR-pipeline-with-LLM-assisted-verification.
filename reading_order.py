"""Geometry-based *provisional* newspaper reading order after layout detection.

A detector identifies text regions, not a reading sequence. For multi-column
newspapers, reading each vertical column from top to bottom is a stronger
default than model output order or global top-to-bottom sorting.

This is NOT article segmentation. Cross-column blocks and ambiguous layouts
are marked for human review.
"""
from __future__ import annotations
from statistics import median


def _box(region):
    b = region["bbox"] if isinstance(region, dict) else region.bbox
    return tuple(map(float, b))


def _text_width(region):
    x0, _, x1, _ = _box(region)
    return x1 - x0


def _height(region):
    _, y0, _, y1 = _box(region)
    return y1 - y0


def _lane_bounds(regions, page_width, min_gap=None):
    """Infer horizontal column intervals from overlap of non-wide text boxes.

    Avoiding wide titles and section banners prevents them from joining
    independent columns. Only gaps supported by multiple text boxes matter.
    """
    if not regions:
        return []
    width = float(page_width)
    # Local section headers can cover two neighboring columns while occupying
    # less than half of the page. Exclude width outliers, not just pagewide text.
    candidate_widths=[_text_width(r) for r in regions
                      if .10*width <= _text_width(r) <= .48*width]
    typical=median(candidate_widths) if candidate_widths else width*.35
    max_column_width=min(.48*width,1.45*typical)
    intervals = []
    for region in regions:
        x0, _, x1, _ = _box(region)
        if x1 - x0 > max_column_width:
            continue
        if x1 - x0 < max(12, .012 * width):
            continue
        intervals.append((x0, x1))
    if not intervals:
        return [(0.0, width)]
    intervals.sort()
    # Merge overlapping x ranges. A small overlap does not automatically mean
    # two newspaper columns should merge: require substantial shared range.
    # Interval spans are first gathered into visually connected bands.
    groups = []
    for left, right in intervals:
        if not groups or left > groups[-1][1]:
            groups.append([left, right, 1])
        else:
            groups[-1][1] = max(groups[-1][1], right)
            groups[-1][2] += 1

    # If small boxes have broken a column into multiple islands, merge islands
    # separated by less than the likely inter-column gutter.
    gap = float(min_gap if min_gap is not None else max(6, width * .012))
    joined = []
    for left, right, count in groups:
        if joined and left - joined[-1][1] <= gap:
            joined[-1][1] = max(joined[-1][1], right)
            joined[-1][2] += count
        else:
            joined.append([left, right, count])

    # A single isolated narrow element (e.g. page number) is not a column.
    # However, retain lanes for tiny pages with only a few regions.
    substantive = [g for g in joined if g[2] >= 2 or len(regions) < 5]
    if not substantive:
        substantive = joined
    return [(g[0], g[1]) for g in substantive]


def order_regions(regions, page_width, page_height):
    """Return sorted region objects with order, column and review metadata.

    The input is not mutated. 'column_index' is 0 for page headers and 1-based
    thereafter. A full-width heading across the top is read before any column;
    body columns then flow left-to-right and top-to-bottom. Original OCR crops,
    confidence and researcher-entered text are preserved.
    """
    if not regions:
        return []
    width = float(page_width)
    height = float(page_height)
    copied = [dict(r) for r in regions]
    if width <= 0 or height <= 0:
        raise ValueError("Page dimensions must be positive")

    broad = []
    for region in copied:
        x0, y0, x1, y1 = _box(region)
        is_wide=(x1-x0) >= .67*width
        if is_wide and y0 < .38*height:
            broad.append(region)

    # A page-wide top story headline is followed by all the page's columns.
    # Only regions above the last early wide band belong to the masthead;
    # body headings within narrow columns stay with their own columns.
    header_bottom=max((_box(r)[3] for r in broad),default=0.0)
    header_bottom=min(header_bottom, .46 * height)
    top=[]
    body=[]
    for region in copied:
        x0,y0,x1,y1=_box(region)
        if header_bottom and (y0+y1)/2 <= header_bottom:
            region["column_index"]=0
            top.append(region)
        else:
            body.append(region)

    # Preserve broad title ordering by lines, using a small vertical tolerance
    # to read left-to-right within a shared metadata row.
    row_tolerance=max(4, .007*height)
    def row_key(r):
        x0,y0,x1,y1=_box(r)
        return (round(y0/row_tolerance), x0, y0)

    top.sort(key=row_key)
    lanes=_lane_bounds(body,width)
    for region in body:
        x0,y0,x1,y1=_box(region)
        overlaps=[max(0,min(x1,right)-max(x0,left)) for left,right in lanes]
        winner=max(range(len(lanes)),key=lambda i:overlaps[i])
        # If it overlaps none, choose the nearest horizontal lane.
        if overlaps[winner]==0:
            mid=(x0+x1)/2
            winner=min(range(len(lanes)),
                       key=lambda i: abs(mid-(lanes[i][0]+lanes[i][1])/2))
        region["column_index"]=winner+1
        lane_left,lane_right=lanes[winner]
        # Wide regions may span several article columns: do not promise their
        # assigned order is unambiguous.
        competing=sum(overlap >= .25*(x1-x0) for overlap in overlaps)
        uncertain=(competing>1 or (x1-x0)>=.55*width)
        region["reading_order_uncertain"]=bool(uncertain)
        if uncertain:
            region["needs_review"]=True
    body.sort(key=lambda r:(r["column_index"],_box(r)[1],_box(r)[0]))
    ordered=top+body
    for i,region in enumerate(ordered,1):
        region["reading_order_guess"]=i
        region["reading_order_method"]="column_major_v1"
        if "reading_order_uncertain" not in region:
            region["reading_order_uncertain"]=False
    return ordered
