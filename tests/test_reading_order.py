"""Regression fixtures for newspaper column reading order.

Coordinates approximate an issue with masthead, wide lead headline,
three/four vertical stories, side headlines and editorial boxes.
"""
import random
import pytest
from reading_order import order_regions, _lane_bounds


def _region(name, box, kind="plain_text"):
    return {"id":name,"bbox":box,"text":name,"type":kind,
            "ocr_confidence":90,"needs_review":False}


def test_newspaper_masthead_then_left_column_then_right():
    items=[
        _region("right_bottom",[690,850,890,1080]),
        _region("middle_bottom",[260,730,460,1150]),
        _region("masthead",[110,95,805,210],"title"),
        _region("left_bottom",[25,780,225,1150]),
        _region("right_top",[690,410,890,810]),
        _region("date",[330,220,465,240]),
        _region("left_top",[20,410,225,755]),
        _region("main_headline",[25,260,895,395],"title"),
        _region("middle_top",[255,410,455,700]),
        _region("price",[710,220,895,240]),
    ]
    random.Random(2026).shuffle(items)
    ordered=order_regions(items,900,1200)
    assert [r["id"] for r in ordered]==[
        "masthead","date","price","main_headline",
        "left_top","left_bottom",
        "middle_top","middle_bottom",
        "right_top","right_bottom"]
    assert [r["reading_order_guess"] for r in ordered]==list(range(1,11))
    assert ordered[4]["column_index"]==1
    assert ordered[6]["column_index"]==2
    assert ordered[8]["column_index"]==3


def test_four_columns_with_fragmented_blocks():
    columns=[(20,215),(245,445),(475,675),(705,895)]
    items=[_region("front_page_title",[100,80,805,180],"title"),
           _region("lead",[25,250,890,375],"title")]
    for i,(x0,x1) in enumerate(columns,1):
        items.extend([
            _region(f"c{i}_subtitle",[x0,415,x1,470],"title"),
            _region(f"c{i}_body1",[x0,490,x1,660]),
            _region(f"c{i}_body2",[x0,685,x1,890]),
            _region(f"c{i}_body3",[x0,905,x1,1150])
        ])
    random.Random(33).shuffle(items)
    output=order_regions(items,920,1250)
    assert [r["id"] for r in output[:2]]==["front_page_title","lead"]
    assert [r["id"] for r in output[2:]]==[
        f"c{i}_{s}" for i in range(1,5)
        for s in ("subtitle","body1","body2","body3")]
    assert len({r["column_index"] for r in output[2:]})==4


def test_article_segments_do_not_interleave_by_vertical_position():
    items=[
        _region("left_early",[35,420,230,520]),
        _region("right_early",[700,410,880,510]),
        _region("left_late",[35,800,230,1000]),
        _region("right_late",[700,700,880,860]),
    ]
    ordered=order_regions(items,950,1100)
    assert [r["id"] for r in ordered]==[
        "left_early","left_late","right_early","right_late"]


def test_ambiguous_cross_column_fragment_is_flagged():
    items=[
        _region("left1",[20,400,230,520]),
        _region("left2",[20,540,230,760]),
        _region("right1",[270,420,470,530]),
        _region("right2",[270,550,470,760]),
        _region("cross_column",[30,780,460,805],"title"),
    ]
    output=order_regions(items,490,900)
    flagged=next(r for r in output if r["id"]=="cross_column")
    assert flagged["needs_review"]
    assert flagged["reading_order_uncertain"]


def test_deterministic_and_non_mutating():
    items=[_region("a",[10,110,120,180]),_region("b",[150,105,250,170])]
    order1=order_regions(items,300,250)
    order2=order_regions(list(reversed(items)),300,250)
    assert [r["id"] for r in order1]==[r["id"] for r in order2]
    assert "reading_order_guess" not in items[0]


@pytest.mark.parametrize("size",[(0,100),(100,0),(-1,100)])
def test_reject_invalid_dimensions(size):
    with pytest.raises(ValueError):
        order_regions([_region("a",[1,2,10,20])],*size)


def test_empty_page():
    assert order_regions([],950,1200)==[]
