from app.menu_parser import OcrBox, extract_items, parse_price


def box(x, y, text, conf=0.9, w=80, h=20):
    return OcrBox(points=[(x, y), (x + w, y), (x + w, y + h), (x, y + h)], text=text, conf=conf)


def test_parse_price_formats():
    assert parse_price("삼겹살 15,000원") == (15000, "삼겹살")
    assert parse_price("₩9000 비빔밥")[0] == 9000
    assert parse_price("공기밥")[0] is None


def test_rows_are_grouped_and_read_left_to_right():
    boxes = [
        box(300, 102, "12,000"), box(20, 100, "불고기"),          # same row, out of order
        box(20, 140, "된장찌개"), box(300, 141, "9,000원"),
        box(150, 300, "3", conf=0.95),                           # lone page number -> dropped
    ]
    items = extract_items(boxes)
    assert [(i.name, i.price) for i in items] == [("불고기", 12000), ("된장찌개", 9000)]


def test_low_confidence_boxes_are_ignored():
    assert extract_items([box(0, 0, "삼겹살", conf=0.1)]) == []


def test_headings_without_price_are_dropped_on_priced_menu():
    boxes = [
        box(20, 10, "샐러드"),                                   # section heading
        box(20, 50, "닭가슴살 샐러드 ........"), box(300, 50, "20,000원"),
        box(20, 90, "감자 샐러드"), box(300, 90, "15,000원"),
    ]
    items = extract_items(boxes)
    assert [(i.name, i.price) for i in items] == [("닭가슴살 샐러드", 20000), ("감자 샐러드", 15000)]


def test_menu_without_prices_keeps_all_rows():
    items = extract_items([box(20, 10, "비빔밥"), box(20, 50, "불고기")])
    assert [i.name for i in items] == ["비빔밥", "불고기"]
