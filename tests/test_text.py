from rag_cskh.bm25 import BM25
from rag_cskh.text import fold, tokens


def test_fold_removes_diacritics():
    assert fold("Bảo hành đổi trả") == "Bao hanh doi tra"


def test_query_without_diacritics_matches_accented_document():
    doc, query = set(tokens("Thời gian bảo hành tối đa 15 ngày")), tokens("thoi gian bao hanh")
    assert "~bao_hanh" in doc and "~bao_hanh" in query


def test_numbers_are_normalised():
    assert "3000000" in tokens("đơn từ 3.000.000 đồng")


def test_bm25_ranks_the_matching_document_first():
    docs = [tokens("Lỗi E21 là lỗi thoát nước"), tokens("Miễn phí giao hàng trong 20 km")]
    assert BM25(docs).rank(tokens("may giat bao loi e21"))[0] == 0
