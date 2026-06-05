import chess

# Phase weights mirror the engine's eval.cpp: P,N,B,R,Q,K -> 0,1,1,2,4,0
_PIECE_PHASE = {
    chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 1,
    chess.ROOK: 2, chess.QUEEN: 4, chess.KING: 0,
}
MAX_PHASE = 24

RESULT_TO_WDL = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}


def game_phase(board: chess.Board) -> int:
    phase = 0
    for pt, w in _PIECE_PHASE.items():
        if w:
            phase += w * (len(board.pieces(pt, chess.WHITE))
                        + len(board.pieces(pt, chess.BLACK)))
    return min(phase, MAX_PHASE)


def board_to_tensor(board: chess.Board):
    #Board -> (12, 8, 8) int8 tensor. Channels 0-5 white P,N,B,R,Q,K; 6-11 black
    import numpy as np
    t = np.zeros((12, 8, 8), dtype=np.int8)
    for sq, piece in board.piece_map().items():
        ch = (piece.piece_type - 1) + (0 if piece.color == chess.WHITE else 6)
        t[ch, chess.square_rank(sq), chess.square_file(sq)] = 1
    return t


def move_to_index(move: chess.Move) -> int:
    return move.from_square * 64 + move.to_square

def passed_pawns(board: chess.Board, color: bool):
    result = []
    enemy = list(board.pieces(chess.PAWN, not color))
    for sq in board.pieces(chess.PAWN, color):
        f, r = chess.square_file(sq), chess.square_rank(sq)
        blocked = False
        for esq in enemy:
            if abs(chess.square_file(esq) - f) > 1:
                continue
            er = chess.square_rank(esq)
            if (color == chess.WHITE and er > r) or (color == chess.BLACK and er < r):
                blocked = True
                break
        if not blocked:
            result.append(sq)
    return result

def _king_edge_dist(sq: int) -> int:
    #Chebyshev distance from a square to the nearest board edge
    r, f = chess.square_rank(sq), chess.square_file(sq)
    return min(r, 7 - r, f, 7 - f)


def _bishop_square_color(board: chess.Board, color: bool) -> int:
    bishops = list(board.pieces(chess.BISHOP, color))
    if len(bishops) != 1:
        return -1
    sq = bishops[0]
    return (chess.square_file(sq) + chess.square_rank(sq)) & 1  # 1=light, 0=dark

_SIG_ORDER = [(chess.QUEEN, "Q"), (chess.ROOK, "R"), (chess.BISHOP, "B"),
            (chess.KNIGHT, "N"), (chess.PAWN, "P")]
_SIG_VALUE = {chess.QUEEN: 9, chess.ROOK: 5, chess.BISHOP: 3,
            chess.KNIGHT: 3, chess.PAWN: 1}


def material_signature(board: chess.Board) -> str:
    #Normalized material signature, stronger side first (e.g. 'KRPvKRP').
    def side(color):
        s = "K"
        for pt, ch in _SIG_ORDER:
            s += ch * len(board.pieces(pt, color))
        return s
    w, b = side(chess.WHITE), side(chess.BLACK)
    wv = sum(_SIG_VALUE[pt] * len(board.pieces(pt, chess.WHITE)) for pt, _ in _SIG_ORDER)
    bv = sum(_SIG_VALUE[pt] * len(board.pieces(pt, chess.BLACK)) for pt, _ in _SIG_ORDER)
    return f"{w}v{b}" if (wv, w) >= (bv, b) else f"{b}v{w}"


def _pawn_structure_counts(board: chess.Board, color: bool):
    file_counts = [0] * 8
    for sq in board.pieces(chess.PAWN, color):
        file_counts[chess.square_file(sq)] += 1
    doubled = isolated = 0
    for f in range(8):
        if file_counts[f] > 1:
            doubled += file_counts[f] - 1
        if file_counts[f] > 0:
            has_adj = ((f > 0 and file_counts[f - 1] > 0)
                    or (f < 7 and file_counts[f + 1] > 0))
            if not has_adj:
                isolated += file_counts[f]
    return doubled, isolated


def _connected_passers(passers) -> int:
    #1 if `color` has two passed pawns on adjacent files (connected passers are far stronger than the sum of two isolated passers), else 0
    files = sorted({chess.square_file(p) for p in passers})
    return int(any(b - a == 1 for a, b in zip(files, files[1:])))


def _rook_features(board: chess.Board, color: bool) -> dict:
    # Rook features: on semi-open file (no own pawns), and distance to the most advanced enemy passer (if any). Both are important for rook activity in endgames.
    f = {"rook_on_semiopen_file": 0, "rook_dist_enemy_passer": 0}
    rooks = list(board.pieces(chess.ROOK, color))
    if not rooks:
        return f
    own_pawn_files = {chess.square_file(p) for p in board.pieces(chess.PAWN, color)}
    for r in rooks:
        if chess.square_file(r) not in own_pawn_files:
            f["rook_on_semiopen_file"] = 1
            break
    enemy_passers = passed_pawns(board, not color)
    if enemy_passers:
        if (not color) == chess.WHITE:
            adv_enemy = max(enemy_passers, key=chess.square_rank)
        else:
            adv_enemy = min(enemy_passers, key=chess.square_rank)
        f["rook_dist_enemy_passer"] = min(
            chess.square_distance(r, adv_enemy) for r in rooks)
    return f


def _side_features(board: chess.Board, color: bool) -> dict:
    f = {}
    f["pawns"] = len(board.pieces(chess.PAWN, color))
    f["knights"] = len(board.pieces(chess.KNIGHT, color))
    f["bishops"] = len(board.pieces(chess.BISHOP, color))
    f["rooks"] = len(board.pieces(chess.ROOK, color))
    f["queens"] = len(board.pieces(chess.QUEEN, color))

    own_k = board.king(color)
    f["king_edge_dist"] = (_king_edge_dist(own_k) if own_k is not None else 0)
    f["bishop_color"] = _bishop_square_color(board, color)

    enemy_k = board.king(not color)
    all_pawns = list(board.pieces(chess.PAWN, color))
    f["king_dist_own_pawn"] = (
        min(chess.square_distance(own_k, p) for p in all_pawns)
        if own_k is not None and all_pawns else 0)

    doubled, isolated = _pawn_structure_counts(board, color)
    f["doubled"] = doubled
    f["isolated"] = isolated

    passers = passed_pawns(board, color)
    f["passed_count"] = len(passers)
    f["connected_passers"] = _connected_passers(passers)

    rf = _rook_features(board, color)
    f["rook_on_semiopen_file"] = rf["rook_on_semiopen_file"]
    f["rook_dist_enemy_passer"] = rf["rook_dist_enemy_passer"]

    if not passers:
        f["pawn_steps"] = 0
        f["own_king_dist_promo"] = 0
        f["enemy_king_dist_promo"] = 0
        f["enemy_king_outside_square"] = 0
        return f

    if color == chess.WHITE:
        sq = max(passers, key=chess.square_rank)
        steps = 7 - chess.square_rank(sq)
        promo = chess.square(chess.square_file(sq), 7)
    else:
        sq = min(passers, key=chess.square_rank)
        steps = chess.square_rank(sq)
        promo = chess.square(chess.square_file(sq), 0)

    f["pawn_steps"] = steps
    f["own_king_dist_promo"] = (chess.square_distance(own_k, promo)
                                if own_k is not None else 0)
    ekd = chess.square_distance(enemy_k, promo) if enemy_k is not None else 0
    f["enemy_king_dist_promo"] = ekd
    # Rule of the square: the defending (enemy) king cannot catch the pawn when its Chebyshev distance to the promotion square exceeds the pawn's steps.
    # The pawn's side moving gains a tempo, so compare against steps-1 when it is `color`'s turn, steps otherwise.
    if enemy_k is not None:
        effective = steps - 1 if board.turn == color else steps
        f["enemy_king_outside_square"] = int(ekd > effective)
    else:
        f["enemy_king_outside_square"] = 0
    return f


# Per-side keys emitted in stm/opp pairs, in vector order.
_SIDE_KEYS = ("pawns", "knights", "bishops", "rooks", "queens",
            "king_edge_dist", "bishop_color",
            "king_dist_own_pawn",
            "doubled", "isolated", "passed_count", "connected_passers",
            "rook_on_semiopen_file", "rook_dist_enemy_passer",
            "pawn_steps", "own_king_dist_promo", "enemy_king_dist_promo",
            "enemy_king_outside_square")

def endgame_features(board: chess.Board) -> dict:
    #STM-relative endgame feature vector.
    stm = board.turn
    s = _side_features(board, stm)
    o = _side_features(board, not stm)

    feats = {}

    # Symmetric king-vs-king geometry (orientation-independent).
    wk, bk = board.king(chess.WHITE), board.king(chess.BLACK)
    if wk is not None and bk is not None:
        king_dist = chess.square_distance(wk, bk)
        feats["king_dist"] = king_dist
        collinear = (chess.square_file(wk) == chess.square_file(bk)
                    or chess.square_rank(wk) == chess.square_rank(bk))
        feats["kings_collinear"] = int(collinear)
        feats["king_gap_parity"] = int(collinear and ((king_dist - 1) % 2 == 1))
    else:
        feats["king_dist"] = 0
        feats["kings_collinear"] = 0
        feats["king_gap_parity"] = 0

    # Opposite-colored bishops (each side exactly one bishop, on opposite square colors)
    cs = _bishop_square_color(board, chess.WHITE)
    co = _bishop_square_color(board, chess.BLACK)
    feats["opposite_bishops"] = int(cs != -1 and co != -1 and cs != co)

    # Per-side quantities, stm first then opp, in fixed order.
    for key in _SIDE_KEYS:
        feats[key + "_stm"] = s[key]
        feats[key + "_opp"] = o[key]
    return feats