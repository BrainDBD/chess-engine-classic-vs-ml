import chess
import chess.pgn
import chess.syzygy
import argparse, json, math, random, sys

from endgame_features import material_signature, move_to_index

RESULT_TO_WDL = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}

def probe_syzygy(tb, board: chess.Board):
    #Return {wdl, stm_result, white_result, dtz} or None if not in tablebase.
    try:
        wdl = tb.probe_wdl(board)
    except (KeyError, ValueError):
        return None  # not in tables, or castling rights / wrong piece count
    if wdl == 2:        stm = 1.0   # real win
    elif wdl == -2:     stm = 0.0   # real loss
    else:               stm = 0.5   # draw, cursed win (+1), blessed loss (-1) -> all practical draws
    out = {
        "wdl": wdl,
        "stm_result": stm,
        "white_result": stm if board.turn == chess.WHITE else 1.0 - stm,
    }
    try:
        dtz = tb.probe_dtz(board)
    except (KeyError, ValueError):
        dtz = None
    out["dtz"] = dtz
    # Normalized DTZ label for analysis / an optional regression target:
    # tanh(|dtz|/100) maps conversion DEPTH to (0,1)
    out["dtz_norm"] = (math.tanh(abs(dtz) / 100.0) if dtz and abs(wdl) == 2 else 0.0)
    return out

def _int_header(game, key):
    try:
        return int(game.headers.get(key, ""))
    except ValueError:
        return None
    
def game_passes_filters(game, cfg) -> bool:
    if game.headers.get("Result", "*") not in RESULT_TO_WDL:
        return False
    if cfg.min_elo > 0:
        we, be = _int_header(game, "WhiteElo"), _int_header(game, "BlackElo")
        if we is None or be is None or we < cfg.min_elo or be < cfg.min_elo:
            return False
    if cfg.normal_only:
        term = game.headers.get("Termination", "")
        if term and term not in ("Normal", "Time forfeit"):
            return False
    return True

def iter_positions(game, cfg, tb=None, game_id=-1):
    #Yield compact feature records for selected positions within one game.
    white_wdl = RESULT_TO_WDL[game.headers["Result"]]
    board = game.board()
    ply = 0

    for move in game.mainline_moves():
        in_check = board.is_check()
        is_capture = board.is_capture(move)
        is_promo = move.promotion is not None

        select = (ply >= cfg.opening_skip)
        if cfg.quiet_only:
            select = select and not in_check and not is_capture and not is_promo
        if select and cfg.max_pieces:
            select = chess.popcount(board.occupied) <= cfg.max_pieces
        if select and cfg.min_pieces:
            select = chess.popcount(board.occupied) >= cfg.min_pieces
        if select and cfg.configs:
            select = material_signature(board) in cfg.configs
        if select and cfg.sample_rate < 1.0:
            select = random.random() < cfg.sample_rate

        if select:
            stm_wdl = white_wdl if board.turn == chess.WHITE else (1.0 - white_wdl)
            record = {
                "fen": board.fen(),
                "game_id": game_id,
                "stm_wdl": stm_wdl,
            }
            if cfg.keep_move:
                record["move_uci"] = move.uci()
                record["move_index"] = move_to_index(move)
            keep = True
            if tb is not None:
                sy = probe_syzygy(tb, board)
                if sy is not None:
                    record["syzygy"] = sy
                elif cfg.syzygy_only:
                    keep = False  # drop positions Syzygy can't label
            if keep:
                yield record

        board.push(move)
        ply += 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pgn", help="input PGN file (use - for stdin)")
    p.add_argument("-o", "--out", required=True, help="output JSONL path")
    p.add_argument("--min-elo", type=int, default=0,
                    help="require BOTH players >= this rating (0 = no filter)")
    p.add_argument("--normal-only", action="store_true",
                    help="drop games with abnormal termination")
    p.add_argument("--opening-skip", type=int, default=8,
                    help="skip the first N plies of each game (book noise)")
    p.add_argument("--sample-rate", type=float, default=1.0,
                    help="probability of keeping each eligible position (0-1)")
    p.add_argument("--quiet-only", action="store_true",
                help="keep only quiet positions (no check/capture/promo)")
    p.add_argument("--max-pieces", type=int, default=0,
                    help="keep only positions with <= N total pieces (incl. kings). "
                        "Use 7 to match 7-man Syzygy. 0 = no filter")
    p.add_argument("--min-pieces", type=int, default=0,
                    help="keep only positions with >= N total pieces (incl. kings). "
                        "Set equal to --max-pieces to extract an exact count. "
                        "0 = no filter")
    p.add_argument("--configs", default=None,
                    help="comma-separated material signatures to keep, e.g. "
                        "'KPPvKPP,KRPvKRP'. Stronger side first, color-independent. "
                        "Restricts to positions in these configs. Default: no filter")
    p.add_argument("--keep-move", action="store_true",
                    help="also store the played move (move_uci/move_index). The "
                        "ML pipeline does not use it -- the geometry features are "
                        "recomputed from the FEN at load time, so no feature block "
                        "is stored -- but a future move-ordering/policy experiment "
                        "might want it. Off by default to keep records lean")
    p.add_argument("--syzygy", default=None, metavar="DIR[,DIR2,...]",
                    help="Syzygy tablebase directory, or several comma-separated "
                        "(e.g. '3-4-5-wdl,6-wdl'); adds oracle labels. Probing a "
                        "6-piece position needs both the 6-man files AND the 5-man "
                        "files reachable by capture, so pass both directories")
    p.add_argument("--syzygy-only", action="store_true",
                    help="drop positions that Syzygy cannot label (requires --syzygy)")
    p.add_argument("--max-games", type=int, default=0,
                    help="stop after this many games read (0 = all)")
    p.add_argument("--max-positions", type=int, default=0,
                    help="stop after this many positions written (0 = all)")
    cfg = p.parse_args(argv)

    if cfg.syzygy_only and not cfg.syzygy:
        p.error("--syzygy-only requires --syzygy")

    # Parse the material-config whitelist into a set (None = no filter).
    cfg.configs = (set(s.strip() for s in cfg.configs.split(",") if s.strip()) if cfg.configs else None)

    tb = None
    if cfg.syzygy:
        dirs = [d.strip() for d in cfg.syzygy.split(",") if d.strip()]
        tb = chess.syzygy.open_tablebase(dirs[0])
        for d in dirs[1:]:
            tb.add_directory(d)   # combine 3-4-5 and 6-man tables for probing
        if len(dirs) > 1:
            print(f"opened {len(dirs)} tablebase directories: {dirs}", file=sys.stderr)

    pgn_in = sys.stdin if cfg.pgn == "-" else open(cfg.pgn, "r", encoding="utf-8", errors="replace")
    games_read = games_kept = positions = 0
    try:
        with open(cfg.out, "w", encoding="utf-8") as out:
            while True:
                game = chess.pgn.read_game(pgn_in)
                if game is None:
                    break
                games_read += 1
                if cfg.max_games and games_read > cfg.max_games:
                    break
                if not game_passes_filters(game, cfg):
                    continue
                games_kept += 1
                for record in iter_positions(game, cfg, tb, game_id=games_read):
                    out.write(json.dumps(record, separators=(",", ":")) + "\n")
                    positions += 1
                    if cfg.max_positions and positions >= cfg.max_positions:
                        raise StopIteration
                if games_read % 2000 == 0:
                    print(f"...{games_read} read, {games_kept} kept, "
                            f"{positions} positions", file=sys.stderr)
    except StopIteration:
        pass
    finally:
        if pgn_in is not sys.stdin:
            pgn_in.close()
        if tb is not None:
            tb.close()

    print(f"Done. {games_read} games read, {games_kept} kept, "
            f"{positions} positions written to {cfg.out}", file=sys.stderr)


if __name__ == "__main__":
    main()