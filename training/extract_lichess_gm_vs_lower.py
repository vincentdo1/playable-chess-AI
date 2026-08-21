"""Extract GM-vs-lower-rated games from a Lichess PGN dump.

Selected games are tagged with the policy side and opponent rating metadata.
"""

import argparse
import io
import os
import re
import sys

import chess.pgn


DEFAULT_BUCKETS = "0-999,1000-1199,1200-1399,1400-1599,1600-1799,1800-1999,2000-2199,2200-2399,2400-9999"
HEADER_RE = re.compile(r'^\[([A-Za-z0-9_]+)\s+"(.*)"\]\s*$')


def _parse_titles(value):
    return {part.strip().upper() for part in value.split(',') if part.strip()}


def _parse_rating_buckets(value):
    buckets = []
    for part in value.split(','):
        part = part.strip()
        if not part:
            continue
        low, high = part.split('-', 1)
        buckets.append((int(low), int(high), f"{int(low)}-{int(high)}"))
    return buckets


def _rating_bucket(elo, buckets):
    for low, high, name in buckets:
        if low <= elo <= high:
            return name
    return "unbucketed"


def _safe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _open_pgn_text(path):
    if path.lower().endswith('.zst'):
        try:
            import zstandard as zstd
        except ImportError as exc:
            raise SystemExit(
                "Missing dependency: zstandard\n"
                "Install it inside your environment with:\n"
                "  pip install zstandard"
            ) from exc

        compressed = open(path, 'rb')
        dctx = zstd.ZstdDecompressor()
        reader = dctx.stream_reader(compressed)
        text = io.TextIOWrapper(reader, encoding='utf-8', errors='replace')
        return compressed, text

    text = open(path, encoding='utf-8', errors='replace')
    return text, text


def _iter_pgn_blocks(text):
    block = []
    for line in text:
        if line.startswith('[Event ') and block:
            yield ''.join(block).strip() + '\n'
            block = [line]
        else:
            block.append(line)
    if block:
        yield ''.join(block).strip() + '\n'


def _parse_headers_from_block(block):
    headers = {}
    for line in block.splitlines():
        if not line.startswith('['):
            break
        match = HEADER_RE.match(line)
        if match:
            key, value = match.groups()
            headers[key] = value.replace(r'\"', '"').replace(r'\\', '\\')
    return headers


def _with_extra_headers(block, extra_headers):
    lines = block.splitlines()
    insert_at = 0
    for idx, line in enumerate(lines):
        if not line.startswith('['):
            insert_at = idx
            break
    else:
        insert_at = len(lines)

    existing = set()
    for line in lines[:insert_at]:
        match = HEADER_RE.match(line)
        if match:
            existing.add(match.group(1))

    extras = [
        f'[{key} "{value}"]'
        for key, value in extra_headers.items()
        if key not in existing
    ]
    return '\n'.join(lines[:insert_at] + extras + lines[insert_at:]) + '\n\n'


def _count_mainline_plies(block):
    game = chess.pgn.read_game(io.StringIO(block))
    if game is None:
        return 0
    return sum(1 for _ in game.mainline_moves())


def _gm_side(headers, gm_titles, gm_min_elo):
    white_title = headers.get('WhiteTitle', '').upper()
    black_title = headers.get('BlackTitle', '').upper()
    white_elo = _safe_int(headers.get('WhiteElo'))
    black_elo = _safe_int(headers.get('BlackElo'))

    white_is_gm = white_title in gm_titles
    black_is_gm = black_title in gm_titles
    if gm_min_elo is not None:
        white_is_gm = white_is_gm or (
            white_elo is not None and white_elo >= gm_min_elo
        )
        black_is_gm = black_is_gm or (
            black_elo is not None and black_elo >= gm_min_elo
        )

    if white_is_gm == black_is_gm:
        return None
    return chess.WHITE if white_is_gm else chess.BLACK


def _is_selected(headers, args, gm_titles):
    if headers.get('Variant', 'Standard') != 'Standard':
        return None
    if args.rated_only and headers.get('Event', '').lower().find('rated') == -1:
        return None
    if args.exclude_bots:
        if headers.get('WhiteTitle', '').upper() == 'BOT':
            return None
        if headers.get('BlackTitle', '').upper() == 'BOT':
            return None

    side = _gm_side(headers, gm_titles, args.gm_min_elo)
    if side is None:
        return None

    opponent_elo = _safe_int(
        headers.get('BlackElo') if side == chess.WHITE else headers.get('WhiteElo')
    )
    if opponent_elo is None:
        return None
    if opponent_elo < args.opponent_min_elo:
        return None
    if opponent_elo > args.opponent_max_elo:
        return None
    return side, opponent_elo


def extract_games(args):
    gm_titles = _parse_titles(args.gm_titles)
    buckets = _parse_rating_buckets(args.rating_buckets)
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    handles = _open_pgn_text(args.input)
    close_handle, pgn_text = handles
    read_games = selected_games = 0
    selected_positions = 0
    bucket_counts = {}

    try:
        with open(args.output, 'w', encoding='utf-8', newline='\n') as out:
            for block in _iter_pgn_blocks(pgn_text):
                read_games += 1
                headers = _parse_headers_from_block(block)

                selected = _is_selected(headers, args, gm_titles)
                if selected is None:
                    if args.max_games_read and read_games >= args.max_games_read:
                        break
                    continue

                gm_side, opponent_elo = selected
                bucket = _rating_bucket(opponent_elo, buckets)
                policy_color = 'White' if gm_side == chess.WHITE else 'Black'
                extra_headers = {
                    'TrainingPolicyColor': policy_color,
                    'StrongSide': policy_color,
                    'OpponentElo': str(opponent_elo),
                    'OpponentRatingBucket': bucket,
                }
                out.write(_with_extra_headers(block, extra_headers))
                selected_games += 1
                selected_positions += _count_mainline_plies(block)
                bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

                if selected_games % args.progress_every == 0:
                    print(
                        f"selected={selected_games:,} read={read_games:,}",
                        flush=True,
                    )

                if args.max_games and selected_games >= args.max_games:
                    break
                if args.max_games_read and read_games >= args.max_games_read:
                    break
    finally:
        close_handle.close()

    print()
    print(f"Read games      : {read_games:,}")
    print(f"Selected games  : {selected_games:,}")
    print(f"Selected plies  : {selected_positions:,}")
    print(f"Output PGN      : {args.output}")
    if bucket_counts:
        print("Rating buckets  :")
        for bucket, count in sorted(bucket_counts.items()):
            print(f"  {bucket}: {count:,}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract GM-vs-lower-rated games from Lichess PGN dumps.'
    )
    parser.add_argument('--input', required=True, help='Path to .pgn or .pgn.zst')
    parser.add_argument('--output', required=True, help='Smaller output .pgn')
    parser.add_argument('--gm_titles', default='GM',
                        help='Comma-separated title tags treated as GM side.')
    parser.add_argument('--gm_min_elo', type=int, default=None,
                        help='Optional Elo fallback for missing title tags.')
    parser.add_argument('--opponent_min_elo', type=int, default=0)
    parser.add_argument('--opponent_max_elo', type=int, default=2200)
    parser.add_argument('--rating_buckets', default=DEFAULT_BUCKETS)
    parser.add_argument('--max_games', type=int, default=100_000)
    parser.add_argument('--max_games_read', type=int, default=0)
    parser.add_argument('--progress_every', type=int, default=1_000)
    parser.add_argument('--rated_only', action='store_true', default=True)
    parser.add_argument('--include_unrated', dest='rated_only',
                        action='store_false')
    parser.add_argument('--exclude_bots', action='store_true', default=True)
    parser.add_argument('--include_bots', dest='exclude_bots',
                        action='store_false')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Input file does not exist: {args.input}", file=sys.stderr)
        raise SystemExit(2)

    extract_games(args)


if __name__ == '__main__':
    main()
