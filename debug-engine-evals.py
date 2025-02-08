import chess
from chess import pgn
from chess.pgn import Game


def read_games(pgn_file):
    with open(pgn_file) as f:
        while True:
            game = pgn.read_game(f)
            if game is None:
                break
            yield game


def extract_eval_from_comment(comment):
    if comment is None:
        return None

    try:
        return float(comment.split("/")[0])
    except ValueError:
        return None


def get_players(game: Game):
    white = game.headers["White"]
    black = game.headers["Black"]

    return white, black

def read_game_evals(game: Game):
    board = game.board()

    position = game.headers["FEN"]
    board.set_fen(position)
    # print(position)

    for node in game.mainline():
        #  print(board.turn == chess.WHITE)
        # turn = node.turn()

        # eval = extract_eval_from_comment(node.comment)
        # if eval == 12.68:
        #     print(game)
        #     print("\n")

        yield board.turn, extract_eval_from_comment(node.comment)
        board.push(node.move)


def collect_evals_from_game(game: Game):
    white, black = get_players(game)

    def transform_game_result(result):
        if result == "1-0":
            return 1
        elif result == "0-1":
            return -1
        else:
            return 0


    game_result = transform_game_result(game.headers["Result"])

    scores = {
        chess.WHITE: [],
        chess.BLACK: []
    }

    players = {
        chess.WHITE: white,
        chess.BLACK: black
    }

    for turn, score in read_game_evals(game):
        if score is not None:
            scores[turn].append(score)

    return game_result, scores, players


def collect_stats(game: Game):
    game_result, scores, players = collect_evals_from_game(game)

    max_white = max(scores[chess.WHITE], default=None)
    max_black = max(scores[chess.BLACK], default=None)

    return {
        players[chess.WHITE]: {
            "max": max_white,
            "is_lost": game_result == -1,
            "scores": scores[chess.WHITE]
        },
        players[chess.BLACK]: {
            "max": max_black,
            "is_lost": game_result == 1,
            "scores": scores[chess.BLACK]
        }
    }


def collect_games_stats(pgn_file):
    lost_stats = {}

    def add_lost(name, max_eval):
        if name not in lost_stats:
            lost_stats[name] = [max_eval]
        else:
            lost_stats[name].append(max_eval)

    for game in read_games(pgn_file):
        try:
            if game.headers["Termination"] is not None:
                continue
        except KeyError:
            pass

        stats = collect_stats(game)

        for player, stat in stats.items():
            if stat["is_lost"]:
                add_lost(player, stat["max"])

    for player, max_evals in lost_stats.items():
        evals = sorted(max_evals, reverse=True)
        print(player, evals)



def measure_search_bug(evals_flow):
    res = []

    index = 1
    while index < len(evals_flow) - 1:
        a = evals_flow[index - 1]
        b = evals_flow[index]
        c = evals_flow[index + 1]

        if b > a > c:
            res.append((abs(b - a) + abs(b - c)) / abs(c - a))
        index += 1

    return res


def find_bugs(pgn_file):
    res = {}

    def addMeasures(player, measures):
        if player not in res:
            res[player] = measures
        else:
            res[player].extend(measures)

    for game in read_games(pgn_file):
        try:
            if game.headers["Termination"] is not None:
                continue
        except KeyError:
            pass

        stats = collect_stats(game)

        for player, stat in stats.items():
            addMeasures(player, measure_search_bug(stat["scores"]))


    # order revered and print
    for player, measure in res.items():
        sorted_res = sorted(measure, reverse=True)
        print(player, sorted_res)


def main():
    pgn_file = "games.pgn"
    # pgn_file = "test-pgn.pgn"
    # collect_games_stats(pgn_file)
    find_bugs(pgn_file)


if __name__ == "__main__":
    main()
