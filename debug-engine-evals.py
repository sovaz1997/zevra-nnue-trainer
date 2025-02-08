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

    for node in game.mainline():
        turn = node.turn()
        board.push(node.move)

        yield turn, extract_eval_from_comment(node.comment)


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



def main():
    pgn_file = "games.pgn"
    for game in read_games(pgn_file):
        print(collect_evals_from_game(game))
        # for gamer, eval in read_game_evals(game):
        #     print(gamer, eval)


if __name__ == "__main__":
    main()
