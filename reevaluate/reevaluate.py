import csv
import chess
import chess.engine
import sys

def evaluate_positions(input_file, output_file, engine_path, nodes_limit, start_line, end_line):
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)

    with open(input_file, newline='', encoding='utf-8') as f_in, \
         open(output_file, 'w', newline='', encoding='utf-8') as f_out:

        reader = csv.reader(f_in, delimiter=',')
        fieldnames = ['fen', 'eval']
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for line_number, row in enumerate(reader, start=1):
            if line_number < start_line or line_number > end_line:
                continue

            if not row:
                continue
            fen = row[0]
            try:
                board = chess.Board(fen)
            except ValueError:
                print(f"Incorrect FEN: {fen}")
                continue

            info = engine.analyse(board, chess.engine.Limit(nodes=nodes_limit))
            score = info["score"].white().score(mate_score=100000)

            writer.writerow({'fen': fen, 'eval': score})

    engine.quit()


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: script.py input_csv output_csv start_line end_line")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    start_line = int(sys.argv[3])
    end_line = int(sys.argv[4])

    stockfish_path = './stockfish-macos-m1-apple-silicon'
    nodes_limit = 10000

    evaluate_positions(input_csv, output_csv, stockfish_path, nodes_limit, start_line, end_line)
