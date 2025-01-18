import csv
import math

import chess
import chess.engine
import sys

import re
import subprocess

_nnue_eval_re = re.compile(r'NNUE evaluation\s+([+-]?\d+\.\d+|none)\s+\((white|black) side\)', re.IGNORECASE)
_final_eval_re = re.compile(r'Final evaluation\s+([+-]?\d+\.\d+|none)\s+\((white|black) side\)|Final evaluation: none \(in check\)', re.IGNORECASE)



def parse_stockfish_line(line):
    nnue_match = _nnue_eval_re.search(line)
    if nnue_match:
        value, side = nnue_match.groups()
        return {'type': 'NNUE', 'value': float(value), 'side': side}

    final_match = _final_eval_re.search(line)
    if final_match:
        value, side = final_match.groups()
        return {'type': 'Final', 'value': float(value), 'side': side}

    return None


class StockfishEvaluator:
    def __init__(self, engine_path='./stockfish-macos-m1-apple-silicon'):
        self.process = subprocess.Popen(
            [engine_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            universal_newlines=True,
            bufsize=1
        )

    def evaluate_position(self, fen):
        self.process.stdin.write(f"position fen {fen}\n")
        self.process.stdin.write("eval\n")
        self.process.stdin.flush()

        nnue_value = None
        final_value = None

        while True:
            line = self.process.stdout.readline()
            if not line.strip():
                continue
            parsed = parse_stockfish_line(line)
            if parsed:
                if parsed['type'] == 'NNUE':
                    nnue_value = parsed['value']
                elif parsed['type'] == 'Final':
                    final_value = parsed['value']

            if nnue_value is not None and final_value is not None:
                break

        return float(nnue_value) * 100

    def close(self):
        self.process.stdin.write("quit\n")
        self.process.stdin.flush()
        self.process.terminate()
#
# def get_eval_from_parsing(fen, engine_path='./stockfish-macos-m1-apple-silicon'):
#     process = subprocess.Popen(
#         [engine_path],
#         stdin=subprocess.PIPE,
#         stdout=subprocess.PIPE,
#         stderr=subprocess.STDOUT,
#         universal_newlines=True
#     )
#
#     process.stdin.write(f"position fen {fen}\n")
#     process.stdin.write("eval\n")
#     process.stdin.write("quit\n")
#     process.stdin.flush()
#
#     nnue_value = None
#     final_value = None
#
#     for line in process.stdout:
#         parsed = parse_stockfish_line(line)
#         if parsed:
#             if parsed['type'] == 'NNUE':
#                 nnue_value = parsed['value']
#             elif parsed['type'] == 'Final':
#                 final_value = parsed['value']
#
#     process.stdout.close()
#     process.stdin.close()
#     process.wait()
#
#     return math.floor(float(nnue_value) * 100)

def evaluate_positions(input_file, output_file, engine_path, nodes_limit, start_line, end_line):
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    evaluator = StockfishEvaluator()

    with open(input_file, 'r', encoding='utf-8') as f_in, \
            open(output_file, 'w', newline='', encoding='utf-8') as f_out:

        fieldnames = ['fen', 'eval']
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        # writer.writeheader()
        row_count = 0

        for line_number, line in enumerate(f_in, start=1):
            row_count += 1
            if line_number < start_line:
                continue
            if end_line is not None and line_number > end_line:
                break

            fen = line.strip()
            if not fen:
                continue

            try:
                board = chess.Board(fen)
            except ValueError:
                print(f"Incorrect FEN: {fen}")

                continue

            try:
                info = engine.analyse(board, chess.engine.Limit(nodes=nodes_limit))
                deep_score = info["score"].white().score(mate_score=100000)
            except Exception as e:
                print(f"Error deep eval {fen}: {e}")
                continue

            try:
                static_eval = evaluator.evaluate_position(fen=fen)
            except Exception as e:
                print(f"Error {fen}: {e}")
                continue

            if abs(static_eval - deep_score) < 60:
                writer.writerow({'fen': fen, 'eval': deep_score})

            f_out.flush()

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
