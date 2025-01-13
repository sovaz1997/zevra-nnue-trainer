from torch import tensor, float32
from torch.utils.data import DataLoader
from src.model.chess_dataset import ChessDataset
from src.model.train_data_manager import TrainDataManager
from src.networks.halfkp.data_manager import HalfKPDataManager
from src.networks.halfkp.network import HalfKPNetwork
from src.networks.simple.data_manager import SimpleNetworkDataManager
from src.networks.simple.network import SimpleNetwork, SimpleDeepNetwork
from src.train import train


def get_positions_distribution(count: int):
    return round(count * 0.8), round(count * 0.2)

def evaluate_position_simple(fen):
    nnue = SimpleNetwork(128)
    manager = SimpleNetworkDataManager()
    nnue.load_weights(22, "trains/simple_screlu_768x128_positions13M_self-play-dataset_with-biases")
    nnue.eval()
    nnue_input = manager.calculate_nnue_input_layer(fen)
    nnue_input = tensor(nnue_input, dtype=float32)
    print(nnue(nnue_input).item())

def evaluate_position_simple_deep(fen):
    nnue = SimpleDeepNetwork(128, 16)
    manager = SimpleNetworkDataManager()
    nnue.load_weights(1, "trains/768x128x16_50Mv2")
    nnue.eval()
    nnue_input = manager.calculate_nnue_input_layer(fen)
    nnue_input = tensor(nnue_input, dtype=float32)
    print(nnue(nnue_input).item())

def evaluate_position_halfkp(fen):
    nnue = HalfKPNetwork(128)
    manager = HalfKPDataManager()
    nnue.load_weights(1, "halfkp")
    nnue.eval()
    nnue_input1, nnue_input2 = manager.calculate_nnue_input_layer(fen)
    nnue_input1 = tensor(nnue_input1, dtype=float32)
    nnue_input2 = tensor(nnue_input2, dtype=float32)
    nnue_input1 = nnue_input1.unsqueeze(0)
    nnue_input2 = nnue_input2.unsqueeze(0)
    print(nnue(nnue_input1, nnue_input2).item())

def create_singlethreaded_data_loader(manager: TrainDataManager, path: str):
    dataset = ChessDataset(path, manager)
    return DataLoader(dataset, batch_size=512, num_workers=0)

def create_data_loader(manager: TrainDataManager, path: str, positions_count: int):
    # return create_singlethreaded_data_loader(manager, path)
    dataset = ChessDataset(path, manager, positions_count)
    return DataLoader(dataset, batch_size=512, num_workers=11, persistent_workers=True, prefetch_factor=2)

def run_simple_train_nnue(
        hidden_size: int,
        train_dataset_path: str,
        validation_dataset_path: str,
        train_directory,
        positions_count: int
):
    # evaluate_position_simple("4k3/8/8/8/8/8/PPPPPPPP/RNBQKBNR w KQ - 0 1")
    # evaluate_position_simple("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    # evaluate_position_simple("1nbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQk - 0 1")
    # return None

    train_count, validation_count = get_positions_distribution(positions_count)

    manager = SimpleNetworkDataManager()

    train(
        SimpleNetwork(hidden_size),
        create_data_loader(manager, train_dataset_path, train_count),
        create_data_loader(manager, validation_dataset_path, validation_count),
        train_directory
    )

def run_halfkp_train_nnue(
        hidden_size: int,
        train_dataset_path: str,
        validation_dataset_path: str,
        train_directory
):
    manager = HalfKPDataManager()

    train(
        HalfKPNetwork(hidden_size),
        create_data_loader(manager, train_dataset_path),
        create_data_loader(manager, validation_dataset_path),
        train_directory
    )


def run_simple_deep_train_nnue(
        hidden_size: int,
        second_hidden_size: int,
        train_dataset_path: str,
        validation_dataset_path: str,
        train_directory,
        positions_count: int
):
    # evaluate_position_simple_deep("2k5/8/8/8/8/8/8/2KBN3 w - - 0 1")
    # evaluate_position_simple_deep("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    # evaluate_position_simple_deep("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
    # evaluate_position_simple_deep("1nb1kbn1/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQha - 0 1")
    # evaluate_position_simple_deep("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    # evaluate_position_simple_deep("4k3/8/8/8/8/8/PPPPPPPP/RNBQKBNR w KQ - 0 1")
    # evaluate_position_simple_deep("4k3/8/8/8/8/8/QQQQQQQQ/QQQQKQQQ w HAha - 0 1")
    # evaluate_position_simple_deep("2kr1br1/p1p2p2/4p2p/3p1np1/8/1PB2P2/P4P1P/2R3RK b - - 3 22")
    # evaluate_position_simple_deep("2kr1br1/p1p2p2/4p2p/3p1np1/8/1PB2P2/P4P1P/2R3RK w - - 3 22")
    # return None

    train_count, validation_count = get_positions_distribution(positions_count)

    manager = SimpleNetworkDataManager()

    train(
        SimpleDeepNetwork(hidden_size, second_hidden_size),
        create_data_loader(manager, train_dataset_path, train_count),
        create_data_loader(manager, validation_dataset_path, validation_count),
        train_directory
    )


SHOULD_TRAIN_SIMPLE = False
SHOULD_TRAIN_HALFKP = False
SHOULD_TRAIN_SIMPLE_DEEP = False
TRAINS_DIR = "trains"

if __name__ == '__main__':
    # run_simple_train_nnue(
    #     1024,
    #     "train_self-play-training-dataset.csv",
    #     "validate_self-play-training-dataset.csv",
    #     f"{TRAINS_DIR}/simple768x1024_positions13M_self-play-dataset",
    #     13000000
    # )

    # run_simple_train_nnue(
    #     256,
    #     "train_self-play-training-dataset.csv",
    #     "validate_self-play-training-dataset.csv",
    #     f"{TRAINS_DIR}/simple768x256_positions13M_self-play-dataset",
    #     13000000
    # )

    run_simple_deep_train_nnue(
        128,
        16,
        "train_100millions_dataset.csv",
        "validate_100millions_dataset.csv",
        f"{TRAINS_DIR}/768x128x16_50Mv2",
        50000000
    )

    # run_simple_deep_train_nnue(
    #     128,
    #     16,
    #     "train_self-play-training-dataset.csv",
    #     "validate_self-play-training-dataset.csv",
    #     f"{TRAINS_DIR}/simple_deep_screlu_768x128_positions13M_self-play-dataset_with-biases",
    #     13000000
    # )

    # run_simple_train_nnue(
    #     128,
    #     "train_self-play-training-dataset.csv",
    #     "validate_self-play-training-dataset.csv",
    #     f"{TRAINS_DIR}/simple_screlu_768x128_positions13M_self-play-dataset_with-biases",
    #     13000000
    # )
    #
    # run_simple_train_nnue(
    #     128,
    #     "train_100millions_dataset.csv",
    #     "validate_100millions_dataset.csv",
    #     f"{TRAINS_DIR}/simple768x128_positions20M",
    #     20000000
    # )

    # run_simple_train_nnue(
    #     64,
    #     "train_100millions_dataset.csv",
    #     "validate_100millions_dataset.csv",
    #     f"{TRAINS_DIR}/simple768x64_positions30M",
    #     30000000
    # )

    # run_simple_train_nnue(
    #     64,
    #     "train_100millions_dataset.csv",
    #     "validate_100millions_dataset.csv",
    #     f"{TRAINS_DIR}/simple768x64_positions10M",
    #     10000000
    # )

    # run_simple_train_nnue(
    #     64,
    #     "train_100millions_dataset.csv",
    #     "validate_100millions_dataset.csv",
    #     f"{TRAINS_DIR}/simple768x64_positions3M",
    #     3000000
    # )

    if SHOULD_TRAIN_SIMPLE:
        run_simple_train_nnue(
            32,
            "train_100millions_dataset.csv",
            "validate_100millions_dataset.csv",
            f"{TRAINS_DIR}/simple_30M"
        )

    if SHOULD_TRAIN_HALFKP:
        run_halfkp_train_nnue(
            128,
            "train_100millions_dataset.csv",
            "validate_100millions_dataset.csv",
            f"{TRAINS_DIR}/halfkp"
        )

    if SHOULD_TRAIN_SIMPLE_DEEP:
        run_simple_deep_train_nnue(
            32,
            "train_100millions_dataset.csv",
            "validate_100millions_dataset.csv",
            f"{TRAINS_DIR}/simple_deep"
        )
