from preprocessing.cornell import process_cornell_data
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cornell", choices=["cornell"], help="Name of the dataset to preprocess.")
    parser.add_argument("--source_file", type=str, default="./data/raw/cornell_movie_dialogues/movie_lines.txt", help="Path to raw dataset file")
    parser.add_argument("--destination_file", type=str, default="./data/corpus/cornell_movies.json", help="Path to corpus output.")
    args = parser.parse_args()

    if args.dataset == "cornell":
        process_cornell_data(args.source_file, args.destination_file)
