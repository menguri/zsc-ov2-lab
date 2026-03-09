import os
import sys

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(DIR))
sys.path.append(os.path.dirname(os.path.dirname(DIR)))

from overcooked_v2_experiments.human_rl.static import (
    CLEAN_2019_HUMAN_DATA_TRAIN,
    CLEAN_2019_HUMAN_DATA_TEST,
    CLEAN_2019_HUMAN_DATA_ALL,
)

from overcooked_v2_experiments.human_rl.imitation.data import (
    preprocess_data,
    save_data,
    LAYOUT_TO_TRIAL_NAME,
    data_exists,
)


def run_preprocessing(split, force=False):

    if split == "train":
        data_path = CLEAN_2019_HUMAN_DATA_TRAIN
    elif split == "test":
        data_path = CLEAN_2019_HUMAN_DATA_TEST
    elif split == "all":
        data_path = CLEAN_2019_HUMAN_DATA_ALL
    else:
        raise ValueError(f"Invalid split: {split}")

    print(f"Preprocessing data for {split} split")

    for layout in LAYOUT_TO_TRIAL_NAME.keys():
        print(f"Preprocessing data for layout {layout}")

        exists = data_exists(layout)

        if exists:
            if force:
                print(
                    f"Data for layout {layout} already exists, but force flag is set, overwriting"
                )
            else:
                print(f"Data for layout {layout} already exists, skipping")
                continue
        else:
            print(f"Data for layout {layout} does not exist, creating")

        inputs, targets = preprocess_data(data_path, layout)
        save_data(layout, inputs, targets)


if __name__ == "__main__":
    run_preprocessing("all", force=True)
