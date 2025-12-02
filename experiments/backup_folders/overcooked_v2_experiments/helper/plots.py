import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path


def visualize_cross_play_matrix(csv_file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Split the policy_labels into two separate columns for better visualization
    df[["policy_1", "policy_2"]] = (
        df["policy_labels"]
        .str.replace("cross-", "", regex=False)
        .str.split("_", expand=True)
    )

    # Pivot the DataFrame to get the average total_reward for each cross-play pair
    pivot_table = df.pivot_table(
        index="policy_1", columns="policy_2", values="total_reward", aggfunc="mean"
    )

    num_seeds = df["annotation"].nunique()

    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        pivot_table,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        cbar=True,
        linewidths=0.5,
        linecolor="black",
        vmin=0,
        vmax=300,
        # vmin=-200,
        # vmax=200,
    )
    plt.title("Cross Play Reward Matrix")
    plt.xlabel("Policy 2")
    plt.ylabel("Policy 1")

    folder_name = os.path.dirname(csv_file)
    plt.figtext(
        0.5,
        0.01,
        f"Folder: {folder_name}; Num Seeds: {num_seeds}",
        ha="center",
        fontsize=8,
        color="gray",
    )

    # plt.show()

    cross_play_matrix_file = str(csv_file).replace(".csv", "_plot.png")
    plt.savefig(cross_play_matrix_file)
    print(f"Cross play matrix saved to {cross_play_matrix_file}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python visualize_cross_play_matrix.py <csv_file>")
        sys.exit(1)

    csv_file = sys.argv[1]
    visualize_cross_play_matrix(Path(csv_file))
