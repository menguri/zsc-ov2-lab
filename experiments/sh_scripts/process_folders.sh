#!/bin/bash
set -e

# Change to script directory
# cd "$(dirname "$0")" || exit 1

for dir in $(ls runs/ | grep '^2025112' | sort | sed -n '/20251126-073602/,/20251126-131745/p'); do
    echo "Processing $dir"
    cd runs/$dir
    
    if [ -f reward_summary_sp.csv ]; then
        echo "  Removing run_3-7 from sp.csv"
        sed -i '/^run_[3456],/d' reward_summary_sp.csv
    fi
    
    if [ -f reward_summary_cross.csv ]; then
        echo "  Filtering cross.csv and regenerating plot"
        head -1 reward_summary_cross.csv > temp.csv
        tail -n +2 reward_summary_cross.csv | grep -v -E 'cross-[0-9]_[3-7]|cross-[3-7]_[0-9]' >> temp.csv
        mv temp.csv reward_summary_cross.csv
        python3 -c "from overcooked_v2_experiments.helper.plots import visualize_cross_play_matrix; from pathlib import Path; visualize_cross_play_matrix(Path('reward_summary_cross.csv'))"
    fi
    
    cd ../..
done

echo "All done"
