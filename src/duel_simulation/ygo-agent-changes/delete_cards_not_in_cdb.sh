#!/bin/bash

while true; do
    # Run the Python command and capture stderr
    output=$(python -u eval.py --deck ../assets/deck/sim/ --deck1 deck_000_c503c6.ydk --deck2 deck_001_4b33bf --num_episodes 32 --strategy greedy --num_envs 16 2>&1)

    # Parse the missing card ID from the error
    card_id=$(echo "$output" | grep -oP 'Card not found: \K\d+')

    if [ -n "$card_id" ]; then
        echo "Missing card ID: $card_id"
        # Delete all .ydk files containing the missing card ID
        grep -l "$card_id" ../assets/deck/sim/*.ydk | xargs rm
        echo "Deleted all decks containing card ID $card_id"
    else
        echo "No missing card ID found in error output. Success!"
        break
    fi
done