#!/bin/bash

export source_dir="data/prop"

export train_dir="data/train"
export val_dir="data/val"
export test_dir="data/test"

export n_train=95
export n_val=10
export n_test=10

##############################

if [ -d "$train_dir" ]; then rm -r "$train_dir"; fi
if [ -d "$val_dir" ]; then rm -r "$val_dir"; fi
if [ -d "$test_dir" ]; then rm -r "$test_dir"; fi

mkdir -p "$train_dir"
mkdir -p "$val_dir"
mkdir -p "$test_dir"

export all_cases=$(find "$source_dir" -maxdepth 1 -mindepth 1 -type d)
export shuffled_cases=$(echo "$all_cases" | shuf)

export train_cases=$(echo "$shuffled_cases" | head -n "$n_train")
export val_cases=$(echo "$shuffled_cases" | head -n "$((n_train + n_val))" | tail -n "$n_val")
export test_cases=$(echo "$shuffled_cases" | tail -n "$n_test")

for case in $train_cases; do
    cp -r "$case" "$train_dir"
done

for case in $val_cases; do
    cp -r "$case" "$val_dir"
done

for case in $test_cases; do
    cp -r "$case" "$test_dir"
done

echo "Train cases:"
echo $train_cases

echo "Validation cases:"
echo $val_cases

echo "Test cases:"
echo $test_cases
