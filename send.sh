for p in 1 2 3
do
    for seed in 10 11 12 13 14
    do
            python src/main_pulser.py --p $p  2>&1 >& seed_12.log &
    done
done