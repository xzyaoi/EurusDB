evaluate() {
    local runid=$1
    echo "Generating data..."
    ./src/utilities/c-generator/lognormal.10k.run
    echo "Evaluating..."
    python3 examples/staged_grid_search.py data/1d_lognormal_10000.csv > lognormal_staged_$runid.log
}
for i in {1..3}; 
do 
    evaluate "$i" &
done