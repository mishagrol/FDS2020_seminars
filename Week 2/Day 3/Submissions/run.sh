  #!/bin/bash
set -x

ML_PKG="os time dask dask-ml json xgboost requests pandas numpy matplotlib.pyplot argparse sklearn joblib"



for pkg in $ML_PKG; do
    if pip -q list installed "$ML_PKG" > /dev/null 2>&1; then
        echo -e "$pkg is already installed"
    else
        pip install $ML_PLG -y
        echo "Successfully installed $pkg"
    fi
done

python3 main.py --test_size 0.3 --results_path results_final.csv