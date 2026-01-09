from pathlib import Path
from datetime import datetime 
import os

import pandas as pd

from utils.paths import VOL_EXPERIMENTS_DIR, BENCHMARKS_DIR

TYPE = 'transformer'

def main():
    all_dfs = []
    # iterate over directory
    for file in os.listdir(VOL_EXPERIMENTS_DIR):
        # restriction
        if TYPE not in file:
            continue

        file_path = Path(VOL_EXPERIMENTS_DIR) / file
        trials = os.listdir(file_path)
        if 'trial_search_best' in trials:
            trial ='trial_search_best'
        else:
            try:
                trial = trials[-1]
            except:
                print(f'No trials found for {file}')

        # define path
        analysis_path = f'{file_path}/{trial}/analysis/fold_avg_metrics.csv'
        

        # read and append
        try:
            df = pd.read_csv(analysis_path)
            
            # Replace NN with name
            df['Model'] = df['Model'].replace({'NN': file})

            all_dfs.append(df)
        except:
            print(f'Cannot find: {analysis_path}')

    # concat all
    df_out = pd.concat(all_dfs, axis=0)
    
    # delete duplicates
    df_out = df_out.drop_duplicates()
    df_out = df_out.sort_values(by='Test MSE', ascending=True)

    print('Shape: ', df_out.shape)

    time = datetime.now() 
    name = f'results_{TYPE}_{time:%d%m%Y_%H%M%S}.csv'
    out_path = BENCHMARKS_DIR / 'tables' /name 
    
    df_out.to_csv(out_path, index=False)
    print('Saved to: ', out_path.resolve())

    return 

if __name__ == '__main__':
    main()

