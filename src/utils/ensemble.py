import os
import pandas as pd


if __name__ == '__main__':
    submit_folder = '../../results/ensemble/'
    if not os.path.exists(submit_folder):
        os.makedirs(submit_folder)
    i = 0
    while True:
        version = 'ensemble_v' + str(i) + '.csv'
        if os.path.exists(os.path.join(submit_folder, version)):
            i += 1
        else:
            save_csv = os.path.join(submit_folder, version)
            break

    csv_list = ['', '', '']

    submit = pd.read_csv(f'{save_csv}', index_col=False)

    temp = pd.DataFrame()
    for idx, csv in enumerate(csv_list):
        temp[f'{idx}'] = pd.read_csv(csv, index_col=False)['ans']

    submit['ans'] = temp.mode(axis=1)[0].astype('int')
    submit.to_csv(os.path.join(submit_folder, 'soft_voting.csv'), index=False)
