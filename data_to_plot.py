import math

import pandas as pd
import matplotlib.pyplot as plt


def create_graphs(df, x_col, y_cols, x_min=None, x_max=None):

    if x_min is not None and x_max is not None:
        df = df[(df[x_col] >= x_min) & (df[x_col] <= x_max)]

    grouped_data = df.groupby(['NCPU', 'ORDER'])

    for (rows, columns), group in grouped_data:
        plt.figure()

        rows_exp = int(math.log10(rows))
        columns_exp = int(math.log10(columns))
        margin_percent = 0.25

        for y_col in y_cols:
            # Calcola i valori minimi e massimi solo all'interno del gruppo corrente
            y_min = group[y_col].min() * (1 - margin_percent)
            y_max = group[y_col].max() * (1 + margin_percent)

            y_range = (y_min, y_max)

            plt.plot(group[x_col], group[y_col])

            plt.title(f'(Matrix 10^{rows_exp} x 10^{columns_exp}) x (Vector 10^{columns_exp})')
            plt.xlabel('Number of threads')
            plt.ylabel(y_col.capitalize())
            plt.ylim(y_range)

            plt.savefig(f'output/plots/plot_{y_col.lower()}_10_{rows_exp}x10_{columns_exp}.png')
            plt.close()


#    for y_col in y_cols:
#        plt.figure()
#
#        order = 0
#
#        for (rows, columns), group in grouped_data:
#            rows_exp = int(math.log10(rows))
#            columns_exp = int(math.log10(columns))
#            plt.plot(group[x_col], group[y_col],
#                     label=f'(Matrix 10^{rows_exp} x 10^{columns_exp}) x (Vector 10^{columns_exp})')
#            order = rows_exp + columns_exp
#
#        plt.title(f'On the order of 10^{order}')
#        plt.xlabel('Number of threads')
#        plt.ylabel(y_col.capitalize())
#        plt.legend()
#        plt.savefig(f'output/plots/plot_{y_col.lower()}_e{order}.png')
#        plt.close()
#

# Esempio di utilizzo con intervallo per le ascisse
csv = pd.read_csv('data.csv')
create_graphs(csv, 'NCPU', ['TIME', 'SPEEDUP', 'EFFICIENCY'], x_min=1, x_max=8)