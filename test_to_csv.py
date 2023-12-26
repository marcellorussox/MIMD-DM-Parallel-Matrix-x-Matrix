import os
import csv
import re

import numpy as np
import pandas as pd


def extract_out_data(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    match_header = re.match(r'<====== NCPU: (\d+) - ORDER: (\d+) - TEST: (\d+) ======>', content)
    if not match_header:
        return None

    ncpu, order, test = map(int, match_header.groups())

    pattern = r'(.*?Overall time: (\d+\.\d+))'
    matches = list(re.finditer(pattern, content, re.DOTALL))

    data = []
    for match in matches:
        test_times = [float(group) for group in match.groups()[1::2]]
        for num_test, time in enumerate(test_times, start=1):
            data.append((ncpu, order, time))

    return data


def process_directory(directory_path):
    out_file_path = os.path.join(directory_path, 'pdc3.out')

    out_data = extract_out_data(out_file_path)

    return out_data


def calculate_average_and_sort_pandas(input_csv):
    df = pd.read_csv(input_csv)

    # Calcola la media della quarta colonna per ogni tupla delle prime tre colonne uguali
    df_result = df.groupby(['NCPU', 'ORDER'])['TIME'].mean().reset_index()

    # Arrotonda i valori a 6 cifre decimali
    df_result['TIME'] = df_result['TIME'].round(6)

    # Ordina il DataFrame in base alle colonne specificate
    df_sorted = df_result.sort_values(by=['NCPU', 'ORDER'], ascending=True)

    # Scrivi il DataFrame ordinato su un nuovo file CSV
    df_sorted.to_csv(input_csv, index=False)


def calculate_speedup_efficiency(input_csv):
    # Leggi il file CSV come DataFrame
    df = pd.read_csv(input_csv)

    # Funzione per calcolare lo speedup
    def calculate_speedup(row):
        base_row = df[(df['NCPU'] == 1) & (df['ORDER'] == row['ORDER'])]
        if len(base_row) > 0:
            return round(base_row['TIME'].values[0] / row['TIME'], 6)
        return np.nan

    # Applica la funzione per calcolare lo speedup
    df['SPEEDUP'] = df.apply(calculate_speedup, axis=1)

    # Calcola l'efficienza solo per i valori di speedup diversi da NaN
    df['EFFICIENCY'] = np.where(df['SPEEDUP'].notna(), round(df['SPEEDUP'] / df['NCPU'], 6), np.nan)

    # Scrivi il DataFrame risultante su un nuovo file CSV
    df.to_csv(input_csv, index=False)


def main(input_directory, output_csv):
    all_data = []

    for subdir in os.listdir(input_directory):
        subdir_path = os.path.join(input_directory, subdir)
        if os.path.isdir(subdir_path):
            out_data = process_directory(subdir_path)

            if out_data is not None:
                all_data.extend(out_data)

    with open(output_csv, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['NCPU', 'ORDER', 'TIME'])
        csv_writer.writerows(all_data)

    print(f"I dati sono stati esportati con successo nel file CSV: {output_csv}")

    calculate_average_and_sort_pandas(output_csv)

    print(f"I dati sono stati aggiornati con la media nel file CSV: {output_csv}")

    calculate_speedup_efficiency(output_csv)

    print(f"Sono stati calcolati i valori di Speedup ed Effcienza nel file CSV: {output_csv}")


# Esempio di utilizzo
main('output/test', 'data.csv')