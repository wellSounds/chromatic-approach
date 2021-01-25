import os, sys
import chromatic_approach as ws
import pandas as pd

if sys.argv[1] == 'chromatic':
    current_label = 'sfx'
    emo_choice = 'low'
    data_dir = 'audio/' + current_label + '/'
    labels = data_dir + '1_filename.csv'
    output = 'chroma_features_' + current_label

    if not os.path.exists(labels):
        with open(data_dir + "1_filename.csv", 'w') as f:
            f.write('filename')
        os.system('cd ' + data_dir + ' && ls >> ../1_filename.csv && cd ../ && mv 1_filename.csv ' + current_label + '/')

    labels_df = pd.read_csv(labels, index_col=False)

    print('\n Wellsounds - chromatic approach \n Project setup for: ' + current_label + '\n')

    if sys.argv[2] == 'pre':
        print('Begin preprocessing...\n')
        ws.normalise_extract_chroma(current_label,labels_df)
        ws.preprocessing_feature_files(output, current_label)
        ws.calculate_tempo(current_label)
    if sys.argv[2] == 'gn':
        print('Generating notes...\n')
        ws.generate_notes(current_label)
        ws.combined_segmented_notes(current_label)
        ws.mix_notes(current_label)
    if sys.argv[2] == 'post':
        print('Post processing...\n')
        ws.apply_eq(current_label)
        ws.apply_compression(current_label)
        ws.normalisation_synthetic(current_label)
        ws.merge_scapes(current_label)
        ws.add_fade(current_label)
    if sys.argv[2] == 'cleanup':
        print('removed uneeded dir')
        ws.clean_working_dir()
else:
    print('call approach e.g., chromatic')
