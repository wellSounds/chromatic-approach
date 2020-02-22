# wellSounds - Baird 2020.
# Audio should be same length.
# python
import numpy as np
import csv, os
import pandas as pd

# audio
import librosa
from pydub.generators import Sine
from pydub import AudioSegment
import scipy.io.wavfile

current_label = 'sfx'
emo_choice = 'high'
data_dir = 'audio/' + current_label + '/'
labels = data_dir + '1_filename.csv'



if not os.path.exists(labels):

    with open(data_dir + "1_filename.csv", 'w') as f:
        f.write('filename')
    os.system('cd ' + data_dir + ' && ls >> ../1_filename.csv && cd ../ && mv 1_filename.csv ' + current_label + '/')

labels_df = pd.read_csv(labels, index_col=False)

print('\n Wellsounds - chromatic approach \n Project setup for: ' + current_label + '\n')


# normalise original audio
def normalise_extract_chroma(current_label,labels_df):
    output = 'chroma_features_' + current_label
    normalise_original_audio = True
    if not os.path.exists(output):
        os.makedirs(output)

    'normalise with librosa '
    if normalise_original_audio == True:
        path = 'audio/' + current_label + '/'
        # row_number = 0
        if not os.path.exists('original_normalised_' + current_label):
            os.makedirs('original_normalised_' + current_label)
        for index, row in labels_df.iterrows():
            audio_file = row['1_filename.csv']
            audio, sr = librosa.core.load(path + audio_file, sr=16000)
            audio = audio * (0.7079 / np.max(np.abs(audio)))
            maxv = np.iinfo(np.int16).max
            audio = (audio * maxv).astype(np.int16)
            scipy.io.wavfile.write('original_normalised_' + current_label + '/' + audio_file, sr, audio)

            print('normalised ', audio_file)

    else:
        print('not normalising original, change normalise original audio to TRUE')

    # process audio files - extract chroma features
    for index, row in labels_df.iterrows():
        audio = row['1_filename.csv']
        print('extracting chroma...' + audio)
        if not os.path.exists(output + '/' + audio):
            os.makedirs(output + '/' + audio)

        y, sr = librosa.load(data_dir + audio)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma = np.interp(chroma, (chroma.min(), chroma.max()), (-100, +100))

        notes = ['C', 'Csh', 'D', 'Dsh', 'E', 'F', 'Fsh', 'G', 'Gsh', 'A', 'Ash', 'B']

        for note in notes:
            if note == 'C':
                current_note = chroma[0:1, 0:264]  # C
            if note == 'Csh':
                current_note = chroma[1:2, 0:264]  # Csh
            if note == 'D':
                current_note = chroma[2:3, 0:264]  # D
            if note == 'Dsh':
                current_note = chroma[3:4, 0:264]  # Dsh
            if note == 'E':
                current_note = chroma[4:5, 0:264]  # E
            if note == 'F':
                current_note = chroma[5:6, 0:264]  # F
            if note == 'Fsh':
                current_note = chroma[6:7, 0:264]  # Fsh
            if note == 'G':
                current_note = chroma[7:8, 0:264]  # G
            if note == 'Gsh':
                current_note = chroma[8:9, 0:264]  # Gsh
            if note == 'A':
                current_note = chroma[9:10, 0:264]  # A
            if note == 'Ash':
                current_note = chroma[10:11, 0:264]  # Ash
            if note == 'B':
                current_note = chroma[11:12, 0:264]  # B

            the_note = pd.DataFrame(current_note)

            the_note.insert(0, 'note', note)
            the_note.insert(1, 'filename', audio)

            the_note.to_csv(output + '/' + audio + '/' + note + '_' + audio + '_chroma.csv', index=False)

        path = output + '/' + audio + '/'
        files_in_dir = [f for f in os.listdir(path) if f.endswith('csv')]

        for filenames in files_in_dir:
            feature_files = pd.read_csv(path + filenames)
            feature_files.to_csv(output + '/' + audio[:-4] + '.csv', mode='a', index=False, header=False)

        with open(output + '/' + audio[:-4] + '.csv', newline='') as f:
            r = csv.reader(f)
            data = [line for line in r]
        with open(output + '/' + audio[:-4] + '.csv', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(
                ['note', 'filename', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14',
                 '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31',
                 '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48',
                 '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65',
                 '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82',
                 '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99',
                 '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113',
                 '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '124', '125', '126', '127',
                 '128', '129', '130', '131', '132', '133', '134', '135', '136', '137', '138', '139', '140', '141',
                 '142', '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', '153', '154', '155',
                 '156', '157', '158', '159', '160', '161', '162', '163', '164', '165', '166', '167', '168', '169',
                 '170', '171', '172', '173', '174', '175', '176', '177', '178', '179', '180', '181', '182', '183',
                 '184', '185', '186', '187', '188', '189', '190', '191', '192', '193', '194', '195', '196', '197',
                 '198', '199', '200', '201', '202', '203', '204', '205', '206', '207', '208', '209', '210', '211',
                 '212', '213', '214', '215', '216', '217', '218', '219', '220', '221', '222', '223', '224', '225',
                 '226', '227', '228', '229', '230', '231', '232', '233', '234', '235', '236', '237', '238', '239',
                 '240', '241', '242', '243', '244', '245', '246', '247', '248', '249', '250', '251', '252', '253',
                 '254', '255', '256', '257', '258', '259', '260', '261', '262', '263'])
            w.writerows(data)

    rm_folders = 'cd ' + output + ' && rm -r */'
    os.system(rm_folders)

def preprocessing_feature_files(output, current_label):
    # read the csv files and transpose it
    path = output
    files_in_dir = [f for f in os.listdir(path) if f.endswith('csv')]

    for filename in files_in_dir:
        data = pd.read_csv(output + '/' + filename, index_col=False)
        data = data.drop(['filename'], axis=1)
        data = data.T
        if not os.path.exists('transposed_' + current_label):
            os.makedirs('transposed_' + current_label)

        data.to_csv('transposed_' + current_label + '/' + filename, index=True, header=False)

def calculate_tempo(current_label):
    """# take mean _ reduce by factor  / calclating the reduction factor by the BPM"""

    path = 'transposed_' + current_label
    files_in_dir = [f for f in os.listdir(path) if f.endswith('csv')]

    for filename in files_in_dir:
        data = pd.read_csv(path + '/' + filename, index_col=False)
        data = data.drop(['note'], axis=1)

        y, sr = librosa.load(data_dir + filename[0:-4] + '.wav')
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
        transpose_value = tempo / 10

        if not os.path.exists('mean_' + current_label):
            os.makedirs('mean_' + current_label)

        data = data.groupby(np.arange(len(data))//transpose_value).mean()

        data.to_csv('mean_' + current_label + '/' + filename, index=False, header=True)
        print('preprocessing complete')

def generate_notes(current_label):
    """generate notes based on full duration """
    for index, row in labels_df.iterrows():
        audio = row['1_filename.csv']
        path = 'mean_' + current_label
        mean_location = [f for f in os.listdir(path) if f.endswith('csv')]

        for filename in mean_location:
            y, sr = librosa.load(data_dir + '/' + audio)
            audio_duration = librosa.get_duration(y=y, sr=sr)
            current_composer = pd.read_csv(path + '/' + filename)
            row_number = 0
            transpose_value = len(current_composer)

            chroma_points = transpose_value
            sample_duration = audio_duration / chroma_points
            samples_ms = sample_duration * 1000



            if not os.path.exists('segments_of_notes_' + current_label):
                os.makedirs('segments_of_notes_' + current_label)
            if not os.path.exists('segments_of_notes_' + current_label + '/' + filename[:-4]):
                os.makedirs('segments_of_notes_' + current_label + '/' + filename[:-4])

            for index, row in current_composer.iterrows():
                row_number = row_number + 1
                row_numb = str(row_number)
                row_numb = row_numb.zfill(2)
                C = row['C']
                Csh = row['Csh']
                D = row['D']
                Dsh = row['Dsh']
                E = row['E']
                F = row['F']
                Fsh = row['Fsh']
                G = row['G']
                Gsh = row['Gsh']
                A = row['A']
                Ash = row['Ash']
                B = row['B']
                if emo_choice == 'low':
                    # C2
                    Ctone = Sine(65.41).to_audio_segment(duration=samples_ms)
                    Cshtone = Sine(69.30).to_audio_segment(duration=samples_ms)
                    Dtone = Sine(73.42).to_audio_segment(duration=samples_ms)
                    Dshtone = Sine(77.78).to_audio_segment(duration=samples_ms)
                    Etone = Sine(82.41).to_audio_segment(duration=samples_ms)
                    Ftone = Sine(87.31).to_audio_segment(duration=samples_ms)
                    Fshtone = Sine(92.50).to_audio_segment(duration=samples_ms)
                    Gtone = Sine(98.00).to_audio_segment(duration=samples_ms)
                    Gshtone = Sine(103.83).to_audio_segment(duration=samples_ms)
                    Atone = Sine(110.00).to_audio_segment(duration=samples_ms)
                    Ashtone = Sine(116.54).to_audio_segment(duration=samples_ms)
                    Btone = Sine(123.47).to_audio_segment(duration=samples_ms)
                if emo_choice == 'high':
                    # C3
                    Ctone = Sine(130.81).to_audio_segment(duration=samples_ms)
                    Cshtone = Sine(138.59).to_audio_segment(duration=samples_ms)
                    Dtone = Sine(146.83).to_audio_segment(duration=samples_ms)
                    Dshtone = Sine(155.56).to_audio_segment(duration=samples_ms)
                    Etone = Sine(164.81).to_audio_segment(duration=samples_ms)
                    Ftone = Sine(174.61).to_audio_segment(duration=samples_ms)
                    Fshtone = Sine(185.00).to_audio_segment(duration=samples_ms)
                    Gtone = Sine(196.00).to_audio_segment(duration=samples_ms)
                    Gshtone = Sine(207.65).to_audio_segment(duration=samples_ms)
                    Atone = Sine(220.00).to_audio_segment(duration=samples_ms)
                    Ashtone = Sine(233.08).to_audio_segment(duration=samples_ms)
                    Btone = Sine(246.94).to_audio_segment(duration=samples_ms)


                volume_reduct = 50
                Cvolume = np.mean(C)
                Cvolume = np.negative(Cvolume) - volume_reduct

                Ctone = Ctone + Cvolume
                Cshvolume = np.mean(Csh)
                Cshvolume = np.negative(Cshvolume) - volume_reduct
                Cshtone = Cshtone + Cshvolume
                Dvolume = np.mean(D)
                Dvolume = np.negative(Dvolume) - volume_reduct
                Dtone = Dtone + Dvolume
                Dshvolume = np.mean(Dsh)
                Dshvolume = np.negative(Dshvolume) - volume_reduct
                Dshtone = Dshtone + Dshvolume
                Evolume = np.mean(E)
                Evolume = np.negative(Evolume) - volume_reduct
                Etone = Etone + Evolume
                Fvolume = np.mean(F)
                Fvolume = np.negative(Fvolume) - volume_reduct
                Ftone = Ftone + Fvolume
                Fshvolume = np.mean(Fsh)
                Fshvolume = np.negative(Fshvolume) - volume_reduct
                Fshtone = Fshtone + Fshvolume
                Gvolume = np.mean(G)
                Gvolume = np.negative(Gvolume) - volume_reduct
                Gtone = Gtone + Gvolume
                Gshvolume = np.mean(Gsh)
                Gshvolume = np.negative(Gshvolume) - volume_reduct
                Gshtone = Gshtone + Gshvolume
                Avolume = np.mean(A)
                Avolume = np.negative(Avolume) - volume_reduct
                Atone = Atone + Avolume
                Ashvolume = np.mean(Ash)
                Ashvolume = np.negative(Ashvolume) - volume_reduct
                Ashtone = Ashtone + Ashvolume
                Bvolume = np.mean(B)
                Bvolume = np.negative(Bvolume) - volume_reduct
                Btone = Btone + Bvolume

                Ctone.export('segments_of_notes_' + current_label + '/' + filename[:-4] + '/' + row_numb + '_C.wav',
                             format="wav")
                Cshtone.export('segments_of_notes_' + current_label + '/' + filename[:-4] + '/' + row_numb + '_Csh.wav',
                               format="wav")
                Dtone.export('segments_of_notes_' + current_label + '/' + filename[:-4] + '/' + row_numb + '_D.wav',
                             format="wav")
                Dshtone.export('segments_of_notes_' + current_label + '/' + filename[:-4] + '/' + row_numb + '_Dsh.wav',
                               format="wav")
                Etone.export('segments_of_notes_' + current_label + '/' + filename[:-4] + '/' + row_numb + '_E.wav',
                             format="wav")
                Ftone.export('segments_of_notes_' + current_label + '/' + filename[:-4] + '/' + row_numb + '_F.wav',
                             format="wav")
                Fshtone.export('segments_of_notes_' + current_label + '/' + filename[:-4] + '/' + row_numb + '_Fsh.wav',
                               format="wav")
                Gtone.export('segments_of_notes_' + current_label + '/' + filename[:-4] + '/' + row_numb + '_G.wav',
                             format="wav")
                Gshtone.export('segments_of_notes_' + current_label + '/' + filename[:-4] + '/' + row_numb + '_Gsh.wav',
                               format="wav")
                Atone.export('segments_of_notes_' + current_label + '/' + filename[:-4] + '/' + row_numb + '_A.wav',
                             format="wav")
                Ashtone.export('segments_of_notes_' + current_label + '/' + filename[:-4] + '/' + row_numb + '_Ash.wav',
                               format="wav")
                Btone.export('segments_of_notes_' + current_label + '/' + filename[:-4] + '/' + row_numb + '_B.wav',
                             format="wav")
            print('segmented audio generated for : ' + filename)






def combined_segmented_notes(current_label):

    # combine the audio files as one merged audio file
    # todo: reduce variables

    path = 'segments_of_notes_' + current_label + '/'

    row_number = 0
    for index, row in labels_df.iterrows():
        audio = row['1_filename.csv']
        audio = audio[:-4]
        ch_dir = 'cd ' + path + audio + '/'
        os.system(ch_dir)
        row_number = 0
        new_path = path + audio + '/'
        segment_location = [f for f in os.listdir(new_path) if f.endswith('wav')]
        numbers = [f for f in os.listdir(new_path) if os.path.isfile(os.path.join(new_path, f))]
        numbers[:] = (elem[:3] for elem in numbers)
        # todo: number list generation to match the segment length

        for number in numbers:
            sound1 = AudioSegment.from_file(new_path + number + 'C.wav')
            sound2 = AudioSegment.from_file(new_path + number + 'Csh.wav')
            sound3 = AudioSegment.from_file(new_path + number + 'D.wav')
            sound4 = AudioSegment.from_file(new_path + number + 'Dsh.wav')
            sound5 = AudioSegment.from_file(new_path + number + 'E.wav')
            sound6 = AudioSegment.from_file(new_path + number + 'F.wav')
            sound7 = AudioSegment.from_file(new_path + number + 'Fsh.wav')
            sound8 = AudioSegment.from_file(new_path + number + 'G.wav')
            sound9 = AudioSegment.from_file(new_path + number + 'Gsh.wav')
            sound10 = AudioSegment.from_file(new_path + number + 'A.wav')
            sound11 = AudioSegment.from_file(new_path + number + 'Ash.wav')
            sound12 = AudioSegment.from_file(new_path + number + 'B.wav')

            combined = sound1.overlay(sound2)
            combined = combined.overlay(sound3)
            combined = combined.overlay(sound4)
            combined = combined.overlay(sound5)
            combined = combined.overlay(sound6)
            combined = combined.overlay(sound7)
            combined = combined.overlay(sound8)
            combined = combined.overlay(sound9)
            combined = combined.overlay(sound10)
            combined = combined.overlay(sound11)
            combined = combined.overlay(sound12)

            if not os.path.exists('combined_notes_' + current_label):
                os.makedirs('combined_notes_' + current_label)
            if not os.path.exists('combined_notes_' + current_label + '/' + audio):
                os.makedirs('combined_notes_' + current_label + '/' + audio)
            combined.export('combined_notes_' + current_label + '/' + audio + '/' + number + '_combined.wav', format='wav')

        print('combination exported:', audio)


def mix_notes(current_label):
    # merge notes files together
    path = 'combined_notes_' + current_label + '/'
    for index, row in labels_df.iterrows():
        audio = row['1_filename.csv']
        audio = audio[:-4]
        new_path = path + audio + '/'
        if not os.path.exists('soundscapes_' + current_label):
            os.makedirs('soundscapes_' + current_label)
        sox_concat = 'cd ' + new_path + ' && sox $(ls *.wav | sort -n) ../../soundscapes_' + current_label + '/' + audio + '_soundscape.wav'
        os.system(sox_concat)
        print('exported:', audio)



def apply_eq(current_label):
    # EQ with Sox
    path = 'soundscapes_' + current_label + '/'
    if current_label == 'hv_ha':
        lowpass_value = 125
    if current_label == 'hv_la':
        lowpass_value = 250
    if current_label == 'lv_la':
        lowpass_value = 125
    if current_label == 'lv_la':
        lowpass_value = 256
    else:
        lowpass_value = 5000
    lowpass_value = str(lowpass_value)
    for index, row in labels_df.iterrows():
        audio = row['1_filename.csv']

        audio = audio[:-4]


        if not os.path.exists('eq_soundscapes_' + current_label):
            os.makedirs('eq_soundscapes_' + current_label)
        eq_sox = 'sox ' + path + audio + '_soundscape.wav ' + 'eq_soundscapes_' + current_label + '/eq_' + audio + '.wav lowpass ' + lowpass_value
        #     print(eq_sox)
        os.system(eq_sox)
        print('EQ done :', audio)

def apply_compression(current_label):
        ### dynamic compression sox
        path = 'eq_soundscapes_' + current_label + '/'
        for index, row in labels_df.iterrows():
            audio = row['1_filename.csv']

            audio = audio[:-4]


            if not os.path.exists('compression_soundscapes_' + current_label):
                os.makedirs('compression_soundscapes_' + current_label)
            dynamic_compresion_sox = 'sox ' + path + 'eq_' + audio + '.wav ' + 'compression_soundscapes_' + current_label + '/compressed_' + audio + '.wav compand 0.1,0.3 -60,-60,-30,-15,-20,-12,-4,-8,-2,-7 -2'
            os.system(dynamic_compresion_sox)
            print('dynamic compression:', audio)

def normalisation_synthetic(current_label):
    ### normalise Sox
    path = 'compression_soundscapes_' + current_label + '/'
    if not os.path.exists('normalised_soundscapes_' + current_label):
        os.makedirs('normalised_soundscapes_' + current_label)
    for index, row in labels_df.iterrows():

        audio_file = row['1_filename.csv']


        audio, sr = librosa.core.load(path + 'compressed_' + audio_file, sr=16000)
        audio = audio * (0.7079 / np.max(np.abs(audio)))
        maxv = np.iinfo(np.int16).max
        audio = (audio * maxv).astype(np.int16)
        scipy.io.wavfile.write('normalised_soundscapes_' + current_label + '/' + 'normalised_' + audio_file, sr, audio)

        print('normalised:', audio_file)




def merge_scapes(current_label):
    """ merge the soundscapes together (original with synthetic)"""
    path = 'normalised_soundscapes_' + current_label + '/'
    for index, row in labels_df.iterrows():
        audio = row['1_filename.csv']
        if not os.path.exists('augmented_soundscapes_' + current_label):
            os.makedirs('augmented_soundscapes_' + current_label)
        normed_sox = 'sox -m ' + path + 'normalised_' + audio + ' original_normalised_' + current_label + '/' + audio + ' augmented_soundscapes_' + current_label + '/' + audio
        os.system(normed_sox)
        print('merged:', audio)




def add_fade(current_label):
    """ fade final soundscape (Sox shell)"""
    path = 'augmented_soundscapes_' + current_label + '/'
    for index, row in labels_df.iterrows():
        audio = row['1_filename.csv']
        audio = audio[:-4]
        if not os.path.exists('final_augmented_' + current_label):
            os.makedirs('final_augmented_' + current_label)
        faded_sox = './fade.sh ' + path + audio + '.wav ' + 'final_augmented_' + current_label + '/augmented_' + audio + '.wav'
        os.system(faded_sox)
        print('faded:', audio)


def clean_working_dir():
     """removes all dire"""
     os.system('rm -r augmented_soundscapes*/')
     os.system('rm -r chroma*/')
     os.system('rm -r combined*/')
     os.system('rm -r compression*/')
     os.system('rm -r eq*/')
     os.system('rm -r mean*/')
     os.system('rm -r normalised*/')
     os.system('rm -r segments*/')
     os.system('rm -r soundscapes*/')
     os.system('rm -r transposed*/')


# if sys.argv[1] == 'pre':
#     print('Begin preprocessing...\n')
#     normalise_extract_chroma(current_label)
#     preprocessing_feature_files(output, current_label)
#     calculate_tempo(current_label)
# if sys.argv[1] == 'gn':
#     print('Generating notes...\n')
#     generate_notes(current_label)
#     combined_segmented_notes(current_label)
#     mix_notes(current_label)
# if sys.argv[1] == 'post':
#     print('Post processing...\n')
#     apply_eq(current_label)
#     apply_compression(current_label)
#     normalisation_synthetic(current_label)
#     merge_scapes(current_label)
#     add_fade(current_label)
# if sys.argv[1] == 'cleanup':
#     print('remove uneeded dir')
#     clean_working_dir()