import numpy as np
import pandas as pd
from collections import defaultdict
import os
import random
import time
import pickle

# file paths
data_path = root_path + "Cough Analytics/CAir/CAir (potential events within episodes features)/"
aug_path = data_path + "noise added features (snr 0.0001 to 10000) -- modified noise/AC1/"

expl_away_fp = data_path + "expl away.csv"
expl_towards_fp = data_path + "explosive.csv"
snore_nose_fp = data_path + "snore nose.csv"
snore_throat_fp = data_path + "snore throat.csv"

    def main():

    # to pd df
    X_expl_away = pd.read_csv(expl_away_fp, delimiter=',')   # 149
    X_expl_towards = pd.read_csv(expl_towards_fp, delimiter=',')   # 135
    X_snore_nose = pd.read_csv(snore_nose_fp, delimiter=',')   #119
    X_snore_throat = pd.read_csv(snore_throat_fp, delimiter=',')   # 72


    # organize by patientID
    expl_away_patientID_dict = defaultdict(list)
    for i in X_expl_away['fileName']:
        expl_away_patientID_dict[i[:2]].append("expl_away_noSNR_"+i)

    expl_towards_patientID_dict = defaultdict(list)
    for i in X_expl_towards['fileName']:
        expl_towards_patientID_dict[i[:2]].append("expl_towards_noSNR_"+i)

    snore_nose_patientID_dict = defaultdict(list)
    for i in X_snore_nose['fileName']:
        snore_nose_patientID_dict[i[:2]].append("snore_nose_noSNR_"+i)

    snore_throat_patientID_dict = defaultdict(list)
    for i in X_snore_throat['fileName']:
        snore_throat_patientID_dict[i[:2]].append("snore_throat_noSNR_"+i)


    len_expl_away_dict = dict()
    for id in expl_away_patientID_dict.keys():
        len_expl_away_dict[id] = len(expl_away_patientID_dict[id])

    len_expl_towards_dict = dict()
    for id in expl_towards_patientID_dict.keys():
        len_expl_away_dict[id] = len(expl_away_patientID_dict[id])

    len_snore_nose_dict = dict()
    for id in snore_nose_patientID_dict.keys():
        len_snore_nose_dict[id] = len(snore_nose_patientID_dict[id])

    len_snore_throat_dict = dict()
    for id in snore_throat_patientID_dict.keys():
        len_snore_throat_dict[id] = len(snore_throat_patientID_dict[id])



    cough_keys = {**expl_away_patientID_dict, **expl_towards_patientID_dict}
    snore_keys = {**snore_nose_patientID_dict, **snore_throat_patientID_dict}
    all_patient_keys = cough_keys.keys()

    # BALANCE SNORE / COUGH CLASSES
    for id in all_patient_keys:

        len_diff = (len(expl_away_patientID_dict[id]) + len(expl_towards_patientID_dict[id])) - (len(snore_throat_patientID_dict[id] + snore_nose_patientID_dict[id]))


        # while more coughs than snores for this patient, pop a cough, increment len_diff down one
        while len_diff > 0:
            rand_int = random.randint(0,1)
            if rand_int is 0:
                if len(expl_away_patientID_dict[id]) is not 0:
                    expl_away_patientID_dict[id].pop(random.randrange(len(expl_away_patientID_dict[id])))
            elif rand_int is 1:
                if len(expl_towards_patientID_dict[id]) is not 0:
                    expl_towards_patientID_dict[id].pop(random.randrange(len(expl_towards_patientID_dict[id])))
            len_diff = len_diff - 1

        while len_diff < 0:
            rand_int = random.randint(0,1)
            if rand_int is 0:
                if len(snore_throat_patientID_dict[id]) is not 0:
                    snore_throat_patientID_dict[id].pop(random.randrange(len(snore_throat_patientID_dict[id])))
            elif rand_int is 1:
                if len(snore_nose_patientID_dict[id]) is not 0:
                    snore_nose_patientID_dict[id].pop(random.randrange(len(snore_nose_patientID_dict[id])))
            len_diff = len_diff + 1

    # get all values for each patient in a dictionary
    snr_list = ["0.0001.csv","0.001.csv","0.01.csv","0.1.csv","10.csv","100.csv","1000.csv","10000.csv"]

    for data_name in os.listdir(aug_path):
        if data_name.startswith("expl away"):
            for snr in snr_list:
                if data_name.endswith(snr):
                    temp_df = pd.read_csv(aug_path+data_name,delimiter=',')
                    for i in temp_df['fileName']:
                        for val in expl_away_patientID_dict.values():
                            for eventID in val:
                                if i in eventID:
                                    expl_away_patientID_dict[i[:2]].append("expl_away_"+snr[:-4]+"_"+i)
                                    break


        elif data_name.startswith("explosive"):
            for snr in snr_list:
                if data_name.endswith(snr):
                    temp_df = pd.read_csv(aug_path+data_name,delimiter=',')
                    for i in temp_df['fileName']:
                        for val in expl_towards_patientID_dict.values():
                            for eventID in val:
                                if i in eventID:
                                    expl_towards_patientID_dict[i[:2]].append("expl_towards_"+snr[:-4]+"_"+i)
                                    break

        elif data_name.startswith("snore nose"):
            for snr in snr_list:
                if data_name.endswith(snr):
                    temp_df = pd.read_csv(aug_path+data_name,delimiter=',')
                    for i in temp_df['fileName']:
                        for val in snore_nose_patientID_dict.values():
                            for eventID in val:
                                if i in eventID:
                                    snore_nose_patientID_dict[i[:2]].append("snore_nose_"+snr[:-4]+"_"+i)
                                    break

        elif data_name.startswith("snore throat"):
            #print("in snore throat...")
            for snr in snr_list:
                if data_name.endswith(snr):
                    #print("ends with worked...")
                    temp_df = pd.read_csv(aug_path+data_name,delimiter=',')
                    for i in temp_df['fileName']:
                        for val in snore_throat_patientID_dict.values():
                            for eventID in val:
                                if i in eventID:
                                    snore_throat_patientID_dict[i[:2]].append("snore_throat_"+snr[:-4]+"_"+i)
                                    break

    #we should have 382 total coughs + snores originally...
    #plus 9 SNR levels, so 382 + 8*382 = 3438 total


    # combine into cough and snore dictionaries
    cough_dict = defaultdict(list)
    snore_dict = defaultdict(list)

    for i in expl_away_patientID_dict.keys():
        cough_dict[i].extend(expl_away_patientID_dict[i])

    for i in expl_towards_patientID_dict.keys():
        cough_dict[i].extend(expl_towards_patientID_dict[i])

    rsum=0
    for i in cough_dict.keys():
        rsum += len(cough_dict[i])

    for i in snore_nose_patientID_dict.keys():
        snore_dict[i].extend(snore_nose_patientID_dict[i])

    for i in snore_throat_patientID_dict.keys():
        snore_dict[i].extend(snore_throat_patientID_dict[i])

    rsum=0
    for i in snore_dict.keys():
        rsum += len(snore_dict[i])

    #save dicts
    save_path = root_path + "Cough Analytics/Data/"
    with open(save_path+'cough_dict'+'.pkl','wb') as f:
        pickle.dump(cough_dict, f, pickle.HIGHEST_PROTOCOL)
    with open(save_path+'snore_dict'+'.pkl','wb') as f:
        pickle.dump(snore_dict, f, pickle.HIGHEST_PROTOCOL)

    # break into train and test sets
    for i in all_patient_keys:

        temp_test_list = []
        temp_train_list = []

        for j in all_patient_keys: #cough_dict and snore_dict have same keys
            if j is i:
                temp_test_list.extend(cough_dict[i])
                temp_test_list.extend(snore_dict[i])
            if j is not i:
                temp_train_list.extend(cough_dict[j])
                temp_train_list.extend(snore_dict[j])

        y_test = []
        X_test = []
        temp_df = pd.read_csv(expl_towards_fp,delimiter=",")
        df_test = pd.DataFrame(columns = temp_df.columns) #creates a new dataframe that's empty

        for dataID in temp_test_list:
            #dataID in form 'expl_away_noSNR_44_20170707_141339#3#4'
            fileName_to_match = dataID.split('_')[3]+"_"+dataID.split("_")[4]+"_"+dataID.split("_")[5]


            if dataID.startswith("expl_towards"):
                y_test.append(0)
                if dataID.split('_')[2] == 'noSNR':
                    temp_df = pd.read_csv(expl_towards_fp,delimiter=",")
                    temp_test_df = temp_df[temp_df['fileName'] == fileName_to_match]

                else:
                    temp_df = pd.read_csv(aug_path+"explosive_snr_"+dataID.split("_")[2]+".csv")
                    temp_test_df = temp_df[temp_df['fileName'] == fileName_to_match]

            elif dataID.startswith("expl_away"):
                y_test.append(0)
                if dataID.split('_')[2] == 'noSNR':
                    temp_df = pd.read_csv(expl_away_fp,delimiter=",")
                    temp_test_df = temp_df[temp_df['fileName'] == fileName_to_match]

                else:
                    temp_df = pd.read_csv(aug_path+"expl away_snr_"+dataID.split("_")[2]+".csv")
                    temp_test_df = temp_df[temp_df['fileName'] == fileName_to_match]

            elif dataID.startswith("snore_nose"):
                y_test.append(1)
                if dataID.split('_')[2] == 'noSNR':
                    temp_df = pd.read_csv(snore_nose_fp,delimiter=",")
                    temp_test_df = temp_df[temp_df['fileName'] == fileName_to_match]

                else:
                    temp_df = pd.read_csv(aug_path+"snore nose_snr_"+dataID.split("_")[2]+".csv")
                    temp_test_df = temp_df[temp_df['fileName'] == fileName_to_match]

            elif dataID.startswith("snore_throat"):
                y_test.append(1)
                if dataID.split('_')[2] == 'noSNR':
                    temp_df = pd.read_csv(snore_throat_fp,delimiter=",")
                    temp_test_df = temp_df[temp_df['fileName'] == fileName_to_match]

                else:
                    temp_df = pd.read_csv(aug_path+"snore throat_snr_"+dataID.split("_")[2]+".csv")
                    temp_test_df = temp_df[temp_df['fileName'] == fileName_to_match]

            df_test = df_test.append(temp_test_df)
            df_test.drop_duplicates(inplace=True) #drop dups, keeps first occurence, inplace updates same df

        X_test = df_test
        y_test_np = np.array(y_test)
        test = X_test
        test['type'] = y_test_np


        y_train = []
        X_train = []
        temp_df = pd.read_csv(expl_towards_fp,delimiter=",")
        df_train = pd.DataFrame(columns = temp_df.columns) #creates a new dataframe that's empty

        for dataID in temp_train_list:
            fileName_to_match = dataID.split('_')[3]+"_"+dataID.split("_")[4]+"_"+dataID.split("_")[5]

            if dataID.startswith("expl_towards"):
                y_train.append(0)
                if dataID.split('_')[2] == 'noSNR':
                    temp_df = pd.read_csv(expl_towards_fp,delimiter=",")
                    temp_train_df = temp_df[temp_df['fileName'] == fileName_to_match]

                else:
                    temp_df = pd.read_csv(aug_path+"explosive_snr_"+dataID.split("_")[2]+".csv")
                    temp_train_df = temp_df[temp_df['fileName'] == fileName_to_match]

            elif dataID.startswith("expl_away"):
                y_train.append(0)
                if dataID.split('_')[2] == 'noSNR':
                    temp_df = pd.read_csv(expl_away_fp,delimiter=",")
                    temp_train_df = temp_df[temp_df['fileName'] == fileName_to_match]

                else:
                    temp_df = pd.read_csv(aug_path+"expl away_snr_"+dataID.split("_")[2]+".csv")
                    temp_train_df = temp_df[temp_df['fileName'] == fileName_to_match]

            elif dataID.startswith("snore_nose"):
                y_train.append(1)
                if dataID.split('_')[2] == 'noSNR':
                    temp_df = pd.read_csv(snore_nose_fp,delimiter=",")
                    temp_train_df = temp_df[temp_df['fileName'] == fileName_to_match]

                else:
                    temp_df = pd.read_csv(aug_path+"snore nose_snr_"+dataID.split("_")[2]+".csv")
                    temp_train_df = temp_df[temp_df['fileName'] == fileName_to_match]

            elif dataID.startswith("snore_throat"):
                y_train.append(1)
                if dataID.split('_')[2] == 'noSNR':
                    temp_df = pd.read_csv(snore_throat_fp,delimiter=",")
                    temp_train_df = temp_df[temp_df['fileName'] == fileName_to_match]

                else:
                    temp_df = pd.read_csv(aug_path+"snore throat_snr_"+dataID.split("_")[2]+".csv")
                    temp_train_df = temp_df[temp_df['fileName'] == fileName_to_match]


            df_train = df_train.append(temp_train_df)
            df_train.drop_duplicates(inplace=True) #drop dups, keeps first occurence, inplace updates same df


        X_train = df_train
        y_train_np = np.array(y_train)
        train = X_train
        train['type'] = y_train_np

        save_path = root_path + "Cough Analytics/Data/"
        folder_name = i+"_patient_as_test/"

        if not os.path.exists(save_path+folder_name):
            os.makedirs(save_path+folder_name)


        print("Saving train, test to ",save_path+folder_name)

        print("train shape:",train.shape) #should be (row length, 43) 42 feat + 1 class
        print("test shape:",test.shape) #should be (row length, 43) 42 feat + 1 class

        train.to_csv(save_path+folder_name+"train_"+i)
        test.to_csv(save_path+folder_name+"test_"+i)


        with open(save_path+folder_name+'test_set_' + i + '.txt','w') as f:
            for item in temp_test_list:
                f.write("%s\n"%item)

        with open(save_path+folder_name+'train_set_' + i + '.txt','w') as f:
            for item in temp_train_list:
                f.write("%s\n"%item)

if __name__ == '__main__':
    main()
