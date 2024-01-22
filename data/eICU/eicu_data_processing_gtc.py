import pandas as pd
import numpy as np
import pickle
import dill
from datetime import datetime, timedelta
import os
from tqdm import tqdm
import random
import dill
SEED = 2020
random.seed(SEED)

def convert_to_icd9(dxStr):
    if dxStr.startswith('E'):
        if len(dxStr) > 4: return 'D_'+dxStr[:4] + '.' + dxStr[4:]
        else: return dxStr
    else:
        if len(dxStr) > 3: return 'D_'+dxStr[:3] + '.' + dxStr[3:]
        else: return 'D_'+dxStr

def convert_to_3digit_icd9(dxStr):
    if dxStr.startswith('E'):
        if len(dxStr) > 4: return 'D_'+dxStr[:4]
        else: return dxStr
    else:
        if len(dxStr) >3: return 'D_'+dxStr[:3]
        else: return 'D_'+dxStr

def conver_to_coarse_icd9(dxStr):
    if '.' in dxStr:
        dx = dxStr.split('.')[0].strip()
        return dx
    else: return dxStr.strip()

##### diag icd code 除了E开头的还有很多其他字母开头的
def diag_conver(dxList):
    new_list = []
    for dx in dxList:
        new_list.append(conver_to_coarse_icd9(dx))
    return new_list
def icd9code_split(dxList):
    new_list = []
    codes = dxList.split(',')
    for code in codes:
        new_list.append(code.strip())
    return new_list

def process_diag(diag_file, nrows=None):
    print("processing diagnosis table...")
    diag_pd = pd.read_csv(diag_file, nrows=nrows)
    diag_pd.dropna(subset=['icd9code'], inplace=True)
    # icd9code_diag = diag_pd.groupby(by=['patientunitstayid','diagnosisoffset'])['icd9code'].unique().reset_index()
    diag_pd['split_icd9code'] = diag_pd.icd9code.map(lambda x: icd9code_split(x))  
    diag_pd['icd9code_3digit'] = diag_pd.split_icd9code.map(lambda x: diag_conver(x))
    print('DONE...')
    return diag_pd


def process_medication(medication_file, nrows=None):
    print("processing medication table...")
    medication_pd = pd.read_csv(medication_file, nrows=nrows)
    medication_drug = medication_pd[['medicationid', 'patientunitstayid', 'drugstartoffset', 'drugname', 'dosage','gtc']]
    medication_drug.drop(index = medication_drug[medication_drug['gtc'] == 0].index, axis=0, inplace=True)
    # medication_drug.dropna(subset=['drugname'], inplace=True)
    print('DONE...')
    return medication_drug


def process_pastHistory(pastHistory_file, nrows=None):
    print("processing pastHistory table....")
    pastH = pd.read_csv(pastHistory_file, nrows=nrows)
    pastH = pastH[['pasthistoryid','patientunitstayid','pasthistoryoffset','pasthistoryenteredoffset','pasthistoryvaluetext']]
    print("DONE...")
    return pastH

def process_patient(patient_file, nrows=None):
    print("processing patient table...")
    patient_pd = pd.read_csv(patient_file, nrows=nrows)
    patient_pd = patient_pd[['uniquepid', 'patienthealthsystemstayid', 'unitdischargeoffset', 'hospitaladmitoffset', 'hospitaldischargeoffset', \
                            'patientunitstayid', 'gender', 'age', 'ethnicity', 'admissionweight', 'apacheadmissiondx']]
    ### stay_id > 2
    # lg2_patient_pd = patient_pd[['patienthealthsystemstayid','patientunitstayid']].groupby(by='patienthealthsystemstayid')['patientunitstayid'].unique().reset_index()
    # lg2_patient_pd['stay_len'] = lg2_patient_pd.patientunitstayid.map(lambda x:len(x))
    # lg2_patient_pd = lg2_patient_pd[lg2_patient_pd.stay_len>=2]
    # lg2_patient_pd.drop(columns=['patientunitstayid'], axis=1, inplace=True)
    # patient_pd = patient_pd.merge(lg2_patient_pd, on='patienthealthsystemstayid', how='inner')

    # patient_pd['age'] = patient_pd['age'].fillna(0)
    patient_pd['gender'] = patient_pd['gender'].fillna(-1)
    patient_pd['ethnicity'] = patient_pd['ethnicity'].fillna(-1)
    patient_pd['hospitaladmitoffset'] = patient_pd['hospitaladmitoffset'].map(lambda x: int(x))
    patient_unitstay_ids = list(set(patient_pd.patientunitstayid.tolist()))
    print('DONE...')
    return patient_pd, patient_unitstay_ids

def process_apachePatientResult(apachePatientResult_file, nrows=None):
    print("processing apachePatientResult table...")
    apachePatientResult = pd.read_csv(apachePatientResult_file, nrows=nrows)
    print(apachePatientResult)
    apachePatientResult = apachePatientResult[['apachepatientresultsid', 'patientunitstayid', 'actualicumortality', \
                                                'actualiculos', 'actualhospitalmortality', 'actualhospitallos']]
    print("DONE...")
    return apachePatientResult

def interaction_op(a, b):
    return list(set(b).intersection(set(a)))

class Voc(object):
    def __init__(self):
        self.idx2word = {}
        self.word2idx = {}
    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word[len(self.word2idx)] = word
            self.word2idx[word] = len(self.word2idx)

def save_dict(dict_object, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(dict_object, f)

def process_seqs_dates_static(eICU_path, voc_file, save_path):
    diagnosis_path = eICU_path+'diagnosis.csv'
    medication_path = eICU_path+'medication.csv'
    lab_path = eICU_path+'lab.csv'
    patient_path = eICU_path+'patient.csv'
    patient, patient_unitstay_ids = process_patient(patient_path)
    diagnosis = process_diag(diagnosis_path)
    medication = process_medication(medication_path)

    print("merge tables with patienthealthsystemstayid...")
    unitid_phealth_dict = {}
    for index, row in tqdm(patient.iterrows(), desc='unitid_phealth_dict:'):
        punitid = row['patientunitstayid']
        phealthid = row['patienthealthsystemstayid']
        unitid_phealth_dict[punitid] = phealthid
    print("unitid_phealth_dict:{}".format(len(unitid_phealth_dict)))

    diagnosis['patienthealthsystemstayid'] = diagnosis.patientunitstayid.map(lambda x: unitid_phealth_dict[x])
    medication['patienthealthsystemstayid'] = medication.patientunitstayid.map(lambda x: unitid_phealth_dict[x])

    patient = patient[~patient.age.isin(['> 89'])]
    patient_lg2 = get_visit_lg2(patient)
    patient_lg2_hid = patient_lg2.patienthealthsystemstayid.tolist()
    patient_lg2_uids = patient_lg2.patientunitstayid.tolist()
    print("patient lg2 health ids:{} {}".format(len(patient_lg2_hid), len(set(patient_lg2_hid))))
    print("patient lg2 unitstay ids:{} {}".format(len(patient_lg2_uids), len(set(patient_lg2_uids))))

    diagnosis_lg2 = get_visit_lg2(diagnosis)
    diagnosis_lg2_uids = diagnosis_lg2.patientunitstayid.tolist()
    print("\nnew_diagnosis_lg2_uids:{} \t{}".format(len(diagnosis_lg2_uids),len(set(diagnosis_lg2_uids))))

    medication_lg2 = get_visit_lg2(medication)
    medication_lg2_uids = medication_lg2.patientunitstayid.tolist()
    print("\nnew_medication_lg2_uids:{}\t{}".format(len(medication_lg2_uids),len(medication_lg2_uids)))

    lg2_inner_uids = interaction_op(list(set(diagnosis_lg2_uids)), list(set(medication_lg2_uids)))

    ### 以lg2_inner_hids作为选择的patientsystemID
    print("lg2 diagnosis, mediction, labtest uids:{} {}".format(len(lg2_inner_uids), len(set(lg2_inner_uids))))
    lg2_inner_uids_patient = patient_lg2[patient_lg2.patientunitstayid.isin(lg2_inner_uids)]
    lg2_inner_hids = list(set(lg2_inner_uids_patient.patienthealthsystemstayid.tolist()))
    lg2_inner_pids = list(set(lg2_inner_uids_patient.uniquepid.tolist()))
    print("lg2_inner_hids:{}\tlg2_inner_pids:{} mean_visit:{} {}".format(len(list(set(lg2_inner_hids))),len(list(set(lg2_inner_pids))), \
        len(lg2_inner_uids)/len(lg2_inner_hids), len(list(set(lg2_inner_uids)))/len(list(lg2_inner_hids))))

    lg2_inner_uids_diagnosis = diagnosis_lg2[diagnosis_lg2.patientunitstayid.isin(lg2_inner_uids)]
    lg2_inner_uids_medication = medication_lg2[medication_lg2.patientunitstayid.isin(lg2_inner_uids)]
    print("diagnosis uids:{}\tmedication uids:{}".format(len(set(lg2_inner_uids_diagnosis.patientunitstayid.tolist())), \
        len(set(lg2_inner_uids_medication.patientunitstayid.tolist()))))


    # ### data sequence: [[[visit1],[visit2], ...,[visitN]],[],..[]]  ---->>> visit=(diagcode, medcode)
    diag_dict = Voc()
    medication_dict = Voc()
    diag_3digit_dict = Voc()
    for index, row in tqdm(lg2_inner_uids_diagnosis.iterrows(), desc='diagnosis dict:'):
        split_icd9code = row['split_icd9code']
        for icd9 in split_icd9code:
            diag_dict.add_word(icd9)
        icd9code_3digit = row['icd9code_3digit']
        for icd9_3 in icd9code_3digit:
            diag_3digit_dict.add_word(icd9_3)
        
    drug_name_list = []
    for index, row in tqdm(lg2_inner_uids_medication.iterrows(), desc='drug dict:'):
        gtc = row['gtc']
        medication_dict.add_word(gtc)

    print("diag_dict:{}\tdiag_3diagit_dict:{}\tmedic_dic:{}\tlab_dict:{}".format(len(diag_dict.word2idx), len(diag_3digit_dict.word2idx), len(medication_dict.word2idx), len(labtest_dict.word2idx)))
    save_dict(diag_dict, save_path+'diag5_voc.pk')
    save_dict(diag_3digit_dict, save_path+'diag3_voc.pk')
    save_dict(medication_dict, save_path+'uniqueMed_voc.pk')
    vocabulary_file = voc_file
    dill.dump(obj={'diag_voc':diag_dict, 'med_voc':medication_dict ,'diag_3digit_voc':diag_3digit_dict}, file=open(vocabulary_file,'wb'))
    # static info 'age', 'admissionweight', 'gender', 'ethnicity'
    obj_feat = ['gender', 'ethnicity']
    feature_dim = []
    feat_val_dic = {}
    for obj in obj_feat:
        vals = list(lg2_inner_uids_patient[obj].unique())
        feature_dim.append(len(vals))
        feat_val_dic[obj] = vals

    pHealth_data_df = pd.DataFrame([], columns=['pid', 'hid', 'icudischargetime', 'admtime', 'dischargetime', 'delta_t', 'global_t' , 'medic', 'diag3', 'diag5'])
    pHealth_static_data_df = pd.DataFrame([], columns=['pid', 'static_info'])
    pHealth_data = []
    pHealth_dates = []
    new_patient_phids = []
    new_patient_puids = []
    for phid in tqdm(lg2_inner_hids, desc='patientHealthID:'):
        sigle_Hpatient = lg2_inner_uids_patient[lg2_inner_uids_patient.patienthealthsystemstayid==phid]
        sigle_Hpatient.sort_values(by='hospitaladmitoffset', ascending=False, inplace=True)    
        # print(sigle_Hpatient[['patienthealthsystemstayid', 'hospitaladmitoffset', 'unitdischargeoffset']])    
        punit_ids = sigle_Hpatient.patientunitstayid.tolist()
        if len(punit_ids) < 2:
            continue
        new_patient_phids.append(phid)
        new_patient_puids.extend(punit_ids)
        # print("phid:{} patient unit stay ids:{}".format(phid, punit_ids))
        '''
        process time gap
        '''
        patient_dates = []
        patient_global_times = []
        patient_dates.append(0.)
        patient_global_times.append(0.)
        for index in range(1, len(punit_ids)):
            # 由于eICU数据的特殊性，因此 是以ICU的入院为单位，其中一次eICU的 ICU住院记录为一次visit
            date_before_icuadm = sigle_Hpatient[sigle_Hpatient.patientunitstayid==punit_ids[index-1]].hospitaladmitoffset.tolist()[0]
            date_before_icudischarge = sigle_Hpatient[sigle_Hpatient.patientunitstayid==punit_ids[index-1]].unitdischargeoffset.tolist()[0]
            date_before = (date_before_icudischarge-date_before_icuadm)
            date_curr_icuadm = sigle_Hpatient[sigle_Hpatient.patientunitstayid==punit_ids[index]].hospitaladmitoffset.tolist()[0]
            date_curr_icudischarge = sigle_Hpatient[sigle_Hpatient.patientunitstayid==punit_ids[index]].unitdischargeoffset.tolist()[0]
            date_curr = (date_curr_icudischarge-date_curr_icuadm)
            time_gap = (date_curr - date_before)  / 1440.
            if time_gap < 0:
                time_gap = 0.0
            patient_dates.append(time_gap)
            patient_global_times.append(date_curr/1440.)        
        '''
        process diagnosis code, drug code, labtest code
        '''
        multi_unit_data = []
        for punit_id, t, g_t in zip(punit_ids, patient_dates, patient_global_times):
            p_row = {}
            p_row['pid'] = phid
            p_row['hid'] = punit_id
            p_row['admtime'] = sigle_Hpatient[sigle_Hpatient.patientunitstayid==punit_id].hospitaladmitoffset.tolist()[0]
            p_row['icudischargetime'] = sigle_Hpatient[sigle_Hpatient.patientunitstayid==punit_id].unitdischargeoffset.tolist()[0]
            p_row['dischargetime'] = sigle_Hpatient[sigle_Hpatient.patientunitstayid==punit_id].hospitaldischargeoffset.tolist()[0]
            p_row['delta_t'] = t
            p_row['global_t'] = g_t
            sigle_unit_data = []
            diagnosis_codes = lg2_inner_uids_diagnosis[lg2_inner_uids_diagnosis.patientunitstayid==punit_id]
            sorted_diag_codes = diagnosis_codes.sort_values(by='diagnosisoffset')
            tmp_split_icd9code = []
            tmp_icd9_3digit = []
            for sicd9code in sorted_diag_codes.split_icd9code.tolist():
                tmp_split_icd9code.extend(sicd9code)
            for icd9_3 in sorted_diag_codes.icd9code_3digit.tolist():
                tmp_icd9_3digit.extend(icd9_3)

            set_diag = list(set(tmp_split_icd9code))
            set_diag.sort(key=tmp_split_icd9code.index)
            set_diag_3digit  =list(set(tmp_icd9_3digit))
            set_diag_3digit.sort(key=tmp_icd9_3digit.index)

            split_icd9code_num = [diag_dict.word2idx[i] for i in set_diag]
            icd9_3digit_num = [diag_3digit_dict.word2idx[i] for i in set_diag_3digit]
            sigle_unit_data.append(split_icd9code_num)
            medication_codes = lg2_inner_uids_medication[lg2_inner_uids_medication.patientunitstayid==punit_id]
            sorted_med_codes = medication_codes.sort_values(by='drugstartoffset')
            medication_items = sorted_med_codes.gtc.tolist()

            set_medic_items = list(set(medication_items))
            set_medic_items.sort(key=medication_items.index)

            medication_items_num = [medication_dict.word2idx[i] for i in set_medic_items]
            if len(medication_items_num) == 0:
                continue
            sigle_unit_data.append(medication_items_num)
            multi_unit_data.append(sigle_unit_data)
            #  只保存纯文本
            p_row['medic'] = set_medic_items
            p_row['diag3'] = set_diag_3digit
            p_row['diag5'] = set_diag
            pHealth_data_df = pHealth_data_df.append(p_row, ignore_index=True)
        
        pHealth_dates.append(patient_dates)
        # pHealth_static_data.append(static_onehot)
        pHealth_data.append(multi_unit_data)
    ## 保存数据
    seq_save_path = save_path+"raw_lab_med_diag3_diag5.csv"
    static_save_path = save_path+"all_static_info.csv"
    pHealth_data_df.to_csv(seq_save_path, index=False)
    pHealth_static_data_df.to_csv(static_save_path, index=False)
    print("all_data_len:{} \t all_dates:{}".format(len(pHealth_data), len(pHealth_dates)))
    pickle.dump(pHealth_data, open(save_path +'seqs.pk', 'wb'), protocol=2)
    pickle.dump(pHealth_dates, open(save_path +'dates.pk', 'wb'), protocol=2)
    with open(save_path +'statistic_info.txt','w') as f:
        f.write("unitid_phealth_dict:{}\n".format(len(unitid_phealth_dict)))
        f.write("patient lg2 health ids:{} {}\n".format(len(patient_lg2_hid), len(set(patient_lg2_hid))))
        f.write("patient lg2 unitstay ids:{} {}\n".format(len(patient_lg2_uids), len(set(patient_lg2_uids))))
        f.write("new_diagnosis_lg2_uids:{} \t{}\n".format(len(diagnosis_lg2_uids),len(set(diagnosis_lg2_uids))))
        f.write("new_medication_lg2_uids:{}\t{}\n".format(len(medication_lg2_uids),len(medication_lg2_uids)))
        # f.write("new_labtest_lg2_uids:{} \t {}\n".format(len(labtest_lg2_uids),len(set(labtest_lg2_uids))))
        f.write("lg2 diagnosis, mediction, labtest uids:{} {}\n".format(len(lg2_inner_uids), len(set(lg2_inner_uids))))
        f.write("diagnosis uids:{}\tmedication uids:{}\n".format(len(set(lg2_inner_uids_diagnosis.patientunitstayid.tolist())), \
            len(set(lg2_inner_uids_medication.patientunitstayid.tolist()))))
        f.write("lg2_inner_hids:{}\tlg2_inner_pids:{}\n".format(len(set(lg2_inner_hids)),len(set(lg2_inner_pids))))
        f.write("diagnosis uids:{}\tmedication uids:{}\n".format(len(set(lg2_inner_uids_diagnosis.patientunitstayid.tolist())), \
            len(set(lg2_inner_uids_medication.patientunitstayid.tolist()))))
        select_pids = list(set(lg2_inner_uids_patient[lg2_inner_uids_patient.patientunitstayid.isin(new_patient_puids)].uniquepid.tolist()))
        f.write("Final selected lg2 hids:{}\tuids:{}\tpids:{}\tmean_visit:{}\n".format(len(new_patient_phids), len(new_patient_puids), len(select_pids), (len(new_patient_phids)/len(new_patient_puids))))
        f.write("**"*20)
        f.write("\nall_data_len:{} \t all_dates:{}\n".format(len(list(pHealth_data_df.pid.unique())), len(pHealth_dates)))
        f.write("average visit:{} {}".format(len(lg2_inner_uids)/len(lg2_inner_hids), len(list(set(lg2_inner_uids)))/len(list(set(lg2_inner_hids)))))

        f.write("diag_dict:{}\tdiag_3diagit_dict:{} \t medic_dic:{}\n".format(len(diag_dict.word2idx), len(diag_3digit_dict.word2idx), \
            len(medication_dict.word2idx)))
        f.write("avg_diag_len:{}\tavg_med_len:{}\tavg_diag3digit_len:{}\n".format(np.mean([len(i) for i in pHealth_data_df['diag5'].tolist()]), 
                                                         np.mean([len(i) for i in pHealth_data_df['medic'].tolist()]),
                                                         np.mean([len(i) for i in pHealth_data_df['diag3'].tolist()])))
    print("Seqs, dates, static_info save DONE....")
    return pHealth_data_df

def get_visit_lg2(data):
    data_lg2 = data.groupby(by=['patienthealthsystemstayid'])['patientunitstayid'].unique().reset_index()
    data_lg2['visit_len'] = data_lg2.patientunitstayid.map(lambda x : len(x))
    data_lg2 = data_lg2[data_lg2.visit_len >= 2]
    data_lg2_unit = data_lg2.patientunitstayid.tolist()
    patient_units_ids = []
    for d in data_lg2_unit:
        d_list = list(d)
        patient_units_ids.extend(d_list)
    new_data_lg2 = data[data.patientunitstayid.isin(patient_units_ids)]
    return new_data_lg2

# create final records
def create_patient_record(df, diag_voc, med_voc, diag_3digit_voc, ehr_sequence_file):
    records = []  # (patient, code_kind:3, codes)  code_kind:diag, proc, med
    for subject_id in df["pid"].unique():
        item_df = df[df["pid"] == subject_id]
        patient = []
        for index, row in item_df.iterrows():
            admission = []
            admission.append([diag_voc.word2idx[i] for i in eval(row["diag5"])])
            admission.append([med_voc.word2idx[i] for i in eval(row["medic"])])
            admission.append([diag_3digit_voc.word2idx[i] for i in eval(row["diag3"])])
            patient.append(admission)
        records.append(patient)
    dill.dump(obj=records, file=open(ehr_sequence_file, "wb"))
    return records


if __name__ == '__main__':
    '''
    得到seqs dates static_info的数据，以病人的healthsystemID作为ID
    '''    
    ### eICU data path 
    eICU_path = "/home/lsc/model/lsc_code/DataSet/eICUData/physionet.org/files/eicu-crd/2.0/"
    ehr_data = process_seqs_dates_static(eICU_path, voc_file='./output/eicu_voc.pk', save_path='./raw_data/')
    
    # final pkl file
    ehr_data = pd.read_csv('./raw_data/raw_lab_med_diag3_diag5.csv')
    voc = dill.load(open('./output/eicu_voc.pk', 'rb'))
    diag_voc, med_voc, diag_3digit_voc = voc['diag_voc'], voc['med_voc'], voc['diag_3digit_voc']
    create_patient_record(ehr_data, diag_voc, med_voc, diag_3digit_voc, ehr_sequence_file='./output/records_final.pkl')


        
















