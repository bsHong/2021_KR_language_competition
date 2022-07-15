import pandas as pd
from tqdm import tqdm
WIC_path = '../00-data/'
train_name = 'NIKL_SKT_WiC_Train.tsv'
dev_name = 'NIKL_SKT_WiC_Dev.tsv'
test_name = 'NIKL_SKT_WiC_Test.tsv'

def mark_span(text, span_st, span_ed, mark):
    return text[:span_st] + mark + text[span_st:span_ed] + mark + text[span_ed:], span_st, span_ed+2

def change2Mark(data_name):
    df = pd.read_csv(WIC_path+data_name, delimiter='\t')
#     mark_df_train = df.copy()
    mark_df_train = pd.DataFrame(columns = ['ID','Target', 'SENTENCE1', 'SENTENCE2',	'ANSWER',	'start_s1',	'end_s1',	'start_s2',	'end_s2'])
    for idx, row in tqdm(df.iterrows()):
        SENTENCE1, start_s1, end_s1 = mark_span(row['SENTENCE1'], row['start_s1'], row['end_s1'], '*')
        SENTENCE2, start_s2, end_s2 = mark_span(row['SENTENCE2'], row['start_s2'], row['end_s2'], '*')
        new_data = {'ID' : row['ID'],	'Target' : row['Target'],	'SENTENCE1' : SENTENCE1,	'SENTENCE2' : SENTENCE2,	'ANSWER':row['ANSWER'],	'start_s1': start_s1,	'end_s1': end_s1,	'start_s2' : start_s2,	'end_s2':end_s2}
#         mark_df_train.loc[idx] = new_data
        mark_df_train = mark_df_train.append(new_data, ignore_index=True)
        if idx == 7 :
            print()
            print(mark_df_train)
            
    mark_df_train.to_csv('./mark_'+data_name[:-4]+'.csv', mode='w', index=None)

# change2Mark(train_name)
# change2Mark(dev_name)

def change2Mark_test(data_name):
    df = pd.read_csv(WIC_path+data_name, delimiter='\t')
#     mark_df_train = df.copy()
    mark_df_train = pd.DataFrame(columns = ['ID','Target', 'SENTENCE1', 'SENTENCE2',	'ANSWER',	'start_s1',	'end_s1',	'start_s2',	'end_s2'])
    for idx, row in tqdm(df.iterrows()):
        SENTENCE1, start_s1, end_s1 = mark_span(row['SENTENCE1'], row['start_s1'], row['end_s1'], '*')
        SENTENCE2, start_s2, end_s2 = mark_span(row['SENTENCE2'], row['start_s2'], row['end_s2'], '*')
        new_data = {'ID' : row['ID'],	'Target' : row['Target'],	'SENTENCE1' : SENTENCE1,	'SENTENCE2' : SENTENCE2,	'ANSWER':row['ANSWER'],	'start_s1': start_s1,	'end_s1': end_s1,	'start_s2' : start_s2,	'end_s2':end_s2}
#         mark_df_train.loc[idx] = new_data
        mark_df_train = mark_df_train.append(new_data, ignore_index=True)
        if idx == 7 :
            print()
            print(mark_df_train)
            
    mark_df_train.to_csv('./mark_'+data_name[:-4]+'.csv', mode='w', index=None)
change2Mark_test(test_name)
