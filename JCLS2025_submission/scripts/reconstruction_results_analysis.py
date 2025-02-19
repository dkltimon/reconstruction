# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 17:50:20 2024
"""

#!pip install evaluate
#!pip install rouge_score
#!pip install sacrebleu
#!pip install jiwer

import evaluate
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_metrics():
    metric1 = evaluate.load("sacrebleu")
    metric2 = evaluate.load("wer")
    metric3 = evaluate.load("rouge")
    return metric1, metric2, metric3

metric1, metric2, metric3 = load_metrics()

test_pred = open(r'orig_reviews.txt', 'r', encoding='utf-8').read().split('\n')
test_ref = open(r'randomized_reviews.txt', 'r', encoding='utf-8').read().split('\n')

wer_score = metric2.compute(predictions=test_pred, references=test_ref)
rougel_score = metric3.compute(predictions=test_pred, references=test_ref)
bleu_score = metric1.compute(predictions=test_pred, references=test_ref)

wer_score_pd = {}
wer_score_pd["wer"] = wer_score
wer_score_pd = pd.DataFrame.from_dict(wer_score_pd, orient="index")
wer_score_pd

rougel_score_pd = pd.DataFrame.from_dict(rougel_score, orient="index")
rougel_score_pd

bleu_score_pd = {}
bleu_score_pd["bleu"] = bleu_score["score"]
bleu_score_pd = pd.DataFrame.from_dict(bleu_score_pd, orient="index")
bleu_score_pd

all_scores = pd.concat([wer_score_pd, rougel_score_pd, bleu_score_pd])
all_scores.T

df = pd.read_csv(r'evaluated_results.csv', sep='\t')

df = pd.melt(df, id_vars=['type'], value_vars=['WER', 'rouge1', 'rouge2', 'rougeL', 'rougeLsum', 'bleu']
             , var_name='measure', value_name='score')

df = pd.read_csv(r'Figure1.csv', sep='\t')

sns.set(font_scale=1.5)
sns.set_style("whitegrid")
f, ax = plt.subplots(figsize = (12,6))
g = sns.barplot(y='score', x='measure', hue='type', data=df)
#ax.set_yscale('log')
plt.show()





import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import os
import evaluate

os.chdir(r'F:\ResearchCaseStudies')

########################################################################################################################
#visualize figure 1

df = pd.read_csv(r'F:\ResearchCaseStudies\inference_results_all.csv', sep='\t')

sns.set(font_scale=1.5)
sns.set_style("whitegrid")
f, ax = plt.subplots(figsize = (14,8))
g = sns.barplot(y='score', hue='type', x='measure', data=df)
g.legend(title='type', loc='best', bbox_to_anchor=(1, 1))

########################################################################################################################
#calculate scores for each of the reviews

translated = open(r'translated_text_75.txt', 'r', encoding='utf-8').read().split('\n')
translated.pop()
orig = open(r'orig_text_75.txt', 'r', encoding='utf-8').read().split('\n')
orig.pop()

def load_metrics():
    metric1 = evaluate.load("sacrebleu")
    metric2 = evaluate.load("wer")
    metric3 = evaluate.load("rouge")
    return metric1, metric2, metric3

metric1, metric2, metric3 = load_metrics()

n = 4094
results = []
while n < 5000:
    pred = [translated[n]]
    ref = [orig[n]]
    wer_score = metric2.compute(predictions=pred, references=ref)
    rouge_score = metric3.compute(predictions=pred, references=ref)
    bleu_score = metric1.compute(predictions=pred, references=ref)
    results.append((wer_score, rouge_score, bleu_score))
    n+=1

########################################################################################################################
#visualize scores of each review

df = pd.read_csv(r'F:\ResearchCaseStudies\75_single_scores.csv', sep='\t')
df['sacreBLEU / 100'] = df['sacreBLEU / 100'].div(100).round(2)
df['WER'] = 1 - df['WER']
df = df.rename(columns={'WER': '1 - WER'})

df = df.sort_values(by=['sacreBLEU'], ascending=False)

to_visual = pd.read_csv(r'F:\ResearchCaseStudies\scores_for_fig2.csv', sep='\t')

sns.set(font_scale=1.3)
sns.set_style("whitegrid")
#sns.set_style("whitegrid", {'axes.grid' : False})
f, ax = plt.subplots(figsize = (12,5))
g = sns.barplot(y='score', x='review', hue='measure', data=to_visual)
g.legend(title='measure', loc='best', bbox_to_anchor=(1, 1))


sns.set(font_scale=1.3)
sns.set_style("whitegrid")
#sns.set_style("whitegrid", {'axes.grid' : False})
f, ax = plt.subplots(figsize = (12,5))
g = sns.boxplot(data=df, showfliers=False)
plt.show()


df.to_csv(r'Figure3.csv', sep='\t', index=False)


########################################################################################################################
#calculate scores for each of the reviews
import os
os.chdir(r'inference_results')
         
df = pd.read_csv(r'translated_reviews_T5Large_chunk_50.csv', sep='\t')

def load_metrics():
    metric1 = evaluate.load("sacrebleu")
    metric2 = evaluate.load("wer")
    metric3 = evaluate.load("rouge")
    return metric1, metric2, metric3

metric1, metric2, metric3 = load_metrics()

n = 0
results = []
while n < len(df):
    pred = [df['translated'][n]]
    ref = [df['orig'][n]]
    wer_score = metric2.compute(predictions=pred, references=ref)
    rouge_score = metric3.compute(predictions=pred, references=ref)
    bleu_score = metric1.compute(predictions=pred, references=ref)
    results.append((wer_score, rouge_score['rouge1'], rouge_score['rouge2'], rouge_score['rougeL'], rouge_score['rougeLsum'], bleu_score['score']))
    n+=1
    
output_df = pd.DataFrame(results, columns=['WER', 'rouge1', 'rouge2', 'rougeL', 'rougeLsum', 'sacreBLEU'])    
output_df['sacreBLEU / 100'] = output_df['sacreBLEU'].div(100).round(2)
output_df['WER'] = 1 - output_df['WER']
output_df = output_df.rename(columns={'WER': '1 - WER'})    
output_df = output_df.drop(columns=['sacreBLEU'])    

output_df.to_csv(r'inference_results_chunk_500.csv', sep='\t', index=False)    
    
results_path = r'results'
filenames = sorted([os.path.join(results_path, fn) for fn in os.listdir(results_path)])
    
    
mean_results = []
for file in filenames:
    df = pd.read_csv(file, sep='\t')
    mean = df.mean().tolist()
    mean_results.append((os.path.basename(file), mean))
    
all_results = pd.read_csv(r'all_inference_results.csv', sep='\t')    
    

all_resultss = pd.melt(all_results, id_vars=['corpus_size', 'chunk_length'], value_vars=['1 - WER', 'rouge1', 'rouge2', 'rougeL', 'rougeLsum', 'sacreBLEU / 100']
             , var_name='measure', value_name='score')    
      
sns.set(font_scale=1.8)
sns.set_style("whitegrid")
g = sns.catplot(
    all_resultss, kind="bar",
    x="chunk_length", y="score", row="corpus_size", hue='measure',
    height=4, aspect=2.5,
)   
plt.show()    
    
    
########################################################################################################################

os.chdir(r'inference_results')
score_df = pd.read_csv(r'inference_results_doc_60_500.csv', sep='\t')

to_visual = pd.read_csv(r'Figure8.csv', sep='\t')

to_visual = pd.melt(to_visual, id_vars=['experiment'], value_vars=['1 - WER','rouge1','rouge2','rougeL','rougeLsum','sacreBLEU / 100'],
                    var_name='measure', value_name='score')

to_visual = to_visual.astype({"score": float})

sns.set(font_scale=1.3)
sns.set_style("whitegrid")
f, ax = plt.subplots(figsize = (12,5))
g = sns.boxplot(data=to_visual, x = 'measure', y ='score', hue = 'experiment', showfliers=False )
plt.show()




sns.set(font_scale=1.3)
sns.set_style("whitegrid")
f, ax = plt.subplots(figsize = (12,5))
g = sns.barplot(hue='measure', data=to_visual)
plt.show()





ref = ["When I was ten years younger than I am now, I acquired a wandering occupation, going out into the countryside to collect folk songs. Throughout the summer of that year, I wandered like a fluttering sparrow through the robin- and sun-filled fields of the village houses. I liked to drink the bitter tea of the peasants, whose tea buckets were placed under the trees on the ridges of the fields, and I had no qualms about scooping up the tea bowls covered with tea scum, filling my own kettle, and talking nonsense with the men working in the fields, and then leaving in the midst of the girls' snickers and laughter at my expense. I once spent an entire afternoon talking to an old man who guarded a melon field, the most melons I'd ever eaten, and when I stood up to say goodbye, I suddenly realized I was gaiting like a pregnant woman. Then I sat on the threshold with a woman who had become a grandmother, and she braided straw shoes and sang me a song called “October Baby”. My favorite part was sitting in front of the peasants' houses when evening came, watching them splash the water from the well they had brought up on the ground to keep down the steaming dust, the rays of the setting sun coming down over the treetops, taking a fan they handed me, tasting their pickles that were as salty as salt, and looking at a few of the young women, talking to the men."]
pred = ["Throughout the summer I wandered through the sun-filled fields and wandered into a village like a sparrow fluttering out into the sun. When I was ten years younger than I am now, I liked to drink the folk songs that I acquired through the occupation of wandering through the countryside. I I had no qualms about talking with the men whose fields were covered with the bitter scum of the fields, and the peasants whose bowls of tea were working under the ridges, scooping up the tea and filling the kettle with the tea, and placing the bowls on my own I realized then that I'd eaten an entire field of melons and stood in the midst of an old man who guarded the girls' laughter at my expense, and when I was suddenly up to say goodbye to the melons I'd spent an entire afternoon talking, and snickers. sat on the threshold of a house and sang a song with a splash of straw on it, like a pregnant woman who had become a favorite of the peasants. Then came the evening when I was sitting with my grandmother and watching them gaiting in front of the “Baby”. a few of the young men were coming down from the treetops to the ground, looking over the steaming water and taking a few salty pickles from the fan, that they had brought to me to keep the dust down, as they were talking, the sun setting on the salty."]
wer_score = metric2.compute(predictions=pred, references=ref)
rouge_score = metric3.compute(predictions=pred, references=ref)
bleu_score = metric1.compute(predictions=pred, references=ref)







    
    
    
    
    
    
    
    
    
    
    