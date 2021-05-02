import pandas as pd
HAFEZ = 0
SAADI = 1
df = pd.read_csv("train_test.csv")
edf = pd.read_csv("evaluate.csv")
df = df.head(int(len(df)*(8/10)))
count_of_poet = df.groupby('label').count()
probablity_of_hafez = count_of_poet.iloc[HAFEZ] /(count_of_poet.iloc[HAFEZ] + count_of_poet.iloc[SAADI])
probablity_of_saadi = count_of_poet.iloc[SAADI] /(count_of_poet.iloc[HAFEZ] + count_of_poet.iloc[SAADI])
hafez_poems = df.loc[df['label'] == 'hafez']
hafez_poems = hafez_poems.join(pd.DataFrame(df.text.str.split(' ').tolist(),columns=['c1','c2','c3','c4','c5','c6','c7','c8','c9','c10','c11','c12','c13']))
words_hafez = pd.DataFrame(columns=['word','poet'])
words_hafez = words_hafez.append(pd.DataFrame({'word':hafez_poems['c1'].tolist(),'poet':hafez_poems['label']}))
words_hafez = words_hafez.append(pd.DataFrame({'word':hafez_poems['c2'].tolist(),'poet':hafez_poems['label']}))
words_hafez = words_hafez.append(pd.DataFrame({'word':hafez_poems['c3'].tolist(),'poet':hafez_poems['label']}))
words_hafez = words_hafez.append(pd.DataFrame({'word':hafez_poems['c4'].tolist(),'poet':hafez_poems['label']}))
words_hafez = words_hafez.append(pd.DataFrame({'word':hafez_poems['c5'].tolist(),'poet':hafez_poems['label']}))
words_hafez = words_hafez.append(pd.DataFrame({'word':hafez_poems['c6'].tolist(),'poet':hafez_poems['label']}))
words_hafez = words_hafez.append(pd.DataFrame({'word':hafez_poems['c7'].tolist(),'poet':hafez_poems['label']}))
words_hafez = words_hafez.append(pd.DataFrame({'word':hafez_poems['c8'].tolist(),'poet':hafez_poems['label']}))
words_hafez = words_hafez.append(pd.DataFrame({'word':hafez_poems['c9'].tolist(),'poet':hafez_poems['label']}))
words_hafez = words_hafez.append(pd.DataFrame({'word':hafez_poems['c10'].tolist(),'poet':hafez_poems['label']}))
words_hafez = words_hafez.append(pd.DataFrame({'word':hafez_poems['c11'].tolist(),'poet':hafez_poems['label']}))
words_hafez = words_hafez.append(pd.DataFrame({'word':hafez_poems['c12'].tolist(),'poet':hafez_poems['label']}))
words_hafez = words_hafez.append(pd.DataFrame({'word':hafez_poems['c13'].tolist(),'poet':hafez_poems['label']}))
words_hafez = words_hafez.dropna()
saadi_poems = df.loc[df['label'] == 'saadi']
saadi_poems = saadi_poems.join(pd.DataFrame(df.text.str.split(' ').tolist(),columns=['c1','c2','c3','c4','c5','c6','c7','c8','c9','c10','c11','c12','c13']))
words_saadi = pd.DataFrame(columns=['word','poet'])
words_saadi = words_saadi.append(pd.DataFrame({'word':saadi_poems['c1'].tolist(),'poet':saadi_poems['label']}))
words_saadi = words_saadi.append(pd.DataFrame({'word':saadi_poems['c2'].tolist(),'poet':saadi_poems['label']}))
words_saadi = words_saadi.append(pd.DataFrame({'word':saadi_poems['c3'].tolist(),'poet':saadi_poems['label']}))
words_saadi = words_saadi.append(pd.DataFrame({'word':saadi_poems['c4'].tolist(),'poet':saadi_poems['label']}))
words_saadi = words_saadi.append(pd.DataFrame({'word':saadi_poems['c5'].tolist(),'poet':saadi_poems['label']}))
words_saadi = words_saadi.append(pd.DataFrame({'word':saadi_poems['c6'].tolist(),'poet':saadi_poems['label']}))
words_saadi = words_saadi.append(pd.DataFrame({'word':saadi_poems['c7'].tolist(),'poet':saadi_poems['label']}))
words_saadi = words_saadi.append(pd.DataFrame({'word':saadi_poems['c8'].tolist(),'poet':saadi_poems['label']}))
words_saadi = words_saadi.append(pd.DataFrame({'word':saadi_poems['c9'].tolist(),'poet':saadi_poems['label']}))
words_saadi = words_saadi.append(pd.DataFrame({'word':saadi_poems['c10'].tolist(),'poet':saadi_poems['label']}))
words_saadi = words_saadi.append(pd.DataFrame({'word':saadi_poems['c11'].tolist(),'poet':saadi_poems['label']}))
words_saadi = words_saadi.append(pd.DataFrame({'word':saadi_poems['c12'].tolist(),'poet':saadi_poems['label']}))
words_saadi = words_saadi.append(pd.DataFrame({'word':saadi_poems['c13'].tolist(),'poet':saadi_poems['label']}))
words_saadi = words_saadi.dropna()
words_hafez = words_hafez.reset_index(drop=True)
words_saadi = words_saadi.reset_index(drop=True)
saadi_poems_count = saadi_poems['label'].count()
hafez_poems_count = hafez_poems['label'].count()
edf['text'] = edf['text'].astype(str)
edf['text'] = edf['text'].str.split(' ')
edf = edf.reset_index(drop=True)
hafez_dict = {}
for i in range(0,words_hafez['word'].count()):
	word = words_hafez.at[i,'word']
	if word in hafez_dict:
		temp=hafez_dict.get(word)
		hafez_dict.update({word:temp+1})
	else:
		hafez_dict.update({word:1})
saadi_dict = {}
for i in range(0,words_saadi['word'].count()):
	word = words_saadi.at[i,'word']
	if word in saadi_dict:
		temp=saadi_dict.get(word)
		saadi_dict.update({word:temp+1})
	else:
		saadi_dict.update({word:1})
predict = []
for i in range(0,edf['text'].count()):
	x = 1
	y = 1
	for j in range(0,len(edf.at[i,'text'])):
		word = edf.at[i,'text'][j]
		saadi_word_count = saadi_dict.get(word)
		hafez_word_count = hafez_dict.get(word)
		if saadi_word_count == None:
			saadi_word_count = 0
		if hafez_word_count == None:
			hafez_word_count = 0
		x = x * (saadi_word_count / saadi_poems_count)
		y = y * (hafez_word_count / hafez_poems_count)	
	if x == 0 and y == 0:
		x = 1
		y = 1
		for j in range(0,len(edf.at[i,'text'])):
			if word in  saadi_dict and word in hafez_dict:
				if word in saadi_dict:
					x = x * (saadi_dict.get(word) / saadi_poems_count)	
				if word in hafez_dict:
					y = y * (hafez_dict.get(word) / hafez_poems_count)
			else:
				if word in saadi_dict:
					x = x * (saadi_dict.get(word) / saadi_poems_count) * 1000000	
				if word in hafez_dict:
					y = y * (hafez_dict.get(word) / hafez_poems_count) * 1000000
	x = x * probablity_of_saadi.iloc[0]
	y = y * probablity_of_hafez.iloc[0]				
	if y > x:
		predict.append("hafez")
	else:
		predict.append("saadi")			
file = open("output.csv","w")
file.write("id,label\n")
for i in range(0,len(predict)):
	file.write(str(i+1)+","+str(predict[i])+"\n")