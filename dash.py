import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from wordcloud import WordCloud,STOPWORDS
import pickle
import pydeck as pdk
import re
from collections import Counter
from PIL import Image

#import variables

#########################  a faire #########################################
# 
#
###########################################################################"


#Variables Correl Description
#becomes 
#variable_x variable_y description


st.set_page_config(layout="wide")


#import des données
@st.cache
def load_data():
	data = pd.read_csv('viz.csv',sep='\t')
	data['flee_reason']=data['flee_reason'].apply(lambda x:'Returnee or Host' if x=='0' else x)
	data['know_leader']=data['know_leader'].apply(lambda x:'Not IDP' if x=='0' else x)
	data['member_EES']=data['member_EES'].apply(lambda x:'Neither IDP nor Returnee Refugee' if x=='0' else x)
	
	return data

datas=load_data()

#st.dataframe(correl)
#st.write(data.columns)
#st.write(correl.shape)

def sankey_graph(data,L,height=600,width=1600):
    """ sankey graph de data pour les catégories dans L dans l'ordre et 
    de hauter et longueur définie éventuellement"""
    
    nodes_colors=["blue","green","grey",'yellow',"coral"]
    link_colors=["lightblue","lightgreen","lightgrey","lightyellow","lightcoral"]
    
    
    labels=[]
    source=[]
    target=[]
    
    for cat in L:
        lab=data[cat].unique().tolist()
        lab.sort()
        labels+=lab
    
    for i in range(len(data[L[0]].unique())): #j'itère sur mes premieres sources
    
        source+=[i for k in range(len(data[L[1]].unique()))] #j'envois sur ma catégorie 2
        index=len(data[L[0]].unique())
        target+=[k for k in range(index,len(data[L[1]].unique())+index)]
        
        for n in range(1,len(L)-1):
        
            source+=[index+k for k in range(len(data[L[n]].unique())) for j in range(len(data[L[n+1]].unique()))]
            index+=len(data[L[n]].unique())
            target+=[index+k for j in range(len(data[L[n]].unique())) for k in range(len(data[L[n+1]].unique()))]
       
    iteration=int(len(source)/len(data[L[0]].unique()))
    value_prov=[(int(i//iteration),source[i],target[i]) for i in range(len(source))]
    
    
    value=[]
    k=0
    position=[]
    for i in L:
        k+=len(data[i].unique())
        position.append(k)
    
   
    
    for triplet in value_prov:    
        k=0
        while triplet[1]>=position[k]:
            k+=1
        
        df=data[data[L[0]]==labels[triplet[0]]].copy()
        df=df[df[L[k]]==labels[triplet[1]]]
        #Je sélectionne ma première catégorie
        value.append(len(df[df[L[k+1]]==labels[triplet[2]]]))
        
    color_nodes=nodes_colors[:len(data[L[0]].unique())]+["black" for i in range(len(labels)-len(data[L[0]].unique()))]
    #print(color_nodes)
    color_links=[]
    for i in range(len(data[L[0]].unique())):
    	color_links+=[link_colors[i] for couleur in range(iteration)]
    #print(L,len(L),iteration)
    #print(color_links)
   
   
    fig = go.Figure(data=[go.Sankey(
    node = dict(
      pad = 15,
      thickness = 30,
      line = dict(color = "black", width = 1),
      label = [i.upper() for i in labels],
      color=color_nodes
      )
      
    ,
    link = dict(
      source = source, # indices correspond to labels, eg A1, A2, A1, B1, ...
      target = target,
      value = value,
      color = color_links))])
    return fig

def count2(abscisse,ordonnée,dataf,title='',legendtitle='',xaxis=''):
    
    agg=dataf[[abscisse,ordonnée]].groupby(by=[abscisse,ordonnée]).aggregate({abscisse:'count'}).unstack().fillna(0)
    agg2=agg.T/agg.T.sum()
    agg2=agg2.T*100
    agg2=agg2.astype(int)
    x=agg.index
    
    if ordonnée.split(' ')[0] in codes['list name'].values:
        colors_code=codes[codes['list name']==ordonnée.split(' ')[0]].sort_values(['coding'])
        labels=colors_code['label'].tolist()
        colors=colors_code['color'].tolist()
        fig = go.Figure()
        #st.write(labels,colors)
        for i in range(len(labels)):
            if labels[i] in dataf[ordonnée].unique():
                fig.add_trace(go.Bar(x=x, y=agg[(abscisse,labels[i])], name=labels[i],\
                           marker_color=colors[i].lower(),customdata=agg2[(abscisse,labels[i])],textposition="inside",\
                           texttemplate="%{customdata} %",textfont_color="black"))
        
    else:
        fig = go.Figure(go.Bar(x=x, y=agg.iloc[:,0], name=agg.columns.tolist()[0][1],marker_color='green',customdata=agg2.iloc[:,0],textposition="inside",\
                           texttemplate="%{customdata} %",textfont_color="black"))
        for i in range(len(agg.columns)-1):
            fig.add_trace(go.Bar(x=x, y=agg.iloc[:,i+1], name=agg.columns.tolist()[i+1][1],customdata=agg2.iloc[:,i+1],textposition="inside",\
                           texttemplate="%{customdata} %",textfont_color="black"))
    
    fig.update_layout(barmode='relative', \
                  xaxis={'title':xaxis,'title_font':{'size':18}},\
                  yaxis={'title':'Persons','title_font':{'size':18}})
    fig.update_layout(legend_title=legendtitle,legend=dict(orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1.01,font=dict(size=18),title=dict(font=dict(size=18))
    ))
    #fig.update_layout(title_text=title)
    
    return fig

def pourcent2(abscisse,ordonnée,dataf,title='',legendtitle='',xaxis=''):
    
    agg2=dataf[[abscisse,ordonnée]].groupby(by=[abscisse,ordonnée]).aggregate({abscisse:'count'}).unstack().fillna(0)
    agg=agg2.T/agg2.T.sum()
    agg=agg.T.round(2)*100
    x=agg2.index
    
    if ordonnée.split(' ')[0] in codes['list name'].values:
        colors_code=codes[codes['list name']==ordonnée.split(' ')[0]].sort_values(['coding'])
        labels=colors_code['label'].tolist()
        colors=colors_code['color'].tolist()
        fig = go.Figure()
        
        for i in range(len(labels)):
            if labels[i] in dataf[ordonnée].unique():
                fig.add_trace(go.Bar(x=x, y=agg[(abscisse,labels[i])], name=labels[i],\
                           marker_color=colors[i].lower(),customdata=agg2[(abscisse,labels[i])],textposition="inside",\
                           texttemplate="%{customdata} persons",textfont_color="black"))
        
    else:
        #st.write(agg)
        #st.write(agg2)
        fig = go.Figure(go.Bar(x=x, y=agg.iloc[:,0], name=agg.columns.tolist()[0][1],marker_color='green',customdata=agg2.iloc[:,0],textposition="inside",\
                           texttemplate="%{customdata} persons",textfont_color="black"))
        for i in range(len(agg.columns)-1):
            fig.add_trace(go.Bar(x=x, y=agg.iloc[:,i+1], name=agg.columns.tolist()[i+1][1],customdata=agg2.iloc[:,i+1],textposition="inside",\
                           texttemplate="%{customdata} persons",textfont_color="black"))
    
    fig.update_layout(barmode='relative', \
                  xaxis={'title':xaxis,'title_font':{'size':18}},\
                  yaxis={'title':'Pourcentage','title_font':{'size':18}})
    fig.update_layout(legend_title=legendtitle,legend=dict(orientation='h',
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1.01,font=dict(size=18),title=dict(font=dict(size=18))
    ))
    #fig.update_layout(title_text=title)
    
    return fig




questions=pd.read_csv('questions.csv',sep='\t')
questions=questions[[i for i in questions.columns if 'Unnamed' not in i]]
codes=pd.read_csv('codes.csv',index_col=None,sep='\t').dropna(how='any',subset=['color'])
continues=pickle.load( open( "cont_feat.p", "rb" ) )
cat_cols=pickle.load( open( "cat_cols.p", "rb" ) )
dummy_cols=pickle.load( open( "dummy.p", "rb" ) )	
questions.set_index('Idquest',inplace=True)
correl=pd.read_csv('graphs.csv',sep='\t')
#st.write(questions)
text=[i for i in questions.columns if questions[i]['Treatment']=='text']
text2=[questions[i]['question'] for i in text]
#st.write(codes)

img1 = Image.open("logoAxiom.png")
img2 = Image.open("logoDRC.png")

def main():	
	
	
	st.sidebar.image(img1,width=200)
	st.sidebar.title("")
	st.sidebar.title("")
	topic = st.sidebar.radio('Select what you want to see?',('General questions','Protection questions','Machine Learning results on questions C31, C32, E1 and E2','Livelihood questions','Wordclouds'))
	
	title2,title3 = st.columns([5,2])
	title3.image(img2)
	
	#st.write(questions)
	#st.write(cat_cols)
	if topic in ['General questions','Protection questions','Livelihood questions']:
		
		title2.title('Correlations uncovered from the database:')
		topics={'General questions':'general','Protection questions':'protection','Livelihood questions':'LH'}
		quest=correl[correl['category']==topics[topic]].copy()
		if topics[topic]=='protection':
			data=datas[datas['section']=='Protection&CCM+(Respondent profile and Overall perception)'].copy()
			title2.title('Protection questions')
		elif topics[topic]=='LH':
			data=datas[datas['section']=='FSL+(Respondent profile and Overall perception)'].copy()
			title2.title('Livelihood questions')
		else:
			data=datas.copy()
			title2.title('General questions')				
		
		
		#st.write(correl)
		#st.write(quest)
		
		
		
		for i in range(len(quest)):
			
			st.markdown("""---""")		
			
			if quest.iloc[i]['graphtype']=='map':
				
				st.subheader('Geographical repartition of responses to question:')
				st.subheader(quest.iloc[i]['description'])
				
				dfmap=data[['position longitude','position latitude',quest.iloc[i]['variable_y']]]
				dfmap['radius']=np.ones(len(dfmap))
				dfmap['position']=dfmap.apply(lambda row: [row['position longitude'],row['position latitude']],axis=1)
				
				caracters=dfmap[quest.iloc[i]['variable_y']].unique().tolist()
				n=len(caracters)
				#st.write(caracters)
				
				layers=[pdk.Layer('ScatterplotLayer',dfmap[dfmap[quest.iloc[i]['variable_y']]==caracters[k]],\
				pickable=True,
    				opacity=0.8,
    				stroked=True,
    				filled=True,
    				auto_highlight=True,
    				radius_scale=6,
    				radius_min_pixels=5,
    				radius_max_pixels=6,\
				get_position='position',\
				get_fill_color=[int(k*255/(n-1)),int(255-k*255/(n-1)), 0,180],get_radius="radius",) for k in range(n)]
				
				st.pydeck_chart(pdk.Deck(map_style='mapbox://styles/mapbox/light-v9',\
				initial_view_state=pdk.ViewState(latitude=9.566,longitude=31.678,zoom=15,height=600),\
				layers=layers,))
				
				st.caption('Yes:Green - No:Red')
				st.write(quest.iloc[i]['description'])
						
			elif quest.iloc[i]['variable_x'] in cat_cols or quest.iloc[i]['variable_y'] in cat_cols:
				
				if quest.iloc[i]['variable_x'] in cat_cols:
					cat,autre=quest.iloc[i]['variable_x'],quest.iloc[i]['variable_y']
				else:
					cat,autre=quest.iloc[i]['variable_y'],quest.iloc[i]['variable_x']
				#st.write('cat: ',cat,' et autre: ',autre)
						
				df=pd.DataFrame(columns=[cat,autre])
				
				catcols=[j for j in data.columns if cat in j]
				cats=[' '.join(i.split(' ')[1:])[:57] for i in catcols]
				
				for n in range(len(catcols)):
					ds=data[[catcols[n],autre]].copy()
					ds=ds[ds[catcols[n]]==1]
					ds[catcols[n]]=ds[catcols[n]].apply(lambda x: cats[n])
					ds.columns=[cat,autre]
					df=df.append(ds)
				df['persons']=np.ones(len(df))		
				#st.write(df)		
				
				#st.write(quest.iloc[i]['graphtype'])
						
									
			else:	
				df=data[[quest.iloc[i]['variable_x'],quest.iloc[i]['variable_y']]].copy()
				df['persons']=np.ones(len(df))
				
			if quest.iloc[i]['graphtype']=='sunburst':
				st.subheader(quest.iloc[i]['title'])
				fig = px.sunburst(df.fillna(''), path=[quest.iloc[i]['variable_x'], quest.iloc[i]['variable_y']], values='persons',color=quest.iloc[i]['variable_y'])
				#fig.update_layout(title_text=quest.iloc[i]['variable_x'] + ' and ' +quest.iloc[i]['variable_y'],font=dict(size=20))
				st.plotly_chart(fig,size=1000)
				
			elif quest.iloc[i]['graphtype']=='treemap':
					
				st.subheader(quest.iloc[i]['title'])
				fig=px.treemap(df, path=[quest.iloc[i]['variable_x'], quest.iloc[i]['variable_y']], values='persons')
				#fig.update_layout(title_text=quest.iloc[i]['title'],font=dict(size=20))
				
				st.plotly_chart(fig,use_container_width=True)
				st.write(quest.iloc[i]['description'])
				k=0
				
					
			elif quest.iloc[i]['graphtype']=='violin':
					
				st.subheader(quest.iloc[i]['title'])
				col1,col2=st.columns([1,1])
				fig = go.Figure()
				
				if quest.iloc[i]['variable_x'].split(' ')[0] in codes['list name'].unique():
					categs = codes[codes['list name']==quest.iloc[i]['variable_x'].split(' ')[0]].sort_values(by='coding')['label'].tolist()				
					
				else:
					categs = df[quest.iloc[i]['variable_x']].unique()
				for categ in categs:
				    fig.add_trace(go.Violin(x=df[quest.iloc[i]['variable_x']][df[quest.iloc[i]['variable_x']] == categ],
                            		y=df[quest.iloc[i]['variable_y']][df[quest.iloc[i]['variable_x']] == categ],
                            		name=categ,
                            		box_visible=True,
                           			meanline_visible=True,points="all",))
				fig.update_layout(showlegend=False)
				fig.update_yaxes(range=[-0.1, df[quest.iloc[i]['variable_y']].max()+1],title=quest.iloc[i]['ytitle'])
					
				st.plotly_chart(fig,use_container_width=True)
				st.write(quest.iloc[i]['description'])
									
			elif quest.iloc[i]['graphtype']=='bar':
					
				st.subheader(quest.iloc[i]['title'])
				
				col1,col2=st.columns([1,1])

				fig1=count2(quest.iloc[i]['variable_x'],quest.iloc[i]['variable_y'],\
				df,xaxis=quest.iloc[i]['xtitle'])
				#fig1.update_layout(title_text=quest.iloc[i]['title'],font=dict(size=20),showlegend=True,xaxis_tickangle=45)
				col1.plotly_chart(fig1,use_container_width=True)
					
				fig2=pourcent2(quest.iloc[i]['variable_x'],quest.iloc[i]['variable_y'],\
				df,xaxis=quest.iloc[i]['xtitle'])
				#fig2.update_layout(title_text=quest.iloc[i]['title'],font=dict(size=20),showlegend=True,xaxis_tickangle=45)
				col2.plotly_chart(fig2,use_container_width=True)
				st.write(quest.iloc[i]['description'])
				
					
			
	
	
##############################################WORDCLOUDS##########################################################"						
						
	elif topic=='Wordclouds':
		
		textgeneral=[questions[i]['question'] for i in ['target_right_explain', 'recomm1']]
		textLH=[questions[i]['question'] for i in ['butchery_related_explain']]
		
		child=False
		
		x, y = np.ogrid[100:500, :600]
		mask = ((x - 300)/2) ** 2 + ((y - 300)/3) ** 2 > 100 ** 2
		mask = 255 * mask.astype(int)
		
		topicwc = st.sidebar.radio('Which question would you like to see?',('General questions','Protection questions','Livelihood question'))
		
		title2.title('Wordclouds for open questions')
		
		if topicwc=='Protection questions':
			data=datas.fillna('')[datas['section']=='Protection&CCM+(Respondent profile and Overall perception)'].copy()
			title2.title('Protection questions')
			feature=st.sidebar.selectbox('Select the question for which you would like to visualize wordclouds of answers',[i for i in text2 if i not in textLH+textgeneral])
		elif topicwc=='Livelihood question':
			data=datas.fillna('')[datas['section']=='FSL+(Respondent profile and Overall perception)'].copy()
			title2.title('Livelihood question')
			feature=st.sidebar.selectbox('Select the question for which you would like to visualize wordclouds of answers',textLH)
		else:
			data=datas.fillna('').copy()
			title2.title('General questions')
			feature=st.sidebar.selectbox('Select the question for which you would like to visualize wordclouds of answers',textgeneral)
		
		#st.write(questions)
		
		if topicwc != 'General questions' or 'recomm' not in feature:
			
			var=[i for i in questions if questions[i]['question']==feature][0]
			
				
			if var == 'discussed_CCCM':
				df=data[data['attended_CCCM']=='Yes'].copy()
			else:
				df=data.copy()
			
			col1, col2, col3 = st.columns([1,4,1])
			col2.title('Wordcloud from question:')
			col2.title(feature)
				
		
			corpus=' '.join(df[var].fillna('').apply(lambda x:'' if x=='0' else x))
			corpus=re.sub('[^A-Za-z ]',' ', corpus)
			corpus=re.sub('\s+',' ', corpus)
			corpus=corpus.lower()
		
			col3.title('')
			col3.title('')
			col3.title('')
			sw=col3.multiselect('Select words you would like to remove from the wordcloud \n\n', [i[0] for i in Counter(corpus.split(' ')).most_common() if i[0] not in STOPWORDS][:20])
		
			if corpus==' ':
	    			corpus='No_response'
			else:
				corpus=' '.join([i for i in corpus.split(' ') if i not in sw])
		
			wc = WordCloud(background_color="#0E1117", repeat=False, mask=mask)		
			wc.generate(corpus)
			col2.image(wc.to_array(), use_column_width = True)	
			
			
			if var=='target_right_explain':
				a,b=st.columns([1,1])
				corpus=' '.join(df[df['section']=='FSL+(Respondent profile and Overall perception)'][var].fillna('').apply(\
				lambda x:'' if x=='0' else x))
				corpus=re.sub('[^A-Za-z ]',' ', corpus)
				corpus=re.sub('\s+',' ', corpus)
				corpus=corpus.lower()
				if corpus==' ' or corpus=='':
					corpus='No_response'
				else:
					corpus=' '.join([i for i in corpus.split(' ') if i not in sw])
				wclh = WordCloud(background_color="#0E1117", repeat=False, mask=mask)		
				wclh.generate(corpus)
				a.subheader('Livelihood beneficiaries')
				a.image(wclh.to_array(), use_column_width = True)
				
				corpus=' '.join(df[df['section']=='Protection&CCM+(Respondent profile and Overall perception)'][var].fillna('').apply(\
				lambda x:'' if x=='0' else x))
				corpus=re.sub('[^A-Za-z ]',' ', corpus)
				corpus=re.sub('\s+',' ', corpus)
				corpus=corpus.lower()
				if corpus==' ' or corpus=='':
					corpus='No_response'
				else:
					corpus=' '.join([i for i in corpus.split(' ') if i not in sw])
				wcprot = WordCloud(background_color="#0E1117", repeat=False, mask=mask)		
				wcprot.generate(corpus)
				b.subheader('Protection beneficiaries')
				b.image(wcprot.to_array(), use_column_width = True)
							
				var2=questions[var]['parent']			
				st.markdown("""---""")	
				st.subheader('Wordclouds according to response to question : '+questions[var2]['question'])
				st.markdown("""---""")	
				
				subcol1,subcol2=st.columns([2,2])
				L=df[var2].unique()
				
				corpus1=corpus2=''
				Corpuses=[corpus1,corpus2]
				
				for i in range(2):		
			
					Corpuses[i]=' '.join(df[df[var2]==L[i]][var].apply(lambda x:'' if x=='0' else x))
					Corpuses[i]=re.sub('[^A-Za-z ]',' ', Corpuses[i])
					Corpuses[i]=re.sub('\s+',' ', Corpuses[i])
					Corpuses[i]=Corpuses[i].lower()
					if Corpuses[i]==' ':
    						Corpuses[i]='No_response'
					else:
						Corpuses[i]=' '.join([i for i in Corpuses[i].split(' ') if i not in sw])
					wc2 = WordCloud(background_color="#0E1117", repeat=False, mask=mask)
					wc2.generate(Corpuses[i])
					if i==0:
						subcol1.write(str(L[i])+' : '+str(len(df[df[var2]==L[i]]))+' '+'repondents')
						subcol1.image(wc2.to_array(), use_column_width = True)
					elif i==1:
						subcol2.write(str(L[i])+' : '+str(len(df[df[var2]==L[i]]))+' '+'repondents')
						subcol2.image(wc2.to_array(), use_column_width = True)
				
				subcol1.subheader('Livelihood beneficiaries')
				subcol2.subheader('Protection beneficiaries')
				
				for sect in ['FSL+(Respondent profile and Overall perception)','Protection&CCM+(Respondent profile and Overall perception)']:
					corpus1=corpus2=''
					Corpuses=[corpus1,corpus2]
					dfbis=df[df['section']==sect].copy()
					for i in range(2):		
			
						Corpuses[i]=' '.join(dfbis[dfbis[var2]==L[i]][var].apply(lambda x:'' if x=='0' else x))
						Corpuses[i]=re.sub('[^A-Za-z ]',' ', Corpuses[i])
						Corpuses[i]=re.sub('\s+',' ', Corpuses[i])
						Corpuses[i]=Corpuses[i].lower()
						if Corpuses[i]==' ':
    							Corpuses[i]='No_response'
						else:
							Corpuses[i]=' '.join([i for i in Corpuses[i].split(' ') if i not in sw])
						
					if sect=='FSL+(Respondent profile and Overall perception)':
						wc2 = WordCloud(background_color="#0E1117", repeat=False, mask=mask)
						wc2.generate(Corpuses[0])
						subcol1.write(str(L[0])+' : '+str(len(dfbis[dfbis[var2]==L[0]]))+' '+'repondents')
						subcol1.image(wc2.to_array(), use_column_width = True)
						wc2 = WordCloud(background_color="#0E1117", repeat=False, mask=mask)
						wc2.generate(Corpuses[1])
						subcol1.write(str(L[1])+' : '+str(len(dfbis[dfbis[var2]==L[1]]))+' '+'repondents')
						subcol1.image(wc2.to_array(), use_column_width = True)
					else:
						wc2 = WordCloud(background_color="#0E1117", repeat=False, mask=mask)
						wc2.generate(Corpuses[0])
						subcol2.write(str(L[0])+' : '+str(len(dfbis[dfbis[var2]==L[0]]))+' '+'repondents')
						subcol2.image(wc2.to_array(), use_column_width = True)
						wc2 = WordCloud(background_color="#0E1117", repeat=False, mask=mask)
						wc2.generate(Corpuses[1])
						subcol2.write(str(L[1])+' : '+str(len(dfbis[dfbis[var2]==L[1]]))+' '+'repondents')
						subcol2.image(wc2.to_array(), use_column_width = True)
				
				if st.checkbox('Would you like to filter Wordcloud according to other questions'):		
					
					
					feature2=st.selectbox('Select one question to filter the wordcloud',\
					[questions[i]['question'] for i in questions.columns if i not in text and i!='UniqueID'])		
					filter2=[i for i in questions if questions[i]['question']==feature2][0]
			
					if filter2 in continues:
						mini=int(data[filter2].fillna(0).min())
						maxi=int(data[filter2].fillna(0).max())
						minimum=st.slider('Select the minimum value you want to visulize', min_value=mini,max_value=maxi)
						maximum=st.slider('Select the maximum value you want to visulize', min_value=minimum,max_value=maxi+1)
						df=df[(df[filter2]>=minimum)&(df[filter2]<=maximum)]	
				
			
					else:
						filter3=st.multiselect('Select the responses you want to include', [i for i in data[filter2].unique()])
						df=df[df[filter2].isin(filter3)]
					
					corpus=' '.join(df[var].apply(lambda x:'' if x=='0' else x))
					corpus=re.sub('[^A-Za-z ]',' ', corpus)
					corpus=re.sub('\s+',' ', corpus)
					corpus=corpus.lower()
			
					if corpus==' ' or corpus=='':
    						corpus='No_response'
					else:
						corpus=' '.join([i for i in corpus.split(' ') if i not in sw])
		
					wc = WordCloud(background_color="#0E1117", repeat=False, mask=mask)
					wc.generate(corpus)
					
					col1, col2, col3 = st.columns([1,4,1])
					col2.image(wc.to_array(), use_column_width = True)
					
					a,b=st.columns([1,1])
					corpus=' '.join(df[df['section']=='FSL+(Respondent profile and Overall perception)'][var].fillna('').apply(\
					lambda x:'' if x=='0' else x))
					corpus=re.sub('[^A-Za-z ]',' ', corpus)
					corpus=re.sub('\s+',' ', corpus)
					corpus=corpus.lower()
					if corpus==' ' or corpus=='':
    						corpus='No_response'
					else:
						corpus=' '.join([i for i in corpus.split(' ') if i not in sw])
					wclh = WordCloud(background_color="#0E1117", repeat=False, mask=mask)		
					wclh.generate(corpus)
					a.subheader('Livelihood beneficiaries')
					a.image(wclh.to_array(), use_column_width = True)
				
					corpus=' '.join(df[df['section']=='Protection&CCM+(Respondent profile and Overall perception)'][var].fillna('').apply(\
					lambda x:'' if x=='0' else x))
					corpus=re.sub('[^A-Za-z ]',' ', corpus)
					corpus=re.sub('\s+',' ', corpus)
					corpus=corpus.lower()
					if corpus==' ' or corpus=='':
    						corpus='No_response'
					else:
						corpus=' '.join([i for i in corpus.split(' ') if i not in sw])
					wcprot = WordCloud(background_color="#0E1117", repeat=False, mask=mask)		
					wcprot.generate(corpus)
					b.subheader('Protection beneficiaries')
					b.image(wcprot.to_array(), use_column_width = True)
								
					var2=questions[var]['parent']			
					st.markdown("""---""")	
					st.subheader('Wordclouds according to response to question : '+questions[var2]['question'])
					st.markdown("""---""")	
				
					subcol1,subcol2=st.columns([2,2])
					L=df[var2].unique()
				
					corpus1=corpus2=''
					Corpuses=[corpus1,corpus2]
				
					for i in range(2):		
				
						Corpuses[i]=' '.join(df[df[var2]==L[i]][var].apply(lambda x:'' if x=='0' else x))
						Corpuses[i]=re.sub('[^A-Za-z ]',' ', Corpuses[i])
						Corpuses[i]=re.sub('\s+',' ', Corpuses[i])
						Corpuses[i]=Corpuses[i].lower()
						if Corpuses[i]==' ':
    								Corpuses[i]='No_response'
						else:
							Corpuses[i]=' '.join([i for i in Corpuses[i].split(' ') if i not in sw])
							if Corpuses[i]=='':
	    							Corpuses[i]='No_response'
						wc2 = WordCloud(background_color="#0E1117", repeat=False, mask=mask)
						wc2.generate(Corpuses[i])
						if i==0:
							subcol1.write(str(L[i])+' : '+str(len(df[df[var2]==L[i]]))+' '+'repondents')
							subcol1.image(wc2.to_array(), use_column_width = True)
						elif i==1:
							subcol2.write(str(L[i])+' : '+str(len(df[df[var2]==L[i]]))+' '+'repondents')
							subcol2.image(wc2.to_array(), use_column_width = True)
				
					subcol1.subheader('Livelihood beneficiaries')
					subcol2.subheader('Protection beneficiaries')
				
					for sect in ['FSL+(Respondent profile and Overall perception)','Protection&CCM+(Respondent profile and Overall perception)']:
						corpus1=corpus2=''
						Corpuses=[corpus1,corpus2]
						dfbis=df[df['section']==sect].copy()
						for i in range(2):		
			
							Corpuses[i]=' '.join(dfbis[dfbis[var2]==L[i]][var].apply(lambda x:'' if x=='0' else x))
							Corpuses[i]=re.sub('[^A-Za-z ]',' ', Corpuses[i])
							Corpuses[i]=re.sub('\s+',' ', Corpuses[i])
							Corpuses[i]=Corpuses[i].lower()
							if Corpuses[i]==' ':
    								Corpuses[i]='No_response'
							else:
								Corpuses[i]=' '.join([i for i in Corpuses[i].split(' ') if i not in sw])
								if Corpuses[i]=='':
	    								Corpuses[i]='No_response'
						if sect=='FSL+(Respondent profile and Overall perception)':
							wc2 = WordCloud(background_color="#0E1117", repeat=False, mask=mask)
							wc2.generate(Corpuses[0])
							subcol1.write(str(L[0])+' : '+str(len(dfbis[dfbis[var2]==L[0]]))+' '+'repondents')
							subcol1.image(wc2.to_array(), use_column_width = True)
							wc2 = WordCloud(background_color="#0E1117", repeat=False, mask=mask)
							wc2.generate(Corpuses[1])
							subcol1.write(str(L[1])+' : '+str(len(dfbis[dfbis[var2]==L[1]]))+' '+'repondents')
							subcol1.image(wc2.to_array(), use_column_width = True)
						else:
							wc2 = WordCloud(background_color="#0E1117", repeat=False, mask=mask)
							wc2.generate(Corpuses[0])
							subcol2.write(str(L[0])+' : '+str(len(dfbis[dfbis[var2]==L[0]]))+' '+'repondents')
							subcol2.image(wc2.to_array(), use_column_width = True)
							wc2 = WordCloud(background_color="#0E1117", repeat=False, mask=mask)
							wc2.generate(Corpuses[1])
							subcol2.write(str(L[1])+' : '+str(len(dfbis[dfbis[var2]==L[1]]))+' '+'repondents')
							subcol2.image(wc2.to_array(), use_column_width = True)
					
					
					
			
			elif col2.checkbox('Would you like to filter Wordcloud according to other questions'):
		
				feature2=col2.selectbox('Select one question to filter the wordcloud',[questions[i]['question'] for i in questions.columns if i not in text and i!='UniqueID'])		
				filter2=[i for i in questions if questions[i]['question']==feature2][0]
			
				if filter2 in continues:
					mini=int(data[filter2].fillna(0).min())
					maxi=int(data[filter2].fillna(0).max())
					minimum=col2.slider('Select the minimum value you want to visulize', min_value=mini,max_value=maxi)
					maximum=col2.slider('Select the maximum value you want to visulize', min_value=minimum,max_value=maxi+1)
					df=df[(df[filter2]>=minimum)&(df[filter2]<=maximum)]	
				
			
				else:
					filter3=col2.multiselect('Select the responses you want to include', [i for i in data[filter2].unique()])
					df=df[df[filter2].isin(filter3)]
			
				corpus=' '.join(df[var].apply(lambda x:'' if x=='0' else x))
				corpus=re.sub('[^A-Za-z ]',' ', corpus)
				corpus=re.sub('\s+',' ', corpus)
				corpus=corpus.lower()
			
				if corpus==' ' or corpus=='':
    					corpus='No_response'
				else:
					corpus=' '.join([i for i in corpus.split(' ') if i not in sw])
		
				wc = WordCloud(background_color="#0E1117", repeat=False, mask=mask)
				wc.generate(corpus)
				col2.image(wc.to_array(), use_column_width = True)
				
				
			
			
			
		
			if questions[var]['parent'] in questions.columns and var!='target_right_explain':
			
				child=True		
				var2=questions[var]['parent']
				
							
				st.markdown("""---""")	
				st.subheader('Wordclouds according to response to question : '+questions[var2]['question'])
				st.markdown("""---""")	
				if var2 not in ['feel_leader_transp','satistaftion CBPN','trust CBPN','satistaftion security_CBPN','rlptcomm']:
					subcol1,subcol2=st.columns([1,1])
				else:
					subcol1,subcol2,subcol3=st.columns([1,1,1])
			
				L=df[var2].unique()
				
				corpus1=corpus2=corpus3=''
				Corpuses=[corpus1,corpus2,corpus3]
				
				for i in range(len(L)):		
			
					Corpuses[i]=' '.join(df[df[var2]==L[i]][var].apply(lambda x:'' if x=='0' else x))
					Corpuses[i]=re.sub('[^A-Za-z ]',' ', Corpuses[i])
					Corpuses[i]=re.sub('\s+',' ', Corpuses[i])
					Corpuses[i]=Corpuses[i].lower()
					if Corpuses[i]==' ':
    						Corpuses[i]='No_response'
					else:
						Corpuses[i]=' '.join([i for i in Corpuses[i].split(' ') if i not in sw])
					wc2 = WordCloud(background_color="#0E1117", repeat=False, mask=mask)
					wc2.generate(Corpuses[i])
					if i==0:
						subcol1.write(str(L[i])+' : '+str(len(df[df[var2]==L[i]]))+' '+'repondents')
						subcol1.image(wc2.to_array(), use_column_width = True)
					elif i==1:
						subcol2.write(str(L[i])+' : '+str(len(df[df[var2]==L[i]]))+' '+'repondents')
						subcol2.image(wc2.to_array(), use_column_width = True)
					else:
						subcol3.write(str(L[i])+' : '+str(len(df[df[var2]==L[i]]))+' '+'repondents')
						subcol3.image(wc2.to_array(), use_column_width = True)
		
			
			
		
	
		##########################################Traitement spécifique Recommandations#######################################################
		else:
			df=data.copy()
			col1, col2, col3 = st.columns([1,1,1])
			
			colonnes=['recomm1','recomm2','recomm3']
			st.title('Wordcloud from question:')
			st.title('E4) What recommendations would you propose to improve the project activities?')
			
			st.title('')
			st.title('')
			st.title('')
			
			corpus=' '.join(data[colonnes[0]].dropna())+\
				' '.join(data[colonnes[1]].dropna())+' '.join(data[colonnes[2]].dropna())
			corpus=re.sub('[^A-Za-z ]',' ', corpus)
			corpus=re.sub('\s+',' ', corpus)
			corpus=corpus.lower()
			sw=st.multiselect('Select words you would like to remove from the wordclouds \n\n', [i[0] for i in Counter(corpus.split(' ')).most_common() if i[0] not in STOPWORDS][:20])
			
			
			for benef in ['All beneficiaries','Livelihood beneficiaries','Protection beneficiaries']:
				if benef=='All beneficiaries':
					df=datas.copy()
				elif benef=='Livelihood beneficiaries':
					df=datas[datas['section']=='FSL+(Respondent profile and Overall perception)']
				else:
					df=datas[datas['section']=='Protection&CCM+(Respondent profile and Overall perception)']
				
				st.subheader(benef)
					
				col1, col2, col3 = st.columns([1,1,1])
					
				for i in range(3):
					col_corpus=' '.join(df[colonnes[i]].dropna())
					col_corpus=re.sub('[^A-Za-z ]',' ', col_corpus)
					col_corpus=re.sub('\s+',' ', col_corpus)
					col_corpus=col_corpus.lower()
					if col_corpus==' ' or col_corpus=='':
			    			col_corpus='No_response'
					else:
						col_corpus=' '.join([i for i in col_corpus.split(' ') if i not in sw])		
					wc = WordCloud(background_color="#0E1117", repeat=False, mask=mask)		
					wc.generate(col_corpus)
					if i==0:
						col1.subheader('Recommandation 1')
						col1.image(wc.to_array(), use_column_width = True)
					elif i==1:
						col2.subheader('Recommandation 2')	
						col2.image(wc.to_array(), use_column_width = True)
					else:
						col3.subheader('Recommandation 3')
						col3.image(wc.to_array(), use_column_width = True)		
					
			if st.checkbox('Would you like to filter Wordcloud according to other questions'):
				
				feature2=st.selectbox('Select one question to filter the wordcloud',[questions[i]['question'] for i in questions.columns if i not in text and i!='UniqueID'])		
				filter2=[i for i in questions if questions[i]['question']==feature2][0]
			
				if filter2 in continues:
					minimum=st.slider('Select the minimum value you want to visulize', 	min_value=datas[filter2].fillna(0).min(),max_value=data[filter2].fillna(0).max())
					maximum=st.slider('Select the maximum value you want to visulize', min_value=minimum,max_value=datas[filter2].fillna(0).max())
					df=datas[(datas[filter2]>=minimum)&(datas[filter2]<=maximum)]	

				else:
					filter3=st.multiselect('Select the responses you want to include', [i for i in datas[filter2].unique()])
					df=datas[datas[filter2].isin(filter3)]
				
								
				
			
				for benef in ['All beneficiaries','Livelihood beneficiaries','Protection beneficiaries']:
					if benef=='All beneficiaries':
						df2=df.copy()
					elif benef=='Livelihood beneficiaries':
						df2=df[df['section']=='FSL+(Respondent profile and Overall perception)']
					else:
						df2=df[df['section']=='Protection&CCM+(Respondent profile and Overall perception)']
					
					st.subheader(benef+' '+str(len(df2))+ ' persons')
					
					col1, col2, col3 = st.columns([1,1,1])
					for i in range(3):
						col_corpus=' '.join(df2[colonnes[i]].dropna())
						col_corpus=re.sub('[^A-Za-z ]',' ', col_corpus)
						col_corpus=re.sub('\s+',' ', col_corpus)
						col_corpus=col_corpus.lower()
						if col_corpus==' ' or col_corpus=='':
				    			col_corpus='No_response'
						else:
							col_corpus=' '.join([i for i in col_corpus.split(' ') if i not in sw])		
						wc = WordCloud(background_color="#0E1117", repeat=False, mask=mask)		
						wc.generate(col_corpus)
						if i==0:
							col1.subheader('Recommandation 1')
							col1.image(wc.to_array(), use_column_width = True)
						elif i==1:
							col2.subheader('Recommandation 2')	
							col2.image(wc.to_array(), use_column_width = True)
						else:
							col3.subheader('Recommandation 3')
							col3.image(wc.to_array(), use_column_width = True)	

############################################mmmmmmmmmmmmmmmmmlllllllllllllllll#########################################################	
			
	elif topic=='Machine Learning results on questions C31, C32, E1 and E2':
		
		title2.title('Machine learning results on models trained on:')
		title2.title('Questions C31, C32, E1 and E2')
		
		
		st.title('')
		st.markdown("""---""")	
		st.subheader('Note:')
		st.write('A machine learning model has been run on the question related to feeling of improvement of the situation thanks to the project, the objective of this was to identify specificaly for these 4 questions. The models are run in order to try to predict as precisely as possible the feeling that the respondents expressed in their responses to these questions. The figures below show for each questions which parameter have a greater impact in the prediction of the model than a normal random aspect (following a statistic normal law')
		st.write('')
		st.write('Each line of the graph represent one feature of the survey that is important to predict the response to the question.')
		st.write('Each point on the right of the feature name represent one person of the survey. A red point represent a high value to the specific feature and a blue point a low value.')
		st.write('SHAP value: When a point is on the right side, it means that it contributed to a better note while on the left side, this specific caracter of the person reduced the final result of the prediction.')
		st.write('')
		st.write('The coding for the responses is indicated under the graph and the interpretation of the graphs is written below.')
		st.markdown("""---""")	
				
		temp = Image.open('changeincome.png')
		image = Image.new("RGBA", temp.size, "WHITE") # Create a white rgba background
		image.paste(temp, (0, 0), temp)
		st.subheader('C31) Since the DRC project my income has…')
		st.image(image, use_column_width = True)
		st.caption('Did not receive any skills training: Did not receive:1 - Did receive:0')
		st.caption('Did not receive Multipurpose Cash Assistance: Did not receive:1 - Did receive:0')
		st.caption('')
		st.write('We can see that the main parameter for feeling that the level of income has increased is the Longitude of the person interviewed. This means that given that in Bentiu (the lowest longitude) people seems to be more confident that their income has increased since the DRC project.')
		st.write('The comes the fact to have received skills training and/or multipurpose cash assistance. Those who received these are also more likely to believe their income has increased')
		st.write('Finaly (but less obvious) people with the lowest incomes seems more likely to believe their income has increased')
		
		st.markdown("""---""")	
		
		temp = Image.open('changefoodsec.png')
		image1 = Image.new("RGBA", temp.size, "WHITE") # Create a white rgba background
		image1.paste(temp, (0, 0), temp)
		st.subheader('C32) Since the DRC project the HHs food security has…')
		st.image(image1, use_column_width = True)		
		st.caption('Did not receive any skills training: Did not receive:1 - Did receive:0')
		st.caption('Member of Seed-Pressing Cooperative: Member:1 - Not a member:0')
		st.caption('')
		st.write('We find on top the fact to have being benefiting from Multipurpose Cash assistance')
		st.write('Then the number of women and girls seems to be an important parameter probably since the project was mainly targeting women')
		st.write('Finaly another important aspect is to be a member of a SPC.')
		
		st.markdown("""---""")	
		
		temp = Image.open('change2LH.png')
		image2 = Image.new("RGBA", temp.size, "WHITE") # Create a white rgba background
		image2.paste(temp, (0, 0), temp)
		st.subheader('E1) Because of the project, I am confident that my livelihood will')
		st.image(image2, use_column_width = True)
		st.caption('Member of Seed-Pressing Cooperative: Member:1 - Not a member:0')
		st.caption('Did not receive any agricultural training: Did not receive:1 - Did receive:0')
		st.caption('Do you know your camp/block leader: Yes:1 - No:0 - Not IDP:0')
		st.caption('Marital status: Married:1 - Single:0 - Widowed:0 - Divorced: 0')
		st.caption('')
		st.write('This time this is the fact to be a member of Seed-Pressing Cooperative that has the main impact.')
		st.write('Then we find again the number of women in the household')
		st.write('Another important factor to feel confident in the fact that the livelihood will increase is the fact to have received agricultural training')
		st.write('Knowing the camp/block leader seems more to be a burden for being optimistic but this is more to be linked with the fact that the model was treating equaly not IDPs and IDPS who were not knowing their leader, so this might better be interpretated as the fact that people who are not IDPs are more optimistics about the improvement of their livelihoods than IDPs. This seems corroborated with the fact that globally in Bentiu people are more optimisting than in Jamjang. In Bentiu most of people intrviewed are host while in Jamjang this is the opposite.')
		st.write('Finaly married people are globaly more optimistic')
		st.write('The level of cash assistance seems to have also a little impact but this is clearly less obvious than the short term improvement we saw just before')
		
		st.markdown("""---""")	
		
		temp = Image.open('change2foodsec.png')
		image3 = Image.new("RGBA", temp.size, "WHITE") # Create a white rgba background
		image3.paste(temp, (0, 0), temp)
		st.subheader('41) Because of the project, my household’s access to food will')
		st.image(image3, use_column_width = True)
		st.caption('Do you know your camp/block leader: Yes:1 - No:0 - Not IDP:0')
		st.caption('Member of Seed-Pressing Cooperative: Member:1 - Not a member:0')
		st.caption('Did not receive any agricultural training: Did not receive:1 - Did receive:0')
		st.caption('Did not receive any skills training: Did not receive:1 - Did receive:0')
		st.caption('')
		st.write('This time we find on top the fact to not know the camp leader and the fact to have been a member of a SPC. Most probably this has to be interpretated as the fact not to be an IDP. This further show the importance of SPCs')
		st.write('Then comes the Longitude with once again people from Bentiu (the lowest longitude) who seem to be more confident that their food acess will increase thanks to the DRC project than people in Jamjam.')
		st.write('The influence of income is not obvious but the composition of the family is. And what comes out is that large families and particularly families with many children under 5 feel less confident')
		st.write('Skills training also seem to help people having conidence in their future.')
		st.write('')
		
		st.title('Conclusion/Recommandations')
		
		st.write('- Cash seems to be the best way to improve the condition of the households on the short term')
		st.write('- For the long term, agricultural and skills trianing have been very much appreciated.')
		st.write('- SPC seem also to play a very appreciated role in the communities.')
		st.write('- On the long term, host communities seem to feel more confident.')
		st.write('- Large household are probably more vulnerable, specialy those with an important number of children.')
		
		
		
		
		
		
		
		
	else:
		st.title('\t DRC South Sudan \t VTC')	


    
 
if __name__== '__main__':
    main()




    
