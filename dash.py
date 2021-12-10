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
	data['financemining']=data['financemining'].apply(lambda x:'Casual labourer' if x=='0' else x)
	data['year_joined']=data['year_joined'].apply(lambda x:'Never joined Gemfair' if x==0 else str(x))
	return data

data=load_data()

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

@st.cache
def count2(abscisse,ordonnée,dataf,title='',legendtitle='',xaxis=''):
    
    agg=dataf[[abscisse,ordonnée]].groupby(by=[abscisse,ordonnée]).aggregate({abscisse:'count'}).unstack().fillna(0)
    agg2=agg.T/agg.T.sum()
    agg2=agg2.T*100
    agg2=agg2.astype(int)
    x=agg.index
    
    if ordonnée.split(' ')[0] in codes['list name'].values:
        colors_code=codes[codes['list name']==ordonnée.split(' ')[0]].sort_values(['code'])
        labels=colors_code['label'].tolist()
        colors=colors_code['color'].tolist()
        fig = go.Figure()
        #st.write(labels,colors)
        for i in range(len(labels)):
            if labels[i] in data[ordonnée].unique():
                fig.add_trace(go.Bar(x=x, y=agg[(abscisse,labels[i])], name=labels[i],\
                           marker_color=colors[i].lower(),customdata=agg2[(abscisse,labels[i])],textposition="inside",\
                           texttemplate="%{customdata} %",textfont_color="black"))
        
    else:
        fig = go.Figure(go.Bar(x=x, y=agg.iloc[:,0], name=agg.columns.tolist()[0][1],marker_color='green'))
        for i in range(len(agg.columns)-1):
            fig.add_trace(go.Bar(x=x, y=agg.iloc[:,i+1], name=agg.columns.tolist()[i+1][1]))
    
    fig.update_layout(barmode='relative', \
                  xaxis={'title':xaxis,'title_font':{'size':18}},\
                  yaxis={'title':'Persons','title_font':{'size':18}})
    fig.update_layout(legend_title=legendtitle,legend=dict(orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1.01,font=dict(size=18),title=dict(font=dict(size=18))
    ))
    fig.update_layout(title_text=title)
    
    return fig

@st.cache
def pourcent2(abscisse,ordonnée,dataf,title='',legendtitle='',xaxis=''):
    
    agg2=dataf[[abscisse,ordonnée]].groupby(by=[abscisse,ordonnée]).aggregate({abscisse:'count'}).unstack().fillna(0)
    agg=agg2.T/agg2.T.sum()
    agg=agg.T.round(2)*100
    x=agg2.index
    
    if ordonnée.split(' ')[0] in codes['list name'].values:
        colors_code=codes[codes['list name']==ordonnée.split(' ')[0]].sort_values(['code'])
        labels=colors_code['label'].tolist()
        colors=colors_code['color'].tolist()
        fig = go.Figure()
        
        for i in range(len(labels)):
            if labels[i] in data[ordonnée].unique():
                fig.add_trace(go.Bar(x=x, y=agg[(abscisse,labels[i])], name=labels[i],\
                           marker_color=colors[i].lower(),customdata=agg2[(abscisse,labels[i])],textposition="inside",\
                           texttemplate="%{customdata} persons",textfont_color="black"))
        
    else:
        fig = go.Figure(go.Bar(x=x, y=agg.iloc[:,0], name=agg.columns.tolist()[0][1],marker_color='green'))
        for i in range(len(agg.columns)-1):
            fig.add_trace(go.Bar(x=x, y=agg.iloc[:,i+1], name=agg.columns.tolist()[i+1][1]))
    
    fig.update_layout(barmode='relative', \
                  xaxis={'title':xaxis,'title_font':{'size':18}},\
                  yaxis={'title':'Pourcentage','title_font':{'size':18}})
    fig.update_layout(legend_title=legendtitle,legend=dict(orientation='h',
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1.01,font=dict(size=18),title=dict(font=dict(size=18))
    ))
    fig.update_layout(title_text=title)
    
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
text2=[questions[i]['question'] for i in text if 'recomm' not in i]+['Recommandation progamming','Recommandation activities'] 
#st.write(text)

img1 = Image.open("logoAxiom.png")
img2 = Image.open("gemfairLogo.png")

def main():	
	
	
	st.sidebar.image(img1,width=200)
	st.sidebar.title("")
	st.sidebar.title("")
	topic = st.sidebar.radio('What do you want to do ?',('Display Machine Learning Results','Chiefdom results','Display other correlations'))
	
	title2,title3 = st.columns([5,2])
	title3.image(img2)
	
	#st.write(questions)
	#st.write(cat_cols)	
	if topic=='Chiefdom results':
		
		quest=correl[correl['variable_x']=='chiefdom'].copy()
		#st.write(quest)
		
		
		
		
			
	
	if topic in ['Display other correlations','Chiefdom results']:
		
		
		if topic=='Display other correlations':
			quest=correl[-(correl['variable_x']=='chiefdom')].copy()
			title2.title('Correlations uncovered from the database:')
			title2.title('Other questions')
		else:
			quest=correl[correl['variable_x']=='chiefdom'].copy()
			title2.title('Differences from one chiefdom to another:')
		st.write('')
		st.write('')
		st.write('')
		st.write('')
		k=0
		
		
		
		for i in range(len(quest)):
			
			st.markdown("""---""")		
			if quest.iloc[i]['variable_x'] in cat_cols or quest.iloc[i]['variable_y'] in cat_cols:

					
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
						
				if quest.iloc[i]['graphtype']=='treemap':
					
					fig=px.treemap(df, path=[quest.iloc[i]['variable_x'], quest.iloc[i]['variable_y']],\
					 values='persons')
					fig.update_layout(title_text=quest.iloc[i]['title'],font=dict(size=20))
					st.write(quest.iloc[i]['description'])
					st.plotly_chart(fig,use_container_width=True)
					k=0
					
					
				elif quest.iloc[i]['graphtype']=='sunburst':
					fig = px.sunburst(df.fillna(''), path=[quest.iloc[i]['variable_x'], quest.iloc[i]['variable_y']], values='persons',color=quest.iloc[i]['variable_y'])
					fig.update_layout(title_text=quest.iloc[i]['title'],font=dict(size=20))
					st.plotly_chart(fig,use_container_width=True)
					st.write(quest.iloc[i]['description'])
					
					k+=1
			
			elif quest.iloc[i]['variable_y2']==quest.iloc[i]['variable_y2'] and quest.iloc[i]['graphtype']=='violin':
				#st.write(quest.iloc[i]['variable_y2'])
				source=quest.iloc[i]['variable_y'][4:]
				df1=data[[quest.iloc[i]['variable_x'],quest.iloc[i]['variable_y']]]
				df2=data[[quest.iloc[i]['variable_x'],quest.iloc[i]['variable_y2']]]
				df1.columns=[quest.iloc[i]['variable_x'],source]
				df2.columns=[quest.iloc[i]['variable_x'],source]
				df1['Season']=np.full(len(df1),'Rainy season')
				df2['Season']=np.full(len(df2),'Dry season')
				df=df1.append(df2)
									
				if quest.iloc[i]['variable_x']=='chiefdom':
					pointpos_un = [-0.9,-0.8,-1,-0.65,-0.75]
					pointpos_deux = [0.75,0.75,1,0.7,0.75]
					show_legend = [True,False,False,False,False]
				else:
					pointpos_un = [-0.9,-0.4,-0.3,-0.2]
					pointpos_deux = [0.8,0.4,0.3,0.2]
					show_legend = [True,False,False,False]
				
				fig = go.Figure()
				
				for k in range(0,len(pd.unique(df[quest.iloc[i]['variable_x']]))):
					fig.add_trace(go.Violin(x=df[quest.iloc[i]['variable_x']][(df['Season'] == 'Rainy season') &
                                        	(df[quest.iloc[i]['variable_x']] == pd.unique(df[quest.iloc[i]['variable_x']])[k])],
                           	 		y=df[source][(df['Season'] == 'Rainy season')&
                                              (df[quest.iloc[i]['variable_x']] == pd.unique(df[quest.iloc[i]['variable_x']])[k])],
                            			legendgroup='Rainy season', scalegroup='Rainy season', name='Rainy season',
                            			side='negative',
                            			pointpos=pointpos_un[k], # where to position points
                            			line_color='blue',
                            			showlegend=show_legend[k])
             					)
					fig.add_trace(go.Violin(x=df[quest.iloc[i]['variable_x']][(df['Season'] == 'Dry season') &
                                        	(df[quest.iloc[i]['variable_x']] == pd.unique(df[quest.iloc[i]['variable_x']])[k])],
                            			y=df[source][(df['Season'] == 'Dry season')&
                                              (df[quest.iloc[i]['variable_x']] == pd.unique(df[quest.iloc[i]['variable_x']])[k])],
                            			legendgroup='Dry season', scalegroup='Dry season', name='Dry season',
                            			side='positive',
                            			pointpos=pointpos_deux[k],
                            			line_color='gold',
                            			showlegend=show_legend[k])
             					)
				
				
				# update characteristics shared by all traces
				fig.update_traces(meanline_visible=True,
                  		points='all', # show all points
                  		jitter=0.05,  # add some jitter on points for better visibility
                  		scalemode='count') #scale violin plot area with total count
				fig.update_layout(
    				title_text=quest.iloc[i]['title'],font=dict(size=20)
    				)
				fig.update_layout(
   				 yaxis_title=quest.iloc[i]['ytitle'])
				fig.update_layout(
   				 xaxis_title=quest.iloc[i]['xtitle'])
				st.plotly_chart(fig,use_container_width=True)
				
				st.write(quest.iloc[i]['description'])
			
			elif quest.iloc[i]['variable_y2']==quest.iloc[i]['variable_y2'] and quest.iloc[i]['graphtype']=='box':
				#st.write(quest.iloc[i]['variable_y2'])
				source=quest.iloc[i]['variable_y'][4:]
				df1=data[[quest.iloc[i]['variable_x'],quest.iloc[i]['variable_y']]]
				df2=data[[quest.iloc[i]['variable_x'],quest.iloc[i]['variable_y2']]]
				df1.columns=[quest.iloc[i]['variable_x'],source]
				df2.columns=[quest.iloc[i]['variable_x'],source]
				df1['Season']=np.full(len(df1),'Rainy season')
				df2['Season']=np.full(len(df2),'Dry season')
			
				
				fig = go.Figure()
				
				fig.add_trace(go.Box(
    				y=df1[source],
    				x=df1[quest.iloc[i]['variable_x']],
    				name='Rainy season',
    				marker_color='blue'
				))
				fig.add_trace(go.Box(
   				y=df2[source],
    				x=df2[quest.iloc[i]['variable_x']],
    				name='Dry season',
    				marker_color='gold'
				))
				fig.update_layout(
   				 yaxis_title=quest.iloc[i]['ytitle'],
    				boxmode='group' # group together boxes of the different traces for each value of x
				)
				st.plotly_chart(fig,use_container_width=True)
				
				
				st.write(quest.iloc[i]['description'])
						
			else:	
				df=data[[quest.iloc[i]['variable_x'],quest.iloc[i]['variable_y']]]
				df['persons']=np.ones(len(df))
				
				if quest.iloc[i]['graphtype']=='sunburst':
					fig = px.sunburst(df.fillna(''), path=[quest.iloc[i]['variable_x'], quest.iloc[i]['variable_y']], values='persons',color=quest.iloc[i]['variable_y'])
					fig.update_layout(title_text=quest.iloc[i]['variable_x'] + ' and ' +quest.iloc[i]['variable_y'],font=dict(size=20))
					st.plotly_chart(fig,size=1000)
					st.write(quest.iloc[i]['description'])
					
				elif quest.iloc[i]['graphtype']=='treemap':
					fig=px.treemap(df, path=[quest.iloc[i]['variable_x'], quest.iloc[i]['variable_y']], values='persons')
					fig.update_layout(title_text=quest.iloc[i]['title'],font=dict(size=20))
					st.plotly_chart(fig,use_container_width=True)
					st.write(quest.iloc[i]['description'])
					k=0
				
				elif quest.iloc[i]['graphtype']=='box':
					fig = px.box(df, x=quest.iloc[i]['variable_x'], y=quest.iloc[i]['variable_y'],points='all')	
					fig.update_layout(title_text=quest.iloc[i]['title'],font=dict(size=20))
					fig.update_yaxes(range=[0, 1000000],title=dict(text=quest.iloc[i]['ytitle'],font=dict(size=14)))
					fig.update_layout(barmode='relative', \
                  				xaxis={'title':quest.iloc[i]['xtitle'],'title_font':{'size':18}},\
                  				yaxis={'title':quest.iloc[i]['ytitle'],'title_font':{'size':18}})
					st.plotly_chart(fig,use_container_width=True)
					st.write(quest.iloc[i]['description'])
					
				elif quest.iloc[i]['graphtype']=='violin':
					col1,col2=st.columns([1,1])
					fig = go.Figure()

					categs = data[quest.iloc[i]['variable_x']].unique()
					if quest.iloc[i]['filtermin']==quest.iloc[i]['filtermin']:
						#st.write(quest.iloc[i]['filtermin'])
						df=df[df[quest.iloc[i]['variable_y']]>=quest.iloc[i]['filtermin']]
					if quest.iloc[i]['filtermax']==quest.iloc[i]['filtermax']:
						#st.write(quest.iloc[i]['filtermax'])
						df=df[df[quest.iloc[i]['variable_y']]<=quest.iloc[i]['filtermax']]

					for categ in categs:
					    fig.add_trace(go.Violin(x=df[quest.iloc[i]['variable_x']][df[quest.iloc[i]['variable_x']] == categ],
                            			y=df[quest.iloc[i]['variable_y']][df[quest.iloc[i]['variable_x']] == categ],
                            			name=categ,
                            			box_visible=True,
                            			meanline_visible=True,points="all",))
					fig.update_layout(title_text=quest.iloc[i]['title'],font=dict(size=20),showlegend=False)
					fig.update_yaxes(range=[df[quest.iloc[i]['variable_y']].min(), df[quest.iloc[i]['variable_y']].max()+1],title=dict(text=quest.iloc[i]['ytitle'],font=dict(size=14)))
					st.plotly_chart(fig,use_container_width=True)
					st.write(quest.iloc[i]['description'])
									
				elif quest.iloc[i]['graphtype']=='bar':
					if quest.iloc[i]['variable_x']=='year_joined':
						df=df[df['year_joined']!='Never joined Gemfair']
						
						
					col1,col2=st.columns([1,1])

					fig1=count2(quest.iloc[i]['variable_x'],quest.iloc[i]['variable_y'],\
					df,xaxis=quest.iloc[i]['xtitle'])
					fig1.update_layout(title_text=quest.iloc[i]['title'],font=dict(size=20),showlegend=False)
					col1.plotly_chart(fig1,use_container_width=True)
					
					fig2=pourcent2(quest.iloc[i]['variable_x'],quest.iloc[i]['variable_y'],\
					df,xaxis='')
					fig2.update_layout(title_text='',legend_title=quest.iloc[i]['legendtitle'],showlegend=True)
					col2.plotly_chart(fig2,use_container_width=True)
					st.write(quest.iloc[i]['description'])	
	
					
	elif topic=='Display Machine Learning Results':
		
		title2.title('Machine learning results on models trained on:')
		title2.title('Questions E11,E12 and E13')
		
		
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
				
		temp = Image.open('shap1.png')
		image = Image.new("RGBA", temp.size, "WHITE") # Create a white rgba background
		image.paste(temp, (0, 0), temp)
		st.subheader('Overall, do you consider your household better off than it was 3 or 4 years ago?')
		st.image(image, use_column_width = True)
		
		
		
		temp = Image.open('shap2.png')
		image1 = Image.new("RGBA", temp.size, "WHITE") # Create a white rgba background
		image1.paste(temp, (0, 0), temp)
		st.subheader('Overall, do you consider your household richer or poorer than the people in this community?')
		st.image(image1, use_column_width = True)		
		
		
		temp = Image.open('shap3.png')
		image2 = Image.new("RGBA", temp.size, "WHITE") # Create a white rgba background
		image2.paste(temp, (0, 0), temp)
		st.subheader('Overall, do you think the GemFair program made your household less or more wealthy?')
		st.image(image2, use_column_width = True)
		


		
		
		
		st.title('Conclusion/Recommandations')
		
		
		
		
		
		
		
	else:
		st.title('\t DRC South Sudan \t VTC')	


    
 
if __name__== '__main__':
    main()




    
