# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 12:30:00 2024

STREAMLIT DASHBOARD for CultEvoSelf project

To run locally type this in Anaconda Prompt:
    streamlit run Dashboard_MeasuringSenseOfControl.py

@author: Matusz Wozniak
"""

#%% import packages
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # only import this one function from matplotlib
from matplotlib.patches import Patch
import seaborn as sns            # 
import scipy as sp               #  

#import statsmodels as sm         # 
#import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import VCSpec
from statsmodels.formula.api import mixedlm
from statsmodels.stats.anova import AnovaRM
from pingouin import rm_anova
import pingouin  as pg

# CLUSTER ANALYSIS
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial import Voronoi, voronoi_plot_2d


#%% SETUP

# Load data
df_long = pd.read_csv('DATA_MeasuringSoA_ALL.csv')


# Create dictionaries with trait and valence labels
experiments_list = [ 'Experiment 1', 'Experiment 1B', 'Experiment 1C', 
                    'Experiment 2', 'Experiment 3', 'Experiment 4', 'Experiment 5' ]



# # Inverted trait_labels dictionary
# inv_map = {v: k for k, v in trait_labels.items()}
# # List of traits from the dictionary
# traits_list  = list(trait_labels.values())
# traits_codes = list(trait_labels.keys())

#%% ################ PLOTS ####################################


# Make single-color color palette for the lines
pal = sns.color_palette(['#CCCCCC'], 60)

# Set global style
sns.set_style('white')

# Figure global settings
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
          #'figure.figsize': (15, 5),
         'axes.labelsize': 26,
         'axes.titlesize': 26,
         'xtick.labelsize': 20,
         'ytick.labelsize': 20}
pylab.rcParams.update(params)





#%% DASHBOARD

### SIDEBAR

st.sidebar.markdown('# **How to measure sense of control with explicit reports**') #' **([preprint](https://osf.io/preprints/psyarxiv/ta9rq))**')

button_radio = st.sidebar.radio("Choose what you want to see:", 
                                ["Introduction", 
                                 "Task description",
                                 "Comparison of experiments", 
                                 "Results of specific experiments",
                                 "Preregistered analyses",
                                 "Variance decomposition analysis",
                                 "Reliability of data from Block 1",
                                 "Cluster analysis",
                                 "Metrics of individual differences",
                                 "Correlations within levels of control", ###
                                 "Explore data from each participant"])

# Select box: experiment
selected_selectbox = "Experiment 1"
selected_selectbox = st.sidebar.selectbox('Which experiment do you want to analyze?', experiments_list)

# Select box: experiment
selectbox_exclude_outliers = "Yes"
selectbox_exclude_outliers = st.sidebar.selectbox('Exclude participants according to the preregistered exclusion criteria?', ["Yes", "No"])



#%% CACLUATE METRICS FOR EACH PARTICIPANT

################################
# Prepare data
################################

selected_exp_num = selected_selectbox[11:]
url_link = f'DATA_MeasuringSoA_long_Exp{selected_exp_num}.csv'
df = pd.read_csv(url_link)
# Adjust the control_level to be on the 0-100 scale 
df['control_level'] = df['control_level']*100
# Make a full_data_copy
data_all = df.copy()
# Exclude participants that do not fulfill the inclusion critia
if selectbox_exclude_outliers == "Yes":
    df = df[df['Include'] == 1]

################################
# Calculate metrics for each subject
################################


# Subset data
data_1 = df.loc[df['block']==1,:]
data_all = df
data_scores = []
resp_b1 = data_1.loc[:, ['SubNum','control_level','response']] 
resp_other = data_all.loc[:, ['SubNum','control_level','response']] 
resp_other_mean = resp_other.groupby(['SubNum', 'control_level']).mean().reset_index()
data_scores = pd.merge(resp_b1, resp_other_mean, how='inner', on=['SubNum','control_level'], suffixes=('_B1', '_All'))
# Correlations within each participant
list_subNums = data_scores['SubNum'].unique()
results_indiv_score = []
df_long_metrics = []
for subject in list_subNums:
    # Scores for averages of all blocks: sum_abs_dev 
    score_sum_dev = np.sum(data_scores.loc[data_scores['SubNum']==subject,'response_All'] - data_scores.loc[data_scores['SubNum']==subject,'control_level']) / np.sum(data_scores.loc[data_scores['SubNum']==subject,'control_level'])
    score_sum_abs_dev = np.sum(np.abs(data_scores.loc[data_scores['SubNum']==subject,'control_level'] - data_scores.loc[data_scores['SubNum']==subject,'response_All'])) / np.sum(data_scores.loc[data_scores['SubNum']==subject,'control_level'])
    score_dev_from_chance = np.sum(np.abs(50 - data_scores.loc[data_scores['SubNum']==subject,'response_All'])) / 550 # automatize it!!!!!!!
    score_chance_ratio = score_sum_abs_dev/(score_dev_from_chance+score_sum_abs_dev) * 2 - 1
    # Scores based on just block 1
    score_b1_sum_dev = np.sum(data_scores.loc[data_scores['SubNum']==subject,'response_B1'] - data_scores.loc[data_scores['SubNum']==subject,'control_level']) / np.sum(data_scores.loc[data_scores['SubNum']==subject,'control_level'])
    score_b1_sum_abs_dev = np.sum(np.abs(data_scores.loc[data_scores['SubNum']==subject,'control_level'] - data_scores.loc[data_scores['SubNum']==subject,'response_B1'])) / np.sum(data_scores.loc[data_scores['SubNum']==subject,'control_level'])
    score_b1_dev_from_chance = np.sum(np.abs(50 - data_scores.loc[data_scores['SubNum']==subject,'response_B1'])) /550
    score_b1_chance_ratio = score_b1_sum_abs_dev/(score_b1_dev_from_chance+score_sum_abs_dev) * 2 - 1
    # Combine into a dataset
    results_indiv_score = results_indiv_score + [[subject, score_sum_dev, score_sum_abs_dev, score_dev_from_chance, score_chance_ratio,
                                                  score_b1_sum_dev, score_b1_sum_abs_dev, score_b1_dev_from_chance, score_b1_chance_ratio ]]
    # Scores for each block
    score_list_sum_dev = []
    score_list_sum_abs_dev = []
    score_list_dev_from_chance = []
    score_list_chance_ratio = []
    for block_num in range(5):
        data_scores_block = df.loc[df['block']==block_num+1,['SubNum','control_level','response']]           
        block_sum_dev = np.sum(data_scores_block.loc[data_scores_block['SubNum']==subject,'response'] - data_scores_block.loc[data_scores_block['SubNum']==subject,'control_level']) / np.sum(data_scores_block.loc[data_scores_block['SubNum']==subject,'control_level'])
        block_sum_abs_dev = np.sum(np.abs(data_scores_block.loc[data_scores_block['SubNum']==subject,'control_level'] - data_scores_block.loc[data_scores_block['SubNum']==subject,'response'])) / np.sum(data_scores_block.loc[data_scores_block['SubNum']==subject,'control_level'])
        block_dev_from_chance = np.sum(np.abs(50 - data_scores_block.loc[data_scores_block['SubNum']==subject,'response'])) / 550 # automatize it!!!!!!!
        block_chance_ratio = block_sum_abs_dev/(block_dev_from_chance+block_sum_abs_dev) * 2 - 1
        
        df_long_metrics = df_long_metrics + [[ subject, block_num+1, block_sum_dev, block_sum_abs_dev, block_dev_from_chance, block_chance_ratio]]
    
# Combine it altogether into a dataframe
results_indiv_score = pd.DataFrame(results_indiv_score)
results_indiv_score = results_indiv_score.rename(columns={0:"SubNum", 1:"SumDev", 2:"SumAbsDev", 3:"DevFromChance", 4:"ChanceRatio", 
                                                          5: "B1_SumDev", 6:"B1_SumAbsDev", 7:"B1_DevFromChance", 8:"B1_ChanceRatio"})
# Sort participants
results_indiv_score_sorted = results_indiv_score.sort_values('SumDev')
#results_indiv_score_sorted['SubNum'] = results_indiv_score_sorted['SubNum'].astype(str)
results_indiv_score_sorted = results_indiv_score_sorted.reset_index().reset_index().rename(columns={"level_0":"sorting_order"})
# Check correlation between the variables:
results_ind_corrs = results_indiv_score.corr()
# Build the long dataset and add sorting
df_long_metrics = pd.DataFrame(df_long_metrics, columns=["SubNum", "block", "SumDev", "SumAbsDev", "DevFromChance", "ChanceRatio"])
df_long_metrics["Bias"] = df_long_metrics["SumDev"]
df_long_metrics["Accuracy"] = 1 - df_long_metrics["SumAbsDev"]
df_long_metrics = df_long_metrics.merge(results_indiv_score_sorted.loc[:,["SubNum","sorting_order"]], how="outer", left_on="SubNum", right_on="SubNum")
# Calculate averages based on df_long_metrics
df_avg_metrics = df_long_metrics.groupby("SubNum").mean().reset_index().drop("block", axis=1)



#%% NO BUTTON PRESSED

# if (button_1 == False & button_2 == False & button_3 == False & button_4 == False):
#     st.title('Dashboard for the result of the CultEvoSelf study')
#     st.markdown('This is a dashboard allowing you to explore the results of the CultEvoSelf study .')


#%% BUTTON == 'Introduction' or none pressed

if button_radio == 'Introduction':
    st.title('Introduction')
    st.markdown('This is a dashboard allowing you to explore the results of our [study on meauring sense of control with explicit reports](https://osf.io/preprints/psyarxiv/ta9rq). In this study we compared several different scales measuring sense of control (sense of agency).')
    st.image(image="img/tense cats.jpg", width=250)
    st.markdown("***Figure 1.*** *This cat has lost sense of control* ")
    
    st.markdown('### What is sense of control and why it matters')
    st.markdown(''' 
                According to the standard definition Sense of Control or Sense of Agency (we will use these terms interchangeably here) 
                refers to the **feeling of control** over actions and their consequences (Moore, 2016). 
                As such, it refers to the conscious experience (feeling) that I am in control. 
                
                Sense of Control is critically important in many practical applications, 
                especially including robotics, automation and artificial intelligence. 
                
                ''')
    # st.markdown(''' 
    #             According to the standard definition Sense of Agency (SoA) refers to the **feeling of control** over actions and their consequences (Moore, 2016). 
    #             As such, it refers to the conscious experience (feeling) that I am in control. 
                
    #             Some authors proposed to distinguish between Feeling of Agency and Judgment of Agency (Synofzik et al., 2008) with the latter referring to
    #             intellectual judgments that I am in control (an objective statement of the state of affairs), while the former refers to the subjective experience 
    #             that "I am in control". In specific situation one might observe one of them in the absence of another:
                
    #             (A) Electrical stimulation of certain parts of the parietal cortex can elicit very strong conscious intention to perform a movement, and even
    #             an experience that one performed a movement (so there is FoA), while such patients are intellectually aware that they have not moved their arm (Desmurget, Sirigu, 2009)
                
    #             (B) Imagine this scenario: I passed through an empty room and afterwards I heard a vase falling. In such situation I do not feel that I caused this event (no FoA), but
    #             because there is no other alternative explanation I judge that I made the vase fall and crash (there is JoA) (Synofzik et al., 2008). Similar effects over one's body movements can
    #             be elicited by stimulating the premotor cortex: an objective movement of one's arm, without the conscious experience that one is in control over this movement (Desmurget, Sirigu, 2009)
                
    #             While these two aspects are strongly related, and in most cases FoA and JoA are equivalent, it is iportant to make this distinction, 
    #             because usually researchers are primarily interested in FoA.
                
    #             ''')
    st.markdown('### How to measure sense of control')
    st.markdown('Sense of Control can be measured either using explicit methods or implicit methods:')
    st.markdown('- **Explicit measures** consist of scales and questions. Participants evaluate their experienced level of control by indicating a point or option on a scale, or by describing their experience verbally')
    st.markdown("""- **Implicit measures** do not involve asking people directly about their experienced level of control, but instead use different measurements that are believed to be directly related to the experience of sense of agency. The most popular ones are: the magnitude of intentional binding and the 
                magnitude of sensory attenuation.""")
    st.markdown("""While explicit measures allow to directly assess one's level of experienced control, they have been criticized on several grounds. The most important criticisms are: **(A)** Explicit measures are unreliable and **(B)** Explicit measures are easily influenced by external factor and types of scales.
                On the other hand implicit measures are much more troublesome to use and have been recently criticized as having even more reliability problems (Corneille, Gawronski, 2024).
                """)
    st.markdown('### The goal of this study')
    st.markdown('''The aim of this research was to assess the reliability of explicit measures of SoA. Specifically, we investigated the 
                influence of the choice of the response scale on participants' reports of SoA.
                ''')
    st.markdown('### References')
    st.markdown(''' 
                Corneille, O., Gawronski, B. (2024). **Self-reports are better measurement instruments than implicit measures**. *Nature Reviews Psychology*. [https://doi.org/10.1038/s44159-024-00376-z](https://doi.org/10.1038/s44159-024-00376-z). 
                
                Haggard, P. (1985). **Sense of agency in the human brain**. *Nature Reviews Neuroscience* 18 (4), 196-207. [https://doi.org/10.1038/nrn.2017.14](https://doi.org/10.1038/nrn.2017.14)  
                
                Moore, J. W. (2016). **What is the sense of agency and why does it matter?**. *Frontiers in psychology, 7*, 1272. [https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2016.01272/full](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2016.01272/full)
                ''')

#%% BUTTON == 'Task description'

if button_radio == 'Task description':
    st.title("Task description")    
    st.markdown(''' 
                The participants' task was to move a red ball around the screen for 8 seconds in each trial. Afterwards they had to rate their 
                experienced level of control. The experiments differed in regard to the scale on which they had to evaluate their experienced control.
                Participants completed 5 blocks of the experimental task. Each block consisted of 11 trials and each trial in a block represented 
                a different level of control: from 0%, through 10%, 20%, up to 100%. The order of trials was random. 
                ''')
    st.markdown("**[LINK TO THE EXPERIMENT](https://measuring-agency.netlify.app/)**")
    st.image(image='img/Figure_1_task.png')
    st.markdown("***Figure 1.*** *Graphical illustration of the procedure of Experiment 1.*")
    
    st.markdown(''' 
                We compared 5 types of scales across Experiments 1 to 5. In all of these experiments the option on the right reflected experiencing 
                being in control, and the option on the left reflected experiencing no control. Moreover, in Experiments 1B and 1C (not visualized) 
                we also tested whether presenting the option "I was fully in control" on the left side and "I had no control at all" on the right affected the results. 
                ''')
    st.image(image='img/Figure_2_task.png')
    st.markdown("***Figure 2.*** *Presentation of response scales used in all experiments.*")
    
    st.markdown(''' 
                ### Assumptions behind the experiment 
                The basic assumption behind the experiment is that subjective experience of control should be related to the objective level of control possessed
                by the participants. In an idealized scenario they should directly map onto each other, as indicated by the red line in Figure 3. However, it is likely 
                that participants might show various types of response biases, for example overestimation or underestimation of their experienced level control 
                (two gray curves). Our main research quetsion was whether we will observe any such biases at the group level and how robust will tey be across different 
                types of scales and how reliable they will be within participants.
                ''')
    st.image(image='img/Figure_3_task.png', width=450)
    st.markdown("***Figure 3.*** *The hypothesized relationships between the objective level of control (x-axis) and the subjectively reported level of control (y-axis).*")
   
#%% BUTTON == 'Comparison of experiments'

if button_radio == 'Comparison of experiments':
    
    # Text
    st.title("Comparison of results between the experiments")
    
    ##########################
    # PREPARE DATA
    ##########################

    # Load data
    df_long = pd.read_csv('DATA_MeasuringSoA_ALL.csv')
    
    # Column names:
    resp = []
    diff = []
    rt = []
    resp_std = []
    for contr_lev in range(11):
        resp.append(f'response_mean_{contr_lev/10:.1f}')
        diff.append(f'difference_mean_{contr_lev/10:.1f}')
        rt.append(f'rt_{contr_lev/10:.1f}')
        resp_std.append(f'response_std_{contr_lev/10:.1f}')
    
    # Differences
    df_long_diff = pd.melt(df_long, id_vars=['SubNum', 'Experiment'], value_vars=diff, var_name='Level of Control', value_name='Difference')
    df_long_diff['Control'] = df_long_diff['Level of Control'].str[-3:]
    df_long_diff['Control'] = pd.to_numeric(df_long_diff['Control'])*100
    
    # Responses
    df_long_resp = pd.melt(df_long, id_vars=['SubNum', 'Experiment'], value_vars=resp, var_name='Level of Control', value_name='Rating of SoA')
    df_long_resp['Control'] = df_long_resp['Level of Control'].str[-3:]
    df_long_resp['Control'] = pd.to_numeric(df_long_resp['Control'])*100

    # RTs
    df_long_rt = pd.melt(df_long, id_vars=['SubNum', 'Experiment'], value_vars=rt, var_name='Level of Control', value_name='RT')
    df_long_rt['Control'] = df_long_rt['Level of Control'].str[-3:]
    df_long_rt['Control'] = pd.to_numeric(df_long_rt['Control'])

    # Response STD
    df_long_resp_std = pd.melt(df_long, id_vars=['SubNum', 'Experiment'], value_vars=resp_std, var_name='Level of Control', value_name='Response STD')
    df_long_resp_std['Control'] = df_long_resp_std['Level of Control'].str[-3:]
    df_long_resp_std['Control'] = pd.to_numeric(df_long_resp_std['Control'])*100

    
    
    
    #############################
    
    # Text
    st.markdown("""The main goal of the study was to investigate whether participants display biases when 
                judging their subjective level of control, and whether these biases will be affected by the 
                type of response scale used to collect these responses. We compared 5 different types of scales 
                (see "Task description" for details): linear scale (Experiment 1), percentage scale (Exp 2), 
                7-point Likert scale (Exp 3), a binary scale (Exp 4), and a 4-point Likert scale (Exp 5). As 
                illustrated on Figure 1 and 2 all of these scales, except for the binary scale, led to the 
                same pattern of results. On average participants tended to underestimate their level of control, 
                especially if the objective level of control was between 30% and 60%, and they only strated to 
                approximate the objective level of control if it reached at least 80%. 
                In contrast, the binary scale led to the pattern of results that was closer to the logistic 
                function. 
                    """)

    ##########################
    # PLOT FIGURE 1: SUBJETCIVE AND OBJECTIVE CONTROL 
    ##########################
    
    fig1, ax1 = plt.subplots()
    fig1.set_size_inches(10, 10)
    
    # Custom palette
    cust_pal = sns.color_palette('Set1')
    pal_exp3clust = [  cust_pal[0], cust_pal[1], cust_pal[2], cust_pal[3], cust_pal[4]]

    # Plot data
    sns.lineplot(data=df_long_resp, x='Control', y='Rating of SoA', hue='Experiment', palette=pal_exp3clust, legend=None)
    sns.scatterplot(data=df_long_resp, x='Control', y='Rating of SoA', hue='Experiment', palette=pal_exp3clust)
    plt.plot([0,100], [0,100], color='k')

    # Labels and appearance
    plt.title('Sense of control for each objective level of control across experiments')
    plt.xlabel('Objective level of control')
    plt.ylabel('Reported sense of control')

    # Show figure
    st.pyplot(fig1)
    
    # Figure caption
    st.markdown("***Figure 1.*** *Subjective reports of experienced control as a function of objective control across experiments.*")

    
    ##########################
    # PLOT FIGURE 2: DIFFERENCE BETWEEN OBJECTIVE AND SUBJECTIVE CONTROL
    ##########################

    # Text
    st.markdown("""The plot below presents the difference between subjective and objective level of control, across 
                objective levels of control. It demonstrates the same bias, but makes it easier to see the relation 
                between the results in different scales.
                """)

    fig2, ax2 = plt.subplots()
    fig2.set_size_inches(16, 8)
    
    # Custom palette
    cust_pal = sns.color_palette('Set1')
    pal_exp3clust = [  cust_pal[0], cust_pal[1], cust_pal[2], cust_pal[3], cust_pal[4]]

    # Plot data
    sns.lineplot(data=df_long_diff, x='Control', y='Difference', hue='Experiment', palette=pal_exp3clust, legend=None)
    sns.scatterplot(data=df_long_diff, x='Control', y='Difference', hue='Experiment', palette=pal_exp3clust)
    plt.plot([0,100], [0,0], color='k')

    # Labels and appearance
    plt.title('Difference between SoA and objective control across experiments')
    plt.xlabel('Objective level of control')
    plt.ylabel('Difference')

    # Show figure
    st.pyplot(fig2)
    
    # Figure caption
    st.markdown("***Figure 2.*** *Difference between subjective and objective control across experiments.*")


    ##########################
    # PLOT FIGURE 3: STANDARD DEVIATIONS
    ##########################

    # Text
    st.markdown("""The next plot illustrates average standard deviations of participants' responses in each experiment. 
                It shows that STDev was similar for all scales, except for the binary sale, in which it was much higher 
                for levels of cotrol between 30 and 60%. 
                """)

    fig3, ax3 = plt.subplots()
    fig3.set_size_inches(16, 8)

    # Custom palette
    cust_pal = sns.color_palette('Set1')
    pal_exp3clust = [  cust_pal[0], cust_pal[1], cust_pal[2], cust_pal[3], cust_pal[4]]

    # Plot data
    sns.lineplot(data=df_long_resp_std, x='Control', y='Response STD', hue='Experiment', palette=pal_exp3clust) # , legend=None)
    #sns.scatterplot(data=df_long_resp_std, x='Control', y='Response STD', hue='Experiment', palette=pal_exp3clust)
    plt.plot([0,100], [0,0], color='k')

    # Labels and appearance
    plt.title('Standard deviation of responses across experiments')
    plt.xlabel('Objective level of control')
    plt.ylabel('Standard deviation')

    # Show figure
    st.pyplot(fig3)
    
    # Figure caption
    st.markdown("***Figure 3.*** *Standard deviation across levels of control in all experiments.*")
    
    
    ##########################
    # PLOT FIGURE 4: HISTOGRAM OF ALL RESPONSES
    ########################## 
    
    # # Multiplot of histograms
    # fig, axs = plt.subplots(2, 1, sharey=True, figsize=(10, 10))
    
    # a = df_long_resp.loc[df_long_resp["Experiment"]==1, "Rating of SoA"]
    # b = df_long_resp.loc[df_long_resp["Experiment"]==2, "Rating of SoA"]
    # c = df_long_resp.loc[df_long_resp["Experiment"]==3, "Rating of SoA"]
    # d = df_long_resp.loc[df_long_resp["Experiment"]==4, "Rating of SoA"]
    # e = df_long_resp.loc[df_long_resp["Experiment"]==5, "Rating of SoA"]
    
    # for i, col in enumerate([a, b]):
    #   ax = axs.flat[i]
    #   #ax.scatter(df['control_level'], df[[_c]])
    #   #ax.hist(df[[col]], bins=20, color='grey')
    #   ax.hist(col, bins=100, color='grey')
    #   #mean_value = df[col].mean()
    #   #ax.axvline(mean_value, color='red', linestyle='dashed', linewidth=1.5)
    #   #ax.set_title(col)
    # st.pyplot(fig)
    # # Figure caption
    # st.markdown("***Figure 4.*** *Histogram of responses in Experiment 1 and 2.*")
    
    ##################
    # STATS: COMPARE EXPERIMENTS
    
    st.markdown("### Results of the statistical tests")
    
    from scipy.stats import f_oneway
    from scipy.stats import tukey_hsd
    a = df_long_resp_std.loc[(df_long_resp_std["Control"]==50) & (df_long_resp_std["Experiment"]==1), "Response STD"]
    b = df_long_resp_std.loc[(df_long_resp_std["Control"]==50) & (df_long_resp_std["Experiment"]==2), "Response STD"]
    c = df_long_resp_std.loc[(df_long_resp_std["Control"]==50) & (df_long_resp_std["Experiment"]==3), "Response STD"]
    d = df_long_resp_std.loc[(df_long_resp_std["Control"]==50) & (df_long_resp_std["Experiment"]==4), "Response STD"]
    e = df_long_resp_std.loc[(df_long_resp_std["Control"]==50) & (df_long_resp_std["Experiment"]==5), "Response STD"]
    F, p = f_oneway(a, b, c, d, e)
    print(f"The results of the 1-way between subjects ANOVA for factor Experiment on Control Level = 5: F={F:03f}, p=={p:03f}")
    st.markdown(f"The results of the 1-way between-subjects ANOVA on STDs for factor Experiment on Control Level = 5: F={F:.03f}, p={p:.03f}")
    
    # Post-hoc tests:
    res = tukey_hsd(a, b, c, d, e)
    print(res)
    
    #res2 = pd.DataFrame(res)
    
    #####
    # BIG ANOVA
    
    # Load data
    df_long_all = pd.read_csv('DATA_MeasuringSoA_long_ALL.csv')
    
    # Perform the ANOVA
    st.markdown(f"#### Big analysis: 5x11 repeated measures ANOVA - means")
    # Prepare data
    df_temp = df_long_all.loc[df_long_all["Include"] == 1,["SubNum", "control_level", "Experiment", "difference"]].groupby(["SubNum","control_level", "Experiment"]).mean().reset_index()
    # PINGOUIN
    # Perform repeated measures ANOVA
    anova_results = rm_anova(data=df_temp, dv='difference', within=['control_level', 'Experiment'], subject='SubNum')
    #st.markdown("PINGUOIN RESULTS:")
    st.table(anova_results)
    
    # Perform the ANOVA
    st.markdown(f"##### Without Exp 4: 4x11 repeated measures ANOVA - means")
    # Prepare data
    df_temp2 = df_temp.loc[df_temp["Experiment"] != 4,:]
    # PINGOUIN
    # Perform repeated measures ANOVA
    anova_results = rm_anova(data=df_temp2, dv='difference', within=['control_level', 'Experiment'], subject='SubNum')
    #st.markdown("PINGUOIN RESULTS:")
    st.table(anova_results)
    
    
    # Perform the ANOVA on STDs
    st.markdown(f"#### Big analysis: 5x11 repeated measures ANOVA - STDs")
    # Prepare data
    df_temp = df_long_all.loc[df_long_all["Include"] == 1,["SubNum", "control_level", "Experiment", "difference"]].groupby(["SubNum","control_level", "Experiment"]).std().reset_index()
    # PINGOUIN
    # Perform repeated measures ANOVA
    anova_results = rm_anova(data=df_temp, dv='difference', within=['control_level', 'Experiment'], subject='SubNum')
    #st.markdown("PINGUOIN RESULTS:")
    st.table(anova_results)

    # Perform the ANOVA
    st.markdown(f"##### Without Exp 4: 4x11 repeated measures ANOVA - STDs")
    # Prepare data
    df_temp2 = df_temp.loc[df_temp["Experiment"] != 4,:]
    # PINGOUIN
    # Perform repeated measures ANOVA
    anova_results = rm_anova(data=df_temp2, dv='difference', within=['control_level', 'Experiment'], subject='SubNum')
    #st.markdown("PINGUOIN RESULTS:")
    st.table(anova_results)
    
    
    
    
#%% BUTTON == 'Results of specific experiments'

if button_radio == 'Results of specific experiments':
    st.title(f"Selected experiment: {selected_selectbox}")
    st.title(f"Main results from the selected experiment")
    st.markdown('''This section shows the stats and plots for a selected experiment. 
                ''')
                
    # ===================================================
    # Prepare data
    selected_exp_num = selected_selectbox[11:]
    url_link = f'DATA_MeasuringSoA_long_Exp{selected_exp_num}.csv'
    df = pd.read_csv(url_link)
    # Adjust the control_level to be on the 0-100 scale 
    df['control_level'] = df['control_level']*100
    # Exclude participants that do not fulfill the inclusion critia
    if selectbox_exclude_outliers == "Yes":
        df = df[df['Include'] == 1]
    
    
    # ===================================================

    
    # PLOT RESULTS 1
    st.markdown('### Subjective reports of sense of control for each objective level of control')
    st.markdown('''Plot the relationship between objective control and subjective reports. 
                ''')
    # Plot results: Plot responses from the expected
    fig1, ax1 = plt.subplots()
    fig1.set_size_inches(10, 10)
    # Plot data
    sns.scatterplot(data=df, x='control_level', y='response', color='k')
    sns.lineplot(data=df, x='control_level', y='response', color='k')
    # Plot a line on 0
    plt.plot([0, 100], [0, 100], color='r')
    # Labels and appearance
    #plt.title(f'Sense of control across levels of control (Exp {selected_exp_num})')
    plt.xlabel('Objective level of control')
    plt.ylabel('Sense of control')
    # Tell streamlit to plot it
    st.pyplot(fig1)
    st.markdown(f"***Figure 1.*** *Average reported sense of control across objective levels of control. Data from Experiment {selected_exp_num}*. ")
    
    
    
    #####################
    # SHOW RESULTS OF THE T-TEST
    
    st.markdown("## **ANOVAs with post hoc tests (on averages across blocks)**")
        
    # Experiment 1C: 2x11 ANOVA
    if selected_exp_num == '1C':
        
        st.markdown(f"#### Experiment {selected_exp_num}: 2x11 repeated measures ANOVA")
        # Prepare data
        df_temp = df.loc[df["Include"] == 1,["SubNum", "control_level", "scale", "difference"]].groupby(["SubNum","control_level", "scale"]).mean().reset_index()

        # PINGOUIN
        # Perform repeated measures ANOVA
        anova_results = rm_anova(data=df_temp, dv='difference', within=['control_level', 'scale'], subject='SubNum')
        #st.markdown("PINGUOIN RESULTS:")
        st.table(anova_results)
        
        
        
    elif selected_exp_num == '4':
        
        st.markdown(f"#### Experiment {selected_exp_num}: Friedman test")
        # Prepare data
        df_temp = df.loc[df["Include"] == 1,["SubNum", "control_level", "difference"]].groupby(["SubNum","control_level"]).mean().reset_index()

        # PINGOUIN
        # Perform repeated measures ANOVA
        friedman_results = pg.friedman(data=df_temp, dv='difference', within='control_level', subject='SubNum')
        #st.markdown("PINGUOIN RESULTS:")
        st.table(friedman_results)
        
        
    else:        
        # Prepare data
        
        st.markdown(f"#### Experiment {selected_exp_num}: Friedman test")
        # Prepare data
        df_temp = df.loc[df["Include"] == 1,["SubNum", "control_level", "difference"]].groupby(["SubNum","control_level"]).mean().reset_index()
        
        # STATSMODELS
        # Fit the model
        # anova = AnovaRM(df_temp, depvar='difference', subject='SubNum', within=['control_level']).fit()
        # st.markdown("STATSMODELS RESULTS:")
        # st.text(anova)
        
        # PINGOUIN
        # Perform repeated measures ANOVA
        anova_results = rm_anova(data=df_temp, dv='difference', within='control_level', subject='SubNum')
        #st.markdown("PINGUOIN RESULTS:")
        st.table(anova_results)
    
    #################################
    ##### Results of the post hoc t-tests
    if selected_exp_num == '4':
        
        # Perform t-tests:
        t_test_results = []
        for control_level in range(11):
            control = control_level * 10
            median_hypothesis  = 0
            data2test = df.loc[df['control_level'] == control, ['SubNum','difference']].groupby('SubNum').mean().reset_index()
            results = sp.stats.wilcoxon(data2test['difference'] - median_hypothesis, axis=0, nan_policy='propagate', alternative='two-sided')
            t_test_results.append(results)
        results_t_tests = pd.DataFrame(t_test_results, columns=['W-value', 'p-value']).reset_index().rename(columns={"index" : "Control"})
        # Calculate Bonferroni-corrected p-values
        results_t_tests['p-value corrected'] = results_t_tests['p-value'].apply(lambda x: min(x*11, 1))
        # Calculate logarithm of the p-value
        results_t_tests['Log p-value'] = np.log10(results_t_tests['p-value'])
        results_t_tests['Control'] = results_t_tests['Control'] * 10
        
        # SHOW RESULTS OF THE T-TEST
        # Show table with the results of t-tests
        st.markdown(f"##### Experiment {selected_exp_num}: Post hoc Wilcoxon results")
        st.table(results_t_tests.loc[:,['Control', 'W-value', 'p-value','p-value corrected']].style.format("{:.3f}"))
        
    else:
        
        # Perform t-tests:
        t_test_results = []
        for control_level in range(11):
            control = control_level * 10
            data2test = df.loc[df['control_level'] == control, ['SubNum','difference']].groupby('SubNum').mean().reset_index()
            results = sp.stats.ttest_1samp(data2test['difference'], popmean=0, axis=0, nan_policy='propagate', alternative='two-sided')
            t_test_results.append(results)
        results_t_tests = pd.DataFrame(t_test_results, columns=['T-value', 'p-value']).reset_index().rename(columns={"index" : "Control"})
        # Calculate Bonferroni-corrected p-values
        results_t_tests['p-value corrected'] = results_t_tests['p-value'].apply(lambda x: min(x*11, 1))
        # Calculate logarithm of the p-value
        results_t_tests['Log p-value'] = np.log10(results_t_tests['p-value'])
        results_t_tests['Control'] = results_t_tests['Control'] * 10
        
        # SHOW RESULTS OF THE T-TEST
        # Show table with the results of t-tests
        st.markdown(f"##### Experiment {selected_exp_num}: T-test results")
        st.table(results_t_tests.loc[:,['Control', 'T-value', 'p-value','p-value corrected']].style.format("{:.3f}"))
    
    
    #####################
    # PLOT 2: RESULTS - STD
    st.markdown('### Standard deviation (of responses across blocks) of reported sense of control')
    st.markdown('''Standard deviation of reports depending on the level of control.
                ''')
    # Plot results: Plot STD of responses
    fig2, ax2 = plt.subplots()
    fig2.set_size_inches(10, 10)
    data2plot = df.loc[:, ['SubNum', 'control_level', 'difference']].groupby(['SubNum', 'control_level']).std().reset_index()
    # Plot data
    sns.scatterplot(data=data2plot, x='control_level', y='difference', color='k')
    sns.lineplot(data=data2plot, x='control_level', y='difference', color='k')

    # Labels and appearance
    #plt.title(f'Standard deviation of sense of control (Exp {selected_exp_num})')
    plt.xlabel('Objective level of control')
    plt.ylabel('STD of Sense of control')
    plt.ylim([0,50])
    
    # Tell streamlit to plot it
    st.pyplot(fig2)
    st.markdown(f"***Figure 2.*** *Standard deviation of reported sense of control across blocks for each objective level of control. Data from Experiment {selected_exp_num}*. ")
    
    if selected_exp_num == "4":
        
        #####################
        # PLOT 3: Plot confidence in response in Experiment 4
        
        # Plot results: Plot STD of responses
        fig3, ax3 = plt.subplots()
        fig3.set_size_inches(10, 10)
        data2plot = df.loc[:, ['SubNum', 'control_level', 'Confidence']].groupby(['SubNum', 'control_level']).mean().reset_index()
        # Plot data
        sns.scatterplot(data=data2plot, x='control_level', y='Confidence', color='k')
        sns.lineplot(data=data2plot, x='control_level', y='Confidence', color='r')

        # Labels and appearance
        #plt.title(f'Standard deviation of sense of control (Exp {selected_exp_num})')
        plt.xlabel('Objective level of control')
        plt.ylabel('Confidence in response [0-100]')
        plt.ylim([0,100])
        
        # Tell streamlit to plot it
        st.pyplot(fig3)
        st.markdown(f"***Figure 3.*** *Mean confidence in reported sense of control for each objective level of control. Data from Experiment {selected_exp_num}*. ")
        
        ###################
        # STATISTICS: STD
        
        # Perform the ANOVA on STDs
        st.markdown(f"#### One-way repeated-measures ANOVA on confidence")
        # Prepare data
        df_temp = df.loc[df["Include"] == 1,["SubNum", "control_level", "Confidence"]].groupby(["SubNum","control_level"]).mean().reset_index()
        # PINGOUIN
        # Perform repeated measures ANOVA
        anova_results = rm_anova(data=df_temp, dv='Confidence', within='control_level', subject='SubNum')
        #st.markdown("PINGUOIN RESULTS:")
        st.table(anova_results)
        
        # Averages
        df_plot = df.loc[df["Include"] == 1,["SubNum", "control_level", "Confidence"]].groupby(["control_level"]).mean().reset_index()
        st.markdown(f"##### Experiment {selected_exp_num}: Average confidence at each control level")
        st.table(df_plot.loc[:,['control_level', 'Confidence']].style.format("{:.3f}"))
        
        
    if selected_exp_num == "1C":
        
        st.markdown('### Comparison (within-subject) between Left-Right and Right-Left scales')
        
        #####################
        # PLOT 3: Plot confidence in response in Experiment 4
        
        # Plot results: Plot STD of responses
        fig3, ax3 = plt.subplots()
        fig3.set_size_inches(12, 10)
        data2plot = df.loc[:, ['SubNum', 'control_level', 'difference', 'scale']].groupby(['SubNum', 'control_level', 'scale']).mean().reset_index()
        # Plot data
        sns.scatterplot(data=data2plot, x='control_level', y='difference', hue="scale")#, color='k')
        sns.lineplot(data=data2plot, x='control_level', y='difference', hue="scale", legend=False)#, color='r')
        plt.plot([0, 100], [0, 0], color='k')

        # Labels and appearance
        #plt.title(f'Standard deviation of sense of control (Exp {selected_exp_num})')
        plt.xlabel('Objective level of control')
        plt.ylabel('Difference')
        plt.ylim([-100,100])
        
        # Tell streamlit to plot it
        st.pyplot(fig3)
        st.markdown(f"""***Figure 3.*** *Comparison between LR and RL scales. Data plots 
                    difference between reported and objective level of control for each objective level of control. 
                    Data from Experiment {selected_exp_num}*. """)
        
        
        
        
    # ############################
    # DISTRIBUTION OF THE RESPONSES
    
    st.markdown("## Histogram of all individual responses")
    
    fig4, ax4 = plt.subplots()
    fig4.set_size_inches(12, 10)

    sns.histplot(data=df, x="response", bins=101)
    st.pyplot(fig4)
    
    st.markdown(f"""***Figure 4.*** *Histogram of all individual responses. It shows whether participants showed 
                the tendency to select more often specific responses. Data from Experiment {selected_exp_num}*. """)
    
        
#%% BUTTON == 'Preregistered analyses'

if button_radio == 'Preregistered analyses':
    
    st.title(f"Selected experiment: {selected_selectbox}")
    st.title("Preregistered analyses")
    st.markdown('''Below we present the results of preregistered analyses.
                ''')
            
    # ANALYSIS 1
    # --------------------------------------
    st.markdown("### **ANALYSIS 1**: performed to evaluate the fit of the data.")
    st.markdown("#### **Analysis 1a**")
    st.markdown("""**Analysis 1a (median)**: A linear regression with Objective level of Control as an independent variable and median 
                reported Sense of Agency as a dependent variable. The dependent variable will be participant’s median response, 
                rather than mean, in order to reduce the influence of potential outliers caused by loss of attention during the task. """)
                
    
    # Prepare data
    #df_temp = df.loc[df["Include"] == 1,["SubNum", "control_level", "response"]].groupby("SubNum").mean()
    df_temp = df.loc[:,["SubNum", "control_level", "response"]].groupby(["SubNum", "control_level"]).median().reset_index()
    X = df_temp[["control_level"]].to_numpy()
    y = df_temp["response"].to_numpy()
    
    
    # Add a constant (for the intercept term)
    X = sm.add_constant(X)
    # Fit the model
    model = sm.OLS(y, X).fit()
    # Print the summary
    print(model.summary())
    st.text(model.summary())
    
    # --------------------------------------
    st.markdown("""**Analysis 1a (mean)**: A linear regression with Objective level of Control as an independent variable and median 
                reported Sense of Agency as a dependent variable. The dependent variable will be participant’s median response, 
                rather than mean, in order to reduce the influence of potential outliers caused by loss of attention during the task. """)
                
    
    # Prepare data
    #df_temp = df.loc[df["Include"] == 1,["SubNum", "control_level", "response"]].groupby("SubNum").mean()
    df_temp = df.loc[:,["SubNum", "control_level", "response"]].groupby(["SubNum", "control_level"]).mean().reset_index()
    X = df_temp[["control_level"]].to_numpy()
    y = df_temp["response"].to_numpy()
    
    
    # Add a constant (for the intercept term)
    X = sm.add_constant(X)
    # Fit the model
    model = sm.OLS(y, X).fit()
    # Print the summary
    print(model.summary())
    st.text(model.summary())
    
    
    
    # --------------------------------------
    st.markdown("""**Analysis 1a (mean from Block1)**: A linear regression with Objective level of Control as an independent variable and median 
                reported Sense of Agency as a dependent variable. The dependent variable will be participant’s median response, 
                rather than mean, in order to reduce the influence of potential outliers caused by loss of attention during the task. """)
                
    
    # Prepare data
    #df_temp = df.loc[df["Include"] == 1,["SubNum", "control_level", "response"]].groupby("SubNum").mean()
    df_temp = df.loc[df["block"] == 1,["SubNum", "control_level", "response"]].groupby(["SubNum", "control_level"]).mean().reset_index()
    X = df_temp[["control_level"]].to_numpy()
    y = df_temp["response"].to_numpy()
    
    
    # Add a constant (for the intercept term)
    X = sm.add_constant(X)
    # Fit the model
    model = sm.OLS(y, X).fit()
    # Print the summary
    print(model.summary())
    st.text(model.summary())
    
    
    # =============================
    # ANALYSIS 1B
    
    # --------------------------------------
    st.markdown("#### **Analysis 1b**")
    st.markdown("""**Analysis 1b**: A linear mixed model (LMM) with Objective level of Control treated as a covariate, and Participant 
                treated as a random effect. The dependent variable will be the reported level of sense of agency (DV1).. """)
                
    
    # Prepare data
    df_temp = df.loc[:,["SubNum", "control_level", "response"]] #.groupby(["control_level"]).median().reset_index()
    # Fit the linear mixed model
    model = mixedlm("response ~ control_level", df, groups="SubNum")
    result = model.fit() 
    
    
    # Extract variance components    
    #variance_components = result.cov_re.diagonal()  # Random effect variances
    residual_variance = result.scale                # Residual variance

    # Print the results
    st.text(result.summary())
    #st.text(f"Variance components: {variance_components}")
    st.text(f"Residual variance: {residual_variance}")
    
    
    
    # =============================
    # ANALYSIS 2
    # =============================
    
    
    # ANALYSIS 2
    st.markdown("### **ANALYSIS 2**: performed to investigate the deviations of sense of agency from the objective level of control.")
    st.markdown("#### **Analysis 2a**")
    st.markdown("""**Analysis 2a**: A one-way ANOVA with factor Objective level of Control (11 levels: from 0% to 100%, every 10%) 
                followed by a series of t-tests comparing median response deviation for each participant from zero. The dependent 
                variable will be participant’s median response deviation, rather than mean response deviation in order to reduce the 
                influence of potential outliers caused by loss of attention during the task.""")
                
    # Prepare data
    df_temp = df.loc[:,["SubNum", "control_level", "difference"]].groupby(["SubNum","control_level"]).median().reset_index()
    # STATSMODELS
    # Fit the model
    anova = AnovaRM(df_temp, depvar='difference', subject='SubNum', within=['control_level']).fit()
    st.markdown("STATSMODELS RESULTS:")
    st.text(anova)
    # PINGOUIN
    # Perform repeated measures ANOVA
    anova_results = rm_anova(data=df_temp, dv='difference', within='control_level', subject='SubNum')
    st.markdown("PINGUOIN RESULTS:")
    st.table(anova_results)
    
    
    
    
    
    
    
        
    ### ANALYSIS 2b
    st.markdown("#### **Analysis 2b**")
    st.markdown("""**Analysis 2b**: A linear mixed model (LMM) with Objective level of Control (11 levels: from 0% to 100%, every 10%) 
                treated as a factor, and Participant treated as a random effect on DV2 (response deviation). """)
                
    # Prepare data
    df_temp = df.loc[:,["SubNum", "control_level", "difference"]] #.groupby(["control_level"]).median().reset_index()
    # Fit the linear mixed model
    model = mixedlm("difference ~ control_level", df, groups="SubNum")
    result = model.fit() 
    # Print the results
    st.text(result.summary())




#%% BUTTON == 'Variance decomposition analysis'

if button_radio == 'Variance decomposition analysis':
    
    st.title(f"Selected experiment: {selected_selectbox}")
    st.title("Variance decomposition analysis")
    st.markdown('''This section presents the results of Variance Decomposition Analysis. As the name implies, it allows to 
                decompose the variance of participants' responses into different sources: variance reflected by different 
                expermental conditions (viariability reflecting different levels of control), variance reflecting differences 
                between different participants  (between-subject variability) and the unexplained variance (error variability). 
                Importantly, here we separate two types of variability reflecting the experimental conditions: variability reflecting 
                the objective level of control (objective LoC), and variability reflecting the group-level bias when evaluating different levels of 
                control (bias in LoC).
                ''')
    st.markdown('''In the first step we calculate the total variability of participants' responses in the task:
                ''')
    
    
    # ===================================================
    # Prepare data
    selected_exp_num = selected_selectbox[11:]
    url_link = f'DATA_MeasuringSoA_long_Exp{selected_exp_num}.csv'
    df = pd.read_csv(url_link)
    # Adjust the control_level to be on the 0-100 scale 
    df['control_level'] = df['control_level']*100
    # Exclude participants that do not fulfill the inclusion critia
    if selectbox_exclude_outliers == "Yes":
        df = df[df['Include'] == 1]
    
    # Calculate variance of responses and deviations
    total_variance_response = df["response"].var()
    total_variance_difference = df["difference"].var()
    var_explained_by_loc = total_variance_response - total_variance_difference
    
    st.markdown("#### **TOTAL VARIANCE OF RESPONSES**")
    st.text(f"Total variance of responses: {total_variance_response:.3f}")
    
    st.markdown("""This value represents the total amount of variance in the data that has to be explained. 
                One source of this variance is the fact that in different trials participants control the red ball over which 
                they have objectively different level of control: 0%, 10%, up to 100%. Hence, one source of variability of responses 
                reporting participants' sense of control comes from the fact that they had objctively different levels of control. 
                This variability can be removed if we look at the variability not of participants' responses, but the variability of 
                the differences between their responses and the objective level of control. In other words: at the variability of their 
                deviations.
                 """)
    
    st.markdown("#### **TOTAL VARIANCE OF DIFFERENCES/DEVIATIONS**")
    st.text(f"Total variance of the deviations: {total_variance_difference:.3f}")
    
    st.markdown("""The variance of deviations is much lower than the variance of responses. This is because we removed 
                the variance caused by the linear effect of the objective level of control. This variance accounted for the following 
                percentage of total variance of participants' responses:
                 """)
    
    #st.markdown("#### **VARIANCE EXPLAINED BY OBJECTIVE LEVEL OF CONTROL**")
    st.text(f"Variance explained by the objective LoC: {100*var_explained_by_loc/total_variance_response:.2f}% ")
    
    # Multiplot of histograms
    fig, axs = plt.subplots(1, 2, sharey=True, figsize=(10, 4))
    
    for i, col in enumerate(df[['response', "difference"]]):
      ax = axs.flat[i]
      #ax.scatter(df['control_level'], df[[_c]])
      ax.hist(df[[col]], bins=20, color='grey')
      mean_value = df[col].mean()
      ax.axvline(mean_value, color='red', linestyle='dashed', linewidth=1.5)
      ax.set_title(col)
    st.pyplot(fig)
    st.markdown(f"""***Figure 1.*** *Histograms of participants' responses (left) and deviations of responses from objective 
                level of control (right). Responses show much higher variability (variance={total_variance_response:.2f}) 
                than differences/deviations ({total_variance_difference:.2f}). Red line shows the mean.*
                 """)
                 
    # # Multiplot with scatterplots
    # cc1, cc2 = st.columns([1,1])
    # with cc1:
    #     st.scatter_chart(data=df, x="control_level", y="response", size=10) 
    # with cc2:
    #     st.scatter_chart(data=df, x="control_level", y="difference", size=10)
    # st.markdown("""***Figure 2.*** *Scatter plots of participants' responses (left) and deviations of responses from objective 
    #             level of control (right).*
    #              """)
           
    
    st.markdown("""The remaining variance is the variance that can be attributed to participants' biases in responses to 
                different levels of the experimental conditions (level of control), their indivividual differences 
                (between-subject variability) and the unexplained variance. These values can be estimated using the 
                Linear Mixed Models approach. 
                 """)
    
    st.markdown("""In the first step we evaluate a null model in which we do not model the effect of the experimental conditions. 
                We perform this model on the deviations (between the reports and the objctive level of control).
                """)
    
    # ------------------
    st.markdown(" ### MODEL 0: Null model with random effect of participants")
    # Prepare data
    df_temp = df.loc[:,["SubNum", "difference", "response"]]
    # Specifying the model
    vc = {"Participant": "1 + C(SubNum)"}
    model0 = sm.MixedLM.from_formula("difference ~ 1", groups="SubNum", vc_formula=vc, data=df_temp)
    result0 = model0.fit()
    # Print the results
    st.text(result0.summary())
    
    # Comparison with total variance of deviations
    st.markdown(f"""The LMM model estimated the variance that can be explained by differences between participants (the last 
                row, "Participant Var": {result0.vcomp[0]:.2f}"), as well as the unexplained variance (here presented under 
                the name of "scale": {result0.scale:.2f})
                """)
    var_in_model0 = result0.scale + result0.vcomp[0]
    st.text(f"Total variance in Model 0: {var_in_model0:.2f}")
    st.text(f"Total variance of deviations: {total_variance_difference:.2f}")
    
    st.markdown("""These values should be very similar, although they do not need to be exactly identical. A small difference 
                might be due to numerical approximations used when performing calculations during estimation of the LMM model.
                """)
    
    st.markdown(" ### MODEL 1: Model with fixed effect of level of control and random effect of participants")
    st.markdown("""The null model doesn't take into consideration the effect of experimental condition. Therefore as the final 
                step we stimated a model that includes level of control as a fixed effect and participants as a random effect:
                """)
    
    # Prepare data
    df_temp = df.loc[:,["SubNum", "control_level", "difference", "response"]]
    # Specifying the model
    vc = {"Participant": "1 + C(SubNum)"}
    model1 = sm.MixedLM.from_formula("difference ~ 1 + C(control_level)", groups="SubNum", vc_formula=vc, data=df_temp)
    result1 = model1.fit()
    # Print the results
    st.text(result1.summary())
    
    st.markdown("""As can be seen below the sum of within-participants variance and the unexplained variance is lower than 
                in Model 0 - that's because we added the fixed factor of level of control, which explained part of the 
                variance from the null model:
                """)
    
    var_in_model1 = result1.scale + result1.vcomp[0]
    var_dev_loc_bias = var_in_model0-var_in_model1
    var_dev_loc_bias_perc = 100*(var_in_model0-var_in_model1)/var_in_model0
    
    st.text(f"Total variance in Model 1: {var_in_model1:.2f}")
    st.text(f"Total variance in Model 0: {var_in_model0:.2f}")
    # st.text(f"Variance of deviations explained by level of control: {var_dev_loc_bias:.2f} ({var_dev_loc_bias_perc:.2f}%)")
    # st.text(f"Variance of deviations explained by the within-subject variability: {result1.vcomp[0]:.2f} ({100*result1.vcomp[0]/var_in_model0:.2f}%)")
    # st.text(f"Unexplained variance of deviations: {result1.scale:.2f} ({100*result1.scale/var_in_model0:.2f}%)")
    
    st.markdown("#### **VARIANCE DECOMPOSITION OF DEVIATIONS**")
    st.markdown("""By subtracting the total varience in Model 1 from total variance in Model 0 we obtained the amount 
                of variance explained by the experimental condition, and we can provide a full decomposition for deviations 
                from the objective level of control:
                """)
                
    vda_difference = pd.DataFrame(data=[
        ["Level of Control: bias", var_dev_loc_bias, 100*var_dev_loc_bias/total_variance_difference],
        ["Within-subject", result1.vcomp[0], 100*result1.vcomp[0]/total_variance_difference],
        ["Unexplained", result1.scale, 100*result1.scale/total_variance_difference],
        ["TOTAL VARIANCE", total_variance_difference, 100]],
        columns=["Variance explained by:", "Variance", "Percentage"])
    st.table(vda_difference.style.format({"Variance":"{:.2f}", "Percentage":"{:.2f}"}))
    
    st.markdown("#### **FULL VARIANCE DECOMPOSITION OF RESPONSES**")
    st.markdown("""Finally, all of these values can be used to provide the total variance decomposition of participants' responses:
                """)
                
    # st.text(f"Variance of responses explained by objective level of control: {total_variance_response-total_variance_difference:.2f} ({100-100*total_variance_difference/total_variance_response:.2f}%)")
    # st.text(f"Variance of responses explained by bias in level of control: {var_dev_loc_bias:.2f} ({100*var_dev_loc_bias/total_variance_response:.2f}%)")
    # st.text(f"Variance of responses explained by the within-subject variability: {result1.vcomp[0]:.2f} ({100*result1.vcomp[0]/total_variance_response:.2f}%)")
    # st.text(f"Unexplained variance of responses: {result1.scale:.2f} ({100*result1.scale/total_variance_response:.2f}%)")
    
    vda_response = pd.DataFrame(data=[
        ["Level of Control: linear", total_variance_response-total_variance_difference, 100-100*total_variance_difference/total_variance_response],
        ["Level of Control: bias", var_dev_loc_bias, 100*var_dev_loc_bias/total_variance_response],
        ["Within-subject", result1.vcomp[0], 100*result1.vcomp[0]/total_variance_response],
        ["Unexplained", result1.scale, 100*result1.scale/total_variance_response],
        ["TOTAL VARIANCE", total_variance_response, 100]],
        columns=["Variance explained by:", "Variance", "Percentage"])
    st.table(vda_response.style.format({"Variance":"{:.2f}", "Percentage":"{:.2f}"}))
    
    
    
    
#%% BUTTON == 'Reliability of data from Block 1'

if button_radio == 'Reliability of data from Block 1':
    
    st.title(f"Selected experiment: {selected_selectbox}")
    st.title("How reliable is data from the first block?")
    st.markdown('''In our study participants rated their sense of control over the red ball 5 times 
                for each level of control. Then we calculated the average from all of these responses 
                as a final measure.
                ''')
    st.markdown('''In this section we wanted to answer the question: Do we need to collect data from all 
                five blocks or is it enough to run only 1 block?
                ''')
    
    # ===================================================
    # Prepare data
    selected_exp_num = selected_selectbox[11:]
    url_link = f'DATA_MeasuringSoA_long_Exp{selected_exp_num}.csv'
    df = pd.read_csv(url_link)
    # Adjust the control_level to be on the 0-100 scale 
    df['control_level'] = df['control_level']*100
    # Exclude participants that do not fulfill the inclusion critia
    if selectbox_exclude_outliers == "Yes":
        df = df[df['Include'] == 1]
    
    
    #####################
    # PLOT 1: Block 1 versus the rest
    st.markdown('### Does data from Block 1 show the same pattern as data from the remaining blocks?')
    st.markdown('''The plot below shows how the results from the first block (red line) compare to the 
                average results from the remaining blocks (black line). The grey line shows the comparison line reflecting no 
                difference between subjective ratings and objective level of control.
                The plot show that the overall pattern of results is very similar between Block 1 responses 
                and the average responses from Blocks 2-5.
                ''')
    
    # Plot results: Plot responses from the expected
    fig1, ax1 = plt.subplots()
    fig1.set_size_inches(18, 10)
    # SPECIFY BLOCK NUMBER
    BlockNum = 1
    # Start plotting
    linecolor = 'r'
    linecolor_all = 'k'
    data2plot = df.loc[df['block']==BlockNum,:]
    data2plot_all = df
    # Plot data
    sns.lineplot(data=data2plot_all, x='control_level', y='difference', color=linecolor_all)
    sns.lineplot(data=data2plot, x='control_level', y='difference', color=linecolor)
    # Plot a line on 0
    plt.plot([0, 100], [0, 0], color='grey')
    # Labels and appearance
    #plt.title(f'The difference between sense of agency and objective control across levels of control (Exp {selected_exp_num}, Block {BlockNum})')
    plt.xlabel('Objective level of control')
    plt.ylabel('Difference: SoA - level of control')
    plt.ylim([-65, 40])
    # Custom legend using patches
    legend_elements = [
        Patch(facecolor='red',   edgecolor='red', label='Block 1'),
        Patch(facecolor='black', edgecolor='black', label='Blocks 2-5') ]
    plt.legend(handles=legend_elements, loc='upper right', title="Legend")
    
    # Tell streamlit to plot it
    st.pyplot(fig1)
    st.markdown(f"***Figure 1.*** *Comparison of results with data only from block 1 (red) and the remaining blocks (black). The difference between sense of control and objective control across levels of control. Data from Experiment {selected_exp_num}* ")
    


    #####################
    # STATISTICS: PERFORM CALCULATIONS
    #####################
            
    # Correlation between block 1 and the rest
    block_num = 1
    blocks_other = [2, 3, 4, 5]
    # Subset data
    data_1 = df.loc[(df['block']==block_num),:]
    data_other = df.loc[(df['block'].isin(blocks_other)),:]
    data_corr = []
    resp_b1 = data_1.loc[:, ['SubNum','control_level','response']] 
    resp_other = data_other.loc[:, ['SubNum','control_level','response']] 
    resp_other_mean = resp_other.groupby(['SubNum', 'control_level']).mean().reset_index()
    data_corr = pd.merge(resp_b1, resp_other_mean, how='inner', on=['SubNum','control_level'], suffixes=('_B1', '_Other'))
    # General correlation
    results_general_corr = data_corr[['response_B1','response_Other']].corr().iloc[0,1]
    
    # Correlations within each participant
    sub_num_list = data_corr['SubNum'].unique()
    results_correlations_subnum = []
    for sub_num in sub_num_list:
        res_corr = data_corr.loc[data_corr['SubNum'] == sub_num,['response_B1','response_Other']].corr() #[0,1]
        results_correlations_subnum = results_correlations_subnum + [[sub_num, res_corr.iloc[0,1]]]
    results_correlations_subnum = pd.DataFrame(results_correlations_subnum)
    results_correlations_subnum.columns = ["participant", "r_correlation"]
    
    # Average correlation for each participant
    blocks_list = [1, 2, 3, 4, 5]
    sub_num_list = data_corr['SubNum'].unique()
    general_corr = []
    sub_corr_list = []
    data_sub_corrs = pd.DataFrame(zip(sub_num_list, [9]*len(sub_num_list), [9]*len(sub_num_list), 
                                      [9]*len(sub_num_list), [9]*len(sub_num_list), [9]*len(sub_num_list) ), 
                                  columns=['SubNum', 'Corr_B1', 'Corr_B2', 'Corr_B3', 'Corr_B4', 'Corr_B5'])
    for block_num in list(range(1,6)):
        exclude = {block_num}
        blocks_other = [num for num in blocks_list if num not in exclude]
        x = df.loc[(df['block']==block_num),['SubNum','control_level','response']]
        y = df.loc[(df['block'].isin(blocks_other)),['SubNum','control_level','response']]
        y = y.groupby(['SubNum', 'control_level']).mean().reset_index()
        xy = pd.merge(x, y, how='inner', on=['SubNum','control_level'], suffixes=('_B', '_Other') )
        # Calculate general correlations
        gen_corr = xy.loc[:,['response_B','response_Other']].corr()
        general_corr.append(gen_corr.iloc[0,1])
        # Calculate results for each participant
        for sub_num in sub_num_list:
            sub_corr = xy.loc[xy['SubNum']==sub_num,['response_B','response_Other']].corr().iloc[0,1]
            data_sub_corrs.loc[data_sub_corrs['SubNum']==sub_num, f'Corr_B{block_num}'] = sub_corr            
    # Calculate the average, i.e. Cronbach's alpha
    data_sub_corrs['cronbach_alpha'] = data_sub_corrs[['Corr_B1', 'Corr_B2', 'Corr_B3', 'Corr_B4', 'Corr_B5']].mean(axis=1)
    #data_sub_corrs[['Corr_B1', 'Corr_B2', 'Corr_B3', 'Corr_B4', 'Corr_B5']].mean(axis=0)
        
    
    
    
    #####################
    # CORRELATIONS BETWEEN SCALES
    
    st.markdown("### Do responses that participants give in block 1 correlate with responses in the remaining blocks?")
    st.markdown(f"""For the majority of participants the ratings from block 1 are very strongly correlated with the average 
                ratings from the remaining blocks. Figure 2 shows the distribution of these correlations. For most participants 
                this correlation falls between 0.75 and 1. The average of these correlations is: **r = {results_general_corr:.03}**. 
                The median is **r = {data_sub_corrs["Corr_B1"].median():.03}**. The range is between {data_sub_corrs["Corr_B1"].min():.03} and
                {data_sub_corrs["Corr_B1"].max():.03}.
                """)
                
    
    
    #####################
    # PLOT 2: Plot distribution of correlations within participants
    
    fig2, ax2 = plt.subplots()
    fig2.set_size_inches(10, 5)
    sns.histplot(data_sub_corrs['Corr_B1'])
    plt.xlim([-1.05,1.05])
    plt.xlabel('Strength of correlation')
    plt.title("Correlation between block 1 and the remaining blocks")
    plt.plot([-1,-1], [0,5], color="red")
    plt.plot([0,0], [0,3], color="red")
    plt.plot([1,1], [0,5], color="red")
    
    st.pyplot(fig2)
    st.markdown(f"***Figure 2A.*** *The distribution of correlations between block 1 and the remaining blocks across participants. Data from Experiment {selected_exp_num}* ")
    
    
    #####################
    # PLOT 2B: Plot distribution of correlations within participants
    
    st.markdown(f"""The plot below shows the average correlation between each single block and the remaining blocks. 
                The average of these correlations is: **r = {data_sub_corrs["cronbach_alpha"].mean():.03}**. 
                The median is **r = {data_sub_corrs["cronbach_alpha"].median():.03}**. The range is 
                between {data_sub_corrs["cronbach_alpha"].min():.03} and {data_sub_corrs["cronbach_alpha"].max():.03}.
                """)
                
    fig2b, ax2b = plt.subplots()
    fig2b.set_size_inches(10, 5)
    sns.histplot(data_sub_corrs['cronbach_alpha'])
    plt.xlim([-1.05,1.05])
    plt.title("Average correlation with the remaining blocks")
    plt.xlabel('Average correlation')
    plt.plot([-1,-1], [0,5], color="red")
    plt.plot([0,0], [0,3], color="red")
    plt.plot([1,1], [0,5], color="red")
    
    st.pyplot(fig2b)
    st.markdown(f"***Figure 2B.*** *The distribution of average correlations between a single block and the remaining blocks. Data from Experiment {selected_exp_num}* ")
    
    
    ### INTRA-CLASS CORRELATION
    
    # icc_result = pg.intraclass_corr(data=df, 
    #                             targets='SubNum', 
    #                             raters='block', 
    #                             ratings='response')
    
    # Do it for each level of control
    for con_lev in range(0,11):
        icc_result = pg.intraclass_corr(data=df.loc[df['control_level'] == con_lev*10,:], 
                                    targets='SubNum', 
                                    raters='block',
                                    ratings='response')
        st.markdown(f"### ICC for control level = {con_lev*10}")
        st.table(icc_result)
    
    #st.markdown("### ICC")
    #st.markdown(f"The ICC is:")
    #st.table(icc_result)
    
#%% BUTTON == 'Correlations within levels of control'

if button_radio == 'Correlations within levels of control':
    
    st.title(f"Selected experiment: {selected_selectbox}")
    st.title("How reliable is data from the first block if we look within each level of control?")
    st.markdown('''The interpretation of this data is a bit more complex.
                ''')
    st.markdown('''In x
                ''')
    
    # ===================================================
    # Prepare data
    selected_exp_num = selected_selectbox[11:]
    url_link = f'DATA_MeasuringSoA_long_Exp{selected_exp_num}.csv'
    df = pd.read_csv(url_link)
    # Adjust the control_level to be on the 0-100 scale 
    df['control_level'] = df['control_level']*100
    # Exclude participants that do not fulfill the inclusion critia
    if selectbox_exclude_outliers == "Yes":
        df = df[df['Include'] == 1]
    
    # Correlation between block 1 and the rest
    block_num = 1
    blocks_other = [2, 3, 4, 5]
    # Subset data
    data_1 = df.loc[(df['block']==block_num),:]
    data_other = df.loc[(df['block'].isin(blocks_other)),:]
    data_corr = []
    resp_b1 = data_1.loc[:, ['SubNum','control_level','response']] 
    resp_other = data_other.loc[:, ['SubNum','control_level','response']] 
    resp_other_mean = resp_other.groupby(['SubNum', 'control_level']).mean().reset_index()
    data_corr = pd.merge(resp_b1, resp_other_mean, how='inner', on=['SubNum','control_level'], suffixes=('_B1', '_Other'))
    # General correlation
    results_general_corr = data_corr[['response_B1','response_Other']].corr().iloc[0,1]
    
    # Correlations within each control level
    results_correlations = []
    for con_lev in range(0,110,10):
        res_corr = data_corr.loc[data_corr['control_level'] == con_lev,['response_B1','response_Other']].corr() #[0,1]
        results_correlations = results_correlations + [[con_lev, res_corr.iloc[0,1]]]
    results_correlations = pd.DataFrame(results_correlations)
    results_correlations.columns = ["control_level", "r_correlation"]
    
    
    
    #####################
    # PLOT 3: Plot correlations for all control levels
    
    # Plot results: Plot responses from the expected
    fig3, ax3 = plt.subplots()
    fig3.set_size_inches(10, 5)
    # Plot general correlation
    sns.barplot(data=results_correlations, x='control_level', y='r_correlation')
    plt.ylim([-1,1])
    plt.plot([-1,11], [0,0], color='k')
    plt.xlim([-1,11])
    plt.xlabel("Level of control")
    plt.ylabel("Correlation size")
    # Tell streamlit to plot it
    st.pyplot(fig3)

    st.table(results_correlations)
    
    #####################
    # PLOT 4: Plot correlations for a given control level
    
    # Check the number of subjects and apply to the slider
    control_level_num = st.slider('Select the objective level of control', min_value=0, max_value=100, step=10)
    data_corr_level = data_corr.loc[data_corr['control_level'] == control_level_num,['response_B1','response_Other']]
    
    # Plot results: Plot responses from the expected
    fig4, ax4 = plt.subplots()
    fig4.set_size_inches(10, 6)
    # Plot general correlation
    sns.scatterplot(data=data_corr_level, x='response_B1', y='response_Other')
    plt.ylim([-5,105])
    plt.plot([0,100], [0,100], color='r')
    plt.xlim([-5,105])
    plt.xlabel("Response in Block 1")
    plt.ylabel("Mean response in Blocks 2-5")
    plt.plot(control_level_num,control_level_num, 'ro', markersize=10)
    # Tell streamlit to plot it
    st.pyplot(fig4)
    

    #####################
    # PLOT 5: Block 1 on X-axis versus the rest on the Y-axis
    
    # Introductory text
    st.markdown('### Plot responses in Block 1 against the remaining responses')
    st.markdown('''Each line is a single participant.
                ''')
    
    # Plot results: Plot responses from the expected
    fig5, ax5 = plt.subplots()
    fig5.set_size_inches(10, 10)
    # Plot general correlation
    sns.lineplot(data=data_corr, x='response_B1', y='response_Other', hue='SubNum', alpha=0.4)
    sns.scatterplot(data=data_corr, x='response_B1', y='response_Other', hue='SubNum', alpha=0.4, legend=False)
    plt.plot([0,100], [0,100], color='r')
    plt.xlabel("Response in Block 1")
    plt.ylabel("Mean response in Blocks 2-5")
    
    # Tell streamlit to plot it
    st.pyplot(fig5)
    
    
    

#%% BUTTON == 'Cluster analysis'

if button_radio == 'Cluster analysis':
    
    st.title(f"Selected experiment: {selected_selectbox}")
    st.title(f"Cluster analysis")
    st.markdown('''Cluster analysis detecting different patterns of responses in the experimental task. 
                ''')
    
    st.markdown(">>> :::DUE TO COMPUTING POWER LIMITATIONS THE FIGURES ARE PRE-RENDERED::: ")
    
    # # PREPARE DATA
    # selected_exp_num = selected_selectbox[11:]
    # url_link = f'DATA_MeasuringSoA_long_Exp{selected_exp_num}.csv'
    # df = pd.read_csv(url_link)
    # # Adjust the control_level to be on the 0-100 scale 
    # df['control_level'] = df['control_level']*100
    # # Exclude participants that do not fulfill the inclusion critia
    # if selectbox_exclude_outliers == "Yes":
    #     df = df[df['Include'] == 1]

    
    
    # # PREPARE DATA FOR CLUSTER ANALYSIS
    # # Prepare datasets
    # avg_diff = df.loc[:,["SubNum", "control_level", "difference"]]
    # x = avg_diff.groupby(["SubNum", "control_level"])
    # x = x.mean().reset_index().pivot(columns="control_level", index="SubNum", values="difference")
    
    # # Create a df_kmeans dataframe that will be useful later
    # df_kmeans = x.copy().reset_index()
    
    
    # ##########################
    # # PERFORM CLUSTER ANALYSIS FOR THREE CLUSTERS
    
    st.markdown("### Perform cluster analysis for 3 clusters")
    
    # # Perform the target cluster analysis   
    # k = 3
    # kmeans = KMeans(init = "k-means++", n_clusters=k, n_init = 20)
    # kmeans.fit(x)
    # # Add to the data
    # df_kmeans['Cluster'] = kmeans.labels_ + 1
    # df_kmeans['Participant'] = df['SubNum']
    # #df_kmeans = pd.concat([df_kmeans, df[resp_mean]], axis=1)

    # cluster_centers = kmeans.cluster_centers_
    # #print('Cluster centres: ',cluster_centers)
    # print('Cluster sizes: ')
    # print(df_kmeans['Cluster'].value_counts())
    
    # # Melt the data into the long format
    # df_k_long = pd.melt(df_kmeans, id_vars=['Participant', 'Cluster'], value_vars=df_kmeans.columns[1:], var_name='Control level', value_name='Difference')
    # df_k_long['Control'] = df_k_long['Control level'].astype(int)
    # #df_k_long['Control'] = pd.to_numeric(df_k_long['Control'])*100

    # # Line plot showing the clusters
    # fig1, ax1 = plt.subplots()
    # fig1.set_size_inches(10, 8)

    # # Custom palette
    # cust_pal = sns.color_palette('Set1')
    # pal_exp3clust = [  cust_pal[2], cust_pal[0], cust_pal[1] ] #, cust_pal[3], ]

    # # Plot data
    # sns.lineplot(data=df_k_long, x='Control', y='Difference', hue='Cluster', palette=pal_exp3clust, legend=None)
    # sns.scatterplot(data=df_k_long, x='Control', y='Difference', hue='Cluster', palette=pal_exp3clust)
    # plt.plot([0,100], [0,0], color='k')
    # plt.ylim([-100,100])

    # # Labels and appearance
    # plt.title(f'Cluster analysis: three clusters solution (Exp {selected_exp_num})')
    # plt.xlabel('Objective level of control')
    # plt.ylabel('Difference')
    # st.pyplot(fig1)
    
    fig_filename = f'figures_all_cluster_analyses/cluster_3clusters_{selected_selectbox}_Exclude_{selectbox_exclude_outliers}.png'
    st.image(image=fig_filename)
    st.markdown(f"***Figure 1.*** *Results of the k-means cluster analysis for a solution with three clusters. Data from Experiment {selected_exp_num}*. ")


    # ##########################
    # # PERFORM MULTIPLE CLUSTER ANALYSES
    
    st.markdown("### Determine the best number of clusters using the Silhouette score")
    # st.markdown(">>> :::WAIT FOR CLUSTER ANALYSIS TO BE COMPLETED (SHOULD TAKE UP TO 30 SECONDS)::: ")
    
    # # Determine the max silhoutte score
    # silhouette_list = np.zeros([11,2])
    # for kk in range(2,11):
    #     kmeans_test = KMeans(init = "k-means++", n_clusters=kk, n_init = 20)
    #     kmeans_test.fit(x)
    #     silh_score = silhouette_score(x, kmeans_test.labels_)
    #     silhouette_list[kk,:] = [kk, silh_score]
    #     print('Silhouette score for n=',str(kk),': ',str(silh_score))
    
    # Text describing the Silhouette score
    st.markdown('''The Silhouette plot allows to select the number of clusters that provide the best fit to the data. 
                The number of clusters that has the highest score 
                ''')
                
    # # Plot silhouette scores for each cluster size
    # fig2, ax2 = plt.subplots()
    # fig2.set_size_inches(10, 4)
    # sns.lineplot(x=silhouette_list[:,0], y=silhouette_list[:,1], color='red').set(title='Silhouette plot', xlabel='Number of clusters', ylabel='Silhouette score')
    # st.pyplot(fig2)
    fig_filename = f'figures_all_cluster_analyses/cluster_silhuoette_{selected_selectbox}_Exclude_{selectbox_exclude_outliers}.png'
    st.image(image=fig_filename)
    st.markdown(f"***Figure 2.*** *Silhouette scores for each k value of the number of clusters. Data from Experiment {selected_exp_num}*. ")
    
    # # Find the number of clusters with the highest silhouette score
    # max_silhouette = silhouette_list[np.argmax(silhouette_list[:, 1]), 0]
    # max_silhouette_first4 = silhouette_list[np.argmax(silhouette_list[:3, 1]), 0]
    
    # # Plot the final cluster analysis results
    # st.markdown("**Results from the Silhuoette plot:**")
    # st.markdown(f"- The optimal number of clusters is: {max_silhouette:.0f} (based on the highest silhouette score)")
    # st.markdown(f"- The optimal number of clusters if cannot be higher than 4 is: {max_silhouette_first4:.0f} (based on the highest silhouette score)")

    
    # ##########################
    # # PERFORM CLUSTER ANALYSIS FOR THE HIGHEST SILHOUETTE SCORE

    # # Perform the target cluster analysis   
    # k = int(max_silhouette_first4)
    # kmeans = KMeans(init = "k-means++", n_clusters=k, n_init = 20)
    # kmeans.fit(x)
    # # Add to the data
    # df_kmeans['Cluster'] = kmeans.labels_ + 1
    # df_kmeans['Participant'] = df['SubNum']
    # #df_kmeans = pd.concat([df_kmeans, df[resp_mean]], axis=1)

    # cluster_centers = kmeans.cluster_centers_
    # #print('Cluster centres: ',cluster_centers)
    # print('Cluster sizes: ')
    # print(df_kmeans['Cluster'].value_counts())
    
    # # Melt the data into the long format
    # df_k_long = pd.melt(df_kmeans, id_vars=['Participant', 'Cluster'], value_vars=df_kmeans.columns[1:], var_name='Control level', value_name='Difference')
    # df_k_long['Control'] = df_k_long['Control level'].astype(int)
    # #df_k_long['Control'] = pd.to_numeric(df_k_long['Control'])*100

    # # Line plot showing the clusters
    # fig3, ax3 = plt.subplots()
    # fig3.set_size_inches(10, 8)

    # # Custom palette
    # cust_pal = sns.color_palette('Set1')
    # pal_exp3clust = [  cust_pal[2], cust_pal[0], cust_pal[1] ] #, cust_pal[3], ]

    # # Plot data
    # sns.lineplot(data=df_k_long, x='Control', y='Difference', hue='Cluster', palette=pal_exp3clust, legend=None)
    # sns.scatterplot(data=df_k_long, x='Control', y='Difference', hue='Cluster', palette=pal_exp3clust)
    # plt.plot([0,100], [0,0], color='k')
    # plt.ylim([-100,100])

    # # Labels and appearance
    # plt.title(f'Cluster analysis: {k} clusters (Exp {selected_exp_num})')
    # plt.xlabel('Objective level of control')
    # plt.ylabel('Difference')
    # st.pyplot(fig3)
    fig_filename = f'figures_all_cluster_analyses/cluster_maxsilh_clusters_{selected_selectbox}_Exclude_{selectbox_exclude_outliers}.png'
    st.image(image=fig_filename)
    st.markdown(f"***Figure 3.*** *Results of the cluster analysis for the number of clusters with the highest Siluoette score. Data from Experiment {selected_exp_num}*. ")
    





#%% BUTTON == 'Metrics of individual differences'

if button_radio == 'Metrics of individual differences':
    
    st.title(f"Selected experiment: {selected_selectbox}")
    st.title("Can we use participants' responses in the task as a measure of individual differences?")
    st.markdown('''The results of the cluster analysis suggest that we can distinguish clusters of participants 
                that are characterized by the tendency to report much lower sense of control than their objective 
                level of control ("underestimators") and another cluster of people that tend to be relatively 
                accurate to to overestimate their control. However, even though cluster analysis in most cases separated them into 
                two clusters, they might correspond to two extremes of a continuum, rather than two qualitatively distinct 
                groups. 
                ''')
    st.markdown('''Here, we developed three metrics that allow to use our task as a measure individual tendency to overestimate 
                or underestimate one's sense of control (**BIAS**), how close one's subjective responses are to the objective level 
                of control (**ACCURACY**), and how close one's responses are to responding at random (**RANDOMNESS**).
                We believe that these matrics, together with our task, might serve as a useful measure of individual differences and
                future research might attempt to determine whether specific values of these metrics are related to other psychological 
                or psychopathological characteristics.
                ''')
    # Describe the metrics
    st.markdown('### Description of the proposed metrics')
    
                
    
    
    ##########################
    # PLOT FIGURE 1
    ##########################
    
    # DESCRIBE BIAS
    st.markdown('''**BIAS (blue in the figures)** Bias reflects whether a given participant has the tendency to underestimate 
                their subjective level of control, to overestimate it or to display no bias. 
                It is calculated as the sum of differences between objective and reported 
                level of control (and then standardized so the values fall between -1 and 1). 
                Negative values indicate the tendency to underestimate one's level of control, with BIAS=-1 indicating that participant always 
                reported experiencing no control at all. Positive values reflect the tendency to overestimate it, with BIAS=1 indicating 
                that the participant always reported being fully in control. When BIAS=0 had equal tendency to under- and overestimate
                their level of control. 
                ''')
    # PLOT
    fig1, ax1 = plt.subplots()
    fig1.set_size_inches(10, 10)
    # Plot all participants
    sns.boxplot(data=df_long_metrics, x='sorting_order', y='Bias', color="blue")
    #sns.scatterplot(data=df_long_metrics, x='sorting_order', y='SumDev')
    # sns.lineplot(data=df_long_metrics, x='sorting_order', y='SumAbsDev', color="orange", alpha=0.5)
    # sns.scatterplot(data=results_indiv_score_sorted, x='sorting_order', y='SumAbsDev', color="orange", alpha=0.5)
    # sns.lineplot(data=df_long_metrics, x='sorting_order', y='ChanceRatio', color="green", alpha=0.5)
    # sns.scatterplot(data=results_indiv_score_sorted, x='sorting_order', y='ChanceRatio', color="green", alpha=0.5)
    plt.plot([0,40], [0,0], color='gray')
    plt.xlabel("Participant (sorted by average BIAS)")
    plt.ylabel("BIAS metric")
    plt.ylim([-1,1])
    plt.xticks([])
    
    # Tell streamlit to plot it
    st.pyplot(fig1)
    st.markdown(f"***Figure 1.*** *Boxplot of the values of BIAS from each participant. The data is sorted according to the level of BIAS. Data from Experiment {selected_exp_num}*. ")
    
    
    #########################
    # PLOT FIGURE 2
    #########################
    
    # DESCRIBE ACCURACY
    st.markdown('''**ACCURACY (orange in the figures)** Accuracy reflects how close participants' responses were to the objective 
                level of control. 
                It is calculated as 1 minus the sum of **absolute** differences between 
                objective and reported level of control (and then standardized so the values fall between 0 and 1). 
                It reflects how accurately participants' subjective reports tracked the objective level of control.
                It can be also understood as how close (on average) participants' responses were to the objective level of control. 
                The value of 1 means that their responses were always the same as objective level of control, while the value of 0 
                reflects the largest possible difference between objective control and subjective reports. 
                ''')
    # PLOT
    fig2, ax2 = plt.subplots()
    fig2.set_size_inches(10, 10)
    # Plot all participants
    sns.lineplot(data=df_long_metrics, x='sorting_order', y='Bias', color="blue", alpha=0.2, err_kws={"alpha": 0.2})
    sns.scatterplot(data=df_avg_metrics, x='sorting_order', y='Bias', color="blue", alpha=0.3)
    #sns.scatterplot(data=df_long_metrics, x='sorting_order', y='SumDev')
    sns.boxplot(data=df_long_metrics, x='sorting_order', y='Accuracy', color="orange") 
    #sns.scatterplot(data=df_avg_metrics, x='sorting_order', y='SumAbsDev', color="orange", alpha=0.5)
    # sns.lineplot(data=df_long_metrics, x='sorting_order', y='ChanceRatio', color="green", alpha=0.5)
    # sns.scatterplot(data=results_indiv_score_sorted, x='sorting_order', y='ChanceRatio', color="green", alpha=0.5)
    plt.plot([0,40], [0,0], color='gray')
    plt.xlabel("Participant (sorted by average BIAS)")
    plt.ylabel("ACCURACY metric")
    plt.ylim([-1,1])
    plt.xticks([])   
    # Tell streamlit to plot it
    st.pyplot(fig2)
    st.markdown(f"***Figure 2.*** *Boxplot of the values of ACCURACY from each participant. Blue line shows bias. The data is sorted according to the level of BIAS. Data from Experiment {selected_exp_num}*. ")
    
    
    #########################
    # PLOT FIGURE 3
    #########################
    
    # DESCRIBE RANDOMNESS
    st.markdown('''**RANDOMNESS (green in the figures)** Randomness reflects whether participants responses were closer to the pattern 
                expected if they responded randomly (in such case all of their responses should be around the value of 50%) or closer to the 
                objective level of control. This metric attempts to indicate which participants were likely not paying attention when doing the task 
                or are characterized by very high uncertainty when doing the task. 
                Positive values indicate that participant's responses were closer to always choosing 50% than to the objective level of control. 
                The value of RANDOMNESS=1 indicates that participants always responded by selecting 50% as their perceived level of control.
                The value of -1 indicates that participants' responses were always equal to the objective level of control. The value of 0 
                indicates that responses were equaly close to these two model situations.
                Participants with values close to 0 and higher are likely to be outliers who did not pay attention when doing the task.
                ''')
    # PLOT
    fig3, ax3 = plt.subplots()
    fig3.set_size_inches(10, 10)
    # Plot all participants
    sns.lineplot(data=df_long_metrics, x='sorting_order', y='Bias', color="blue", alpha=0.2, err_kws={"alpha": 0.2})
    sns.scatterplot(data=df_avg_metrics, x='sorting_order', y='Bias', color="blue", alpha=0.3)
    sns.lineplot(data=df_long_metrics, x='sorting_order', y='Accuracy', color="orange", alpha=0.2, err_kws={"alpha": 0.2}) #, alpha=0.5)
    sns.scatterplot(data=df_avg_metrics, x='sorting_order', y='Accuracy', color="orange", alpha=0.3)
    sns.boxplot(data=df_long_metrics, x='sorting_order', y='ChanceRatio', color="green") #, alpha=0.5)
    #sns.scatterplot(data=results_indiv_score_sorted, x='sorting_order', y='ChanceRatio', color="green", alpha=0.5)
    plt.plot([0,40], [0,0], color='gray')
    plt.xlabel("Participant (sorted by average BIAS)")
    plt.ylabel("RANDOMNESS metric")
    plt.ylim([-1,1])
    plt.xticks([])   
    # Tell streamlit to plot it
    st.pyplot(fig3)
    st.markdown(f"***Figure 3.*** *Boxplot of the values of RANDOMNESS from each participant. Blue line shows bias, orange line shows accuracy. The data is sorted according to the level of BIAS. Data from Experiment {selected_exp_num}*. ")
    
    
    
#%% BUTTON = "Explore data from each participant"
    
if button_radio == 'Explore data from each participant':
    
    st.title(f"Selected experiment: {selected_selectbox}")
    st.title("Explore data from each participant")
    st.markdown('''This section allows to investigate one-by-one the data from each participant. The participants 
                are sorted according to their response bias: from the largest negative bias (participants who strongly 
                underestimated their level of control) to the highest positive bias (participants who most strongly overestimated it). 
                It also presents data from participants that were excluded based on the preregistered exclusion criteria.
                ''')
    st.markdown('''The plot below displays participant's responses for each level of control. The blue line corresponds to the average 
                responses across all blocks and the red line reflects the responses in Block 1. Moreover, in the top left corner 
                we display metrics calculated for each participant (see the previous section of the dashboard for explanation) as well as 
                correlations between the first and the remaining blocks.
                ''')
                
    # PREPARE DATA
    selected_exp_num = selected_selectbox[11:]
    url_link = f'DATA_MeasuringSoA_long_Exp{selected_exp_num}.csv'
    df = pd.read_csv(url_link)
    # Adjust the control_level to be on the 0-100 scale 
    df['control_level'] = df['control_level']*100
    # Do not exclude participants - it's an overview of all of them
    data_all = df.copy()
    
        
    ################################
    # Calculate metrics for all subjects
    ################################

    # Subset data
    data_1 = df.loc[df['block']==1,:]
    data_scores = []
    resp_b1 = data_1.loc[:, ['SubNum','control_level','response']] 
    resp_other = data_all.loc[:, ['SubNum','control_level','response']] 
    resp_other_mean = resp_other.groupby(['SubNum', 'control_level']).mean().reset_index()
    data_scores = pd.merge(resp_b1, resp_other_mean, how='inner', on=['SubNum','control_level'], suffixes=('_B1', '_All'))
    # Correlations within each participant
    list_subNums = data_all['SubNum'].unique()
    results_indiv_score = []
    df_long_metrics = []
    for subject in list_subNums:
        # Scores for averages of all blocks: sum_abs_dev 
        score_sum_dev = np.sum(data_scores.loc[data_scores['SubNum']==subject,'response_All'] - data_scores.loc[data_scores['SubNum']==subject,'control_level']) / np.sum(data_scores.loc[data_scores['SubNum']==subject,'control_level'])
        score_sum_abs_dev = np.sum(np.abs(data_scores.loc[data_scores['SubNum']==subject,'control_level'] - data_scores.loc[data_scores['SubNum']==subject,'response_All'])) / np.sum(data_scores.loc[data_scores['SubNum']==subject,'control_level'])
        score_dev_from_chance = np.sum(np.abs(50 - data_scores.loc[data_scores['SubNum']==subject,'response_All'])) / 550 # automatize it!!!!!!!
        score_chance_ratio = score_sum_abs_dev/(score_dev_from_chance+score_sum_abs_dev) * 2 - 1
        # Scores based on just block 1
        score_b1_sum_dev = np.sum(data_scores.loc[data_scores['SubNum']==subject,'response_B1'] - data_scores.loc[data_scores['SubNum']==subject,'control_level']) / np.sum(data_scores.loc[data_scores['SubNum']==subject,'control_level'])
        score_b1_sum_abs_dev = np.sum(np.abs(data_scores.loc[data_scores['SubNum']==subject,'control_level'] - data_scores.loc[data_scores['SubNum']==subject,'response_B1'])) / np.sum(data_scores.loc[data_scores['SubNum']==subject,'control_level'])
        score_b1_dev_from_chance = np.sum(np.abs(50 - data_scores.loc[data_scores['SubNum']==subject,'response_B1'])) /550
        score_b1_chance_ratio = score_b1_sum_abs_dev/(score_b1_dev_from_chance+score_sum_abs_dev) * 2 - 1
        # Combine into a dataset
        results_indiv_score = results_indiv_score + [[subject, score_sum_dev, score_sum_abs_dev, score_dev_from_chance, score_chance_ratio,
                                                      score_b1_sum_dev, score_b1_sum_abs_dev, score_b1_dev_from_chance, score_b1_chance_ratio ]]
        # Scores for each block
        score_list_sum_dev = []
        score_list_sum_abs_dev = []
        score_list_dev_from_chance = []
        score_list_chance_ratio = []
        for block_num in range(5):
            data_scores_block = df.loc[df['block']==block_num+1,['SubNum','control_level','response']]           
            block_sum_dev = np.sum(data_scores_block.loc[data_scores_block['SubNum']==subject,'response'] - data_scores_block.loc[data_scores_block['SubNum']==subject,'control_level']) / np.sum(data_scores_block.loc[data_scores_block['SubNum']==subject,'control_level'])
            block_sum_abs_dev = np.sum(np.abs(data_scores_block.loc[data_scores_block['SubNum']==subject,'control_level'] - data_scores_block.loc[data_scores_block['SubNum']==subject,'response'])) / np.sum(data_scores_block.loc[data_scores_block['SubNum']==subject,'control_level'])
            block_dev_from_chance = np.sum(np.abs(50 - data_scores_block.loc[data_scores_block['SubNum']==subject,'response'])) / 550 # automatize it!!!!!!!
            block_chance_ratio = block_sum_abs_dev/(block_dev_from_chance+block_sum_abs_dev) * 2 - 1
            df_long_metrics = df_long_metrics + [[ subject, block_num+1, block_sum_dev, block_sum_abs_dev, block_dev_from_chance, block_chance_ratio]]

    # Combine it altogether into a dataframe
    results_indiv_score = pd.DataFrame(results_indiv_score)
    results_indiv_score = results_indiv_score.rename(columns={0:"SubNum", 1:"SumDev", 2:"SumAbsDev", 3:"DevFromChance", 4:"ChanceRatio", 
                                                              5: "B1_SumDev", 6:"B1_SumAbsDev", 7:"B1_DevFromChance", 8:"B1_ChanceRatio"})
    # Sort participants
    results_indiv_score_sorted = results_indiv_score.sort_values('SumDev')
    #results_indiv_score_sorted['SubNum'] = results_indiv_score_sorted['SubNum'].astype(str)
    results_indiv_score_sorted = results_indiv_score_sorted.reset_index().reset_index().rename(columns={"level_0":"sorting_order"})
    # Check correlation between the variables:
    results_ind_corrs = results_indiv_score.corr()
    # Build the long dataset and add sorting
    df_long_metrics = pd.DataFrame(df_long_metrics, columns=["SubNum", "block", "SumDev", "SumAbsDev", "DevFromChance", "ChanceRatio"])
    df_long_metrics["Bias"] = df_long_metrics["SumDev"]
    df_long_metrics["Accuracy"] = 1 - df_long_metrics["SumAbsDev"]
    df_long_metrics = df_long_metrics.merge(results_indiv_score_sorted.loc[:,["SubNum","sorting_order"]], how="outer", left_on="SubNum", right_on="SubNum")
    # Calculate averages based on df_long_metrics
    df_avg_metrics = df_long_metrics.groupby("SubNum").mean().reset_index().drop("block", axis=1)
    df_avg_metrics["sorting_order"] = df_avg_metrics["sorting_order"] + 1

    ########################
    # Calculate correlations
    ########################
    
    # Correlation between block 1 and the rest
    block_num = 1
    blocks_other = [2, 3, 4, 5]
    # Subset data
    data_1 = df.loc[(df['block']==block_num),:]
    data_other = df.loc[(df['block'].isin(blocks_other)),:]
    data_corr = []
    resp_b1 = data_1.loc[:, ['SubNum','control_level','response']] 
    resp_other = data_other.loc[:, ['SubNum','control_level','response']] 
    resp_other_mean = resp_other.groupby(['SubNum', 'control_level']).mean().reset_index()
    data_corr = pd.merge(resp_b1, resp_other_mean, how='inner', on=['SubNum','control_level'], suffixes=('_B1', '_Other'))
    # General correlation
    results_general_corr = data_corr[['response_B1','response_Other']].corr().iloc[0,1]
    
    # Correlations within each participant
    sub_num_list = data_corr['SubNum'].unique()
    results_correlations_subnum = []
    for sub_num in sub_num_list:
        res_corr = data_corr.loc[data_corr['SubNum'] == sub_num,['response_B1','response_Other']].corr() #[0,1]
        results_correlations_subnum = results_correlations_subnum + [[sub_num, res_corr.iloc[0,1]]]
    results_correlations_subnum = pd.DataFrame(results_correlations_subnum)
    results_correlations_subnum.columns = ["participant", "r_correlation"]
    

    ############################
    # PLOT 1: DATA FROM EACH INDIVIDUAL PARTICIPANT
    ############################
    
    # Check the number of subjects and apply to the slider
    number_of_subjects = df['SubNum'].max()
    order_num = st.slider('Select participant (sorted from highest negative to highest positive bias)', 1, number_of_subjects)
    
    # Dictionary linking subject number and the order basd on BIAS
    dict_order2subnum = df_avg_metrics.set_index("sorting_order")["SubNum"].to_dict()
    sub_num = dict_order2subnum[order_num]
    
    # Subset data to get just Block 1
    data_B1 = data_all.loc[data_all['block']==1,:]
    sub_metrics = df_avg_metrics.loc[df_avg_metrics['SubNum']==sub_num,:]
    
    # PLOT SPECIFIC PARTICIPANT
    fig1, ax1 = plt.subplots()
    fig1.set_size_inches(10, 9)
    sns.lineplot(data=data_all.loc[data_all['SubNum'] == sub_num,:], x='control_level', y='response', color="blue", alpha=0.5, legend=False)
    #sns.lineplot(data=data_B1.loc[data_all['SubNum'] == sub_num,:], x='control_level', y='response', color="red", alpha=1, legend=False)
    plt.xlim([0,100])
    plt.ylim([0,100])
    plt.xlabel("Objective level of control")
    plt.ylabel("Subjective level of control")
    plt.plot([0,100],[0,100], color='black')
    # Add text annotations
    if df.loc[df["SubNum"]==sub_num,"Include"].iloc[0] == 0:
        plt.text(5, 96, "PARTICIPANT EXCLUDED", fontsize=16, color="red")
    num_value = sub_metrics['SumDev'].iloc[0]
    plt.text(5, 92, f"Participant number: {sub_num}, {selected_selectbox}", fontsize=16, color="black")
    num_value = sub_metrics['Bias'].iloc[0]
    plt.text(5, 88, f"BIAS: {num_value:.02}", fontsize=12, color="black")
    num_value = sub_metrics['Accuracy'].iloc[0]
    plt.text(5, 85, f"ACCURACY: {num_value:.02}", fontsize=12, color="black")
    num_value = sub_metrics['ChanceRatio'].iloc[0]
    plt.text(5, 82, f"RANDOMNESS: {num_value:.02}", fontsize=12, color="black")
    num_value = results_correlations_subnum.loc[results_correlations_subnum["participant"]==sub_num,'r_correlation'].iloc[0]
    plt.text(5, 78, f"Block1-Rest correlation: {num_value:.02}", fontsize=12, color="black")

    # Tell streamlit to plot it
    st.pyplot(fig1)
    st.markdown(f"***Figure 1.*** *Average responses (blue line) and responses from Block 1 (red line) for each objective level of control. Data from Experiment {selected_exp_num}*. ")
    
    



#%% TEST

# 