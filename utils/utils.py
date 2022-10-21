# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import seaborn as sns

class utilities:
    
    def __init__(self):
        self.activity_id = {0:'rope jumping', 1:'lying', 2:'sitting',
                      3:'standing', 4:'walking', 5:'running', 6:'cycling',
                      7:'Nordic walking', 8:'asceding stairs', 9:'descending stairs',
                      10:'vaccum cleaning', 11:'ironing'}
        
        self.pamap2_list = self.activity_id.values()
        
        self.protocol_acts = [0,1,2,3,4,5,6,7,8,9,10,11]
        
    def confusion_matrix(self,predict_labels,real_labels):
        pred_results = {}
        base_dict = {}
        
        for i in self.protocol_acts:
            base_dict[i] = 0
        
        for i in self.protocol_acts:
            pred_results[i] = base_dict.copy()
            
        for pl,tl in list(zip(predict_labels,  real_labels)):
            pred_results[pl][tl]+=1
        
        self.pred_results_df = pd.DataFrame(pred_results)
        
        self.pred_results_df.columns=[self.activity_id[x] for x in self.protocol_acts]
        self.pred_results_df.index = [self.activity_id[x] for x in self.protocol_acts]
        
        # return self.pred_results_df
    

    def save_conf_mat(self,path):
        cm = self.pred_results_df.astype('float') / self.pred_results_df.sum(axis=1)[:, np.newaxis]
        df_cm = pd.DataFrame(cm,dtype="float", index=self.pamap2_list, 
                             columns=self.pamap2_list)
        sns.set(font_scale=1.4)
        sns.set(rc={'figure.figsize':(15,15)})
        image = sns.heatmap(df_cm,cmap="YlGnBu", annot=True,fmt=".2f", annot_kws={"size": 16})
        image.get_figure().savefig(path)
