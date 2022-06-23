import pandas as pd
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


data = pd.read_csv('gdc_data.csv')
data['file_type'] = data.file_name.apply(lambda x: x.split('-')[5][:2])
data['section_type'] = data.file_type.apply(lambda x: {
    'DX': 'Biopsy', 'TS': 'Frozen', 'BS': 'Frozen', 'MS': 'Unknown'}[x])

section_type='Biopsy'  # Frozen or Biopsy

idxx=data['section_type']==section_type

data=data.loc[idxx,:]

data = data.reset_index(drop=True)

data['health_type'] = data.sample_type.apply(lambda x: 'Healthy' 
                                             if x == 'Solid Tissue Normal' else 'Unhealthy')

data = data[data.health_type == 'Unhealthy']



data['case_id'] = data.file_name.apply(lambda x: '-'.join(x.split('-')[:3]))

data['file_type'] = data.file_name.apply(lambda x: x.split('-')[5][:2])
data['section_type'] = data.file_type.apply(lambda x: {'TS':'Frozen', 
                                                       'BS': 'Frozen', 
                                                       'DX': 'Biopsy', 
                                                       'MS': 'Unknown' }[x])


data = data.reset_index(drop=True)



new_data = data
new_data.rename(columns={'primary_diagnosis': 'primary_diagnosis_x'}, inplace=True)
new_data.rename(columns={'primary_site': 'primary_site_x'}, inplace=True)

abbrevations = '''LAML 	Acute Myeloid Leukemia
ACC 	Adrenocortical carcinoma
BLCA 	Bladder Urothelial Carcinoma
LGG 	Brain Lower Grade Glioma
BRCA 	Breast invasive carcinoma
CESC 	Cervical squamous cell carcinoma and endocervical adenocarcinoma
CHOL 	Cholangiocarcinoma
LCML 	Chronic Myelogenous Leukemia
COAD 	Colon adenocarcinoma
CNTL 	Controls
ESCA 	Esophageal carcinoma
FPPP 	FFPE Pilot Phase II
GBM 	Glioblastoma multiforme
HNSC 	Head and Neck squamous cell carcinoma
KICH 	Kidney Chromophobe
KIRC 	Kidney renal clear cell carcinoma
KIRP 	Kidney renal papillary cell carcinoma
LIHC 	Liver hepatocellular carcinoma
LUAD 	Lung adenocarcinoma
LUSC 	Lung squamous cell carcinoma
DLBC 	Lymphoid Neoplasm Diffuse Large B-cell Lymphoma
MESO 	Mesothelioma
MISC 	Miscellaneous
OV 	Ovarian serous cystadenocarcinoma
PAAD 	Pancreatic adenocarcinoma
PCPG 	Pheochromocytoma and Paraganglioma
PRAD 	Prostate adenocarcinoma
READ 	Rectum adenocarcinoma
SARC 	Sarcoma
SKCM 	Skin Cutaneous Melanoma
STAD 	Stomach adenocarcinoma
TGCT 	Testicular Germ Cell Tumors
THYM 	Thymoma
THCA 	Thyroid carcinoma
UCS 	Uterine Carcinosarcoma
UCEC 	Uterine Corpus Endometrial Carcinoma
UVM 	Uveal Melanoma'''
abbrevations_map = {abbr.split('\t')[1].strip().lower(): abbr.split('\t')[0].strip() 
 for abbr in abbrevations.split('\n')}

new_data['primary_diagnosis_abbrv'] = new_data.\
    primary_diagnosis_x.apply(lambda x:abbrevations_map[x.lower()])


tumor_type_mapping = {
    "AML": "Haematopoietic",
    "DLBC": "Haematopoietic",
    "THYM": "Haematopoietic",
    
    "CESC": "Gynaecological",
    "UCS": "Gynaecological",
    "UCEC": "Gynaecological",
    "OV": "Gynaecological",
    
    "BLCA": "Urinary tract",
    "KICH": "Urinary tract",
    "KIRC": "Urinary tract",
    "KIRP": "Urinary tract",
    
    "PRAD": "Prostate/testis",
    "TGCT": "Prostate/testis",
    
    "THCA": "Endocrine",
    "ACC": "Endocrine",
    "PCPG": "Endocrine",
    
    "BRCA": "Breast",
    
    "ESCA": "Gastrointestinal tract",
    "STAD": "Gastrointestinal tract",
    "COAD": "Gastrointestinal tract",
    "READ": "Gastrointestinal tract",
    
    "CHOL": "Liver, pancreaticobiliary",
    "LIHC": "Liver, pancreaticobiliary",
    "PAAD": "Liver, pancreaticobiliary",
    
    "LUSC": "Pulmonary",
    "LUAD": "Pulmonary",
    "MESO": "Pulmonary",
    
    "HNSC": "Head and neck",
    
    "SKCM": "Melanocytic malignancies",
    "UVM": "Melanocytic malignancies",
    
    "LGG": "Brain",
    "GBM": "Brain",
    
    "SARC": "Mesenchymal"
} 

new_data['tumor_type'] = new_data.primary_diagnosis_abbrv.apply(
    lambda x: tumor_type_mapping[x])


new_data = new_data.loc[(new_data["tumor_type"] != 'Breast') & (new_data["tumor_type"] != 'Mesenchymal')&(new_data["tumor_type"] != 'Head and neck')]

thresh=int(.6*new_data.shape[0])
train_new_data=new_data.iloc[0:thresh,:]
test_new_data=new_data.iloc[thresh:,:]


train_new_data = train_new_data.reset_index(drop=True)
test_new_data = test_new_data.reset_index(drop=True)



case_id_maps = []
for i, el in new_data.iterrows():
    case_id_maps.append(list(new_data[new_data.case_id == el.case_id].index))

new_data['case_grp'] = case_id_maps


train_case_id_maps = []
for i, el in train_new_data.iterrows():
    train_case_id_maps.append(list(train_new_data[train_new_data.case_id == el.case_id].index))

train_new_data['case_grp'] = train_case_id_maps

test_case_id_maps = []
for i, el in test_new_data.iterrows():
    test_case_id_maps.append(list(test_new_data[test_new_data.case_id == el.case_id].index))

test_new_data['case_grp'] = test_case_id_maps



## create primary diagnosis labels per WSI
pd_dict = {pd: i for i, pd in enumerate(train_new_data.primary_diagnosis_x.unique())}
one_hot_encoder = np.eye(len(train_new_data.primary_diagnosis_x.unique()))

train_labels = [one_hot_encoder[pd_dict[diagnosis]] for diagnosis in train_new_data.primary_diagnosis_x]
train_labels=np.array([lbl.tolist() for lbl in train_labels])
train_labels=train_labels.astype(np.float16)

test_labels = [one_hot_encoder[pd_dict[diagnosis]] for diagnosis in test_new_data.primary_diagnosis_x]
test_labels=np.array([lbl.tolist() for lbl in test_labels])
test_labels=test_labels.astype(np.float16)



## create features and primary site labels per WSI
def read_features(file_path):
    file_path=os.getcwd().replace(os.sep, '/') +file_path.split('entire_gdc_index')[1]
    #file_path=file_path.replace('/','\\')
    with open(file_path.replace('_barcode', '_feature'), 'r') as f:
        features = f.read().split('\n')
        arr = np.array([list(map(float, feature.strip().split(' '))) 
                        for feature in features if feature.strip()])
    
    return arr




#features = [read_features(b_file) for b_file in new_data.barcode_file]
train_features = [read_features(b_file) for b_file in train_new_data.barcode_file]
test_features = [read_features(b_file) for b_file in test_new_data.barcode_file]


#features_label=[ps for ps in data.primary_site] ###### conditioning on primary site
#features_label=[ps for ps in new_data.tumor_type] ###### conditioning on tumor type

train_features_label=[ps for ps in train_new_data.tumor_type] ###### conditioning on tumor type
test_features_label=[ps for ps in test_new_data.tumor_type] ###### conditioning on tumor type

# features_label is ps label

#np_features=np.array(features)
#np_features_label=np.array(features_label)


train_features=np.array(train_features)
test_features=np.array(test_features)
train_features_label=np.array(train_features_label)
test_features_label=np.array(test_features_label)



#Create flattened primary diagnosis labels
train_flat_pd_onehot_labels=[]
test_flat_pd_onehot_labels=[]
for i,fe in enumerate(train_features):
    train_flat_pd_onehot_labels.append(train_labels[i]*np.ones((len(fe),1)))
    
for i,fe in enumerate(test_features):
    test_flat_pd_onehot_labels.append(test_labels[i]*np.ones((len(fe),1))) 
    
    
train_flat_pd_onehot_labels = [item for sublist in train_flat_pd_onehot_labels for item in sublist]
train_flat_pd_onehot_labels=np.array(train_flat_pd_onehot_labels)
train_flat_pd_onehot_labels=train_flat_pd_onehot_labels.astype(np.float16)
train_flat_pd_onehot_labels.shape



test_flat_pd_onehot_labels = [item for sublist in test_flat_pd_onehot_labels for item in sublist]
test_flat_pd_onehot_labels=np.array(test_flat_pd_onehot_labels)
test_flat_pd_onehot_labels=test_flat_pd_onehot_labels.astype(np.float16)
test_flat_pd_onehot_labels.shape



#Create flattened features from list to np array

train_flat_features = [item for sublist in train_features for item in sublist]
train_flat_features=np.array(train_flat_features)
train_flat_features=train_flat_features.astype(np.float16)


test_flat_features = [item for sublist in test_features for item in sublist]
test_flat_features=np.array(test_flat_features)
test_flat_features=test_flat_features.astype(np.float16)



from sklearn.preprocessing import StandardScaler,MinMaxScaler
scaler=StandardScaler()
scaler.fit(train_flat_features)

train_flat_features=scaler.transform(train_flat_features)
test_flat_features=scaler.transform(test_flat_features)


from sklearn.preprocessing import OneHotEncoder, LabelEncoder

le=LabelEncoder()
ps_train_num_label=le.fit_transform(train_features_label)
ps_test_num_label=le.transform(test_features_label)


ps_train_flat_num_label=[]
for i,fe in enumerate(train_features):
    for j in range(len(fe)):
        ps_train_flat_num_label.append(ps_train_num_label[i])
        
        
ps_test_flat_num_label=[]
for i,fe in enumerate(test_features):
    for j in range(len(fe)):
        ps_test_flat_num_label.append(ps_test_num_label[i])
        
        
ps_train_flat_num_label=np.array(ps_train_flat_num_label)

ps_test_flat_num_label=np.array(ps_test_flat_num_label)
ps_test_flat_num_label.shape


ps_train_flat_num_label=ps_train_flat_num_label.reshape(ps_train_flat_num_label.shape[0],1)
ps_test_flat_num_label=ps_test_flat_num_label.reshape(ps_test_flat_num_label.shape[0],1)

enc=OneHotEncoder(sparse=False)
enc.fit(ps_train_flat_num_label)





ps_train_flat_onehot_label=enc.transform(ps_train_flat_num_label)
ps_train_flat_num_label=ps_train_flat_num_label.astype(dtype=np.float16)
ps_train_flat_onehot_label=ps_train_flat_onehot_label.astype(dtype=np.float16)

ps_test_flat_onehot_label=enc.transform(ps_test_flat_num_label)
ps_test_flat_num_label=ps_test_flat_num_label.astype(dtype=np.float16)
ps_test_flat_onehot_label=ps_test_flat_onehot_label.astype(dtype=np.float16)




ps_train_num_label=ps_train_num_label.reshape(ps_train_num_label.shape[0],1)
ps_train_onehot_num_label=enc.transform(ps_train_num_label)
ps_train_onehot_num_label=ps_train_onehot_num_label.astype(dtype=np.float16)

ps_test_num_label=ps_test_num_label.reshape(ps_test_num_label.shape[0],1)
ps_test_onehot_num_label=enc.transform(ps_test_num_label)
ps_test_onehot_num_label=ps_test_onehot_num_label.astype(dtype=np.float16)





vae={}

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon




D = train_flat_features.shape[1]
latent_dim = 50
alpha=0.00001 



feature_inputs = tf.keras.Input(shape=(D,))
cond_input=tf.keras.Input(shape=(ps_train_flat_onehot_label.shape[1],))
encoder_inputs = tf.keras.layers.concatenate([feature_inputs, cond_input],  axis=1)



x = layers.Dense(512,activation="selu",kernel_regularizer=tf.keras.regularizers.l2(l=0.1))(feature_inputs) # feature_inputs and encoder_inputs are option
x = layers.Dropout(0.25)(x)
x = layers.Dense(256, activation="selu",kernel_regularizer=tf.keras.regularizers.l2(l=0.1))(x)
x = layers.Dropout(0.25)(x)
x = layers.Dense(128, activation="selu",kernel_regularizer=tf.keras.regularizers.l2(l=0.1))(x)
x = layers.Dropout(0.25)(x)


pd_classifier_output = layers.Dense(train_flat_pd_onehot_labels.shape[1], activation="softmax", name='pd_out')(x)

z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

z = Sampling()([z_mean, z_log_var])


z_cond=tf.concat([z, cond_input], axis=1)
z_cond2=tf.concat([z_cond, pd_classifier_output], axis=1)



encoder = keras.Model([feature_inputs,cond_input], [z_mean, z_log_var, z_cond2,pd_classifier_output], name="encoder")
encoder.summary()


latent_inputs = tf.keras.Input(shape=(latent_dim+ps_train_flat_onehot_label.shape[1]+train_flat_pd_onehot_labels.shape[1],))

x = layers.Dense(128, activation="selu",kernel_regularizer=tf.keras.regularizers.l2(l=0.1))(latent_inputs)
x = layers.Dropout(0.25)(x)
x = layers.Dense(256, activation="selu",kernel_regularizer=tf.keras.regularizers.l2(l=0.1))(x)
x = layers.Dropout(0.25)(x)
x = layers.Dense(512, activation="selu",kernel_regularizer=tf.keras.regularizers.l2(l=0.1))(x)
x = layers.Dropout(0.25)(x)



decoder_outputs= layers.Dense(D , activation="linear")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()
cce = tf.keras.losses.CategoricalCrossentropy()

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.pd_classification_loss_tracker = keras.metrics.Mean(name="classification_loss")
        self.rec_kl_class_loss_tracker = keras.metrics.Mean(name="rec_kl_class_loss")
        self.grad_loss_tracker = keras.metrics.Mean(name="grad_loss")
        self.epoch_accuracy_tracker = tf.keras.metrics.CategoricalAccuracy(name="classification_accuracy")

        
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.pd_classification_loss_tracker,
            self.rec_kl_class_loss_tracker,
            self.grad_loss_tracker,
            self.epoch_accuracy_tracker
        ]

    def train_step(self, data):
        x, y = data
        print(tf.shape(y))
        just_feature=x[0]
        grad_loss=0
        total_grad=[]
        B=[]

        with tf.GradientTape() as tape2:
                
            with tf.GradientTape() as tape1:
                z_mean, z_log_var, z_cond,  predicted_pd = self.encoder(x)



                reconstruction = self.decoder(z_cond)
                reconstruction_loss = tf.reduce_mean(
                        tf.keras.losses.MSE(just_feature, reconstruction))

                classification_loss = tf.reduce_mean(cce(y,predicted_pd))

                kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
                kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
                rec_kl_class_loss = 2*reconstruction_loss +  .1*kl_loss + .5*classification_loss
                
            main_loss_grads = tape1.gradient(rec_kl_class_loss, self.trainable_weights)
            
            print('grad_loss:',grad_loss)
            for ii in range(len(main_loss_grads)):
                B=tf.sign(main_loss_grads[ii])
                grad_loss=grad_loss+ tf.norm(tf.math.subtract(main_loss_grads[ii],B),ord=2)      

                
                #grad_loss=grad_loss+ tf.norm(main_loss_grads[ii],ord=1)    
        
        total_loss=alpha*grad_loss + rec_kl_class_loss

        grad_of_gradloss=tape2.gradient(grad_loss, self.trainable_weights)



        for jj in range(len(grad_of_gradloss)):
            total_grad.append(alpha*grad_of_gradloss[jj]+main_loss_grads[jj])
            


        self.optimizer.apply_gradients(zip(total_grad, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.grad_loss_tracker.update_state(grad_loss)
        self.rec_kl_class_loss_tracker.update_state(rec_kl_class_loss)
        self.pd_classification_loss_tracker.update_state(classification_loss)
        self.epoch_accuracy_tracker.update_state(y,predicted_pd)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "grad_loss":self.grad_loss_tracker.result(),
            "classification_loss":self.pd_classification_loss_tracker.result(),
            "classification_accuracy":self.epoch_accuracy_tracker.result(),
           
        }
    
    
    def test_step(self, data):
        xx, yy = data
        just_feature=xx[0]
        #print(tf.shape(yy))
        
        total_grad=[]
        
        
        z_mean, z_log_var, z_cond,  val_predicted_pd = self.encoder(xx)

        reconstruction = self.decoder(z_cond)
        val_reconstruction_loss = tf.reduce_mean(
                tf.keras.losses.MSE(just_feature, reconstruction))  #+tf.reduce_mean(tf.keras.losses.MAE(just_feature, reconstruction))
        #reconstruction_loss = tf.reduce_mean(keras.losses.binary_crossentropy(data, reconstruction))
        val_classification_loss = tf.reduce_mean(cce(yy,val_predicted_pd))

        val_kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        val_kl_loss = tf.reduce_mean(tf.reduce_sum(val_kl_loss, axis=1))
        val_rec_kl_class_loss = 2*val_reconstruction_loss +  .1*val_kl_loss + .5*val_classification_loss

        
        self.reconstruction_loss_tracker.update_state(val_reconstruction_loss)
        self.pd_classification_loss_tracker.update_state(val_classification_loss)
        self.epoch_accuracy_tracker.update_state(yy,val_predicted_pd)
        self.kl_loss_tracker.update_state(val_kl_loss)
        return {
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "classification_loss":self.pd_classification_loss_tracker.result(),
            "classification_accuracy":self.epoch_accuracy_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }    




vae = VAE(encoder, decoder)


vae.compile(optimizer=keras.optimizers.Adam(lr=.0001))
history_alpha_non_0=vae.fit([train_flat_features,ps_train_flat_onehot_label],train_flat_pd_onehot_labels, epochs=150, batch_size=256, validation_data=([test_flat_features,ps_test_flat_onehot_label],test_flat_pd_onehot_labels))





def fisher_vector(bag_of_patch_features_primarysite, vae):
    
    bag_of_patch_features,ps=bag_of_patch_features_primarysite
    
    
    with tf.GradientTape() as tape:
        z_mean, z_log_var, z, pd = vae.encoder(bag_of_patch_features_primarysite)
        reconstruction = vae.decoder(z)
        
        reconstruction_loss = tf.reduce_mean(
                    tf.keras.losses.MSE(bag_of_patch_features, reconstruction)) #+tf.reduce_mean(tf.keras.losses.MAE(bag_of_patch_features, reconstruction))
        

    grads = tape.gradient(reconstruction_loss, vae.trainable_weights)


    mygrads=[]

    for i in range(len(grads)):

        try:
            tmp=grads[i].numpy()
            tmp=tmp.astype(np.float16)
        except:

            continue

        tmp=np.matrix.flatten(tmp)

        mygrads.extend(list(tmp))
    return mygrads


# Delete unnecessary variables to free some memory
del train_flat_features
del ps_train_flat_onehot_label
del train_flat_pd_onehot_labels
del test_flat_features
del ps_test_flat_onehot_label
del test_flat_pd_onehot_labels

del ps_train_onehot_num_label
del ps_train_num_label
del ps_test_num_label


del ps_test_flat_num_label
del ps_train_flat_num_label


# seperate data based on tumortype
pr_s=train_new_data.loc[:,'tumor_type'].unique()# tumor_type


features_dic={}
conditioned_data={}

for i ,j in enumerate(pr_s):
    temp=[]
    idx=train_new_data.loc[:,'tumor_type']==j  
    for ff in train_new_data.loc[idx,'barcode_file']:
        temp.append(read_features(ff))
    
    features_dic[j]=temp

selected_idx={}
Dim_of_fisher=40000
for i,j in enumerate(pr_s): 
    print(i)
    tmp=[]
   
    
    for ii in range(len(features_dic[j])):
        tempo=np.array(features_dic[j][ii])
        tempo=tempo.astype(np.float16)        
        
        tempo=scaler.transform(tempo)
        
        label=enc.transform([le.transform([j])])*np.ones((tempo.shape[0],1))                
        tmp.append(fisher_vector([tempo,label],vae))    
    
    
    tmp=np.array(tmp)
    tmp=tmp.astype(dtype=np.float32)
    stds=np.var(tmp,axis=0)
    idxxx=np.argsort(-1*stds)
    selected_idx['idx_'+j]=idxxx[0:Dim_of_fisher]     

fisher_feature = []

for i in range(len(test_features)):    
    print(i)

    
    label=ps_test_onehot_num_label[i]*np.ones((test_features[i].shape[0],1))

    
    tempo=test_features[i]
    tempo=tempo.astype(np.float16)
    tempo=scaler.transform(tempo)
    
    tmp=fisher_vector([tempo,label],vae)
    tmp=np.array(tmp)
    tmp=tmp.astype(dtype=np.float16)
    

    
    #tmp=tmp[selected_idx['idx_'+test_features_label[i]]]   #Uncomment if you want to reduce the dimensionality
    
    fisher_feature.append(tmp)      


fisher_feature = np.array(fisher_feature)

from sklearn.preprocessing import normalize



fisher_feature=np.sign(fisher_feature)*(np.abs(fisher_feature)**.7)
fisher_feature=normalize(fisher_feature)    


import timeit
start_time = timeit.default_timer()
from scipy.spatial.distance import pdist, squareform
dist_mat=pdist(fisher_feature,metric='hamming')


dist_mat=dist_mat.astype(dtype=np.float16)
dist_mat = squareform(dist_mat)

for i in range(dist_mat.shape[0]):
    dist_mat[i, test_case_id_maps[i]] = np.float('inf')

searcheable_indxs = np.argsort(dist_mat, axis=1)

top_n = 3

search_indxs = searcheable_indxs[:, :top_n]


search_results = np.array(test_new_data.primary_site_x)[search_indxs]


from collections import Counter

def get_winner(lst):
    b = Counter(lst)
    prev_freq = -1
    winner = []
    for el, freq in b.most_common():
        if prev_freq == -1 or prev_freq == freq:
            winner.append(el)
            prev_freq = freq
            
    return winner

winners = [get_winner(lst) for lst in search_results]

prediction_correct = []
for i, ps in enumerate(test_new_data.primary_site_x.tolist()):
    prediction_correct.append(ps in winners[i])
    
test_new_data['correct_ps_prediction'] = np.array(prediction_correct)*1.

test_new_data.groupby('primary_site_x').agg({'correct_ps_prediction': 'mean'})



for i in range(dist_mat.shape[0]):
    dist_mat[i, test_case_id_maps[i]] = np.float('inf')
    
pss = test_new_data.tumor_type.unique()

ps_id_map = {ps:list(test_new_data[test_new_data.tumor_type == ps].index) for ps in pss}


all_index = set(list(test_new_data.index))



for i in range(dist_mat.shape[0]):
    this_ps = test_new_data.loc[i].tumor_type
    
    dist_mat[i, list(all_index - set(ps_id_map[this_ps])) ] = np.float('inf')
    
    
    
searcheable_indxs = np.argsort(dist_mat, axis=1)

search_indxs = searcheable_indxs[:, :top_n]

search_results = np.array(test_new_data.primary_diagnosis_x)[search_indxs]

from collections import Counter

def get_winner(lst):
    b = Counter(lst)
    prev_freq = -1
    winner = []
    for el, freq in b.most_common():
        if prev_freq == -1 or prev_freq == freq:
            winner.append(el)
            prev_freq = freq
            
    return winner

winners = [get_winner(lst) for lst in search_results]


import pandas as pd
from sklearn.metrics import classification_report

ground_truth = list(test_new_data.primary_diagnosis_x)
predictions = [g if g in ws else ws[0]  for g, ws in zip(ground_truth, winners)]


report = classification_report(ground_truth, predictions, output_dict=True)
recall_df = pd.DataFrame(report).transpose()

winners = np.array(winners)

prediction_correct = []
for i, ps in enumerate(test_new_data.primary_diagnosis_x.tolist()):
    prediction_correct.append(ps in winners[i])
    

    
test_new_data['correct_pd_prediction'] = np.array(prediction_correct)*1.

test_new_data.groupby('tumor_type').agg({'correct_pd_prediction': 'mean'})


result_df = test_new_data.groupby('primary_diagnosis_x').agg({'correct_pd_prediction': 'mean', 'tumor_type': 'max'}).reset_index()\
        .sort_values(by='tumor_type')

result_df.set_index(['tumor_type', 'primary_diagnosis_x'])

result_df['recall'] = result_df.primary_diagnosis_x.apply(lambda x: recall_df.loc[x]['recall'])
result_df['precision'] = result_df.primary_diagnosis_x.apply(lambda x: recall_df.loc[x]['precision'])
result_df['f1_score'] = result_df.primary_diagnosis_x.apply(lambda x: recall_df.loc[x]['f1-score'])

#result_df.set_index('primary_diagnosis_x').set_index('tumor_type')
result_df.set_index(['tumor_type', 'primary_diagnosis_x'])

print(result_df)         