############################################################################
## Deep K-Means for Network Intrusion Detection — NSL-KDD
## Swarm Learning Example
############################################################################
import os
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

data_dir  = os.getenv('DATA_DIR',  '/platform/swarmml/data')
model_dir = os.getenv('MODEL_DIR', '/platform/swarmml/model')
max_epochs = int(os.getenv('MAX_EPOCHS', '100'))
min_peers  = int(os.getenv('MIN_PEERS',  '2'))

col_names = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
    'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
    'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
    'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'label', 'difficulty'
]

attack_map = {
    'normal': 'normal',
    'back': 'dos', 'land': 'dos', 'neptune': 'dos', 'pod': 'dos',
    'smurf': 'dos', 'teardrop': 'dos', 'mailbomb': 'dos',
    'apache2': 'dos', 'processtable': 'dos', 'udpstorm': 'dos',
    'ipsweep': 'probe', 'nmap': 'probe', 'portsweep': 'probe',
    'satan': 'probe', 'mscan': 'probe', 'saint': 'probe',
    'ftp_write': 'r2l', 'guess_passwd': 'r2l', 'imap': 'r2l',
    'multihop': 'r2l', 'phf': 'r2l', 'spy': 'r2l', 'warezclient': 'r2l',
    'warezmaster': 'r2l', 'sendmail': 'r2l', 'named': 'r2l',
    'snmpgetattack': 'r2l', 'snmpguess': 'r2l', 'xlock': 'r2l',
    'xsnoop': 'r2l', 'httptunnel': 'r2l',
    'buffer_overflow': 'u2r', 'loadmodule': 'u2r', 'perl': 'u2r',
    'rootkit': 'u2r', 'ps': 'u2r', 'sqlattack': 'u2r', 'xterm': 'u2r'
}

# ── FIXED categories from full dataset — ensures all nodes produce same shape ──
PROTOCOL_CATS = ['icmp', 'tcp', 'udp']
SERVICE_CATS  = ['IRC','X11','Z39_50','aol','auth','bgp','courier','csnet_ns',
                 'ctf','daytime','discard','domain','domain_u','echo','eco_i',
                 'ecr_i','efs','exec','finger','ftp','ftp_data','gopher',
                 'harvest','hostnames','http','http_2784','http_443','http_8001',
                 'imap4','iso_tsap','klogin','kshell','ldap','link','login',
                 'mtp','name','netbios_dgm','netbios_ns','netbios_ssn','netstat',
                 'nnsp','nntp','ntp_u','other','pm_dump','pop_2','pop_3',
                 'printer','private','red_i','remote_job','rje','shell','smtp',
                 'sql_net','ssh','sunrpc','supdup','systat','telnet','tftp_u',
                 'tim_i','time','urh_i','urp_i','uucp','uucp_path','vmnet','whois']
FLAG_CATS     = ['OTH','REJ','RSTO','RSTOS0','RSTR','S0','S1','S2','S3','SF','SH']

# ── Load data ──────────────────────────────────────────────────────────────
print("Loading dataset...")
file_path = os.path.join(data_dir, 'KDDTrain+.txt')

with open(file_path, 'r') as f:
    first_line = f.readline().strip()
has_header = first_line.startswith('duration')

if has_header:
    train_df = pd.read_csv(file_path, low_memory=False)
    labels   = train_df['label'].values
    train_df = train_df.drop(['label'], axis=1)
else:
    train_df = pd.read_csv(file_path, names=col_names, low_memory=False)
    train_df['label'] = train_df['label'].map(attack_map).fillna('unknown')
    labels   = train_df['label'].values
    train_df = train_df.drop(['label', 'difficulty'], axis=1)

print(f"Dataset shape: {train_df.shape}")
print("Label distribution:")
unique_l, counts_l = np.unique(labels, return_counts=True)
for u, c in zip(unique_l, counts_l):
    print(f"  {u}: {c}")

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)
print(f"Encoded classes: {label_encoder.classes_}")

# ── Preprocess with FIXED categories ──────────────────────────────────────
categorical_cols = ['protocol_type', 'service', 'flag']
numeric_cols     = [c for c in train_df.columns if c not in categorical_cols]

preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(
        categories=[PROTOCOL_CATS, SERVICE_CATS, FLAG_CATS],
        sparse_output=False,
        handle_unknown='ignore'
    ), categorical_cols),
    ('num', StandardScaler(), numeric_cols)
])

X = preprocessor.fit_transform(train_df)
print(f"Final feature shape: {X.shape}")
print("Preprocessing complete!\n")

# ── Build models ───────────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

input_dim    = X.shape[1]
encoding_dim = 10
n_clusters   = 5
batch_size   = 256

# Encoder
enc_input  = keras.Input(shape=(input_dim,), name='enc_input')
e          = layers.Dense(64, activation='relu')(enc_input)
e          = layers.Dense(32, activation='relu')(e)
embedding  = layers.Dense(encoding_dim, activation='linear', name='embedding')(e)
encoder    = Model(inputs=enc_input, outputs=embedding, name='encoder')

# AutoEncoder
d          = layers.Dense(32, activation='relu')(embedding)
d          = layers.Dense(64, activation='relu')(d)
recon_out  = layers.Dense(input_dim, activation='linear', name='recon_out')(d)
autoencoder = Model(inputs=enc_input, outputs=recon_out, name='autoencoder')

# ClusteringLayer
class ClusteringLayer(layers.Layer):
    def __init__(self, n_clusters, embedding_dim, **kwargs):
        super().__init__(**kwargs)
        self.n_clusters    = n_clusters
        self.embedding_dim = embedding_dim

    def build(self, input_shape):
        self.clusters = self.add_weight(
            name='clusters',
            shape=(self.n_clusters, self.embedding_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        z        = tf.expand_dims(inputs, axis=1)
        mu       = tf.expand_dims(self.clusters, axis=0)
        distance = tf.reduce_sum(tf.square(z - mu), axis=2)
        q        = 1.0 / (1.0 + distance)
        q        = q / tf.reduce_sum(q, axis=1, keepdims=True)
        return q

    def get_config(self):
        config = super().get_config()
        config.update({'n_clusters': self.n_clusters,
                       'embedding_dim': self.embedding_dim})
        return config

# Combined model
combined_input  = keras.Input(shape=(input_dim,), name='combined_input')
emb_out         = encoder(combined_input)
cluster_out     = ClusteringLayer(n_clusters, encoding_dim, name='clustering')(emb_out)
d2              = layers.Dense(32, activation='relu', name='dec1')(emb_out)
d2              = layers.Dense(64, activation='relu', name='dec2')(d2)
recon_out2      = layers.Dense(input_dim, activation='linear', name='recon')(d2)

combined_model  = Model(
    inputs=combined_input,
    outputs=[cluster_out, recon_out2],
    name='combined_model'
)

print(f"AutoEncoder input/output: {input_dim}")
autoencoder.summary()

# ── PHASE 1: AutoEncoder Pretraining ──────────────────────────────────────
print('\n' + '='*70)
print('PHASE 1: AUTOENCODER PRETRAINING')
print('='*70)

autoencoder.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse', metrics=['mae']
)

history = autoencoder.fit(
    X, X,
    epochs=50,
    batch_size=batch_size,
    validation_split=0.1,
    verbose=1,
    shuffle=True
)

print(f'\nFinal train loss : {history.history["loss"][-1]:.4f}')
print(f'Final val loss   : {history.history["val_loss"][-1]:.4f}')

os.makedirs(model_dir, exist_ok=True)
encoder.save(os.path.join(model_dir, 'pretrained_encoder.keras'))
print('Pretrained encoder saved.')

combined_model.get_layer('encoder').set_weights(encoder.get_weights())

print('\nGenerating embeddings...')
embeddings = encoder.predict(X, verbose=0)
print(f'Embeddings shape: {embeddings.shape}')

from sklearn.cluster import KMeans
print('Running K-Means to initialize cluster centers...')
kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
kmeans.fit(embeddings)
initial_centers = kmeans.cluster_centers_

unique_k, counts_k = np.unique(kmeans.labels_, return_counts=True)
print('Initial cluster sizes:')
for cid, cnt in zip(unique_k, counts_k):
    print(f'  Cluster {cid}: {cnt} ({cnt/len(embeddings)*100:.1f}%)')

combined_model.get_layer('clustering').set_weights([initial_centers])
print('Cluster centers loaded.')
print('='*70)
print('PHASE 1 COMPLETE')
print('='*70 + '\n')

# ── PHASE 2: Joint Training with SwarmCallback ─────────────────────────────
print('='*70)
print('PHASE 2: JOINT DEEP K-MEANS TRAINING WITH SWARM')
print('='*70)

def target_distribution(q):
    weight = q ** 2 / q.sum(axis=0)
    return (weight.T / weight.sum(axis=1)).T

combined_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss=['kld', 'mse'],
    loss_weights=[0.1, 1.0]
)

try:
    from swarmlearning.tf import SwarmCallback
    swarmCallback = SwarmCallback(
        syncFrequency=512,
        minPeers=min_peers,
        useAdaptiveSync=False,
        adsValData=(X[:100], X[:100]),
        adsValBatchSize=batch_size
    )
    swarmCallback.logger.setLevel(logging.DEBUG)
    callbacks = [swarmCallback]
    print(f'SwarmCallback enabled — syncFrequency=512, minPeers={min_peers}')
except Exception:
    callbacks = []
    print('SwarmCallback not available — running in local mode')

update_interval = 140
print('\nComputing initial P...')
q_pred, _ = combined_model.predict(X, verbose=0)
p = target_distribution(q_pred)
print(f'Q shape: {q_pred.shape} | P shape: {p.shape}')
print('\nStarting training...\n')

step = 0
for epoch in range(max_epochs):
    if step % update_interval == 0:
        q_pred, _ = combined_model.predict(X, verbose=0)
        p = target_distribution(q_pred)
        assignments = np.argmax(q_pred, axis=1)
        unique_e, counts_e = np.unique(assignments, return_counts=True)
        dist_str = ' | '.join([f'C{c}:{n}' for c, n in zip(unique_e, counts_e)])
        print(f'  [Step {step:4d}] Clusters: {dist_str}')

    h = combined_model.fit(X, [p, X], epochs=1,
                           batch_size=batch_size,
                           verbose=0,
                           callbacks=callbacks)
    step += len(X) // batch_size

    if (epoch + 1) % 10 == 0:
        print(f'  Epoch {epoch+1:3d}/{max_epochs} | Loss: {h.history["loss"][0]:.4f}')
        combined_model.save(os.path.join(model_dir, f'model_epoch_{epoch+1}.keras'))

final_path = os.path.join(model_dir, 'deep_kmeans_final.keras')
combined_model.save(final_path)
print(f'\nFinal model saved: {final_path}')

q_final, _ = combined_model.predict(X, verbose=0)
final_assignments = np.argmax(q_final, axis=1)
unique_f, counts_f = np.unique(final_assignments, return_counts=True)
print('\nFinal cluster distribution:')
for cid, cnt in zip(unique_f, counts_f):
    print(f'  Cluster {cid}: {cnt} ({cnt/len(X)*100:.1f}%)')

print('\n' + '='*70)
print('PHASE 2 COMPLETE')
print('='*70 + '\n')
