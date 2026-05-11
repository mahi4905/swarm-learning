############################################################################
## Deep K-Means for Network Intrusion Detection using NSL-KDD Dataset
## Swarm Learning Example
############################################################################

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# ── Environment variables (Swarm injects these) ──────────────────────────
data_dir  = os.getenv('DATA_DIR',  '/platform/swarmml/data')
model_dir = os.getenv('MODEL_DIR', '/platform/swarmml/model')

# ── Column names for NSL-KDD ─────────────────────────────────────────────
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

# ── Attack family mapping ─────────────────────────────────────────────────
attack_map = {
    'normal': 'normal',
    # DoS
    'back': 'dos', 'land': 'dos', 'neptune': 'dos', 'pod': 'dos',
    'smurf': 'dos', 'teardrop': 'dos', 'mailbomb': 'dos',
    'apache2': 'dos', 'processtable': 'dos', 'udpstorm': 'dos',
    # Probe
    'ipsweep': 'probe', 'nmap': 'probe', 'portsweep': 'probe',
    'satan': 'probe', 'mscan': 'probe', 'saint': 'probe',
    # R2L
    'ftp_write': 'r2l', 'guess_passwd': 'r2l', 'imap': 'r2l',
    'multihop': 'r2l', 'phf': 'r2l', 'spy': 'r2l', 'warezclient': 'r2l',
    'warezmaster': 'r2l', 'sendmail': 'r2l', 'named': 'r2l',
    'snmpgetattack': 'r2l', 'snmpguess': 'r2l', 'xlock': 'r2l',
    'xsnoop': 'r2l', 'httptunnel': 'r2l',
    # U2R
    'buffer_overflow': 'u2r', 'loadmodule': 'u2r', 'perl': 'u2r',
    'rootkit': 'u2r', 'ps': 'u2r', 'sqlattack': 'u2r', 'xterm': 'u2r'
}

# ── Load data ─────────────────────────────────────────────────────────────
print("Loading dataset...")
train_df = pd.read_csv(os.path.join(data_dir, 'KDDTrain+.txt'),
                       names=col_names)

print(f"Dataset shape: {train_df.shape}")
print(f"\nLabel distribution:")
print(train_df['label'].value_counts())

# ── Map attacks to 5 families ─────────────────────────────────────────────
train_df['attack_family'] = train_df['label'].map(attack_map)
train_df['attack_family'] = train_df['attack_family'].fillna('unknown')

print(f"\nAttack family distribution:")
print(train_df['attack_family'].value_counts())

# ── Save labels separately (for evaluation only) ──────────────────────────
labels = train_df['attack_family'].values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)
print(f"\nEncoded classes: {label_encoder.classes_}")

# ── Drop label and difficulty columns ─────────────────────────────────────
train_df = train_df.drop(['label', 'difficulty', 'attack_family'], axis=1)

# ── One-hot encode categorical columns ───────────────────────────────────
categorical_cols = ['protocol_type', 'service', 'flag']
numeric_cols = [c for c in train_df.columns if c not in categorical_cols]

print(f"\nCategorical columns: {categorical_cols}")
print(f"Numeric columns count: {len(numeric_cols)}")

preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'),
     categorical_cols),
    ('num', StandardScaler(), numeric_cols)
])

X = preprocessor.fit_transform(train_df)
print(f"\nFinal feature shape after preprocessing: {X.shape}")
print("Preprocessing complete!")
