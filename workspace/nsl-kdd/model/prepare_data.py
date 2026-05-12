############################################################################
## prepare_data.py — Creates 5 biased non-IID node splits
############################################################################
import os
import numpy as np
import pandas as pd

data_dir = os.getenv('DATA_DIR', '/platform/swarmml/data')

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

print("Loading dataset...")
df = pd.read_csv(os.path.join(data_dir, 'KDDTrain+.txt'),
                 names=col_names, low_memory=False)

# map to 5 families — keep label column for nsl_kdd.py to use
df['label'] = df['label'].map(attack_map).fillna('unknown')

# drop only difficulty
df = df.drop(['difficulty'], axis=1)

# separate by family
normal = df[df['label'] == 'normal']
dos    = df[df['label'] == 'dos']
probe  = df[df['label'] == 'probe']
r2l    = df[df['label'] == 'r2l']
u2r    = df[df['label'] == 'u2r']

print(f"Normal:{len(normal)} DoS:{len(dos)} Probe:{len(probe)} R2L:{len(r2l)} U2R:{len(u2r)}")

def make_node(dominant, others, dominant_ratio=0.65, node_size=20000):
    dominant_n = int(node_size * dominant_ratio)
    other_n    = node_size - dominant_n
    dominant_sample = dominant.sample(n=min(dominant_n, len(dominant)), random_state=42)
    other_parts = []
    per_other = other_n // len(others)
    for o in others:
        other_parts.append(o.sample(n=min(per_other, len(o)), random_state=42))
    node_df = pd.concat([dominant_sample] + other_parts)
    return node_df.sample(frac=1, random_state=42).reset_index(drop=True)

node1 = make_node(normal, [dos, probe, r2l, u2r])
node2 = make_node(dos,    [normal, probe, r2l, u2r])
node3 = make_node(probe,  [normal, dos, r2l, u2r])
node4 = make_node(r2l,    [normal, dos, probe, u2r])
node5 = make_node(u2r,    [normal, dos, probe, r2l])

for i, node in enumerate([node1, node2, node3, node4, node5], 1):
    out_path = os.path.join(data_dir, f'node{i}')
    os.makedirs(out_path, exist_ok=True)
    # save WITH header, WITH label column
    node.to_csv(os.path.join(out_path, 'KDDTrain+.txt'), index=False)
    print(f"Node {i}: {len(node)} rows | label dist: {node['label'].value_counts().to_dict()}")

print("\nAll node splits created successfully!")
