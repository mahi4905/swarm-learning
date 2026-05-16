############################################################################
## evaluate.py — Test the federated Deep K-Means model
## Usage: python3 evaluate.py
############################################################################
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import normalized_mutual_info_score, accuracy_score
from scipy.optimize import linear_sum_assignment
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

# ── Paths ──────────────────────────────────────────────────────────────────
data_dir  = os.path.expanduser('~/swarm-learning/workspace/nsl-kdd/data')
model_dir = os.path.expanduser('~/swarm-learning/workspace/nsl-kdd/model/output')

# ── Column names ───────────────────────────────────────────────────────────
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

# ── Hungarian matching ─────────────────────────────────────────────────────
def hungarian_match(y_pred, y_true, n_clusters):
    """Match cluster IDs to true class IDs using Hungarian algorithm."""
    confusion = np.zeros((n_clusters, n_clusters), dtype=int)
    for p, t in zip(y_pred, y_true):
        if t < n_clusters:
            confusion[p][t] += 1
    row_ind, col_ind = linear_sum_assignment(-confusion)
    mapping = {r: c for r, c in zip(row_ind, col_ind)}
    return np.array([mapping.get(p, 0) for p in y_pred])

# ── ClusteringLayer (needed to load model) ─────────────────────────────────
class ClusteringLayer(layers.Layer):
    def __init__(self, n_clusters, embedding_dim, **kwargs):
        super().__init__(**kwargs)
        self.n_clusters    = n_clusters
        self.embedding_dim = embedding_dim

    def build(self, input_shape):
        self.clusters = self.add_weight(
            name='clusters',
            shape=(self.n_clusters, self.embedding_dim),
            initializer='glorot_uniform', trainable=True
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

# ══════════════════════════════════════════════════════════════════════════
# PART 1 — Evaluate on KDDTest+.txt
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("PART 1: EVALUATING ON KDDTest+.txt")
print("="*70)

# Load test data
test_df = pd.read_csv(os.path.join(data_dir, 'KDDTest+.txt'),
                      names=col_names, low_memory=False)

test_df['label'] = test_df['label'].map(attack_map).fillna('unknown')
true_labels = test_df['label'].values
test_df = test_df.drop(['label', 'difficulty'], axis=1)

# Preprocess
categorical_cols = ['protocol_type', 'service', 'flag']
numeric_cols     = [c for c in test_df.columns if c not in categorical_cols]

preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(
        categories=[PROTOCOL_CATS, SERVICE_CATS, FLAG_CATS],
        sparse_output=False, handle_unknown='ignore'
    ), categorical_cols),
    ('num', StandardScaler(), numeric_cols)
])

X_test = preprocessor.fit_transform(test_df)
print(f"Test data shape: {X_test.shape}")

# Encode true labels
le = LabelEncoder()
y_true = le.fit_transform(true_labels)
print(f"True classes: {le.classes_}")

# Load model
print("\nLoading federated model...")
model_path = os.path.join(model_dir, 'deep_kmeans_final.keras')
model = keras.models.load_model(
    model_path,
    custom_objects={'ClusteringLayer': ClusteringLayer}
)
print("Model loaded successfully!")

# Get predictions
print("Getting cluster assignments...")
outputs = model.predict(X_test, verbose=0)
if isinstance(outputs, list):
    q_pred = outputs[0]
else:
    q_pred = outputs

cluster_assignments = np.argmax(q_pred, axis=1)

# Hungarian matching
n_classes = len(le.classes_)
matched = hungarian_match(cluster_assignments, y_true, n_classes)

# Metrics
nmi = normalized_mutual_info_score(y_true, cluster_assignments)
acc = accuracy_score(y_true, matched)

print(f"\n{'='*70}")
print(f"RESULTS ON TEST SET ({len(X_test)} rows)")
print(f"{'='*70}")
print(f"NMI  (Normalized Mutual Information): {nmi:.4f}  (higher is better, max=1.0)")
print(f"ACC  (Clustering Accuracy):           {acc:.4f}  ({acc*100:.1f}%)")
print(f"{'='*70}")

# Per-class breakdown
print("\nPer-class cluster assignment breakdown:")
print(f"{'True Label':<12} {'Count':>8} {'Correctly Matched':>18} {'Accuracy':>10}")
print("-"*52)
for i, cls in enumerate(le.classes_):
    mask = y_true == i
    total = mask.sum()
    correct = (matched[mask] == i).sum()
    cls_acc = correct / total if total > 0 else 0
    print(f"{cls:<12} {total:>8} {correct:>18} {cls_acc*100:>9.1f}%")

# ══════════════════════════════════════════════════════════════════════════
# PART 2 — Test with your own custom rows
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("PART 2: PREDICTING ON CUSTOM ROWS")
print("="*70)

# Family names for output
family_names = {0: 'dos', 1: 'normal', 2: 'probe', 3: 'r2l', 4: 'u2r'}

def predict_row(row_dict, label="?"):
    """Predict attack family for a single network connection."""
    row_df = pd.DataFrame([row_dict])
    X_row  = preprocessor.transform(row_df)
    out    = model.predict(X_row, verbose=0)
    q      = out[0] if isinstance(out, list) else out
    probs  = q[0]
    cluster = np.argmax(probs)
    matched_class = family_names.get(cluster, f"cluster_{cluster}")
    print(f"\nRow: {label}")
    print(f"  Probabilities: " + " | ".join([f"C{i}:{p:.3f}" for i, p in enumerate(probs)]))
    print(f"  Predicted cluster: {cluster} → Attack family: {matched_class.upper()}")
    print(f"  Confidence: {probs.max()*100:.1f}%")
    return matched_class

# Example 1 — DoS attack (neptune-like)
print("\n--- Custom Row 1: DoS Attack (neptune-like) ---")
predict_row({
    'duration': 0, 'protocol_type': 'tcp', 'service': 'private',
    'flag': 'REJ', 'src_bytes': 0, 'dst_bytes': 0, 'land': 0,
    'wrong_fragment': 0, 'urgent': 0, 'hot': 0, 'num_failed_logins': 0,
    'logged_in': 0, 'num_compromised': 0, 'root_shell': 0, 'su_attempted': 0,
    'num_root': 0, 'num_file_creations': 0, 'num_shells': 0,
    'num_access_files': 0, 'num_outbound_cmds': 0, 'is_host_login': 0,
    'is_guest_login': 0, 'count': 229, 'srv_count': 10, 'serror_rate': 0.0,
    'srv_serror_rate': 0.0, 'rerror_rate': 1.0, 'srv_rerror_rate': 1.0,
    'same_srv_rate': 0.04, 'diff_srv_rate': 0.06, 'srv_diff_host_rate': 0.0,
    'dst_host_count': 255, 'dst_host_srv_count': 10, 'dst_host_same_srv_rate': 0.04,
    'dst_host_diff_srv_rate': 0.06, 'dst_host_same_src_port_rate': 0.0,
    'dst_host_srv_diff_host_rate': 0.0, 'dst_host_serror_rate': 0.0,
    'dst_host_srv_serror_rate': 0.0, 'dst_host_rerror_rate': 1.0,
    'dst_host_srv_rerror_rate': 1.0
}, label="DoS attack (neptune)")

# Example 2 — Normal traffic (http browsing)
print("\n--- Custom Row 2: Normal HTTP Traffic ---")
predict_row({
    'duration': 0, 'protocol_type': 'tcp', 'service': 'http',
    'flag': 'SF', 'src_bytes': 181, 'dst_bytes': 5450, 'land': 0,
    'wrong_fragment': 0, 'urgent': 0, 'hot': 0, 'num_failed_logins': 0,
    'logged_in': 1, 'num_compromised': 0, 'root_shell': 0, 'su_attempted': 0,
    'num_root': 0, 'num_file_creations': 0, 'num_shells': 0,
    'num_access_files': 0, 'num_outbound_cmds': 0, 'is_host_login': 0,
    'is_guest_login': 0, 'count': 8, 'srv_count': 8, 'serror_rate': 0.0,
    'srv_serror_rate': 0.0, 'rerror_rate': 0.0, 'srv_rerror_rate': 0.0,
    'same_srv_rate': 1.0, 'diff_srv_rate': 0.0, 'srv_diff_host_rate': 0.0,
    'dst_host_count': 9, 'dst_host_srv_count': 9, 'dst_host_same_srv_rate': 1.0,
    'dst_host_diff_srv_rate': 0.0, 'dst_host_same_src_port_rate': 0.11,
    'dst_host_srv_diff_host_rate': 0.0, 'dst_host_serror_rate': 0.0,
    'dst_host_srv_serror_rate': 0.0, 'dst_host_rerror_rate': 0.0,
    'dst_host_srv_rerror_rate': 0.0
}, label="Normal HTTP traffic")

# Example 3 — Port scan (portsweep-like)
print("\n--- Custom Row 3: Probe Attack (portsweep-like) ---")
predict_row({
    'duration': 1, 'protocol_type': 'tcp', 'service': 'private',
    'flag': 'REJ', 'src_bytes': 0, 'dst_bytes': 0, 'land': 0,
    'wrong_fragment': 0, 'urgent': 0, 'hot': 0, 'num_failed_logins': 0,
    'logged_in': 0, 'num_compromised': 0, 'root_shell': 0, 'su_attempted': 0,
    'num_root': 0, 'num_file_creations': 0, 'num_shells': 0,
    'num_access_files': 0, 'num_outbound_cmds': 0, 'is_host_login': 0,
    'is_guest_login': 0, 'count': 1, 'srv_count': 1, 'serror_rate': 0.0,
    'srv_serror_rate': 0.0, 'rerror_rate': 1.0, 'srv_rerror_rate': 1.0,
    'same_srv_rate': 1.0, 'diff_srv_rate': 0.0, 'srv_diff_host_rate': 0.0,
    'dst_host_count': 255, 'dst_host_srv_count': 1, 'dst_host_same_srv_rate': 0.0,
    'dst_host_diff_srv_rate': 0.01, 'dst_host_same_src_port_rate': 0.01,
    'dst_host_srv_diff_host_rate': 0.0, 'dst_host_serror_rate': 0.0,
    'dst_host_srv_serror_rate': 0.0, 'dst_host_rerror_rate': 1.0,
    'dst_host_srv_rerror_rate': 1.0
}, label="Probe attack (portsweep)")

print("\n" + "="*70)
print("EVALUATION COMPLETE")
print("="*70 + "\n")
