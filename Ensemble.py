import argparse
import numpy as np
from skopt import gp_minimize
import pandas as pd
import pickle

def objective(weights):
    pred = np.zeros_like(r[0])

    for i in range(len(weights)):
        pred += r[i] * weights[i]

    pred = pred.argmax(axis=1)

    correct = (pred == label).sum()
    acc = correct / len(label)
    print(acc)
    return -acc

def open_file(file):

    if file.find('te') ==-1:
        fr = open(file, 'rb')
        inf = pickle.load(fr)
        df = pd.DataFrame(inf)
        df = pd.DataFrame(df.values.T, index=df.columns, columns=df.index)
        score = df.values
    else:
        score = np.load(file, allow_pickle=True)
    return score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='multi-stream ensemble')
    parser.add_argument(
        '--ctr_joint',
        type=str,
        default='mix_data/val/ctr_J_val_42.55.pkl'),  # ctr_joint
    parser.add_argument(
        '--ctr_bone',
        type=str,
        default='mix_data/val/ctr_B_val_43.10.pkl'),  # ctr_bone
    parser.add_argument(
        '--ctr_joint_bone',
        type=str,
        default='mix_data/val/ctr_JB_val_42.60.pkl'),  # ctr_joint_bone

    parser.add_argument(
        '--ctr_A',
        type=str,
        default='mix_data/val/ctr_A_val_33.80.pkl'),

    parser.add_argument(
        '--ctr_jm',
        type=str,
        default='mix_data/val/ctr_jm_val_35.85.pkl'),
    parser.add_argument(
        '--ctr_bm',
        type=str,
        default='mix_data/val/ctr_bm_val_35.80.pkl'),
    parser.add_argument(
        '--ctr_jbm',
        type=str,
        default='mix_data/val/ctr_jbm_val_36.30.pkl'),

    parser.add_argument(
        '--td_joint',
        type=str,
        default='mix_data/val/td_j_val_44.20.npy'),  # td_joint
    parser.add_argument(
        '--td_bone',
        type=str,
        default='mix_data/val/td_b_val_43.80.npy'),  # td_bone
    parser.add_argument(
        '--td_joint_bone',
        type=str,
        default='mix_data/val/td_jb_val_44.npy'),  # td_joint_bone
    parser.add_argument(
        '--td_A',
        type=str,
        default='mix_data/val/td_A_val_36.70.npy'),

    parser.add_argument(
        '--td_jm',
        type=str,
        default='mix_data/val/td_jm_val_34.05.npy'),
    parser.add_argument(
        '--td_bm',
        type=str,
        default='mix_data/val/td_bm_val_35.60.npy'),
    parser.add_argument(
        '--td_jbm',
        type=str,
        default='mix_data/val/td_jbm_val_35.80.npy'),

    parser.add_argument(
        '--mst_j',
        type=str,
        default='mix_data/val/mst_j_val_42.65.npy'),
    parser.add_argument(
        '--mst_b',
        type=str,
        default='mix_data/val/mst_b_val_40.30.npy'),
    parser.add_argument(
        '--mst_jb',
        type=str,
        default='mix_data/val/mst_jb_val_41.85.npy'),
    parser.add_argument(
        '--mst_A',
        type=str,
        default='mix_data/val/mst_A_val_34.2.npy'),
    parser.add_argument(
        '--mst_jm',
        type=str,
        default='mix_data/val/mst_jm_val_35.60.npy'),
    parser.add_argument(
        '--mst_bm',
        type=str,
        default='mix_data/val/mst_bm_val_36.15.npy'),
    parser.add_argument(
        '--mst_jbm',
        type=str,
        default='mix_data/val/mst_jbm_val_37.7.npy'),

    parser.add_argument(
        '--te_joint',
        type=str,
        default='mix_data/val/te_j_val_42.65.npy'),  # te_joint
    parser.add_argument(
        '--te_bone',
        type=str,
        default='mix_data/val/te_b_val_42.25.npy'),  # te_bone
    parser.add_argument(
        '--te_joint_bone',
        type=str,
        default='mix_data/val/te_jb_val_43.50.npy'),  # te_joint_bone
    parser.add_argument(
        '--te_A',
        type=str,
        default='mix_data/val/te_A_val_36.40.npy'),
    parser.add_argument(
        '--te_jm',
        type=str,
        default='mix_data/val/te_jm_val_34.60.npy'),
    parser.add_argument(
        '--te_bm',
        type=str,
        default='mix_data/val/te_bm_val_33.20.npy'),
    parser.add_argument(
        '--te_jbm',
        type=str,
        default='mix_data/val/te_jbm_val_34.25.npy'),

    # former参数
    parser.add_argument(
        '--former_joint',
        type=str,
        default='mix_data/val/former_j_val_43.80.pkl'),
    parser.add_argument(
        '--former_bone',
        type=str,
        default='mix_data/val/former_b_val_43.55.pkl'),
    parser.add_argument(
        '--former_k2',
        type=str,
        default='mix_data/val/former_k2_val_43.85.pkl'),
    parser.add_argument(
        '--former_joint_motion',
        type=str,
        default='mix_data/val/former_jm_val_35.60.pkl'),
    parser.add_argument(
        '--former_bone_motion',
        type=str,
        default='mix_data/val/former_bm_val_35.40.pkl'),
    parser.add_argument(
        '--former_k2_motion',
        type=str,
        default='mix_data/val/former_k2m_val_35.00.pkl'),

    arg = parser.parse_args()

    label = np.load('val_label.npy')

    r = []

    r.append(open_file(arg.ctr_joint))

    r.append(open_file(arg.ctr_bone))

    r.append(open_file(arg.ctr_joint_bone))

    r.append(open_file(arg.ctr_jm))

    r.append(open_file(arg.ctr_bm))

    r.append(open_file(arg.ctr_jbm))

    r.append(open_file(arg.td_joint))

    r.append(open_file(arg.td_bone))

    r.append(open_file(arg.td_joint_bone))

    r.append(open_file(arg.td_jm))

    r.append(open_file(arg.td_bm))

    r.append(open_file(arg.td_jbm))

    r.append(open_file(arg.mst_j))

    r.append(open_file(arg.mst_b))

    r.append(open_file(arg.mst_jb))

    r.append(open_file(arg.mst_jm))

    r.append(open_file(arg.mst_bm))

    r.append(open_file(arg.mst_jbm))

    r.append(open_file(arg.te_joint))

    r.append(open_file(arg.te_bone))

    r.append(open_file(arg.te_joint_bone))

    r.append(open_file(arg.te_jm))

    r.append(open_file(arg.te_bm))

    r.append(open_file(arg.te_jbm))

    r.append(open_file(arg.ctr_A))

    r.append(open_file(arg.td_A))

    r.append(open_file(arg.mst_A))

    r.append(open_file(arg.te_A))

    r.append(open_file(arg.former_joint))

    r.append(open_file(arg.former_bone))

    r.append(open_file(arg.former_k2))

    r.append(open_file(arg.former_joint_motion))

    r.append(open_file(arg.former_bone_motion))

    r.append(open_file(arg.former_k2_motion))

    space = [(0.2, 1.2) for i in range(34)]

    result = gp_minimize(objective, space, n_calls=200, random_state=1)

    print('Maximum accuracy: {:.4f}%'.format(-result.fun * 100))
    print('Optimal weights: {}'.format(result.x))

    final_pred = np.zeros_like(r[0])

    for i in range(len(result.x)):
        final_pred += r[i] * result.x[i]

    np.save('ensemble_score.npy', final_pred)
