'''
@PackageName: TE-GCN-main - ronghe_gpu_6.py
@author: Weizhetao
@since 2024/10/12 17:20
'''

import torch
import pickle
import argparse
import numpy as np
import pandas as pd


def get_parser():
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
        default='mix_data/val/ctr_JB_val_42.60.pkl'),    # ctr_joint_bone

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
        default='mix_data/val/te_jb_val_43.50.npy'),   # te_joint_bone
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

    #former参数
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


    parser.add_argument(
        '--val_sample',
        type=str,
        default='val_label.npy'),

    return parser


def Cal_Score(File, Rate, ntu60XS_num, Numclass):
    final_score = torch.zeros(ntu60XS_num, Numclass).cuda()
    for idx, file in enumerate(File):
        #关于te的npy需要特殊处理
        if file.find('te') == -1:
            fr = open(file, 'rb')
            inf = pickle.load(fr)
            df = pd.DataFrame(inf)
            df = pd.DataFrame(df.values.T, index=df.columns, columns=df.index)
            score = torch.tensor(data=df.values, device='cuda')
        else:
            inf = np.load(file, allow_pickle=True)
            score = torch.tensor(data=inf, device='cuda')
        final_score += Rate[idx] * score
    return final_score


def Cal_Acc(final_score, true_label):
    wrong_index = []
    _, predict_label = torch.max(final_score, 1)
    for index, p_label in enumerate(predict_label):
        if p_label!= true_label[index]:
            wrong_index.append(index)

    wrong_num = np.array(wrong_index).shape[0]
    total_num = true_label.shape[0]
    Acc = (total_num - wrong_num) / total_num
    return Acc


def gen_label(val_npy_path):
    true_label = np.load(val_npy_path)
    return torch.from_numpy(true_label).cuda()


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    # Mix_GCN Score File
    ctr_j = args.ctr_joint
    ctr_b = args.ctr_bone
    ctr_jb = args.ctr_joint_bone
    ctr_A = args.ctr_A
    ctr_jm=args.ctr_jm
    ctr_bm = args.ctr_bm
    ctr_jbm = args.ctr_jbm

    td_j = args.td_joint
    td_b = args.td_bone
    td_jb = args.td_joint_bone
    td_A = args.td_A
    td_jm = args.td_jm
    td_bm = args.td_bm
    td_jbm = args.td_jbm

    mst_j=args.mst_j
    mst_b = args.mst_b
    mst_jb = args.mst_jb
    mst_A = args.mst_A
    mst_jm = args.mst_jm
    mst_bm = args.mst_bm
    mst_jbm = args.mst_jbm

    te_j = args.te_joint
    te_b = args.te_bone
    te_jb = args.te_joint_bone
    te_A = args.te_A
    te_jm=args.te_jm
    te_bm = args.te_bm
    te_jbm = args.te_jbm

    # former
    former_joint = args.former_joint
    former_bone = args.former_bone
    former_k2 = args.former_k2
    former_joint_motion = args.former_joint_motion
    former_bone_motion = args.former_bone_motion
    former_k2_motion = args.former_k2_motion

    val_npy_path = args.val_sample

    File = [ctr_j, ctr_b, ctr_jb,
            ctr_jm, ctr_bm, ctr_jbm,

            td_j, td_b, td_jb,
            td_jm, td_bm, td_jbm,

            mst_j, mst_b, mst_jb,
            mst_jm, mst_bm, mst_jbm,

            te_j, te_b, te_jb,
            te_jm, te_bm, te_jbm,

            ctr_A, td_A,mst_A, te_A,

            former_joint, former_bone, former_k2,
            former_joint_motion, former_bone_motion,former_k2_motion

            ]

    best_acc=0

    Numclass = 155
    Sample_Num = 2000
    most_acc = 0.01
    best_rate = None
    # Rate=[1.1136316307090173, 1.2, 1.2, 1.2, 0.657831076391617, 1.2, 1.2, 1.2, 0.2, 0.2, 0.2, 0.47594429049588594, 0.541482720033379, 1.2, 0.34408644315747755, 0.2356160419253492, 0.2, 1.0397240225156603]
    # Rate=[0.2, 0.5725989373366909, 0.8163957496293421, 1.2, 1.2, 0.5652162330294188, 1.2, 1.2, 0.2, 1.095305744466307, 0.46106536506028134, 1.2, 0.7055282229382498, 0.33992074166768904, 0.2887626162336633, 0.7854354135055648, 0.8831192334915743, 1.2, 0.726232384279272, 0.2, 0.2, 0.9760195316621711]
    Rate=[0.2, 0.2, 1.0044082220696935, 0.6140449624234021, 0.4378685586399178, 0.5762757486983721, 1.2, 1.2, 0.8267967489415304, 0.2, 0.6178611916616141, 0.5808327288434987, 0.2, 0.2, 0.40951185317191635, 0.4478994237345658, 1.2, 0.2, 0.2, 1.2, 0.8573175670402788, 0.7287741978336104, 0.2, 1.2, 0.5821752488199325, 0.21236058808087854, 0.41940823897400215, 1.2, 0.5979987115597925, 0.9823094728841417, 1.2, 0.41644639439411574, 1.2, 0.2]

    final_score = Cal_Score(File, Rate, Sample_Num, Numclass)
    true_label = gen_label(val_npy_path)
    Acc = Cal_Acc(final_score, true_label)
    # np.save('pred1.npy', final_score.detach().cpu().numpy())
    print(Acc)