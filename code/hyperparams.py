import argparse

def set_args():
    parser = argparse.ArgumentParser()

    # =============== dataset prep ============== #
    parser.add_argument('--max_seq_len',  default=200,  type=int,   help='The max number of tokens in a dial')
    parser.add_argument('--spe_utt_turn', default=6,    type=int,   help='The number of turns in a dial for test->HE or Distinct')
    parser.add_argument('--min_utt_turn', default=4,    type=int,   help='The min number of utterances in a dial')
    parser.add_argument('--min_resp_len', default=5,    type=int,   help='The min number of tokens in a resp')
    parser.add_argument('--prof_ol',      default=0.85, type=float, help='The prof overlap rate->remove the same prof')

    # ============= common setting ============= #
    parser.add_argument('--log_path',        default='../save_load/log/log.log',                 type=str)
    parser.add_argument('--save_model_path', default='../save_load/model',                       type=str)
    parser.add_argument('--GPT2_tokenizer',  default='../save_load/pre_weight/gpt2-zh',          type=str)
    parser.add_argument('--Bert_tokenizer',  default='../save_load/pre_weight/bert-zh',          type=str)
    parser.add_argument('--GPT2_checkp',     default='../save_load/model/agent/epoch5',          type=str)
    parser.add_argument('--VarB_checkp',     default='../save_load/model/agent/epoch5/VarB.pth', type=str)
    parser.add_argument('--BiEs_checkp',     default='../save_load/model/rl/R1/epoch5',          type=str)
    parser.add_argument('--CaLM_checkp',     default='../save_load/model/rl/R2/epoch5/CaLM.pth', type=str)
    parser.add_argument('--AtEc_checkp',     default='../save_load/model/rl/R3/epoch5/AtEc.pth', type=str)
    parser.add_argument('--Sele_checkp',     default='../save_load/model/rl/RS/epoch3000',       type=str)

    parser.add_argument('--device',     default='0',    type=str)
    parser.add_argument('--forwa_only', default=False,  type=bool)
    parser.add_argument('--batch_size', default=32,     type=int)
    parser.add_argument('--epochs',     default=30,     type=int)
    parser.add_argument('--lr',         default=2.6e-5, type=float)
    parser.add_argument('--warm_step',  default=3000,   type=int)
    parser.add_argument('--n_worker',   default=2,      type=int)
    parser.add_argument('--patience',   default=3,      type=int,  help='The number of patience times')
    parser.add_argument('--early_stop', default=False,  type=bool, help='Whether to set an early stop, patience needs to be set if used')
    parser.add_argument('--niter_save', default=1,      type=int,  help='Save model after set epoch, early_stop needs to be set to False if used')
    parser.add_argument('--n_generat',  default=5,      type=int,  help='The number of resps for selecting, should>=1, for selector, inference, interact')

    # =============== train agent =============== #
    parser.add_argument('--GPT2_pretrained', default='../save_load/pre_weight/gpt2-zh',             type=str)
    parser.add_argument('--GPT2_config',     default='../save_load/pre_weight/gpt2-zh/config.json', type=str)
    parser.add_argument('--train_dial_path', default='../dataset/agent/train/dial-train.txt',       type=str)
    parser.add_argument('--train_cont_path', default='../dataset/agent/train/cont-train.txt',       type=str)
    parser.add_argument('--train_prof_path', default='../dataset/agent/train/prof-train.txt',       type=str)
    parser.add_argument('--train_resp_path', default='../dataset/agent/train/resp-train.txt',       type=str)
    parser.add_argument('--train_keyw_path', default='../dataset/agent/train/keyw-train.txt',       type=str)
    parser.add_argument('--valid_dial_path', default='../dataset/agent/valid/dial-valid.txt',       type=str)
    parser.add_argument('--valid_cont_path', default='../dataset/agent/valid/cont-valid.txt',       type=str)
    parser.add_argument('--valid_prof_path', default='../dataset/agent/valid/prof-valid.txt',       type=str)
    parser.add_argument('--valid_resp_path', default='../dataset/agent/valid/resp-valid.txt',       type=str)
    parser.add_argument('--valid_keyw_path', default='../dataset/agent/valid/keyw-valid.txt',       type=str)

    parser.add_argument('--pos_class', default=3350, type=int,   help='The positive class recall weight')
    parser.add_argument('--n_layer',   default=3,    type=int,   help='The number of transformer layers in Var model')
    parser.add_argument('--n_head',    default=8,    type=int,   help='The number of transformer heads in Var model')
    parser.add_argument('--d_model',   default=1024, type=int,   help='The embedd size of Var model')
    parser.add_argument('--dis_obj',   default=0.15, type=float, help='The distinction objective')

    # =============== train reward ============== #
    parser.add_argument('--BiEs_pretrained', default='../save_load/pre_weight/bert-zh', type=str)
    parser.add_argument('--rl_trda_path',    default='../dataset/rl/trda-rl.txt',       type=str)
    parser.add_argument('--rl_vada_path',    default='../dataset/rl/vada-rl.txt',       type=str)

    parser.add_argument('--rl_code',     default="R3", type=str, help='R1->P(S,T) R2->P(T) R3->AtEc')
    parser.add_argument('--clm_n_head',  default=8,    type=int, help='The number of transformer heads in P(T) model')
    parser.add_argument('--clm_d_model', default=512,  type=int, help='The embedd size of the P(T) model')
    parser.add_argument('--ate_d_model', default=512,  type=int, help='The embedd size of the AtEc model')

    # ============== train selector ============= #
    parser.add_argument('--Sele_pretrained', default='../save_load/pre_weight/bert-zh', type=str)
    parser.add_argument('--rl_sali_path',    default='../dataset/rl/sali-rl.txt',       type=str)
    parser.add_argument('--rl_stin_path',    default='../dataset/rl/stin-rl.txt',       type=str)

    parser.add_argument('--n_episode', default=3000, type=int,   help='The number of episodes of RL')
    parser.add_argument('--n_step',    default=10,   type=int,   help='The number of actions in an episode of RL')
    parser.add_argument('--n_prompt',  default=2,    type=int,   help='The number of conts are considered')
    parser.add_argument('--gamma',     default=0.9,  type=float, help='Affect decay rate of reward')

    # ========== inference & interactio ========= #
    parser.add_argument('--save_resu_path', default='../save_load/result',                 type=str)
    parser.add_argument('--inte_stin_path', default='../dataset/interact/dain-it.txt',     type=str)
    parser.add_argument('--test_dial_path', default='../dataset/agent/test/dial-test.txt', type=str)
    parser.add_argument('--test_cont_path', default='../dataset/agent/test/cont-test.txt', type=str)
    parser.add_argument('--test_prof_path', default='../dataset/agent/test/prof-test.txt', type=str)
    parser.add_argument('--test_resp_path', default='../dataset/agent/test/resp-test.txt', type=str)
    parser.add_argument('--test_keyw_path', default='../dataset/agent/test/keyw-test.txt', type=str)

    parser.add_argument('--max_resp_len',   default=50,      type=int)
    # parser.add_argument('--ignore_tks_ids', default=[1312, 198, 628, 44320], type=list)
    parser.add_argument('--ignore_tks_ids', default=[42467], type=list)
    parser.add_argument('--repeti_penalty', default=2.0,     type=float)
    parser.add_argument('--temperature',    default=1.0,     type=float)
    parser.add_argument('--topp',           default=0.7,     type=float)
    parser.add_argument('--topk',           default=4,       type=int)
    parser.add_argument('--n_dial',         default=10,      type=int)
    parser.add_argument('--n_turn',         default=5,       type=int)

    # =========== va & re & it analysis ========= #
    parser.add_argument('--var_poste_save', default='../dataset/analysis/posterior',   type=str)
    parser.add_argument('--var_prior_save', default='../dataset/analysis/prior',       type=str)
    parser.add_argument('--dm_dial_path',   default='../dataset/analysis/dial-dm.txt', type=str)
    parser.add_argument('--al_cont_path',   default='../dataset/analysis/cont-al.txt', type=str)
    parser.add_argument('--al_prof_path',   default='../dataset/analysis/prof-al.txt', type=str)
    parser.add_argument('--dm_resp_path',   default='../dataset/analysis/resp-dm.txt', type=str)
    parser.add_argument('--dm_keyw_path',   default='../dataset/analysis/keyw-dm.txt', type=str)
    parser.add_argument('--sali_path',      default='../dataset/analysis/sali-re.txt', type=str)
    parser.add_argument('--mosa_path',      default='../dataset/analysis/mosa-re.txt', type=str)
    parser.add_argument('--rand_path',      default='../dataset/analysis/rand-re.txt', type=str)
    parser.add_argument('--itge_path',      default='../dataset/analysis/itge-it.txt', type=str)

    parser.add_argument('--n_sample', default=5,    type=int, help='The times of Zp sampling')
    parser.add_argument('--n_prof',   default=1146, type=int, help='The number of profs to analyze->Zp')
    parser.add_argument('--n_cont',   default=1000, type=int, help='The number of conts to analyze->Zp or Za')

    parser.add_argument('--re_code', default="R2", type=str,   help='R2->P(T) R3->AtEc')
    parser.add_argument('--it_code', default="DL", type=str,   help='DL->The dial len calcul CH->The coherence calcul')
    parser.add_argument('--op_thre', default=0.8,  type=float, help='The overlapping threshold of utterances')
    parser.add_argument('--di_thre', default=0.9,  type=float, help='The vector distance threshold of utterances')

    return parser.parse_args()