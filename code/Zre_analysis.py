from model_reward import Causal_LM, AutoEncoder
from load_data import load_rean_dataset
from hyperparams import set_args
from utils import*

def main():
    device_assign(args, logger)

    pad_id = tokenizer.pad_token_id
    n_posit = args.max_seq_len + 2

    if args.re_code is "R2":
        CaLM = Causal_LM(args.clm_d_model, args.clm_n_head, n_posit, len(tokenizer), pad_id, args.device)
        CaLM.load_state_dict(torch.load(args.CaLM_checkp)["CaLM"])
        logger.info("Restored the Causal_LM model from the check point")
        CaLM.to(args.device)
        CaLM.eval()

        salis, rands = load_rean_dataset(logger, sali_path=args.sali_path, rand_path=args.rand_path)
        logger.info('Starting Causal_LM model analysis')
        sali_list, rand_list = [], []

        for sali, rand in zip(salis, rands):
            sali_enco = tokenizer(sali, return_tensors="pt").to(args.device)
            rand_enco = tokenizer(rand, return_tensors="pt").to(args.device)
            sali_pro = CaLM.forward(sali_enco)
            rand_pro = CaLM.forward(rand_enco)
            sali_r2 = sali_pro.item()
            rand_r2 = rand_pro.item()
            print("sali:", sali_r2)
            print("rand:", rand_r2)
            sali_list.append(sali_r2)
            rand_list.append(rand_r2)

        print("sali-mean:", sum(sali_list)/len(salis))
        print("rand-mean:", sum(rand_list)/len(rands))

    elif args.re_code is "R3":
        AtEc = AutoEncoder(args.ate_d_model, len(tokenizer), pad_id, args.device)
        AtEc.load_state_dict(torch.load(args.AtEc_checkp)["AtEc"])
        logger.info("Restored the AutoEncoder model from the check point")
        AtEc.to(args.device)
        AtEc.eval()

        salis, mosas, rands = load_rean_dataset(logger, sali_path=args.sali_path, mosa_path=args.mosa_path, rand_path=args.rand_path)
        logger.info('Starting AutoEncoder model analysis')
        samo_list, sara_list = [], []

        for sali, mosa, rand in zip(salis, mosas, rands):
            sali_enco = tokenizer(sali, return_tensors="pt").to(args.device)
            mosa_enco = tokenizer(mosa, return_tensors="pt").to(args.device)
            rand_enco = tokenizer(rand, return_tensors="pt").to(args.device)
            sali_vec = AtEc.forward(sali_enco, if_train=False)
            mosa_vec = AtEc.forward(mosa_enco, if_train=False)
            rand_vec = AtEc.forward(rand_enco, if_train=False)
            s_v = F.normalize(sali_vec, dim=-1)
            m_v = F.normalize(mosa_vec, dim=-1)
            r_v = F.normalize(rand_vec, dim=-1)
            samo_distance = torch.abs(s_v.matmul(m_v.permute([1, 0])))
            sara_distance = torch.abs(s_v.matmul(r_v.permute([1, 0])))
            samo_r3 = samo_distance.item()
            sara_r3 = sara_distance.item()
            print("samo:", samo_r3)
            print("sara:", sara_r3)
            samo_list.append(samo_r3)
            sara_list.append(sara_r3)

        print("samo-mean:", sum(samo_list)/len(mosas))
        print("sara-mean:", sum(sara_list)/len(rands))

    else:
        logger.info("Invalid re_code, please reset it")
        sys.exit()


if __name__ == '__main__':
    args = set_args()
    logger = create_logger(args)
    tokenizer = buil_BertTokenizer(args)
    main()