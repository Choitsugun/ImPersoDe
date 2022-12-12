from transformers import BertForNextSentencePrediction
from model_reward import AutoEncoder
from load_data import load_itan_dataset
from hyperparams import set_args
from utils import*

def dial_len_anal():
    Tab = '\t'
    nul_str = ''
    op = args.op_thre
    di = args.di_thre
    device_assign(args, logger)

    pad_id = tokenizer.pad_token_id
    AtEc = AutoEncoder(args.ate_d_model, len(tokenizer), pad_id, args.device)
    AtEc.load_state_dict(torch.load(args.AtEc_checkp)["AtEc"])
    logger.info("Restored the AutoEncoder model from the check point")
    AtEc.to(args.device)
    AtEc.eval()

    itges, salis = load_itan_dataset(logger, args.itge_path, args.sali_path)
    logger.info('Starting the length of dialogue analysis')
    count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    len_sum = 0

    for itge in itges:
        dial = itge.split(Tab)
        len_prop_count = 1
        len_scop_count = 2
        len_vedi_count = 1
        len_turn_count = args.n_turn * 2

        # Empty resp confirm
        for i, utte in enumerate(dial):
            if utte is nul_str:
                len_turn_count = i
                break

        # Post-resp overlapping confirm
        for i, utte in enumerate(dial[:-1]):
            tokens_b = set(utte.split())
            tokens_n = set(dial[1:][i].split())
            len_seki = len(tokens_b & tokens_n)

            if len(tokens_n) == 0 or len(tokens_b) == 0:
                break

            if len_seki / len(tokens_b) >= op or len_seki / len(tokens_n) >= op:
                break

            len_prop_count = len_prop_count + 1

        # Self-consecutive overlapping confirm
        for i, utte in enumerate(dial[:-2]):
            tokens_b = set(utte.split())
            tokens_n = set(dial[2:][i].split())
            len_seki = len(tokens_b & tokens_n)

            if len(tokens_n) == 0 or len(tokens_b) == 0:
                break

            if len_seki / len(tokens_b) >= op or len_seki / len(tokens_n) >= op:
                break

            len_scop_count = len_scop_count + 1

        # Vector space distance confirm
        for utte in dial[1:]:
            if utte is nul_str:
                break

            fg = False
            utte_enco = tokenizer(utte, return_tensors="pt").to(args.device)

            for sali in salis:
                sali_enco = tokenizer(sali, return_tensors="pt").to(args.device)
                sali_vec = AtEc.forward(sali_enco, if_train=False)
                utte_vec = AtEc.forward(utte_enco, if_train=False)
                s_v = F.normalize(sali_vec, dim=-1)
                u_v = F.normalize(utte_vec, dim=-1)
                distance = torch.abs(s_v.matmul(u_v.permute([1, 0])))

                if distance.item() >= di:
                    fg = True
                    break

            if fg:
                break

            len_vedi_count = len_vedi_count + 1

        turn = min(len_prop_count, len_scop_count, len_vedi_count, len_turn_count)
        count[turn-1] = count[turn-1] + 1
        len_sum = len_sum + turn

    print("len-mean:", len_sum / len(itges))
    print("len-samples:", count)


def coherence_anal():
    Tab = '\t'
    nul_str = ''
    device_assign(args, logger)

    BiEs = BertForNextSentencePrediction.from_pretrained(args.BiEs_checkp)
    logger.info("Restored the Binary Estimation model from the check point")
    BiEs.to(args.device)
    BiEs.eval()

    itges = load_itan_dataset(logger, args.itge_path)
    logger.info('Starting the coherence analysis')
    count_P = 0

    for itge in itges:
        dial = itge.split(Tab)

        for i, utte in enumerate(dial):
            if i % 2 == 0:
                prompt = utte
            else:
                if utte is nul_str:
                    break
                else:
                    next_sentence = utte
                    encoding = tokenizer(prompt, next_sentence, return_tensors='pt').to(args.device)
                    outputs = BiEs.forward(**encoding)
                    output = F.softmax(outputs.logits, dim=-1)  # 1 2
                    score = output[0, 0].item()

                    if score > 0.5:
                        count_P = count_P + 1

    print("score > 0.5:", count_P)


if __name__ == '__main__':
    args = set_args()
    logger = create_logger(args)
    tokenizer = buil_BertTokenizer(args)

    if args.it_code == "DL":
        dial_len_anal()

    if args.it_code == "CH":
        coherence_anal()