from transformers import BertForNextSentencePrediction
from load_data import load_rele_dataset, Collater_R1, Collater_R2R3
from model_reward import Causal_LM, AutoEncoder
from torch.utils.data import DataLoader
from hyperparams import set_args
from tqdm import tqdm
import transformers
from utils import*
import os

def save_model(RewM, epoch):
    if args.rl_code is "R1":
        RewM.save_pretrained(args.save_model_path + "/rl/{}/epoch{}/".format(args.rl_code, epoch))

    if args.rl_code is "R2":
        save_path = args.save_model_path + "/rl/{}/epoch{}".format(args.rl_code, epoch)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save({'CaLM': RewM.state_dict()}, save_path + "/CaLM.pth")

    if args.rl_code is "R3":
        save_path = args.save_model_path + "/rl/{}/epoch{}".format(args.rl_code, epoch)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save({'AtEc': RewM.state_dict()}, save_path + "/AtEc.pth")

    logger.info("Saved the reward model:{} of epoch:{}".format(args.rl_code, epoch))


def train_epoch(RewM, train_dataloader, optimizer, scheduler, epoch):
    RewM.train()

    total_loss = 0
    batch_step = len(train_dataloader)

    for batch_idx, batch_data in enumerate(tqdm(train_dataloader)):
        try:
            if args.rl_code is "R1":
                enco, labe = batch_data
                outputs = RewM.forward(**enco, labels=labe)
                loss = outputs.loss

            if args.rl_code in ["R2", "R3"]:
                seq = batch_data
                loss = RewM.forward(seq)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += float(loss)

        except RuntimeError as exception:
            if "out of memory" in str(exception):
                logger.info("WARNING: ran out of memory")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                logger.info(str(exception))
                raise exception

    nll = total_loss / batch_step
    logger.info("Training epoch:{} Loss:{}".format(epoch, nll))


def valid_epoch(RewM, valid_dataloader, epoch):
    RewM.eval()

    total_loss = 0
    total_accu = 0
    total_prec = 0
    total_reca = 0
    total_F1   = 0
    batch_step = len(valid_dataloader)

    for batch_idx, batch_data in enumerate(tqdm(valid_dataloader)):
        try:
            if args.rl_code is "R1":
                enco, labe = batch_data
                outputs = RewM.forward(**enco, labels=labe)

                if args.forwa_only:
                    output = F.softmax(outputs.logits, dim=-1)    # N 2
                    output = torch.where(output[:, 0]>=0.5, 0, 1)    # N
                    hit = torch.where((output-labe)==0, 1, 0)
                    accu = torch.sum(hit) / torch.numel(labe)
                    TP = hit * torch.where(labe==0, 1, 0)
                    prec = torch.sum(TP) / (torch.numel(output) - torch.sum(output))
                    FN = hit * labe
                    reca = torch.sum(TP) / (torch.sum(TP) + torch.sum(FN))
                    F1 = 2 / (1 / prec + 1 / reca)

                    total_accu += float(accu)
                    total_prec += float(prec)
                    total_reca += float(reca)
                    total_F1   += float(F1)
                else:
                    loss = outputs.loss
                    total_loss += float(loss)

            if args.rl_code in ["R2", "R3"]:
                seq = batch_data
                loss = RewM.forward(seq)
                total_loss += float(loss)

        except RuntimeError as exception:
            if "out of memory" in str(exception):
                logger.info("WARNING: ran out of memory")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                logger.info(str(exception))
                raise exception

    if args.forwa_only and args.rl_code is "R1":
        logger.info("Accuracy:{}".format(total_accu/batch_step))
        logger.info("Precision:{}".format(total_prec/batch_step))
        logger.info("Recall:{}".format(total_reca/batch_step))
        logger.info("F1:{}".format(total_F1/batch_step))
    else:
        nll = total_loss / batch_step
        logger.info("Validating epoch:{} Loss:{}".format(epoch, nll))

        return nll


def train(RewM, train_data, valid_data):
    patience = 0
    best_val_loss = float('Inf')

    if args.rl_code is "R1":
        collate_fn = Collater_R1(tokenizer, args.device)

    if args.rl_code in ["R2", "R3"]:
        collate_fn = Collater_R2R3(tokenizer, args.device)

    train_dataloader = \
    DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.n_worker, collate_fn=collate_fn)
    valid_dataloader = \
    DataLoader(valid_data, batch_size=args.batch_size, shuffle=True, num_workers=args.n_worker, collate_fn=collate_fn)

    if args.forwa_only:
        # ========== eval ========== #
        logger.info('Starting validating')
        valid_epoch(RewM, valid_dataloader, None)
        logger.info('Validating finished')
    else:
        # ========== train ========== #
        t_total = len(train_dataloader) * args.epochs
        optimizer = transformers.AdamW(list(RewM.parameters()), lr=args.lr)
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer, args.warm_step, t_total)
        logger.info('Starting training')

        for epoch in range(1, args.epochs+1):
            train_epoch(RewM, train_dataloader, optimizer, scheduler, epoch)
            val_loss = valid_epoch(RewM, valid_dataloader, epoch)

            if (val_loss < best_val_loss and args.early_stop) or args.niter_save <= epoch:
                # Save rl model and variational model
                save_model(RewM, epoch)
                best_val_loss = val_loss
                patience = 0
            else:
                # This variable is useless when early_stop is False
                patience = patience + 1

            if args.patience < patience and args.early_stop:
                logger.info("Early stop due to run out of patience")
                break

        logger.info('Training finished')


def main():
    device_assign(args, logger)

    if args.rl_code in ["R1", "R2", "R3"]:
        train_data = load_rele_dataset(args.rl_code, logger, dial_path=args.rl_trda_path)
        valid_data = load_rele_dataset(args.rl_code, logger, dial_path=args.rl_vada_path)
    else:
        # There is no usecase for setting rl_code as RS
        logger.info("Invalid rl_code, please reset it")
        sys.exit()

    if args.rl_code is "R1":
        if args.forwa_only:
            RewM = BertForNextSentencePrediction.from_pretrained(args.BiEs_checkp)
            logger.info("Restored the Binary Estimation model from the check point")
        else:
            RewM = BertForNextSentencePrediction.from_pretrained(args.BiEs_pretrained)
            logger.info("Initialized the Binary Estimation model from the pretrained weight (Bert)")

    if args.rl_code is "R2":
        pad_id = tokenizer.pad_token_id
        n_posit = args.max_seq_len + 2    # +2 for adding [BOS] and [EOS]
        if args.forwa_only:
            RewM = Causal_LM(args.clm_d_model, args.clm_n_head, n_posit, len(tokenizer), pad_id, args.device)
            RewM.load_state_dict(torch.load(args.CaLM_checkp)["CaLM"])
            logger.info("Restored the Causal_LM model from the check point")
        else:
            RewM = Causal_LM(args.clm_d_model, args.clm_n_head, n_posit, len(tokenizer), pad_id, args.device)
            logger.info("Initialized the Causal_LM model")

    if args.rl_code is "R3":
        pad_id = tokenizer.pad_token_id
        if args.forwa_only:
            RewM = AutoEncoder(args.ate_d_model, len(tokenizer), pad_id, args.device)
            RewM.load_state_dict(torch.load(args.AtEc_checkp)["AtEc"])
            logger.info("Restored the AutoEncoder model from the check point")
        else:
            RewM = AutoEncoder(args.ate_d_model, len(tokenizer), pad_id, args.device)
            logger.info("Initialized the AutoEncoder model")

    RewM = to_device(args.device, logger, RewM)
    train(RewM, train_data, valid_data)


if __name__ == '__main__':
    args = set_args()
    logger = create_logger(args)
    tokenizer = buil_BertTokenizer(args)
    main()