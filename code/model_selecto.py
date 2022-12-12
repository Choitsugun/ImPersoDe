from transformers import GPT2LMHeadModel, BertForMultipleChoice, BertForNextSentencePrediction
from load_data import load_rele_dataset, Collater_RS
from model_reward import Causal_LM, AutoEncoder
from model_bayvar import Variational_Bayes
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import transformers
from utils import*
import random

class Self_Play(nn.Module):
    def __init__(self, args, tokenizer):
        super().__init__()
        self.Tab = '\t'
        self.Sap = ' '
        self.cont = ""
        self.args = args
        self.tokenizer = tokenizer

        # Restore the GPT-2 model and the variational Bayes model
        self.GPT2 = GPT2LMHeadModel.from_pretrained(args.GPT2_checkp)
        self.VarB = Variational_Bayes(args.d_model, args.n_head, args.n_layer, len(tokenizer), args.device)
        self.VarB.load_state_dict(torch.load(args.VarB_checkp)["VarB"])
        self.GPT2.eval()
        self.VarB.eval()

    def receiver(self, message:str=None):
        if message is not None:
            self.cont = self.cont + self.Tab + message

        while True:
            if len(self.cont.split()) > self.args.max_seq_len:
                cont_list = self.cont.split(self.Tab)[1:]
                self.cont = self.Tab.join(cont_list)
            else:
                break

        bos_token = self.tokenizer.bos_token
        eos_token = self.tokenizer.eos_token
        z_token = self.tokenizer.additional_special_tokens[0]
        a_token = self.tokenizer.additional_special_tokens[-1]

        cont_input = \
        z_token + self.Sap + a_token + self.Sap + bos_token + self.Sap + self.cont.replace(self.Tab, eos_token + self.Sap) + eos_token
        cont_enco = self.tokenizer(cont_input, padding=False, return_tensors="pt").to(self.args.device)
        cont_input_ids = cont_enco["input_ids"]
        role = make_role(cont_enco, self.tokenizer.eos_token_id, False)
        prior_mu, prior_logvar, c_pe, c_ie, c_ro, c_am = \
        self.VarB.prior_net(self.GPT2.transformer.wpe, self.GPT2.transformer.wte, cont_enco, role)

        if message is not None:
            latent_z = sample_gaussian(prior_mu, prior_logvar, self.args.device)
            latent_a = \
            self.VarB.prior_net(latent_z=latent_z, posit_embeds=c_pe, input_embeds=c_ie, role_embeds=c_ro, attent_masks=c_am)
            resp = generator(self.args, self.GPT2, self.tokenizer, cont_input_ids, latent_z, latent_a, role)
            self.cont = self.cont + self.Tab + resp

            return resp
        else:
            action_list = []
            for _ in range(self.args.n_generat):
                latent_z = sample_gaussian(prior_mu, prior_logvar, self.args.device)
                latent_a = \
                self.VarB.prior_net(latent_z=latent_z, posit_embeds=c_pe, input_embeds=c_ie, role_embeds=c_ro, attent_masks=c_am)
                action = generator(self.args, self.GPT2, self.tokenizer, cont_input_ids, latent_z, latent_a, role)
                action_list.append(action)

            return action_list


class Reinforce(nn.Module):
    def __init__(self, args, tokenizer, logger, if_train=True, if_interact=False):
        super().__init__()
        self.Tab = '\t'
        self.nul_str = ''
        self.args = args
        self.tokenizer = tokenizer

        if if_train:
            pad_id = tokenizer.pad_token_id
            n_posit = args.max_seq_len + 2

            self.BiEs = BertForNextSentencePrediction.from_pretrained(args.BiEs_checkp)
            logger.info("Restored the Binary Estimation model from the check point")
            self.CaLM = Causal_LM(args.clm_d_model, args.clm_n_head, n_posit, len(tokenizer), pad_id, args.device)
            self.CaLM.load_state_dict(torch.load(args.CaLM_checkp)["CaLM"])
            logger.info("Restored the Causal_LM model from the check point")
            self.AtEc = AutoEncoder(args.ate_d_model, len(tokenizer), pad_id, args.device)
            self.AtEc.load_state_dict(torch.load(args.AtEc_checkp)["AtEc"])
            logger.info("Restored the AutoEncoder model from the check point")

            self.BiEs.eval()
            self.CaLM.eval()
            self.AtEc.eval()

            self.Selec = BertForMultipleChoice.from_pretrained(args.Sele_pretrained)
            self.optimizer = transformers.AdamW(list(self.Selec.parameters()), lr=args.lr)
            logger.info("Initialized the Selector model from the pretrained weight")
            self.sali_stin(args, tokenizer, logger)
            self.Selec.train()
        else:
            self.Selec = BertForMultipleChoice.from_pretrained(args.Sele_checkp)
            logger.info("Restored the Selector model from the check point")

            if if_interact:
                self.sali_stin(args, tokenizer, logger, if_interact=True)

            self.Selec.eval()

    def sali_stin(self, args, tokenizer, logger, if_interact=False):
        if if_interact:
            _, stin = load_rele_dataset("RS", logger, sali_path=args.rl_sali_path, stin_path=args.inte_stin_path)
            self.stin = stin
            return
        else:
            sali, stin = load_rele_dataset("RS", logger, sali_path=args.rl_sali_path, stin_path=args.rl_stin_path)
            self.stin = stin

        collate_fn = Collater_RS(tokenizer, args.device)
        sali_dataloader = DataLoader(sali, batch_size=args.batch_size, collate_fn=collate_fn)
        self.AtEc.to(args.device)
        self.hiddens = torch.tensor([]).to(args.device)

        logger.info("Starting compute the vector spaces of the safe resp list")
        for batch_idx, batch_data in enumerate(tqdm(sali_dataloader)):
            seq = batch_data
            output = self.AtEc.forward(seq, if_train=False)
            self.hiddens = torch.cat((self.hiddens, output), dim=0)    # N C

        logger.info("Computing finished")

    def state_init(self):
        # Note: Please ensure that all the dial of stin comes from the training set to avoid the appearance of UNK token
        i = random.randint(0, len(self.stin)-1)
        return self.stin[i]

    def select_action(self, state, actions):
        prompts = []
        prompt = state.split(self.Tab)[-self.args.n_prompt:]
        prompt = self.tokenizer.sep_token.join(prompt)

        for _ in range(self.args.n_generat):
            prompts.append(prompt)

        encoding = self.tokenizer(prompts, actions, return_tensors='pt', padding=True).to(self.args.device)
        outputs = self.Selec.forward(**{k: v.unsqueeze(0) for k, v in encoding.items()})    # batch size is 1
        output = F.softmax(outputs.logits, dim=-1).log()    # 1 n_c
        topv, topi = output.topk(self.args.n_generat)       # 1 n_generat

        if len(actions) == 1:
            action = actions[0]
            log_prob = topv.squeeze()

            return action, log_prob

        for i in range(self.args.n_generat):
            action = actions[topi.squeeze()[i].item()]

            if action is not self.nul_str:
                log_prob = topv.squeeze()[i]

                return action, log_prob
            else:
                if i == self.args.n_generat - 1:
                    log_prob = topv.squeeze()[i]

                    return action, log_prob

    def reward_eval(self, state, action, resp):
        prompt = state.split(self.Tab)[-1]
        next_sentence = action
        encoding = self.tokenizer(prompt, next_sentence, return_tensors='pt').to(self.args.device)
        outputs = self.BiEs.forward(**encoding)
        output = F.softmax(outputs.logits, dim=-1).log()    # 1 2
        r1 = output[0, 0].item()    # logP(S,T)

        act_enco = self.tokenizer(action, return_tensors="pt").to(self.args.device)
        act_pro = self.CaLM.forward(act_enco)
        r2 = act_pro.item()    # -logP(T)

        resp_enco = self.tokenizer(resp, return_tensors="pt").to(self.args.device)
        resp_vec = self.AtEc.forward(resp_enco, if_train=False)    # 1 C
        r_s = F.normalize(resp_vec, dim=-1)
        h_s = F.normalize(self.hiddens, dim=-1)
        distance = torch.abs(h_s.matmul(r_s.permute([1, 0])))    # N 1
        r3 = -torch.mean(distance).log().item()    # -logCos()

        return r1+r2+r3

    def update_parameter(self, rewards, log_probs, gamma):
        R = torch.tensor(1.).to(self.args.device)
        loss = torch.tensor(0.).to(self.args.device)
        cum_reward = 0

        for idx, i in enumerate(reversed(range(len(rewards)))):
            R = gamma * R + rewards[i]
            loss = loss - (R-gamma**(idx+1)) * log_probs[i]
            cum_reward += rewards[i]

        loss = loss / len(rewards)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss), cum_reward