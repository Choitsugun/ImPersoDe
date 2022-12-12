from torch.utils.data import Dataset
from tqdm import tqdm
import codecs
import random
import torch

# ==================== For train_agent.py ==================== #
class Collater():
    def __init__(self, tokenizer, device):
        self.tokenizer = tokenizer
        self.device = device

    def __call__(self, batch):
        dial, cont, prof, resp, keyw = tokenizer_process(batch, self.tokenizer, self.device)
        return dial, cont, prof, resp, keyw


class MyDataset(Dataset):
    def __init__(self, dial, cont, prof, resp, keyw):
        self.dial = dial
        self.cont = cont
        self.prof = prof
        self.resp = resp
        self.keyw = keyw

    def __getitem__(self, index):
        b_dial = self.dial[index]
        b_cont = self.cont[index]
        b_prof = self.prof[index]
        b_resp = self.resp[index]
        b_keyw = self.keyw[index]
        return b_dial, b_cont, b_prof, b_resp, b_keyw

    def __len__(self):
        return len(self.dial)


def format_process(l_dial, l_cont, l_prof, l_resp, l_keyw, tokenizer):
    bos_token = tokenizer.bos_token
    eos_token = tokenizer.eos_token
    z_token = tokenizer.additional_special_tokens[0]
    a_token = tokenizer.additional_special_tokens[-1]
    dial = []
    cont = []
    prof = []
    resp = []
    keyw = []
    Tab = '\t'
    Sap = ' '

    # Feed into the generator
    # [z] [a] [BOS] utterance1[EOS] utterance2[EOS] response[EOS]
    for dialog in tqdm(l_dial):
        d = z_token + Sap + a_token + Sap + bos_token + Sap + dialog.replace(Tab, eos_token+Sap) + eos_token
        dial.append(d)

    # Feed into the prior net
    # [z] [a] [BOS] utterance1[EOS] utterance2[EOS]
    for context in tqdm(l_cont):
        d = z_token + Sap + a_token + Sap + bos_token + Sap + context.replace(Tab, eos_token+Sap) + eos_token
        cont.append(d)

    # Feed into the posterior net of z
    # [z] [BOS] persona1[EOS] persona2[EOS]
    for profile in tqdm(l_prof):
        d = z_token + Sap + bos_token + Sap + profile.replace(Tab, eos_token+Sap) + eos_token
        prof.append(d)

    # Feed into the posterior net of a
    # response
    for response in tqdm(l_resp):
        d = response
        resp.append(d)

    # Feed into the posterior net of a
    # keyword1 keyword2 keyword3
    for keyword in tqdm(l_keyw):
        d = keyword
        keyw.append(d)

    assert len(dial) == len(cont)
    assert len(dial) == len(prof)
    assert len(dial) == len(resp)
    assert len(dial) == len(keyw)
    assert len(cont) == len(prof)
    assert len(cont) == len(resp)
    assert len(cont) == len(keyw)
    assert len(prof) == len(resp)
    assert len(prof) == len(keyw)
    assert len(resp) == len(keyw)

    return dial, cont, prof, resp, keyw


def load_dataset(dial_path, cont_path, prof_path, resp_path, keyw_path, tokenizer, logger):
    logger.info("Loading dataset and staring format process")

    l_dial = [line.strip() for line in codecs.open(dial_path, 'r', 'utf-8').readlines() if line.strip()]
    l_cont = [line.strip() for line in codecs.open(cont_path, 'r', 'utf-8').readlines() if line.strip()]
    l_prof = [line.strip() for line in codecs.open(prof_path, 'r', 'utf-8').readlines() if line.strip()]
    l_resp = [line.strip() for line in codecs.open(resp_path, 'r', 'utf-8').readlines() if line.strip()]
    l_keyw = [line.strip() for line in codecs.open(keyw_path, 'r', 'utf-8').readlines() if line.strip()]

    dial, cont, prof, resp, keyw = format_process(l_dial, l_cont, l_prof, l_resp, l_keyw, tokenizer)
    dataset = MyDataset(dial, cont, prof, resp, keyw)
    logger.info("Dataset format process are finshed")

    return dataset


def load_test_dataset(dial_path, cont_path, prof_path, resp_path, keyw_path, tokenizer, logger):
    logger.info("Loading test dataset and staring format process")

    l_dial = [line.strip() for line in codecs.open(dial_path, 'r', 'utf-8').readlines() if line.strip()]
    l_cont = [line.strip() for line in codecs.open(cont_path, 'r', 'utf-8').readlines() if line.strip()]
    l_prof = [line.strip() for line in codecs.open(prof_path, 'r', 'utf-8').readlines() if line.strip()]
    l_resp = [line.strip() for line in codecs.open(resp_path, 'r', 'utf-8').readlines() if line.strip()]
    l_keyw = [line.strip() for line in codecs.open(keyw_path, 'r', 'utf-8').readlines() if line.strip()]

    dial, cont, prof, resp, keyw = format_process(l_dial, l_cont, l_prof, l_resp, l_keyw, tokenizer)
    logger.info("Dataset format process are finshed")

    return cont, prof, resp


def tokenizer_process(batch, tokenizer, device):
    b_dial, b_cont, b_prof, b_resp, b_keyw = zip(*batch)

    b_dial = tokenizer(b_dial, padding=True, return_tensors="pt").to(device)
    b_cont = tokenizer(b_cont, padding=True, return_tensors="pt").to(device)
    b_prof = tokenizer(b_prof, padding=True, return_tensors="pt").to(device)
    b_resp = tokenizer(b_resp, padding=True, return_tensors="pt").to(device)
    b_keyw = tokenizer(b_keyw, padding=True, return_tensors="pt").to(device)

    return b_dial, b_cont, b_prof, b_resp, b_keyw


# ========= For train_reward.py & train_selector.py ========= #
class Collater_R1():
    def __init__(self, tokenizer, device):
        self.tokenizer = tokenizer
        self.device = device

    def __call__(self, batch):
        enco, labe = tokenizer_process_R1(batch, self.tokenizer, self.device)
        return enco, labe


class Collater_R2R3():
    def __init__(self, tokenizer, device):
        self.tokenizer = tokenizer
        self.device = device

    def __call__(self, batch):
        pool = tokenizer_process_R2R3(batch, self.tokenizer, self.device)
        return pool


class Collater_RS():
    def __init__(self, tokenizer, device):
        self.tokenizer = tokenizer
        self.device = device

    def __call__(self, batch):
        sali = tokenizer_process_RS(batch, self.tokenizer, self.device)
        return sali


class MyDataset_R1(Dataset):
    def __init__(self, prom, nese, labe):
        self.prom = prom
        self.nese = nese
        self.labe = labe

    def __getitem__(self, index):
        b_prom = self.prom[index]
        b_nese = self.nese[index]
        b_labe = self.labe[index]

        return b_prom, b_nese, b_labe

    def __len__(self):
        return len(self.prom)


class MyDataset_R2R3(Dataset):
    def __init__(self, pool):
        self.pool = pool

    def __getitem__(self, index):
        b_pool = self.pool[index]
        return b_pool

    def __len__(self):
        return len(self.pool)


class MyDataset_RS(Dataset):
    def __init__(self, sali):
        self.sali = sali

    def __getitem__(self, index):
        b_sali = self.sali[index]
        return b_sali

    def __len__(self):
        return len(self.sali)


def rl_format(l_dial):
    # Note: Need not to deal with the space of Chinese word segmentation, since Bert tokenizer will ignores them.
    prom = []
    nese = []
    labe =[]
    pool = []
    Tab = '\t'

    for dial in tqdm(l_dial):
        utte = dial.split(Tab)
        i = random.randint(0, len(utte)-2)
        prom.append(utte[i])
        nese.append(utte[i+1])
        labe.append(0)    # 0 indicates sequence B is a continuation of sequence A
        pool.extend(utte)

    for _ in tqdm(l_dial):
        i = random.randint(0, len(pool)-1)
        prom.append(pool[i])
        i = random.randint(0, len(pool)-1)
        nese.append(pool[i])
        labe.append(1)    # 1 indicates sequence B is a random sequence

    return prom, nese, labe, pool


def load_rele_dataset(rl_code, logger, dial_path=None, sali_path=None, stin_path=None):
    logger.info("Loading reinforcement learning dataset and staring format process")

    if rl_code in ["R1", "R2", "R3"]:
        l_dial = [line.strip() for line in codecs.open(dial_path, 'r', 'utf-8').readlines() if line.strip()]
        prom, nese, labe, pool = rl_format(l_dial)

        if rl_code is "R1":
            logger.info("Construct the MyDataset of MMI->P(S,T)")
            dataset = MyDataset_R1(prom, nese, labe)

        if rl_code is "R2":
            logger.info("Construct the MyDataset of MMI->P(T)")
            dataset = MyDataset_R2R3(pool)

        if rl_code is "R3":
            logger.info("Construct the MyDataset of AutoEncoder")
            dataset = MyDataset_R2R3(pool)

        logger.info("Dataset format process are finshed")

        return dataset

    if rl_code is "RS":
        l_sali = [line.strip() for line in codecs.open(sali_path, 'r', 'utf-8').readlines() if line.strip()]
        logger.info("Construct the MyDataset of safe resp list")
        dataset = MyDataset_RS(l_sali)
        logger.info("Load the utterances for init state")
        l_stin = [line.strip() for line in codecs.open(stin_path, 'r', 'utf-8').readlines() if line.strip()]
        logger.info("Dataset format process are finshed")

        return dataset, l_stin


def load_rean_dataset(logger, sali_path=None, mosa_path=None, rand_path=None):
    logger.info("Loading analysis dataset of reward model")

    l_sali = [line.strip() for line in codecs.open(sali_path, 'r', 'utf-8').readlines() if line.strip()]
    l_rand = [line.strip() for line in codecs.open(rand_path, 'r', 'utf-8').readlines() if line.strip()]
    assert len(l_sali) == len(l_rand)

    if mosa_path is None:
        logger.info("Loading process for analysis of Causal_LM model are finshed")

        return l_sali, l_rand
    else:
        l_mosa = [line.strip() for line in codecs.open(mosa_path, 'r', 'utf-8').readlines() if line.strip()]
        assert len(l_sali) == len(l_mosa)
        logger.info("Loading process for analysis of AutoEncoder model are finshed")

        return l_sali, l_mosa, l_rand


def load_itan_dataset(logger, itge_path, sali_path=None):
    logger.info("Loading analysis dataset of interaction")

    if sali_path is not None:
        l_sali = [line.strip() for line in codecs.open(sali_path, 'r', 'utf-8').readlines() if line.strip()]
        l_itge = [line.strip() for line in codecs.open(itge_path, 'r', 'utf-8').readlines() if line.strip()]
        logger.info("Loading process for the length of dialogue analysis of interaction are finshed")

        return l_itge, l_sali
    else:
        l_itge = [line.strip() for line in codecs.open(itge_path, 'r', 'utf-8').readlines() if line.strip()]
        logger.info("Loading process for the coherence analysis of interaction are finshed")

        return l_itge


def tokenizer_process_R1(batch, tokenizer, device):
    b_prom, b_nese, b_labe = zip(*batch)
    b_enco = tokenizer(b_prom, b_nese, padding=True, return_tensors="pt").to(device)
    b_labe = torch.tensor(b_labe).to(device)

    return b_enco, b_labe


def tokenizer_process_R2R3(batch, tokenizer, device):
    b_pool = batch
    b_pool = tokenizer(b_pool, padding=True, return_tensors="pt").to(device)

    return b_pool


def tokenizer_process_RS(batch, tokenizer, device):
    b_sali = batch
    b_sali = tokenizer(b_sali, padding=True, return_tensors="pt").to(device)

    return b_sali