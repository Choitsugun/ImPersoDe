from model_selecto import Reinforce, Self_Play
from hyperparams import set_args
from utils import*
import codecs
import os

def main():
    args = set_args()
    logger = create_logger(args)
    device_assign(args, logger)

    inter = Self_Play(args, buil_GPT2Tokenizer(args))
    agent = Reinforce(args, buil_BertTokenizer(args), logger, if_train=False, if_interact=True)

    inter, agent = to_device(args.device, logger, inter, agent)
    logger.info('Starting interaction')

    if not os.path.exists(args.save_resu_path):
        os.makedirs(args.save_resu_path)
    file = codecs.open(os.path.join(args.save_resu_path, "interaction"), 'w', 'utf8')

    n_generat = args.n_generat
    for _ in range(1, args.n_dial+1):
        inter.cont = agent.state_init()

        # agent <---> agent + RS
        inter.args.n_generat = n_generat
        latch_cont = inter.cont
        for _ in range(args.n_turn):
            cont = inter.cont
            actions = inter.receiver()
            action, _ = agent.select_action(cont, actions)
            _ = inter.receiver(message=action)    # Updata the state

        file.write("- genera_RS: " + inter.cont + "\n")

        # agent <---> agent
        inter.args.n_generat = 1
        inter.cont = latch_cont
        for _ in range(args.n_turn):
            cont = inter.cont
            actions = inter.receiver()
            action, _ = agent.select_action(cont, actions)
            _ = inter.receiver(message=action)    # Updata the state

        file.write("- genera_AA: " + inter.cont + "\n\n")

    file.close()
    logger.info('Interaction finished')


if __name__ == '__main__':
    main()