from model_selecto import Reinforce, Self_Play
from hyperparams import set_args
from utils import*

def main():
    args = set_args()
    logger = create_logger(args)
    device_assign(args, logger)

    inter = Self_Play(args, buil_GPT2Tokenizer(args))
    agent = Reinforce(args, buil_BertTokenizer(args), logger)

    inter, agent = to_device(args.device, logger, inter, agent)
    logger.info('Starting reinforce learning')

    for episode in range(1, args.n_episode+1):
        inter.cont = agent.state_init()
        log_probs = []
        rewards = []

        for _ in range(args.n_step):
            cont = inter.cont
            actions = inter.receiver()
            action, log_prob = agent.select_action(cont, actions)
            resp = inter.receiver(message=action)    # Updata the state
            rewards.append(agent.reward_eval(cont, action, resp))
            log_probs.append(log_prob)

        loss, cum_reward = agent.update_parameter(rewards, log_probs, args.gamma)
        logger.info("Episode:{} Loss:{} Cumulatived reward:{}".format(episode, loss, cum_reward))

        if episode % 500 == 0:
            agent.Selec.save_pretrained(args.save_model_path + "/rl/RS/epoch{}/".format(episode))

    logger.info("Saved the Selector model")


if __name__ == '__main__':
    main()