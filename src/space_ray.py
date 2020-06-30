import ray
from ray.tune.registry import register_env
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.utils import try_import_torch

from point_mass_wrapper import CPMWrapper
import csv
import numpy as np
import logging
import os
import argparse
from functools import partial

torch, _ = try_import_torch()


def env_creator(env_config):
    path = env_config["path"]
    wrapper_func = env_config["wrapper_func"]
    feats = []
    with open(path, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            f = []
            for i in range(len(row)):
                if i != 0:
                    try:
                        f.append(float(row[i]))
                    except:
                        f.append(row[i])
            feats.append(f)
    return wrapper_func(instance_feats=feats, test=env_config["test"],)


def order_instances(trainer, num_instances):
    evals = []
    env = trainer.workers.local_worker().env
    model = trainer.get_policy().model
    set, _ = env.get_instance_set()
    for i in range(num_instances):
        env.set_instance_set([i])
        obs = np.reshape(np.array(env.reset()), (1, -1))
        obs = torch.from_numpy(obs).float().cuda()
        m_out = model.__call__({"obs": obs})[0]
        val = model.value_function().detach().cpu().numpy()
        evals.append(val[0])
    env.set_instance_set(set)
    return np.argsort(evals)

def get_mean_v(trainer, n_insts):
    vs = []
    env = trainer.workers.local_worker().env
    model = trainer.get_policy().model
    for i in range(n_insts):
        obs = np.reshape(env.reset(), (1, -1))
        obs = torch.from_numpy(obs).float().cuda()
        m_out = model.__call__({"obs": obs})[0]
        val = model.value_function().detach().cpu().numpy()
        vs.append(val)
    return np.mean(vs)


def eval_hook(trainer, eval_env, s, outdir, name):
    step = 1
    to_eval = eval_env.get_num_instances()
    train_reward = 0
    policies = []
    for _ in range(to_eval):
        policies.append([])
    rewards = np.zeros(to_eval)
    for n in range(to_eval):
        print(f"Evaluating instance {n}")
        obs = eval_env.reset()
        done = False
        rews = 0
        pol = [eval_env.inst_id]
        while not done:
            action = trainer.compute_action(obs)
            pol.append(action)
            obs, r, done, _ = eval_env.step(action)
            rews += r
        print("Done")
        rewards[eval_env.inst_id] = rews
        train_reward += rews
        policies[eval_env.inst_id] = pol
    train_reward = train_reward / to_eval
    with open(os.path.join(outdir, f"{name}_reward.txt"), "a") as fh:
        fh.writelines(
            str(train_reward)
            + "\t"
            + str(step)
            + "\t"
            + str(s)
            + "\t"
            + str(to_eval)
            + "\n"
        )

    with open(os.path.join(outdir, f"{name}_per_instance.txt"), "a") as fh:
        r_string = ""
        for r in rewards:
            r_string += str(r) + "\t"
        r_string += str(step) + "\t" + str(s) + "\n"
        fh.writelines(r_string)

    with open(os.path.join(outdir, f"{name}_policies.txt"), "a") as fh:
        for p in policies:
            p_string = str(p[0]) + " "
            for i in range(1, len(p)):
                p_string += str(p[i])
            fh.writelines(p_string + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Ray Ape-X RR/CL/SPL")
    parser.add_argument(
        "--mode", choices=["rr", "spl"], default="spl", help=" "
    )
    parser.add_argument(
        "--outdir",
        default="./space_log",
        help="Directory in which to save metrics",
    )
    parser.add_argument(
        "--eval", default=1, type=int, help="Evaluation and Test interval"
    )
    parser.add_argument(
        "--iter", default=1e5, type=int, help="Number of training iterations"
    )
    parser.add_argument(
        "--test",
        default="/home/eimer/Dokumente/dac_spl/features/pointmass_test.csv",
        help="Test instance file",
    )
    parser.add_argument(
        "--instances",
        default="/home/eimer/Dokumente/dac_spl/features/pointmass_train.csv",
        help="Instance file",
    )
    args = parser.parse_args()

    env_func = partial(CPMWrapper)

    outdir = args.outdir
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    env_creator_name = "cpm_" + args.mode
    register_env(env_creator_name, env_creator)
    logger = logging.getLogger(__name__)

    eval_env = env_creator(
        {"path": args.instances, "wrapper_func": env_func, "test": True,}
    )
    test_env = env_creator(
        {"path": args.test, "wrapper_func": env_func, "test": True,},
    )

    if args.mode == "rr":
        test = True
    else:
        test = False

    ray.init(object_store_memory=10 ** 9)
    trainer = PPOTrainer(
        env=env_creator_name,
        config={
            "env_config": {
                "path": args.instances,
                "wrapper_func": env_func,
                "test": test,
            },
            "use_pytorch": True,
        },
    )

    steps = 0
    eval_hook(trainer, eval_env, steps, outdir, "eval")
    eval_hook(trainer, test_env, steps, outdir, "test")

    delta_v = -np.inf
    last_v = 0
    n_instances = trainer.workers.local_worker().env.get_instance_set_size()
    training_steps = n_instances
    eval_factor = 1

    for i in range(int(args.iter)):
        for j in range(training_steps):
            trainer.train()
        steps += training_steps

        if not args.mode == "rr":
            #Computer difference in value function
            mean_v = get_mean_v(
                trainer, trainer.workers.local_worker().env.get_instance_set_size()
            )
            delta_v = np.abs(np.abs(mean_v) - np.abs(last_v))
            last_v = mean_v

            #If agent converges, increase instance set size
            if delta_v <= 0.1 * np.abs(last_v):
                trainer.workers.foreach_worker(
                    lambda ev: ev.foreach_env(
                        lambda env: env.increase_set_size()
                    )
                )
                training_steps = (
                    trainer.workers.local_worker().env.get_instance_set_size()
                )

            #Recompute instance set
            indices = order_instances(trainer, n_instances)
            trainer.workers.foreach_worker(
                lambda ev: ev.foreach_env(
                    lambda env: env.set_instance_set(indices)
                )
            )

        #Evaluation on train and test set
        if steps >= eval_factor * args.eval:
            eval_factor += 1
            eval_hook(trainer, eval_env, steps, outdir, "eval")
            eval_hook(trainer, test_env, steps, outdir, "test")
