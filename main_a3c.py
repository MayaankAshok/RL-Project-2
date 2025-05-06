import torch.multiprocessing as mp
import torch
from a3c_agent import ActorCritic, worker_func
from Env import UAVEnv
import numpy as np
# from uav_gym_env import UAVGymEnv

import matplotlib.pyplot as plt
import os
import queue  # <-- Add this to handle timeout

if __name__ == "__main__":
    env = UAVEnv()
    # env = UAVGymEnv()

    obs_dim = env.get_observation_space_info()['shape'][0]
    act_dim = env.get_action_space_info()['shape'][0]

    global_net = ActorCritic(obs_dim, act_dim)
    global_net.share_memory()
    optimizer = torch.optim.Adam(global_net.parameters(), lr=1e-5)

    global_ep, global_ep_r = mp.Value('i', 0), mp.Value('d', 0.)
    res_queue = mp.Queue()

    num_workers = mp.cpu_count() - 4
    processes = []
    print("num_worker=====",num_workers)

    os.makedirs("models", exist_ok=True)
    rewards = []
    success_list, crash_list, timeout_list = [], [], []  ## success (passed through window), 
                                                         ## crash (hit wall), timeout (neither passed nor crashed) 

    # Start all workers
    for i in range(num_workers):
        p = mp.Process(target=worker_func,
                       args=(global_net, optimizer, global_ep, global_ep_r, res_queue, i))
        p.start()
        processes.append(p)

    done_count = 0
    timeout_sec = 60  # Wait up to 60 seconds for a result

    while True:
        try:
            result = res_queue.get(timeout=timeout_sec)
            if result is not None:
                r, event_type = result  # Unpack reward and event type
                rewards.append(r)

                # Track events
                if event_type == 'success':
                    success_list.append(1)
                    crash_list.append(0)
                    timeout_list.append(0)
                elif event_type == 'crash':
                    success_list.append(0)
                    crash_list.append(1)
                    timeout_list.append(0)
                else:  # timeout
                    success_list.append(0)
                    crash_list.append(0)
                    timeout_list.append(1)

                # Logging
                if len(rewards) % 10 == 0:
                    avg_last_10 = sum(rewards[-10:]) / 10
                    print(f"[Main] Episode {len(rewards)}, Avg Reward (last 10): {avg_last_10:.2f}")

                if len(rewards) % 100 == 0:
                    torch.save(global_net.state_dict(), f"models/a3c_uav_ep{len(rewards)}.pth")
            else:
                done_count += 1
                if done_count == num_workers:
                    break
        except queue.Empty:
            print(f"[Main] Timeout waiting for result after {timeout_sec} seconds. A worker may be stuck.")
            break  # Exit gracefully if a worker is likely stuck

    for p in processes:
        p.join(timeout=10)
        if p.is_alive():
            print(f"[Main] Worker {p.pid} is still alive. Terminating.")
            p.terminate()
            p.join()

    # Plotting reward curve
    plt.plot(rewards)    
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('A3C UAV Training Reward Curve')
    plt.grid()
    plt.savefig("reward_plot.png")
    plt.show()

    # Plot success/crash/timeout counts
    plt.figure()
    plt.plot(success_list, label="Success")
    plt.plot(crash_list, label="Crash")
    plt.plot(timeout_list, label="Timeout")
    plt.xlabel('Episode')
    plt.ylabel('Event Occurrence')
    plt.title('A3C UAV Event Type per Episode')
    plt.legend()
    plt.grid()
    plt.savefig("event_types_plot.png")
    plt.show()


    with open("training_summary.txt", "w") as f:
        f.write(f"Total Episodes: {len(rewards)}\n")
        f.write(f"Successes: {sum(success_list)}\n")
        f.write(f"Crashes: {sum(crash_list)}\n")
        f.write(f"Timeouts: {sum(timeout_list)}\n")


