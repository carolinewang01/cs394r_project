import os
import subprocess
import time

import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def submit_to_condor(env_id, exec_cmd, results_dir, job_name, expt_params, num_trials, sleep_time=0):
    '''purpose of this function is to submit a script to condor that runs num_trials instances
    '''    
    if num_trials == 0: 
        # print(f"0 jobs submitted to condor for {results_dir + job_name}, {env_id}")
        return 

    condor_log_dir = os.path.join(results_dir, 'condor_logs')
    if not os.path.exists(condor_log_dir):
        os.makedirs(condor_log_dir)
    notification = "Never" # ["Complete", "Never", "Always", "Error"]

    condor_contents = \
f"""Executable = {exec_cmd} 
Universe = vanilla
Getenv = true
+GPUJob = true
Requirements = (TARGET.GPUSlot) && InMastodon

+Group = "GRAD" 
+Project = "AI_ROBOTICS"
+ProjectDescription = "{job_name} {env_id}"

Input = /dev/null
Error = {condor_log_dir}/{job_name}_$(CLUSTER).err
Output = {condor_log_dir}/{job_name}_$(CLUSTER).out
Log = {condor_log_dir}/{job_name}_$(CLUSTER).log

Notify_user = caroline.l.wang@utexas.edu
Notification = {notification}

arguments = \
""" 
    for k, v in expt_params.items():
        condor_contents += f" --{k} {v}" 
    condor_contents += f"\nQueue {num_trials}"

    # submit to condor
    proc = subprocess.Popen('condor_submit', stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    proc.stdin.write(condor_contents.encode())
    proc.stdin.close()

    time.sleep(sleep_time)
    # print("CONDOR SUB SCRIPT IS \n", condor_contents)
    print(f"Submitted {num_trials} jobs for {results_dir + job_name}, {env_id} to condor")


def check_tb_logs(savefile_dir, expected_rew_len):
    f = glob.glob(savefile_dir + "/events*")[0]
    iterator = EventAccumulator(f).Reload()
    tag = "test/reward"
    events = iterator.Scalars(tag)
    rewards = [e.value for e in events]
    if len(rewards) < expected_rew_len:
        return False
    return True

def check_single_dir(savefile_dir, check_tb_log=False, expected_rew_len=None):
    if not os.path.exists(savefile_dir):  # directory doesn't even exist
        return False
    if not os.listdir(savefile_dir):  # directory exists but is empty
        return False
    if check_tb_log:
        return check_tb_logs(savefile_dir, expected_rew_len=expected_rew_len)

    return True

def count_nonempty_dirs(savefile_dirlist):
    '''
    Checks number of nonempty and/or existing directories in the list of directories
    '''
    done = []
    for savefile_dir in savefile_dirlist:
        done.append(check_single_dir(savefile_dir))
    return sum(done)
