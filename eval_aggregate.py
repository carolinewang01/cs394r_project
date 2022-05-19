import glob
import json
import numpy as np
def main():
    log_dir='./evaluate_results/'
    fs=glob.glob(log_dir+'/*')
    results={}
    aggr_results={}
    for f in fs:
        r=json.load(open(f,'r'))
        exp_tag=list(r.keys())[0]
        if exp_tag not in results.keys():
            results[exp_tag]={}
            results[exp_tag] = {}
            results[exp_tag]['rews'] = []
            results[exp_tag]['winrates'] = []
            results[exp_tag]['wintierates'] = []
            aggr_results[exp_tag] = {}
        results[exp_tag]['rews'].append(r[exp_tag]['mean_rews'])
        results[exp_tag]['winrates'].append(r[exp_tag]['mean_winrates'])
        results[exp_tag]['wintierates'].append(r[exp_tag]['mean_wintierates'])
        aggr_results[exp_tag]['mean_rews']=np.mean(results[exp_tag]['rews'])
        aggr_results[exp_tag]['mean_winrates']=np.mean(results[exp_tag]['winrates'])
        aggr_results[exp_tag]['mean_wintierates']=np.mean(results[exp_tag]['wintierates'])
        aggr_results[exp_tag]['std_rews']=np.std(results[exp_tag]['rews'])
        aggr_results[exp_tag]['std_winrates']=np.std(results[exp_tag]['winrates'])
        aggr_results[exp_tag]['std_wintierates']=np.std(results[exp_tag]['wintierates'])

 
    print(aggr_results)


if __name__=='__main__':
    main()
