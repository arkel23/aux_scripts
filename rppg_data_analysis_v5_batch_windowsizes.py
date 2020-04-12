import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import os
import argparse
from pathlib import Path
from pathlib import PurePath
from bisect import bisect_left
plt.ioff()

EPSILON = 0.01 # s = 10 ms * 50 diff between calc time and time truth, 10 ms for calc time and time frame

def get_closests(df, val):
    lower_idx = bisect_left(df, val) #val is in the list
    return lower_idx

def undersampling(time_calc, hr_calc, time_truth, hr_truth, len_calc_orig, len_true_orig):
    if len_true_orig > len_calc_orig:
        time_ss = np.reshape(time_calc, (-1, 1))
        hr_ss = np.reshape(hr_calc, (-1, 1))
        time_us = np.zeros((len_calc, 1))
        hr_us = np.zeros((len_calc, 1))
        len_same = len_calc_orig
        len_bigger = len_true_orig

        j = 0
        for i in range(len_bigger):
            if (abs(time_calc[j] - time_truth[i]) < EPSILON*50):
                time_us[j] = time_truth[i]
                hr_us[j] = hr_truth[i]
                j += 1
                print(j, time_us[j], hr_us[j], time_ss[j], hr_ss[j])
                if (j == len_same):
                    break;
    else:
        time_ss = np.reshape(time_truth, (-1, 1))
        hr_ss = np.reshape(hr_truth, (-1, 1))
        time_us = np.zeros((len_true_orig, 1))
        hr_us = np.zeros((len_true_orig, 1))
        len_same = len_true_orig
        len_bigger = len_calc_orig

        j = 0
        for i in range(len_bigger):
            print(time_calc[i], hr_calc[i], time_truth[j], hr_truth[j])
            if (abs(time_truth[j] - time_calc[i]) < EPSILON*50):
                time_us[j] = time_calc[i]
                hr_us[j] = hr_calc[i]
                print(j, time_us[j], hr_us[j], time_ss[j], hr_ss[j])
                j += 1
                if (j == len_same):
                    break;

    return time_us, hr_us, time_ss, hr_ss, len_same, len_bigger

def plots_metrics(rmse, mae, pear, study):
    # original order: 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 1, 20, 2, 3, 4, 5, 6, 7, 8, 9
    idx = [10, 12, 13, 14, 15, 16, 17, 18, 19, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11]
    rmse = rmse[idx]
    mae = mae[idx]
    pear = pear[idx]
    
    n = len(rmse)
    x = np.arange(n)
    #print(x.shape, x, rmse.shape, rmse)
    fig, axs = plt.subplots(3,1)
    fig.set_size_inches(10, 6, forward=True)
    fig.subplots_adjust(hspace=1)
    fig.suptitle('Accuracy metrics for {}'.format(study)
    , fontsize = 24)
    axs[0].plot(x, rmse)
    axs[1].plot(x, mae)
    axs[2].plot(x, pear)
    
    axs[0].set_xlabel('Window Size (s)', fontsize = 16)
    axs[1].set_xlabel('Window Size (s)', fontsize = 16)
    axs[2].set_xlabel('Window Size (s)', fontsize = 16)
    axs[0].set_ylabel('RMSE', fontsize = 18)
    axs[0].set_ylim(0, max(rmse)+np.std(rmse))
    axs[1].set_ylabel('MAE', fontsize = 16)
    axs[1].set_ylim(0, max(mae)+np.std(mae))
    axs[2].set_ylabel('Pearson r', fontsize = 16)
    axs[2].set_ylim(-1, 1)
    
    axs[0].grid(True)
    axs[1].grid(True)
    axs[2].grid(True)
    
    ticks = ['1', '2', '3', '4', '5', '6', '7' '8', '9', '10',
    '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']
    
    axs[0].set_xticks(x) 
    axs[0].set_xticklabels(ticks, fontsize=14)
    axs[1].set_xticks(x) 
    axs[1].set_xticklabels(ticks, fontsize=14)
    axs[2].set_xticks(x) 
    axs[2].set_xticklabels(ticks, fontsize=14)
    fig.canvas.draw()
    out_dir = './results/'
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir,'results_summary_{}.png'.format(study)), bbox_inches='tight', 
    dpi=300, format='png')
    
    df = pd.DataFrame(data={'rmse': rmse[:,0], 'mae': mae[:,0], 'pear': pear[:,0]}, index=np.arange(1, n+1))
    df.to_csv(os.path.join(out_dir, 'results_summary_{}.txt'.format(study)), sep='\t')

def plots_individual(time_truth, hr_truth, time_calc, hr_calc, calc_name, sub_name):
    
    fig, axs = plt.subplots(2, 1)
    #fig.set_size_inches((640/10, 480/10)) 
    fig.subplots_adjust(hspace=0.5)
                       
    fig.suptitle('Calculated file info for {}: {}'.format(sub_name, calc_name))
    axs[0].plot(time_truth, hr_truth)
    axs[1].plot(time_calc, hr_calc)
    axs[0].set_ylim(np.min(hr_truth)-np.std(hr_truth), np.max(hr_truth)+np.std(hr_truth))
    axs[1].set_ylim(np.min(hr_calc)-np.std(hr_calc), np.max(hr_calc)+np.std(hr_calc))
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Heart rate (BPM)')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Heart rate (BPM)')
    axs[0].set_title('Ground truth values')
    axs[1].set_title('Calculated values')
    axs[0].grid(True)
    axs[1].grid(True)
    fig.canvas.draw()
    out_dir = './results/'
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir,'{}_{}.png'.format(sub_name, calc_name)))
    
def metrics(time_truth, hr_truth, time_calc, hr_calc, calc_name, sub_name, printFlag):
    N = len(hr_truth)
    M = len(hr_calc)
    if (N == M):

        #root mean squared error
        rmse_top_root = np.sum(np.power(hr_truth - hr_calc,2))
        rmse = np.sqrt(rmse_top_root/N)
        #mean absolute error
        mae = np.sum(np.abs(hr_truth - hr_calc))/N
        
        mean_hr_truth = np.mean(hr_truth)
        mean_hr_calc = np.mean(hr_calc)
        #print(mean_hr_truth, mean_hr_calc)
        #covariance
        #definition from vision-based heart rate estimation wang 2019
        #pear_corr_top = np.sum(hr_truth - mean_hr_truth)*np.sum(hr_calc - mean_hr_calc)
        #definition from wikipedia
        pear_corr_top = np.sum((hr_truth - mean_hr_truth)*(hr_calc - mean_hr_calc))
        #standard deviations
        std_dev_truth = np.sqrt(np.sum(np.power(hr_truth - mean_hr_truth,2)))
        std_dev_calc = np.sqrt(np.sum(np.power(hr_calc - mean_hr_calc,2)))
        #print(std_dev_truth, std_dev_calc)
        pear_corr = pear_corr_top/(std_dev_truth*std_dev_calc)
            
        if (printFlag):
            print('Results for {}, window size: {}'.format(sub_name, calc_name))
            print('RMSE: ', rmse)
            print('MAE: ', mae)
            print('Pearsons: ', pear_corr)
        
    return rmse, mae, pear_corr

def load_HR(path_calc_str, path_truth_str, low_time):

    # Problems that I see with reading these CSV files is that in average they have around 2 K something samples, which means
    # a sampling rate of 30 Hz, pretty similar to the fps but it doesn't exactly match the frames
    
    # read the ground_truth and the calculated values and store them
    path_calc = Path(path_calc_str)
    df_calc = pd.read_csv(path_calc, delim_whitespace=True)
    df_calc.columns = df_calc.columns.str.lower() #MAKE SURE ALL BPMs are written in LOWER CASE
    time_calc = np.array(df_calc['time']/1000) #ms to seconds
    hr_calc = np.array(df_calc['bpm']) 

    path_truth =  Path(path_truth_str)
    df_truth = pd.read_csv(path_truth, header=None, delim_whitespace=True)
    hr_truth = np.array(df_truth.iloc[1])
    time_truth = np.array(df_truth.iloc[2])

    base = os.path.basename(os.path.normpath(path_calc))
    calc_name = os.path.splitext(base)[0] 
    base_gt = os.path.basename(os.path.dirname(os.path.normpath(path_truth)))
    sub_name = os.path.splitext(base_gt)[0]
    
    #print(hr_truth.shape, time_truth.shape, hr_calc.shape, time_calc.shape)
    #print(hr_truth.dtype, time_truth.dtype, hr_calc.dtype, time_calc.dtype)
    #print(time_truth[0], time_calc[0], time_truth[-1], time_calc[-1])

    #section the dataframes to only include values between
    # low_time s and high_time s
    high_time = time_calc[-20] #a few frames before the end of the video
    low_slice_calc = get_closests(time_calc, low_time)
    high_slice_calc = get_closests(time_calc, high_time)           
    low_slice_true = get_closests(time_truth, low_time)
    high_slice_true = get_closests(time_truth, high_time)
    #print(low_slice_calc, high_slice_calc, low_slice_true, high_slice_true)

    hr_calc = hr_calc[low_slice_calc:high_slice_calc]
    time_calc = time_calc[low_slice_calc:high_slice_calc]
    hr_truth = hr_truth[low_slice_true:high_slice_true]
    time_truth = time_truth[low_slice_true:high_slice_true]

    #print(hr_truth.shape, time_truth.shape, hr_calc.shape, time_calc.shape)
    #print(time_truth[0], time_calc[0], time_truth[-1], time_calc[-1])

    len_calc_orig = len(hr_calc)
    len_true_orig = len(hr_truth)

    time_us, hr_us, time_ss, hr_ss, len_same, len_bigger = undersampling(time_calc, hr_calc, time_truth, hr_truth, len_calc_orig, len_true_orig)
    
    #print(hr_us.shape, time_us.shape, hr_ss.shape, time_ss.shape)
    #print(time_us[0], time_ss[0], time_us[-1], time_ss[-1])

    if len_calc_orig == len_same:
        hr_calc_f = hr_ss
        time_calc_f = time_ss
        hr_tr_f = hr_us
        time_tr_f = time_us
    else:
        hr_calc_f = hr_us
        time_calc_f = time_us
        hr_tr_f = hr_ss
        time_tr_f = time_ss

    return time_tr_f, hr_tr_f, time_calc_f, hr_calc_f, calc_name, sub_name
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-low', required=False, dest='low', default=23, type=int, 
    help='Time to begin analysis in seconds. Must be higher than the window size. Default is 23.')
    parser.add_argument('-r', required=True, dest='res', type=str, help='Path (absolute or relative to folder containing results.)')
    parser.add_argument('-print', dest='printFlag', default=False, type=bool, help='''Print individual accuracy metrics or not. 
    True or False. Default is False''')
    parser.add_argument('-study', required=True, dest='study', type=str, 
    help='Nature of analysis. Describe in few words what the comparison is about (asides from window sizes).')
    args = parser.parse_args()
    
    p = Path(args.res)
    bpm_paths = []
    truth_paths = []
    i = 0
    for x in p.iterdir():
        bpm_paths.append([])
        if x.is_dir(): #subject list level
            for y in x.iterdir(): #subject data level
                if y.is_dir():
                    for z in y.iterdir():
                        if (PurePath(z).match('bpm/*.txt')):
                            bpm_paths[i].append(z)
                else:
                    truth_paths.append(y)
            i += 1
    
    print('No of subjects to go through: ', len(truth_paths))
    #print(truth_paths)
    #print(bpm_paths)         
       
    rmse_sub = []; mae_sub = []; 
    pear_sub = []; 

    for subject, val in enumerate(truth_paths):
        rmse_sub.append([])
        mae_sub.append([])
        pear_sub.append([])
        for windowSize, val in enumerate(bpm_paths[subject]):
            calc = bpm_paths[subject][windowSize]
            truth = truth_paths[subject]
            time_tr_f, hr_tr_f, time_calc_f, hr_calc_f, calc_name, sub_name = load_HR(calc, truth, args.low)
            #Calculate metrics for each individual and window size
            #print('Calculating for {}'.format(bpm_paths[subject][windowSize]))
            rmse_temp, mae_temp, pear_temp = metrics(time_tr_f, hr_tr_f, time_calc_f, hr_calc_f,
            calc_name, sub_name, args.printFlag)
            rmse_sub[subject].append(rmse_temp)
            mae_sub[subject].append(mae_temp)
            pear_sub[subject].append(pear_temp)
            #Plot for each individual and window size
            if (args.printFlag == True):
                plots_individual(time_tr_f, hr_tr_f, time_calc_f, hr_calc_f, calc_name, sub_name)  

    #average accuracy measurements through subjects
    rmse_avg = np.reshape(np.mean(np.asarray(rmse_sub), axis=0),(-1,1))
    mae_avg = np.reshape(np.mean(np.asarray(mae_sub), axis=0),(-1,1))
    pear_avg = np.reshape(np.mean(np.asarray(pear_sub), axis=0),(-1,1))
    
    #Plot metrics averaged over individual as a function of window size
    plots_metrics(rmse_avg, mae_avg, pear_avg, args.study) #all subjects    

main()
